"""
PRO COA AI Agent — Reasoning Engine
University of Cambridge × Evinova (AstraZeneca)

Architecture:
  Step 1: Analyzer (Haiku)      — extracts + infers trial parameters from free text
  Step 2: KG Queries            — retrieves evidence from Neo4j
  Step 3: Scoring Engine        — programmatic 100-pt regulatory scale
  Step 4: Battery Optimizer     — selects non-redundant instrument combination
  Step 5: Narrative Cleaning    — Haiku cleans messy KG text fields
  Step 6: Reasoner (Sonnet)     — synthesises KG evidence + live web search
  Step 7: Logging               — saves every query for evaluation
"""

import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from graph import Neo4jConnection

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)


# =============================================================================
# CONSTANTS — INDICATION-SPECIFIC CORE DOMAINS
# Source: FDA (2021) "Core Patient-Reported Outcomes in Cancer Clinical Trials"
# =============================================================================
INDICATION_CORE_DOMAINS = {
    "multiple myeloma": ["bone pain", "physical function", "fatigue"],
    "mm": ["bone pain", "physical function", "fatigue"],
    "rrmm": ["bone pain", "physical function", "fatigue", "treatment tolerability"],
    "nsclc": ["dyspnea", "cough", "chest pain", "physical function"],
    "non-small cell lung": ["dyspnea", "cough", "chest pain", "physical function"],
    "lung cancer": ["dyspnea", "cough", "physical function"],
    "crpc": ["pain", "urinary function", "physical function"],
    "prostate cancer": ["pain", "urinary function", "physical function"],
    "metastatic castration-resistant": ["pain", "urinary function", "physical function"],
    "breast cancer": ["fatigue", "pain", "physical function", "emotional function"],
    "colorectal": ["nausea", "appetite loss", "bowel function", "fatigue"],
    "crc": ["nausea", "appetite loss", "bowel function", "fatigue"],
    "ovarian": ["abdominal pain", "bloating", "fatigue", "physical function"],
    "lymphoma": ["fatigue", "night sweats", "physical function"],
    "leukemia": ["fatigue", "physical function", "emotional function"],
    "aml": ["fatigue", "physical function", "emotional function"],
    "default": ["physical function", "fatigue", "pain"]
}


# =============================================================================
# CONSTANTS — HTA/PAYER INSTRUMENT PREFERENCES
# Sources cited per entry
# =============================================================================
HTA_PREFERENCES = {
    "NICE": {
        "required_instruments": ["EQ-5D"],   # Wildcard — EQ-5D-5L OR EQ-5D-3L satisfies this
        "preferred_version": "EQ-5D-5L",
        "accepted_versions": ["EQ-5D-5L", "EQ-5D-3L"],
        "notes": (
            "NICE requires a preference-based EQ-5D measure for cost-utility analysis. "
            "EQ-5D-5L is preferred per NICE position statement (October 2019). "
            "Without EQ-5D, QALY calculation is impossible and UK reimbursement is severely compromised."
        ),
        "reference": "NICE DSU Technical Support Document 2 (2011, updated 2019); NICE EQ-5D-5L position statement (2019)"
    },
    "ICER": {
        "required_instruments": [],
        "preferred_instruments": ["EQ-5D-5L", "SF-36", "SF-6D"],
        "notes": (
            "ICER uses utility-based measures for cost-effectiveness analysis in US value assessments. "
            "EQ-5D-5L is strongly preferred for QALY calculation."
        ),
        "reference": "ICER Value Assessment Framework (2020)"
    },
    "EUnetHTA": {
        "required_instruments": [],
        "preferred_instruments": ["EQ-5D-5L", "EORTC QLQ-C30"],
        "notes": (
            "EU HTA Regulation 2021/2282 Joint Clinical Assessments increasingly require standardised "
            "PRO instruments for cross-country comparison. EQ-5D-5L required for HTA utility analysis."
        ),
        "reference": "EU HTA Regulation 2021/2282; EUnetHTA 21 methodology guidelines"
    },
    "SMC": {
        "required_instruments": ["EQ-5D"],
        "preferred_version": "EQ-5D-5L",
        "notes": "Scottish Medicines Consortium aligns with NICE on EQ-5D requirement.",
        "reference": "SMC Modifiers and PACE framework"
    }
}


# =============================================================================
# CONSTANTS — GEOGRAPHIC LANGUAGE REQUIREMENTS
# IMPORTANT: FDA does NOT specify a minimum number of languages.
# Source: FDA PRO Guidance (2009) Section IV.A — requires linguistically validated
# translations for languages used in the trial. No numeric minimum is stated.
# Source: EMA Reflection Paper on PRO (2005) — requires translations for each
# EU member state language where the trial is conducted.
# =============================================================================
GEOGRAPHIC_LANGUAGE_REQUIREMENTS = {
    "Global": {
        "min_languages": 15,
        "key_languages": [
            "English", "Spanish", "French", "German", "Italian",
            "Japanese", "Mandarin", "Portuguese", "Russian", "Korean",
            "Polish", "Dutch", "Swedish", "Turkish", "Arabic"
        ],
        "regulatory_note": (
            "FDA PRO Guidance (2009) Section IV.A requires linguistically validated translations "
            "for each language used in the trial. There is no FDA-specified minimum number of languages. "
            "EMA requires translations for each EU member state language where the trial is conducted."
        ),
        "reference": "FDA PRO Guidance (2009) Section IV.A; EMA Reflection Paper on PRO (2005)"
    },
    "EU": {
        "min_languages": 10,
        "key_languages": [
            "English", "French", "German", "Spanish", "Italian",
            "Dutch", "Polish", "Swedish", "Danish", "Finnish",
            "Czech", "Romanian", "Hungarian", "Portuguese", "Greek"
        ],
        "regulatory_note": (
            "EMA requires validated translations for each member state language where the trial "
            "is conducted per EMA Reflection Paper on PRO (2005). "
            "EU HTA Regulation 2021/2282 requires standardised instruments for Joint Clinical Assessments."
        ),
        "reference": "EMA Reflection Paper on PRO (2005); EU HTA Regulation 2021/2282"
    },
    "US-only": {
        "min_languages": 1,
        "key_languages": ["English"],
        "regulatory_note": (
            "English linguistic validation required. "
            "If trial population includes non-English speakers, additional translations required "
            "per FDA PRO Guidance (2009) Section IV.A."
        ),
        "reference": "FDA PRO Guidance (2009) Section IV.A"
    }
}


# =============================================================================
# CONSTANTS — INSTRUMENT RECALL PERIODS (days)
# ONLY instruments with published, citable recall periods are listed.
# For instruments NOT listed: RECALL_PERIOD_UNKNOWN = -1 sentinel value.
# The recall bias penalty does NOT fire on unknown instruments —
# Sonnet is instructed to verify via web search.
# =============================================================================
RECALL_PERIOD_UNKNOWN = -1

INSTRUMENT_RECALL_PERIODS = {
    # EuroQol Group official documentation — "TODAY"
    "eq-5d": 0,
    "eq-5d-5l": 0,
    "eq-5d-3l": 0,
    # Cleeland & Ryan (1994) Pain 62(3):173-182 — "past 24 hours"
    "bpi-sf": 1,
    "bpi": 1,
    # Mendoza et al. (1999) Cancer 85(5):1186-1196 — "right now" and "past 24 hours"
    "bfi": 1,
    # IMMPACT recommendations Dworkin et al. (2005) Pain 113(1-2):9-19 — "right now"
    "nrs": 1,
    "vas": 1,
    # Clinician/patient global impression — current state or since last visit
    "pgis": 1,
    "pgic": 1,
    # NCI PRO-CTCAE User Manual v1.0 — "past 7 days"
    "pro-ctcae": 7,
    # FACIT.org official documentation — "past 7 days"
    "fact-p": 7,
    "fact-g": 7,
    "fact-b": 7,
    "fact-l": 7,
    "facit-fatigue": 7,
    # EORTC Quality of Life Group manual — "during the past week"
    "eortc qlq-c30": 7,
    "eortc qlq-lc13": 7,
    "eortc qlq-my20": 7,
    "eortc qlq-pr25": 7,
    "eortc qlq-hn35": 7,
    # Zigmond & Snaith (1983) Acta Psychiatr Scand 67(6):361-370 — "past week"
    "hads": 7,
    # Spitzer et al. (2006) Arch Intern Med 166(10):1092-1097 — "last 2 weeks"
    "gad-7": 14,
    # Kroenke et al. (2001) J Gen Intern Med 16(9):606-613 — "last 2 weeks"
    "phq-9": 14,
    # Ware & Sherbourne (1992) Med Care 30(6):473-483 — "past 4 weeks"
    "sf-36": 28,
    "sf-12": 28,
}


# =============================================================================
# CONSTANTS — KNOWN LANGUAGE COUNTS (approximate, for reporting)
# Sources: instrument developer documentation and published translations registries
# Used for REPORTING only — not for pass/fail thresholds
# =============================================================================
KNOWN_LANGUAGE_COUNTS = {
    "eq-5d": 150,     # EuroQol Group — 150+ validated translations
    "eq-5d-5l": 150,
    "eq-5d-3l": 150,
    "eortc qlq-c30": 100,   # EORTC — 100+ languages
    "eortc qlq-my20": 80,
    "fact-g": 60,           # FACIT.org — 60+ languages
    "fact-p": 60,
    "fact-b": 60,
    "bpi-sf": 40,           # MD Anderson — 40+ languages
    "bpi": 40,
    "pro-ctcae": 30,        # NCI — 30+ languages
    "sf-36": 80,            # QualityMetric — 80+ languages
    "sf-12": 80,
    "bfi": 9,               # MD Anderson — approximately 9 languages (Mendoza 1999 + translations)
    "hads": 30,             # Multiple translated versions available
    "pgis": 15,
    "pgic": 15,
}


# =============================================================================
# DOMAIN SYNONYM MAP
# Allows broad instruments stored as "HRQoL" to match "physical function" etc.
# Source: FDA (2021) Core PRO Guidance domain definitions
# =============================================================================
DOMAIN_SYNONYMS = {
    "bone pain": ["pain", "bone", "analgesic", "bpi", "nrs", "aches", "skeletal"],
    "physical function": ["physical", "function", "activity", "mobility", "performance", "adl", "karnofsky"],
    "fatigue": ["fatigue", "tiredness", "energy", "exhaustion", "asthenia", "vitality"],
    "dyspnea": ["dyspnea", "breathlessness", "breathing", "respiratory", "shortness of breath"],
    "cough": ["cough", "respiratory", "pulmonary"],
    "pain": ["pain", "analgesic", "bpi", "nrs", "aches", "discomfort", "bone"],
    "nausea": ["nausea", "vomiting", "gi", "gastrointestinal", "emesis"],
    "urinary function": ["urinary", "urology", "bladder", "ipss", "micturition"],
    "emotional function": ["emotional", "anxiety", "depression", "psychological", "mental", "hads", "phq"],
    "appetite loss": ["appetite", "anorexia", "eating", "weight"],
    "bowel function": ["bowel", "diarrhoea", "constipation", "gastrointestinal"],
    "treatment tolerability": ["tolerability", "adverse", "toxicity", "ctcae", "symptom", "side effect", "crs", "cytokine release", "icans"],
"cytokine release syndrome (crs) symptoms": ["crs", "cytokine", "ctcae", "icans", "pro-ctcae", "tolerability", "adverse"],
    "hrqol": ["hrqol", "quality of life", "health-related", "wellbeing", "function"],
    "disease-specific symptoms": ["disease", "specific", "myeloma", "cancer-specific", "my20", "symptom"],
}


# =============================================================================
# KNOWN BROAD INSTRUMENTS (implicitly cover all core domains for their indication)
# =============================================================================
BROAD_INSTRUMENTS = {
    "eortc qlq-c30": ["hrqol", "physical function", "emotional function", "fatigue", "pain", "nausea"],
    "eortc qlq-my20": ["bone pain", "physical function", "fatigue", "disease-specific symptoms"],
    "fact-g": ["physical function", "emotional function", "fatigue", "hrqol"],
    "fact-p": ["pain", "physical function", "emotional function", "fatigue", "hrqol"],
    "sf-36": ["physical function", "emotional function", "fatigue", "pain", "hrqol"],
    "promis": ["physical function", "fatigue", "pain", "emotional function"],
    "eq-5d": ["physical function", "pain", "hrqol", "emotional function"],
    "eq-5d-5l": ["physical function", "pain", "hrqol", "emotional function"],
}


# =============================================================================
# GLOSSARY LOADING
# =============================================================================
GLOSSARY_TEXT = ""
try:
    glossary_df = pd.read_csv("PRO_Terminology_Glossary.csv")
    if "Importance_Rank" in glossary_df.columns:
        glossary_df = glossary_df.sort_values("Importance_Rank")
    rows = []
    for _, row in glossary_df.head(20).iterrows():
        rows.append(" | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val)))
    GLOSSARY_TEXT = "\n".join(rows)
except Exception as e:
    GLOSSARY_TEXT = "Glossary unavailable. Use standard COA terminology."
    logging.warning(f"Glossary load failed: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_full_stop(text: str) -> str:
    """Ensure text ends with a full stop."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def clean_mcid(raw_mcid) -> tuple:
    """
    Clean a raw MCID string from the KG into (display_value, full_text).
    Removes PMC IDs, patient context, and returns a clean numeric threshold.
    """
    if not raw_mcid:
        return "", ""
    text = " ".join(str(m) for m in raw_mcid) if isinstance(raw_mcid, list) else str(raw_mcid)
    text = text.strip()
    if text.lower() in ["none", "nan", "", "null", "not established", "unknown", "n/a", "tbd"]:
        return "", ""
    # Remove PMC references: pmc12345678 (2024)
    text_clean = re.sub(r'\bpmc\d+\b\s*\(\d{4}\)', '', text, flags=re.IGNORECASE).strip()
    # Remove "in [X] cancer patients" context
    text_clean = re.sub(r'\s+in\s+[\w/\s]+cancer\s+patients?.*$', '', text_clean, flags=re.IGNORECASE).strip()
    text_clean = re.sub(r'\s+in\s+[\w/\s]+patients?.*$', '', text_clean, flags=re.IGNORECASE).strip()
    # Extract first numeric threshold
    match = re.search(r'(\d+\.?\d*)\s*(points?|units?|%|score)', text_clean, re.IGNORECASE)
    if match:
        for sentence in re.split(r'[.\n]', text_clean):
            if match.group(0).lower() in sentence.lower():
                short = sentence.strip().rstrip(',;').strip()
                return (short[:100] + "..." if len(short) > 100 else short), text_clean
    short = text_clean[:80].rstrip(',;').strip()
    return (short + "..." if len(text_clean) > 80 else short), text_clean


def _to_str(value) -> str:
    """Safely convert any KG field to a lowercase string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()

# =============================================================================
# HAIKU SYSTEM PROMPT
# =============================================================================
HAIKU_SYSTEM_PROMPT = """You are a clinical trial parameter extractor for oncology COA strategy.
Parse the user input into a strict JSON object.
For any missing field, infer the most likely 2026 industry standard based on context,
flag it in 'assumptions_made', and explain your reasoning.

Core Domain Lookup (from FDA 2021 Core PRO Guidance):
- Multiple Myeloma / RRMM: bone pain, physical function, fatigue, treatment tolerability
- NSCLC: dyspnea, cough, chest pain, physical function
- CRPC/Prostate Cancer: pain, urinary function, physical function
- Breast Cancer: fatigue, pain, physical function, emotional function
- Colorectal Cancer: nausea, appetite loss, bowel function, fatigue
- Ovarian Cancer: abdominal pain, bloating, fatigue, physical function
- Default (unknown oncology): physical function, fatigue, pain

IMPORTANT — population_subtype: Use the exact clinical term from the input.
Do NOT reduce clinical terms to generic categories.
Examples: "Relapsed/Refractory" NOT "Symptomatic", "Newly Diagnosed" NOT "Symptomatic",
"Smoldering" NOT "Asymptomatic/Smoldering".
Use "Symptomatic" only when no specific clinical term is available.

Inference Rules:
1. If tpp_claims missing: output ["treatment tolerability", "physical function maintenance"] + note
2. If population_subtype missing: default to "Symptomatic" + note
3. If phase missing: default to "Phase 3" + note
4. If drug_class is Bispecific or CAR-T: infer administration as "Step-up dosing" + note
5. If geographic_footprint missing: Phase 3 = "Global", Phase 2 = "EU", Phase 1 = "US-only"
6. If hta_markets missing: Global = ["NICE","ICER","EUnetHTA"], EU = ["NICE","EUnetHTA"], US = ["ICER"]
When inferring missing information, explain the reasoning in plain clinical language.
Do not reference numbered rules. Instead, explain WHY the inference was made.
Examples of good assumption or inference text:
- "Administration inferred as Step-up dosing: BCMA bispecific antibodies require step-up dosing to manage cytokine release syndrome risk, as established across all FDA-approved T-cell engaging bispecifics including teclistamab, elranatamab, and linvoseltamab. Alternative dosing would change CRS monitoring requirements."
- "EUnetHTA added to HTA markets: A global Phase 3 trial with EMA approval goal falls under EU Regulation 2021/2282, which requires Joint Clinical Assessments for new medicines. EUnetHTA is the body coordinating these assessments and has specific PRO instrument requirements for cross-country comparison."
- "Geographic footprint inferred as Global: Phase 3 oncology trials seeking both FDA and EMA approval are conducted across multiple continents by standard practice, necessitating translations for all major trial site languages."

Return ONLY valid JSON. No markdown fences. No explanation outside the JSON."""


# =============================================================================
# STEP 1: ANALYZER
# =============================================================================
def analyze_trial_context(user_text: str) -> dict:
    """Extract trial parameters using Claude Haiku. Returns structured JSON context."""
    expected_format = """{
  "indication": "primary cancer type",
  "indication_synonyms": ["synonyms for KG search, e.g. MM, RRMM, myeloma"],
  "population_subtype": "exact clinical term e.g. Relapsed/Refractory, Newly Diagnosed, Smoldering",
  "phase": "Phase 1 | Phase 2 | Phase 3",
  "drug_class": "e.g. Bispecific, Proteasome Inhibitor, ICI, CDK4/6 inhibitor",
  "administration": "Step-up dosing | Subcutaneous | IV | Oral | Unknown",
  "dosing_frequency": "Weekly | Biweekly | Monthly | Unknown",
  "tpp_claims": ["desired label claims — be specific"],
  "core_domains_required": ["indication-specific core domains PLUS domains needed to prove TPP claims"],
  "geographic_footprint": "Global | EU | US-only | Unknown",
  "hta_markets": ["NICE", "ICER", "EUnetHTA", "SMC"],
  "trial_duration_cycles": "number or Unknown",
  "assumptions_made": ["each inference with reasoning"]
}"""

    _default = {
        "indication": "unknown",
        "indication_synonyms": [],
        "population_subtype": "Symptomatic",
        "phase": "Phase 3",
        "drug_class": "Unknown",
        "administration": "Unknown",
        "tpp_claims": ["treatment tolerability", "physical function maintenance"],
        "core_domains_required": ["physical function", "fatigue", "pain"],
        "geographic_footprint": "Global",
        "hta_markets": ["NICE", "ICER", "EUnetHTA"],
        "assumptions_made": ["All defaults applied — analyzer failed."],
        "dosing_frequency": "Unknown",
        "trial_duration_cycles": "Unknown"
    }

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=HAIKU_SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                f"Extract trial parameters. Return only JSON.\n\n{user_text}\n\nFormat:\n{expected_format}"}]
        )
        raw = response.content[0].text.strip().replace("```json","").replace("```","").strip()
        context = json.loads(raw)
        logging.info(f"Analyzer: indication={context.get('indication')} phase={context.get('phase')}")
        return context
    except json.JSONDecodeError as e:
        logging.error(f"Analyzer JSON parse failed: {e}")
        _default["assumptions_made"] = [f"JSON parse failed — all defaults applied."]
        return _default
    except Exception as e:
        logging.error(f"Analyzer API failed: {e}")
        _default["assumptions_made"] = [f"Analyzer API failed ({e}) — all defaults applied."]
        return _default


# =============================================================================
# STEP 2 — UTILITY: KG WRAPPERS
# =============================================================================
def _get_conn():
    return Neo4jConnection(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )


def get_instruments_by_indication(indication="", phase="", endpoint=""):
    try:
        conn = _get_conn()
        try:
            return conn.get_instruments_by_indication(
                indications=[indication] if indication else [""],
                phase=phase, endpoint=endpoint
            )
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_instruments_by_indication: {e}")
        return []


def get_regulatory_evidence(indication="", agency=""):
    try:
        conn = _get_conn()
        try:
            return conn.get_regulatory_evidence(
                indications=[indication] if indication else [""],
                agency=agency
            )
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_regulatory_evidence: {e}")
        return []


def get_instrument_reference(instrument_name=""):
    try:
        conn = _get_conn()
        try:
            return conn.get_instrument_reference(instrument_name=instrument_name)
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_instrument_reference: {e}")
        return []


def get_regulatory_rules(indication="", lifecycle_stage="", decision_type=""):
    try:
        conn = _get_conn()
        try:
            return conn.get_regulatory_rules(
                indication=indication,
                lifecycle_stage=lifecycle_stage,
                decision_type=decision_type
            )
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_regulatory_rules: {e}")
        return []


def get_regulatory_evidence_for_instrument(instrument_name=""):
    try:
        conn = _get_conn()
        try:
            return conn.get_regulatory_evidence_for_instrument(instrument_name=instrument_name)
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_regulatory_evidence_for_instrument: {e}")
        return []


# =============================================================================
# STEP 3: SCORING ENGINE
# =============================================================================
def score_evidence(context_json: dict, kg_records: list) -> list:
    """
    Score each instrument on a 0-100 scientific scale plus operational bonuses.
    All penalties are evidence-based with regulatory citations.
    Score is floored at 0. Risk Level set independently to carry penalty severity.
    """
    indication = _to_str(context_json.get("indication"))
    population = str(context_json.get("population_subtype", "Symptomatic"))
    phase = str(context_json.get("phase", "Phase 3"))
    administration = str(context_json.get("administration", "Unknown"))
    tpp_claims = [
        str(c).lower().replace("inferred: ", "").strip()
        for c in context_json.get("tpp_claims", [])
    ]
    core_domains = [str(d).lower() for d in context_json.get("core_domains_required", [])]
    geographic_footprint = str(context_json.get("geographic_footprint", "Global"))
    hta_markets = context_json.get("hta_markets", [])
    drug_class = _to_str(context_json.get("drug_class"))

    results = []

    for record in kg_records:
        instrument_name = str(record.get("instrument_name", "Unknown"))
        instrument_lower = instrument_name.lower()

        # Build comprehensive domain search text from multiple fields
        domain_search_parts = [
            _to_str(record.get("instrument_domain")),
            _to_str(record.get("domains_measured")),
            _to_str(record.get("key_finding")),
            _to_str(record.get("subscale_results")),
            _to_str(record.get("instrument_subscales_assessed")),
            _to_str(record.get("strengths")),
            # Add known broad instrument domains
            " ".join(BROAD_INSTRUMENTS.get(instrument_lower, [])),
        ]
        instrument_domains = " ".join(domain_search_parts)
        instrument_domains_list = [
            d.strip().lower()
            for part in domain_search_parts
            for d in re.split(r'[|,;]', part) if d.strip()
        ]

        mode_options = _to_str(record.get("mode_options"))
        source_documents = _to_str(record.get("source_documents"))
        developer_str = _to_str(record.get("developer"))
        endpoint_role = _to_str(record.get("endpoint_role"))
        prespecified = _to_str(record.get("prespecified"))
        significance = _to_str(record.get("significance"))
        mid_met = _to_str(record.get("mid_met"))
        direction = _to_str(record.get("direction"))
        regulatory_acceptance = _to_str(record.get("regulatory_acceptance"))
        fda_alignment = _to_str(record.get("fda_alignment"))
        drug_class_match = _to_str(record.get("disease_classification"))

        # total_items: extract integer via regex
        total_items_raw = str(record.get("total_items", ""))
        num_match = re.search(r'\d+', total_items_raw)
        total_items = int(num_match.group()) if num_match else 0

        # Recall period: look up by instrument name; unknown = sentinel -1
        recall_period = next(
            (days for key, days in INSTRUMENT_RECALL_PERIODS.items() if key in instrument_lower),
            RECALL_PERIOD_UNKNOWN
        )
        recall_period_key = next(
            (key for key in INSTRUMENT_RECALL_PERIODS if key in instrument_lower),
            None
        )

        # Language count: use known counts first, then KG data
        known_lang = next(
            (count for key, count in KNOWN_LANGUAGE_COUNTS.items() if key in instrument_lower),
            None
        )
        if known_lang is not None:
            language_count = known_lang
        else:
            languages_val = record.get("languages", "")
            languages_str = str(languages_val).lower()
            if "85+" in languages_str or "100+" in languages_str or "all major" in languages_str:
                language_count = 100
            elif isinstance(languages_val, list):
                language_count = len([l for l in languages_val if l])
            else:
                language_count = len([l for l in languages_str.split("|") if l.strip()])

        # --- MCID ---
        raw_mcid = record.get("mcid", "")
        mcid_str = _to_str(raw_mcid)
        mcid_null = ["none", "not established", "unknown", "nan", "n/a", "tbd", "not reported", "pending", "null", ""]
        mcid_valid = (
            mcid_str.strip() != "" and
            mcid_str.strip() not in mcid_null and
            not any(t in mcid_str for t in ["not established", "not reported", "unknown", "pending"])
        )
        mcid_display, _ = clean_mcid(raw_mcid)

        raw_score = 0
        operational_bonus = 0
        flags = []

        # ── POSITIVE WEIGHTS (max 100) ────────────────────────────

        # 1. TPP / Core Domain Fit (+35)
        # Source: FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials
        tpp_match = False
        # Check TPP claims against instrument domains using synonym expansion
        for claim in tpp_claims:
            claim_synonyms = DOMAIN_SYNONYMS.get(claim, [claim])
            all_claim_terms = [claim] + claim_synonyms
            if any(term in instrument_domains for term in all_claim_terms):
                tpp_match = True
                break
        # Check core domains using synonym expansion
        if not tpp_match:
            for domain in core_domains:
                domain_synonyms = DOMAIN_SYNONYMS.get(domain, [domain])
                all_domain_terms = [domain] + domain_synonyms
                if any(term in instrument_domains for term in all_domain_terms):
                    tpp_match = True
                    break
        if tpp_match:
            raw_score += 35
            flags.append(ensure_full_stop(
                "TPP/Core Fit (+35): Instrument domains align with TPP claims and "
                "FDA-defined core domains for this indication "
                "[FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials]."
            ))

        # 2. Regulatory Trust (+25)
        # Source: FDA PRO Guidance (2009) Section V; EMA Reflection Paper on PRO (2005)
        if any(t in regulatory_acceptance for t in ["fda", "ema", "accepted", "approved", "strong"]):
            raw_score += 25
            flags.append(ensure_full_stop(
                "Regulatory Trust (+25): Instrument has documented FDA/EMA regulatory acceptance "
                "[FDA PRO Guidance (2009) Section V; EMA Reflection Paper on PRO (2005)]."
            ))
        elif any(t in regulatory_acceptance for t in ["moderate", "conditional", "exploratory"]):
            raw_score += 12
            flags.append(ensure_full_stop(
                "Regulatory Trust (+12, partial): Moderate regulatory acceptance documented."
            ))

        # 3. Competitor / SoC Benchmark (+20)
        # Source: FDA PRO Guidance (2009) Section III.B — prior regulatory familiarity
        # Check by disease classification AND by trial prevalence
        soc_match = (
            (drug_class and drug_class_match and (drug_class in drug_class_match or drug_class_match in drug_class))
            or (record.get("trial_prevalence") and "high" in _to_str(record.get("trial_prevalence")))
        )
        if soc_match:
            raw_score += 20
            flags.append(ensure_full_stop(
                "Competitor Benchmark (+20): Instrument used in standard-of-care or high-prevalence trials "
                "— indicates regulatory familiarity "
                "[FDA PRO Guidance (2009) Section III.B]."
            ))

        # 4. MoA-Specific Sensitivity (+20)
        # Source: FDA PFDD Guidance 1 (2017) — instrument must be sensitive to drug mechanism
        MOA_KEYWORDS = {
            "bispecific": ["cytokine release", "crs", "fatigue", "neurotoxicity", "icans", "infection"],
            "car-t": ["cytokine release", "crs", "fatigue", "neurotoxicity", "icans"],
            "proteasome inhibitor": ["peripheral neuropathy", "neuropathy", "fatigue"],
            "ici": ["fatigue", "immune-related", "diarrhea", "endocrine", "colitis"],
            "cdk4/6": ["fatigue", "nausea", "neutropenia"],
            "antibody drug conjugate": ["nausea", "fatigue", "neuropathy", "alopecia"],
            "bcma": ["fatigue", "infection", "crs", "neurotoxicity", "cytokine release"],
        }
        moa_fired = False
        for class_key, tox_domains in MOA_KEYWORDS.items():
            if class_key in drug_class:
                if any(tox in instrument_domains for tox in tox_domains):
                    raw_score += 20
                    flags.append(ensure_full_stop(
                        f"MoA Sensitivity (+20): Instrument captures class-specific toxicity domains "
                        f"for {drug_class} "
                        f"[FDA PFDD Guidance 1 (2017)]."
                    ))
                    moa_fired = True
                    break

        # 5. Validated MCID (+10)
        # Source: FDA PRO Guidance (2009) Section V.C
        if mcid_valid and mcid_display:
            raw_score += 10
            flags.append(ensure_full_stop(
                f"Validated MCID (+10): MCID established — {mcid_display}. "
                f"Enables responder analysis required for label claims "
                f"[FDA PRO Guidance (2009) Section V.C]."
            ))
        elif mcid_valid:
            raw_score += 10
            flags.append(ensure_full_stop(
                "Validated MCID (+10): MCID established. "
                "Enables responder analysis [FDA PRO Guidance (2009) Section V.C]."
            ))
        else:
            flags.append(ensure_full_stop(
                "No Validated MCID: MCID not established — responder analysis impossible, "
                "limiting label claim language to mean change statistics "
                "[FDA PRO Guidance (2009) Section V.C]."
            ))

        raw_score = min(raw_score, 100)

        # ── CONDITIONAL PENALTIES ─────────────────────────────────
        penalty_total = 0
        risk_level = "LOW"

        # PENALTY 1: Missing Core (-50, CRITICAL)
        # Source: FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials
        # Applies to symptomatic populations AND clinical terms that imply active disease.
        SYMPTOMATIC_TERMS = [
            "symptomatic", "relapsed", "refractory", "relapsed/refractory",
            "metastatic", "advanced", "progressive", "active disease",
            "previously treated", "heavily pretreated", "rrmm", "rrbc",
            "first-line", "second-line", "later-line", "newly diagnosed",
            "treatment-naive"
        ]
        is_symptomatic = any(term in population.lower() for term in SYMPTOMATIC_TERMS)

        # HTA utility instruments (EQ-5D, SF-6D) are exempt from Missing Core penalty.
        # They are not intended to be the primary disease measurement instrument —
        # they provide health utility values for cost-effectiveness analysis.
        HTA_UTILITY_INSTRUMENTS = ["eq-5d", "eq5d", "sf-6d", "sf-36"]
        is_hta_utility = any(h in instrument_lower for h in HTA_UTILITY_INSTRUMENTS)

        if is_symptomatic and core_domains and not is_hta_utility:
            missing_cores = []
            for domain in core_domains:
                domain_synonyms = DOMAIN_SYNONYMS.get(domain.lower(), [domain])
                all_terms = [domain] + domain_synonyms
                found = any(term in instrument_domains for term in all_terms)
                if not found:
                    missing_cores.append(domain)
            if len(missing_cores) >= len(core_domains) / 2:
                penalty_total += 50
                risk_level = "CRITICAL"
                flags.append(ensure_full_stop(
                    f"MISSING CORE PENALTY (-50, CRITICAL): Instrument does not measure "
                    f"required core domains: {missing_cores}. "
                    f"Per FDA (2021) 'Core Patient-Reported Outcomes in Cancer Clinical Trials', "
                    f"failure to measure core domains risks Refusal to File or PRO label claim rejection. "
                    f"This instrument should NOT be ranked first for this indication."
                ))

        # PENALTY 2: Recall Bias (-40, CRITICAL)
        # Source: FDA PFDD Guidance 3 (2022) — recall period must match symptom fluctuation
        # Threshold: >7 days is incompatible with step-up dosing (CRS events within 24-72h)
        # ONLY fires for KNOWN recall periods. Unknown instruments are flagged for Sonnet to verify.
        STEP_UP_ADMINS = ["step-up dosing", "weekly iv", "weekly"]
        is_step_up = any(a in administration.lower() for a in STEP_UP_ADMINS)

        if is_step_up:
            if recall_period == RECALL_PERIOD_UNKNOWN:
                flags.append(ensure_full_stop(
                    f"RECALL PERIOD UNKNOWN: {instrument_name} recall period not in reference database. "
                    f"For {administration}, FDA PFDD Guidance 3 (2022) requires recall to match "
                    f"symptom fluctuation — CRS/ICANS events occur within 24-72 hours of dosing. "
                    f"Sonnet has been instructed to verify the official recall period via web search."
                ))
            elif recall_period > 7:
                penalty_total += 40
                risk_level = "CRITICAL"
                citation = f"per {recall_period_key} validation (see INSTRUMENT_RECALL_PERIODS)" if recall_period_key else "per published validation"
                flags.append(ensure_full_stop(
                    f"RECALL BIAS PENALTY (-40, CRITICAL): {recall_period}-day recall period "
                    f"incompatible with {administration} ({citation}). "
                    f"CRS/ICANS events occur within 24-72 hours of dosing. "
                    f"Per FDA PFDD Guidance 3 (2022), recall must match symptom fluctuation pattern. "
                    f"Data would likely be characterised as exploratory-only by FDA."
                ))
            else:
                source = recall_period_key or "published validation"
                flags.append(ensure_full_stop(
                    f"Recall period compatible: {instrument_name} has {recall_period}-day recall "
                    f"({source}), compatible with {administration} "
                    f"[FDA PFDD Guidance 3 (2022)]."
                ))

        # PENALTY 3: Pre-specification / Alpha Control (-35, HIGH)
        # Source: FDA PRO Guidance (2009) Section V; ICH E9 (1998) Section 2.2.5
        # Only fires when KG record EXPLICITLY shows non-pre-specification
        has_explicit_record = instrument_name != "Unknown" and (prespecified != "" or endpoint_role != "")
        if has_explicit_record and prespecified not in ["yes", "true", "1"] and endpoint_role in ["exploratory", "unknown"]:
            penalty_total += 35
            if risk_level not in ["CRITICAL"]:
                risk_level = "HIGH"
            flags.append(ensure_full_stop(
                "PRE-SPECIFICATION PENALTY (-35, HIGH): KG record shows instrument was not "
                "pre-specified in SAP with alpha controlled. Results will be exploratory only — "
                "cannot support formal label claims "
                "[FDA PRO Guidance (2009) Section V; ICH E9 (1998) Section 2.2.5]. "
                "This is the most common cause of PRO data failing to reach the label, "
                "as documented in the abiraterone COU-AA-301/302 precedent in this KG."
            ))

        # PENALTY 4: Estimand Burden (-30, HIGH)
        # Source: ICH E9(R1) Addendum (2019); FDA PRO Guidance (2009) Section IV.B
        if ("phase 3" in phase.lower() or "phase iii" in phase.lower()) and total_items > 30:
            penalty_total += 30
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "HIGH"
            flags.append(ensure_full_stop(
                f"ESTIMAND BURDEN PENALTY (-30, HIGH): {total_items}-item instrument in Phase 3. "
                f"ICH E9(R1) Addendum (2019) requires Treatment Policy estimand — PRO collection "
                f"must continue post-discontinuation. Instruments >30 items show lower completion "
                f"rates in this setting, generating missing data. "
                f"Consider subscale approach or shorter companion instrument."
            ))

        # PENALTY 5: No Validated MCID (-20, MODERATE)
        # Source: FDA PRO Guidance (2009) Section V.C
        if not mcid_valid:
            penalty_total += 20
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "MODERATE"
            flags.append(ensure_full_stop(
                "NO MCID PENALTY (-20, MODERATE): No validated MCID established. "
                "Without MCID, responder analysis is impossible and label language is restricted "
                "to mean change statistics — a weaker regulatory argument "
                "[FDA PRO Guidance (2009) Section V.C]."
            ))

        # PENALTY 6: Asymptomatic Burden (-20, MODERATE)
        # Source: FDA PRO Guidance (2009) Section IV.B; FDA PFDD Guidance 2 (2018)
        SYMPTOM_HEAVY = ["bpi", "bone pain", "nrs", "pain intensity", "symptom"]
        is_symptom_heavy = any(s in instrument_lower or s in instrument_domains for s in SYMPTOM_HEAVY)
        if "asymptomatic" in population.lower() or "smoldering" in population.lower():
            if is_symptom_heavy:
                penalty_total += 20
                if risk_level not in ["CRITICAL", "HIGH"]:
                    risk_level = "MODERATE"
                flags.append(ensure_full_stop(
                    "ASYMPTOMATIC BURDEN PENALTY (-20, MODERATE): Symptom-heavy instrument "
                    "applied to asymptomatic population. "
                    "Measuring symptoms the patient does not have causes questionnaire fatigue "
                    "[FDA PRO Guidance (2009) Section IV.B; FDA PFDD Guidance 2 (2018)]. "
                    "Consider HRQoL-focused instrument (EQ-5D-5L, FACT-G)."
                ))

        # ── OPERATIONAL BONUSES (independent of 100-pt cap) ──────

        # eCOA Ready (+10)
        # Source: FDA eCOA Guidance (2023)
        if any(t in mode_options for t in ["ecoa", "electronic", "app", "tablet", "digital"]):
            operational_bonus += 10
            flags.append(ensure_full_stop(
                "eCOA Ready (+10 operational): Electronic mode supported — "
                "reduces transcription error and enables real-time monitoring "
                "[FDA eCOA Guidance (2023)]."
            ))

        # Open Access (+5)
        OPEN_ACCESS_DEVS = [
            "eortc", "nci", "national cancer institute", "facit",
            "rand", "who", "world health organization", "nih", "pcori"
        ]
        if any(d in developer_str or d in source_documents or d in instrument_lower
               for d in OPEN_ACCESS_DEVS):
            operational_bonus += 5
            flags.append(ensure_full_stop(
                "Open Access (+5 operational): Instrument from open-access developer — "
                "no commercial licensing fees, reduces trial setup time."
            ))

        # Language coverage — REPORT only, no invented pass/fail threshold
        # Source: FDA PRO Guidance (2009) Section IV.A; EMA Reflection Paper on PRO (2005)
        if geographic_footprint in ["Global", "EU"]:
            geo = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(geographic_footprint,
                  GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
            key_langs = geo["key_languages"]
            if language_count >= 50:
                flags.append(ensure_full_stop(
                    f"Language coverage: {instrument_name} has approximately {language_count} "
                    f"validated translations — strong coverage for {geographic_footprint} trial. "
                    f"Verify specific language availability for trial sites "
                    f"[FDA PRO Guidance (2009) Section IV.A]."
                ))
            elif language_count > 0:
                operational_bonus -= 5
                flags.append(ensure_full_stop(
                    f"Language coverage note (-5 operational): {instrument_name} has approximately "
                    f"{language_count} validated translations. "
                    f"For {geographic_footprint} trial, verify coverage for: "
                    f"{', '.join(key_langs[:6])}. "
                    f"Commission additional translations if needed — typically 6-12 months "
                    f"[FDA PRO Guidance (2009) Section IV.A; ISPOR ePRO Task Force (2009)]."
                ))
            else:
                operational_bonus -= 10
                flags.append(ensure_full_stop(
                    f"LANGUAGE DATA UNAVAILABLE (-10 operational): No translation data found "
                    f"for {instrument_name}. "
                    f"Sonnet instructed to verify via web search or PROQOLID. "
                    f"Linguistically validated translations required for all trial languages "
                    f"[FDA PRO Guidance (2009) Section IV.A]."
                ))

        # HTA Alignment notes
        if "NICE" in hta_markets:
            if any(u in instrument_lower for u in ["eq-5d", "eq5d"]):
                flags.append(ensure_full_stop(
                    "HTA Alignment: EQ-5D included — supports QALY calculation for NICE "
                    "cost-utility analysis "
                    "[NICE DSU Technical Support Document 2 (2019)]."
                ))
            else:
                flags.append(ensure_full_stop(
                    "HTA NOTE — NICE: This instrument alone cannot support QALY-based "
                    "cost-utility analysis. EQ-5D-5L must be included alongside this instrument "
                    "for UK market access "
                    "[NICE DSU Technical Support Document 2 (2019)]."
                ))

        if "ICER" in hta_markets:
            if any(u in instrument_lower for u in ["eq-5d", "eq5d", "sf-6d", "sf-36"]):
                flags.append(ensure_full_stop(
                    "HTA Alignment: Utility-based measure included — supports ICER "
                    "cost-effectiveness analysis "
                    "[ICER Value Assessment Framework (2020)]."
                ))

        # Final score
        scientific_score = max(0, raw_score - penalty_total)

        results.append({
            "instrument_name": instrument_name,
            "scientific_score": scientific_score,
            "raw_positive_score": raw_score,
            "penalty_total": penalty_total,
            "operational_bonus": operational_bonus,
            "final_adjusted_score": scientific_score + operational_bonus,
            "risk_level": risk_level,
            "flags": flags,
            "drug_name": record.get("drug_name", ""),
            "trial_name": record.get("trial_name", ""),
            "nct_id": record.get("nct_id", ""),
            "phase": record.get("phase", ""),
            "disease_area": record.get("disease_area", ""),
            "patient_population": record.get("patient_population", ""),
            "pro_position": record.get("pro_position", ""),
            "key_finding": record.get("key_finding", ""),
            "compliance_rate": record.get("compliance_rate", ""),
            "assessment_schedule": record.get("assessment_schedule", ""),
            "publication_doi": record.get("publication_doi", ""),
            "publication_year": record.get("publication_year", ""),
            "p_value": record.get("p_value", ""),
            "effect_size": record.get("effect_size", ""),
            "fda_label_url": record.get("fda_label_url", ""),
            "ema_label_url": record.get("ema_label_url", ""),
            "key_toxicities": record.get("key_toxicities", ""),
            "validation_status": record.get("validation_status", ""),
            "strengths": record.get("strengths", ""),
            "limitations": record.get("limitations", ""),
            "recall_period": recall_period,
            "language_count": language_count,
        })

    risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    results.sort(key=lambda x: (risk_order.get(x["risk_level"], 4), -x["scientific_score"]))
    return results


# =============================================================================
# STEP 4: BATTERY OPTIMIZER
# =============================================================================
def build_battery(scored_instruments: list, context_json: dict) -> dict:
    """
    Assemble an optimal non-redundant COA battery.
    Selects highest-scoring instrument per domain category.
    Adds HTA-required instruments regardless of score.
    Caps at 5 instruments (clinical standard).
    """
    hta_markets = context_json.get("hta_markets", [])
    core_domains = [str(d).lower() for d in context_json.get("core_domains_required", [])]

    DOMAIN_CATEGORIES = {
        "hrqol_general": ["hrqol", "quality of life", "health-related", "wellbeing"],
        "disease_specific": ["disease-specific", "myeloma", "cancer-specific", "my20", "symptom burden"],
        "fatigue": ["fatigue", "tiredness", "energy", "asthenia"],
        "pain": ["pain", "bone pain", "analgesic", "nrs", "bpi"],
        "physical_function": ["physical function", "physical well-being", "mobility", "adl"],
        "treatment_tolerability": ["adverse", "tolerability", "toxicity", "ctcae", "symptom", "crs", "cytokine", "pro-ctcae"],
        "health_utility": ["utility", "eq-5d", "eq5d", "sf-6d", "qaly"],
        "neuropathy": ["neuropathy", "neurotoxicity", "ntx", "peripheral"],
        "emotional": ["emotional", "anxiety", "depression", "psychological", "hads", "phq"],
    }

    HTA_REQUIRED_WILDCARD = {
        "NICE": "eq-5d",
        "ICER": "eq-5d",
        "EUnetHTA": "eq-5d",
        "SMC": "eq-5d",
    }

    selected = []
    selected_names = set()
    covered_domains = set()

    # Pass 1: best non-critical instrument per domain category
    for category, keywords in DOMAIN_CATEGORIES.items():
        best = None
        for inst in scored_instruments:
            if inst["instrument_name"] in selected_names:
                continue
            if inst["risk_level"] == "CRITICAL":
                continue
            inst_lower = inst["instrument_name"].lower()
            # Search instrument name, flags text, AND known broad instrument domains
            broad_domains = " ".join(BROAD_INSTRUMENTS.get(inst_lower, []))
            inst_text = " ".join([
                inst_lower,
                " ".join(inst.get("flags", [])).lower(),
                broad_domains,
            ])
            if any(kw in inst_text for kw in keywords):
                if best is None or inst["scientific_score"] > best["scientific_score"]:
                    best = inst
        if best:
            # Deduplicate by instrument family — do not add a variant if the family is already covered
            INSTRUMENT_FAMILIES = {
                "eq-5d": ["eq-5d", "eq-5d-5l", "eq-5d-3l", "eq5d"],
            }
            already_family_covered = False
            for family_key, family_members in INSTRUMENT_FAMILIES.items():
                # Check if best instrument is in this family
                if any(member in best["instrument_name"].lower() for member in family_members):
                    # Check if any family member already selected
                    if any(
                        any(member in sel["instrument_name"].lower() for member in family_members)
                        for sel in selected
                    ):
                        already_family_covered = True
                        break
            if already_family_covered:
                continue
            selected.append({**best, "battery_role": category.replace("_", " ").title()})
            selected_names.add(best["instrument_name"])
            covered_domains.add(category)

    # Pass 2: HTA-required instruments
    hta_additions = []
    for market in hta_markets:
        wildcard = HTA_REQUIRED_WILDCARD.get(market, "")
        if not wildcard:
            continue
        already = any(wildcard in n.lower() for n in selected_names)
        if not already:
            for inst in scored_instruments:
                if wildcard in inst["instrument_name"].lower():
                    # Override risk level: HTA-required instruments are not risky to include —
                    # they are mandatory. The risk is in OMITTING them.
                    inst_copy = dict(inst)
                    inst_copy["risk_level"] = "LOW"
                    hta_additions.append({
                        **inst_copy,
                        "battery_role": f"HTA Required ({market})",
                        "battery_note": (
                            f"Required for {market} cost-utility analysis regardless of scientific score. "
                            f"Without EQ-5D-5L, QALY calculation is impossible and UK reimbursement "
                            f"is severely compromised [NICE DSU TSD 2, 2019]."
                        )
                    })
                    selected_names.add(inst["instrument_name"])
                    break

    # Check domain gaps at BATTERY level 
    # A domain is covered if ANY instrument in the battery covers it.
    # Use domain synonyms and BROAD_INSTRUMENTS for matching.
    gaps = []
    all_battery_instruments = selected + hta_additions

    for domain in core_domains:
        domain_synonyms_list = DOMAIN_SYNONYMS.get(domain.lower(), [domain])
        all_terms = [domain.lower()] + [s.lower() for s in domain_synonyms_list]

        domain_covered = False
        for battery_inst in all_battery_instruments:
            inst_lower = battery_inst["instrument_name"].lower()
            # Check broad instruments dict
            broad = [b.lower() for b in BROAD_INSTRUMENTS.get(inst_lower, [])]
            # Check instrument name and flags
            flags_text = " ".join(battery_inst.get("flags", [])).lower()
            search_text = inst_lower + " " + flags_text + " " + " ".join(broad)
            if any(term in search_text for term in all_terms):
                domain_covered = True
                break

        if not domain_covered:
            gaps.append(domain)

    final_battery = (selected + hta_additions)[:5]

    return {
        "battery": final_battery,
        "battery_names": [b["instrument_name"] for b in final_battery],
        "covered_domains": list(covered_domains),
        "gaps": gaps,
        "hta_additions": [b["instrument_name"] for b in hta_additions],
    }


# =============================================================================
# STEP 5: KG NARRATIVE CLEANER
# =============================================================================
def clean_kg_narratives(records: list) -> list:
    """
    Use Claude Haiku to clean messy narrative fields in KG records.
    Removes PMC IDs, fixes typos, converts to clean sentences.
    Gracefully returns original records if cleaning fails.
    """
    if not records:
        return records

    dirty = []
    for i, r in enumerate(records[:15]):  # Cap at 15 to control token cost
        dirty.append({
            "idx": i,
            "instrument_name": r.get("instrument_name", ""),
            "mcid": str(r.get("mcid", ""))[:300],
            "key_finding": str(r.get("key_finding", ""))[:300],
            "regulatory_acceptance": str(r.get("regulatory_acceptance", ""))[:200],
            "strengths": str(r.get("strengths", ""))[:200],
            "limitations": str(r.get("limitations", ""))[:200],
        })

    system = (
        "You are a medical editor cleaning raw database records for a clinical trials tool. "
        "Return a JSON array of cleaned records with the same idx values. "
        "For each record: "
        "(1) mcid: extract only the numeric threshold and unit, e.g. '1.33 points on 0-10 scale'. Remove PMC IDs, patient context, and source names. "
        "(2) key_finding: one clean sentence in active voice ending with a full stop. No brackets, no raw IDs. "
        "(3) regulatory_acceptance: one clean sentence ending with a full stop. "
        "(4) strengths: one clean sentence ending with a full stop. "
        "(5) limitations: one clean sentence ending with a full stop. "
        "If a field is empty or noise only, return empty string. "
        "Do NOT add information not in the original text. "
        "Return ONLY valid JSON array. No markdown."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content":
                f"Clean these records. Return only JSON array:\n{json.dumps(dirty)}"}]
        )
        raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        cleaned = json.loads(raw)
        cleaned_map = {c["idx"]: c for c in cleaned}
        result = []
        for i, rec in enumerate(records):
            r = dict(rec)
            if i in cleaned_map:
                cf = cleaned_map[i]
                for field in ["mcid", "key_finding", "regulatory_acceptance", "strengths", "limitations"]:
                    if cf.get(field):
                        r[field] = cf[field]
            result.append(r)
        logging.info(f"KG narrative cleaning: {len(cleaned)} records cleaned.")
        return result
    except Exception as e:
        logging.warning(f"KG cleaning failed, using raw records: {e}")
        return records


# =============================================================================
# STEP 6: REASONER — MAIN ORCHESTRATOR
# =============================================================================
def get_recommendation(user_text: str) -> dict:
    """
    Main pipeline: Analyzer → KG queries → Scoring → Battery → Cleaning → Sonnet + web search.
    Returns structured result dict for the Streamlit UI.
    """
    error_status = None
    kg_records = []
    reg_records = []
    inst_refs = []
    reg_rules = []
    inst_regulatory_precedents = {}
    battery_result = {"battery": [], "battery_names": [], "covered_domains": [], "gaps": [], "hta_additions": []}

    # --- STEP A: Analyze trial context ---
    context_json = analyze_trial_context(user_text)
    indication = context_json.get("indication", "")
    synonyms = context_json.get("indication_synonyms") or [indication]
    phase = context_json.get("phase", "Phase 3")

    # --- STEP B: Query Knowledge Graph ---
    try:
        # Instruments: search primary indication + synonyms
        search_terms = list(dict.fromkeys([indication] + synonyms[:3]))
        for term in search_terms:
            records = get_instruments_by_indication(indication=term, phase=phase, endpoint="")
            kg_records.extend(records)
        # Deduplicate instruments by name
        kg_records = list({r.get("instrument_name", "Unknown"): r for r in kg_records}.values())

        # Regulatory evidence: search all synonym terms
        all_reg = []
        for term in search_terms:
            all_reg.extend(get_regulatory_evidence(indication=term, agency=""))
        seen_ids = set()
        for r in all_reg:
            rid = r.get("review_id") or (r.get("drug_name","") + r.get("agency",""))
            if rid not in seen_ids:
                seen_ids.add(rid)
                reg_records.append(r)

        # Regulatory rules (all stages — actual values are Instrument_Selection etc.)
        reg_rules = get_regulatory_rules(indication=indication, lifecycle_stage="", decision_type="")

        logging.info(f"KG: {len(kg_records)} instruments, {len(reg_records)} reviews, {len(reg_rules)} rules")
    except Exception as e:
        error_status = f"Knowledge Graph offline: {e}"
        logging.error(f"KG query failed: {e}")

    # --- STEP C: Score instruments ---
    scored = score_evidence(context_json, kg_records) if kg_records else []

    # --- STEP D: Build battery ---
    if scored:
        battery_result = build_battery(scored, context_json)
    top_5 = battery_result["battery"] if battery_result["battery"] else scored[:5]

    # --- STEP E: Fetch instrument refs + per-instrument regulatory precedent ---
    if not error_status:
        try:
            for inst in top_5:
                name = inst["instrument_name"]
                refs = get_instrument_reference(instrument_name=name)
                if refs:
                    inst_refs.extend(refs if isinstance(refs, list) else [refs])
                precedents = get_regulatory_evidence_for_instrument(instrument_name=name)
                if precedents:
                    inst_regulatory_precedents[name] = precedents
        except Exception as e:
            logging.error(f"Instrument ref/precedent fetch failed: {e}")

    # --- STEP F: Clean KG narrative fields ---
    if kg_records:
        try:
            kg_records = clean_kg_narratives(kg_records)
        except Exception as e:
            logging.warning(f"KG cleaning skipped: {e}")

    # --- STEP G: Build evidence block for Sonnet ---
    kg_block_lines = []
    if error_status:
        kg_block_lines.append(f"⚠️ KNOWLEDGE GRAPH OFFLINE — {error_status}")
        kg_block_lines.append("Rely entirely on web search. State this clearly in the response.")
    else:
        # Battery summary
        if battery_result["battery"]:
            kg_block_lines.append("=== RECOMMENDED COA BATTERY (domain coverage optimiser) ===\n")
            for b in battery_result["battery"]:
                kg_block_lines.append(
                    f"  → {b['instrument_name']} | Role: {b.get('battery_role','')} | "
                    f"Score: {b['scientific_score']}/100 | Risk: {b['risk_level']}"
                    + (f"\n    Note: {b.get('battery_note','')}" if b.get("battery_note") else "")
                )
            if battery_result["gaps"]:
                kg_block_lines.append(
                    f"\n  ⚠️ DOMAIN GAPS — no instrument found for: {', '.join(battery_result['gaps'])}. "
                    f"Search the web to identify instruments for these domains."
                )

        # Scored instrument ranking
        kg_block_lines.append(f"\n=== ALL SCORED INSTRUMENTS ({len(scored)} evaluated) ===\n")
        for i, inst in enumerate(scored[:8], 1):
            nct = inst.get("nct_id","")
            doi = inst.get("publication_doi","")
            fda_url = inst.get("fda_label_url","")
            ema_url = inst.get("ema_label_url","")
            source_note = ""
            if nct and str(nct).startswith("NCT"):
                source_note = f"ClinicalTrials.gov: https://clinicaltrials.gov/study/{nct}"
            elif doi:
                source_note = f"DOI: https://doi.org/{doi}"
            elif fda_url and str(fda_url).startswith("http"):
                source_note = f"FDA label: {fda_url}"
            kg_block_lines.append(
                f"[TI-{i:03d}] {inst['instrument_name']} in {inst.get('trial_name','')} "
                f"({inst.get('nct_id','')}) for {inst.get('drug_name','')} — {inst.get('phase','')}\n"
                f"  Score: {inst['scientific_score']}/100 | Risk: {inst['risk_level']}\n"
                f"  Key finding: {inst.get('key_finding','Not recorded')}\n"
                f"  Source URL for this record: {source_note or 'Not available'}\n"
                f"  When citing this record, use: [TI-{i:03d}] with URL {source_note or 'N/A'}\n"
            )

        # Regulatory reviews
        kg_block_lines.append(f"\n=== REGULATORY REVIEWS ({len(reg_records)} records) ===\n")
        for i, rr in enumerate(reg_records[:10], 1):
            rr_fda = rr.get("fda_label_url","")
            rr_ema = rr.get("ema_label_url","")
            rr_url = rr_fda if str(rr_fda).startswith("http") else (rr_ema if str(rr_ema).startswith("http") else "")
            kg_block_lines.append(
                f"[RR-{i:03d}] {rr.get('agency','')} review of {rr.get('drug_name','')} | "
                f"Decision: {rr.get('decision','')} | "
                f"Instruments accepted: {rr.get('instruments_accepted','')} | "
                f"Claim type: {rr.get('claim_type','')} | "
                f"Source URL: {rr_url or 'See drug label'} | "
                f"When citing: use [RR-{i:03d}] with URL {rr_url or rr.get('drug_name','')}"
            )

        # Instrument reference data
        kg_block_lines.append(f"\n=== INSTRUMENT REFERENCE DATA ({len(inst_refs)} records) ===\n")
        for i, ir in enumerate(inst_refs[:8], 1):
            kg_block_lines.append(
                f"[IR-{i:03d}] {ir.get('short_name','')} | "
                f"Domains: {ir.get('domains','')} | MCID: {ir.get('mcid','')} | "
                f"Validation: {ir.get('validation','')} | "
                f"Regulatory acceptance: {ir.get('regulatory_acceptance','')}"
            )

        # Per-instrument regulatory precedents
        if inst_regulatory_precedents:
            kg_block_lines.append(f"\n=== INSTRUMENT REGULATORY PRECEDENTS ===\n")
            kg_block_lines.append("Whether each recommended instrument has prior FDA/EMA acceptance.\n")
            for inst_name, reviews in inst_regulatory_precedents.items():
                kg_block_lines.append(f"\nInstrument: {inst_name}")
                for j, rev in enumerate(reviews[:3], 1):
                    accepted = inst_name.lower() in str(rev.get("instruments_accepted","")).lower()
                    kg_block_lines.append(
                        f"  [PREC-{j}] {rev.get('agency','')} | Drug: {rev.get('drug_name','')} | "
                        f"Decision: {rev.get('decision','')} | "
                        f"{'✅ ACCEPTED' if accepted else 'Reviewed — acceptance unclear'}\n"
                        f"  Claim type: {rev.get('claim_type','')}\n"
                        f"  Label language: {str(rev.get('label_language',''))[:200]}"
                    )
                    if rev.get("rejection_reason_primary"):
                        kg_block_lines.append(
                            f"  ⚠️ Rejection risk: {rev.get('rejection_reason_primary','')}"
                        )

        # Rejection reason analysis
        rejections = [r for r in reg_records if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")]
        kg_block_lines.append(f"\n=== REJECTION REASON ANALYSIS ({len(rejections)} records with rejection data) ===\n")
        kg_block_lines.append("Source: actual FDA/EMA medical review documents. Cite [REJ-XXX] when referencing.\n")
        for i, rr in enumerate(reg_records, 1):
            if rr.get("rejection_reason_primary") or rr.get("rejection_reason_detailed"):
                kg_block_lines.append(
                    f"[REJ-{i:03d}] Drug: {rr.get('drug_name','')} | Agency: {rr.get('agency','')} | "
                    f"Decision: {rr.get('decision','')}\n"
                    f"  Primary reasons: {rr.get('rejection_reason_primary','')}\n"
                    f"  Detailed: {str(rr.get('rejection_reason_detailed',''))[:500]}\n"
                    f"  Missing data issues: {rr.get('missing_data_issue','')}\n"
                    f"  Alpha controlled: {rr.get('alpha_controlled','')}\n"
                    f"  Final approved label language: {rr.get('label_language','Not specified')}\n"
                )

        # Regulatory rules 
        if reg_rules:
            kg_block_lines.append(f"\n=== REGULATORY RULES ({len(reg_rules)} rules) ===\n")
            for i, rule in enumerate(reg_rules, 1):
                kg_block_lines.append(
                    f"[RULE-{i:03d}] {rule.get('source_document','')} Section {rule.get('section','')} | "
                    f"Stage: {rule.get('lifecycle_stage','')} | Type: {rule.get('decision_type','')}\n"
                    f"  Rule: {rule.get('rule_text','')}\n"
                )
        
    # Build citation index — maps labels like "TI-001" to source data
    # This allows the UI to render clickable links next to each citation label
    citation_index = {}

    for i, inst in enumerate(scored[:8], 1):
        label = f"TI-{i:03d}"
        nct = inst.get("nct_id", "")
        doi = inst.get("publication_doi", "")
        drug = inst.get("drug_name", "")
        trial = inst.get("trial_name", "")
        fda_url = inst.get("fda_label_url", "")
        ema_url = inst.get("ema_label_url", "")
        links = []
        if nct and str(nct).startswith("NCT"):
            links.append({"label": f"ClinicalTrials.gov: {trial or nct}", "url": f"https://clinicaltrials.gov/study/{nct}"})
        if doi:
            links.append({"label": f"Publication ({inst.get('publication_year','')})", "url": f"https://doi.org/{doi}"})
        if fda_url and str(fda_url).startswith("http"):
            links.append({"label": f"FDA label: {drug}", "url": fda_url})
        if ema_url and str(ema_url).startswith("http"):
            links.append({"label": f"EMA label: {drug}", "url": ema_url})
        if not links and drug:
            links.append({"label": f"FDA DailyMed: {drug}", "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(' ', '+')}"})
        citation_index[label] = {
            "type": "trial_instrument",
            "instrument": inst.get("instrument_name",""),
            "trial": trial,
            "drug": drug,
            "phase": inst.get("phase",""),
            "links": links
        }

    for i, rr in enumerate(reg_records[:10], 1):
        label = f"RR-{i:03d}"
        links = []
        fda_url = rr.get("fda_label_url", "")
        ema_url = rr.get("ema_label_url", "")
        drug = rr.get("drug_name", "")
        if fda_url and str(fda_url).startswith("http"):
            links.append({"label": f"FDA label: {drug}", "url": fda_url})
        if ema_url and str(ema_url).startswith("http"):
            links.append({"label": f"EMA label: {drug}", "url": ema_url})
        if not links and drug:
            links.append({"label": f"FDA DailyMed: {drug}", "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(' ', '+')}"})
        citation_index[label] = {
            "type": "regulatory_review",
            "drug": drug,
            "agency": rr.get("agency",""),
            "decision": rr.get("decision",""),
            "links": links
        }

    for i, rr in enumerate(reg_records, 1):
        if rr.get("rejection_reason_primary") or rr.get("rejection_reason_detailed"):
            label = f"REJ-{i:03d}"
            drug = rr.get("drug_name","")
            links = []
            fda_url = rr.get("fda_label_url","")
            ema_url = rr.get("ema_label_url","")
            if fda_url and str(fda_url).startswith("http"):
                links.append({"label": f"FDA label: {drug}", "url": fda_url})
            if ema_url and str(ema_url).startswith("http"):
                links.append({"label": f"EMA label: {drug}", "url": ema_url})
            citation_index[label] = {
                "type": "rejection",
                "drug": drug,
                "agency": rr.get("agency",""),
                "primary_reason": rr.get("rejection_reason_primary",""),
                "links": links
            }

    kg_evidence_block = "\n".join(kg_block_lines)

    # --- STEP H: HTA context block ---
    hta_lines = ["\n=== HTA/PAYER CONTEXT ===\n"]
    for body in context_json.get("hta_markets", []):
        if body in HTA_PREFERENCES:
            h = HTA_PREFERENCES[body]
            hta_lines.append(
                f"{body}: Required — {h.get('required_instruments',[])} | "
                f"Notes: {h['notes']} | Ref: {h['reference']}"
            )
    hta_block = "\n".join(hta_lines)

    # --- STEP I: Build Sonnet system prompt ---
    sonnet_system = f"""You are a senior COA strategist writing a protocol-ready recommendation memo.
The Python scoring engine has already selected the optimal instrument battery.
Your job is to JUSTIFY the battery with evidence and provide a concrete action plan.
You are NOT re-analysing data. You are explaining decisions already made.

ABSOLUTE RULES — violation means the response is invalid:

RULE 1 (One job only): Justify the pre-built battery. Do not recommend instruments outside it
unless you find specific web evidence that a critical domain is genuinely uncovered after
checking all instruments in the battery collectively.

RULE 2 (Citations — exact format required, no exceptions):
For EVERY factual claim, provide an inline citation immediately after the claim in one of these formats:

Format A — Web source (URL known):
[Descriptive source name](https://complete-url.com/path)
Example: "EQ-5D-5L is required by NICE [NICE DSU Technical Support Document 2](https://www.nice.org.uk/about/what-we-do/our-programmes/nice-guidance/technology-appraisal-guidance/changes-to-how-we-make-decisions)."

Format B — Web source (URL not found, but source known):
[Author Year Journal](https://pubmed.ncbi.nlm.nih.gov/?term=author+year+journal+title)
Example: "MCID is 10 points [Osoba 1998 J Clin Oncol](https://pubmed.ncbi.nlm.nih.gov/?term=Osoba+1998+QLQ-C30+meaningful)."

Format C — KG record:
[TI-001], [RR-003], [REJ-012] etc. using the label from the evidence block above.

PROHIBITED citation formats (will be treated as uncited claims):
- [13-4] or [45-6] or any number-dash-number format — these are internal search indices, not citations
- (Author Year) without a URL — not clickable, not verifiable
- Bare parenthetical like (NICE DSU TSD 2) — must include URL
- Empty brackets like [] or trailing spaces after claims

NEVER hallucinate any citation. All claims must be backed by a real source. If you cannot find a source, do not make the claim.

RULE 3 (Precision about what is exploratory):
When describing regulatory outcomes, always state WHAT was designated exploratory.
CORRECT: "The PRO endpoint using MySIm-Q was designated exploratory by FDA [REJ-XXX]."
WRONG: "Ciltacabtagene autoleucel was designated exploratory."
The drug receives approval or rejection. The PRO endpoint receives exploratory designation.

RULE 4 (Relevance — no orphaned statistics):
Every piece of evidence must connect to a specific implication for this trial.
Structure every evidence point as:
"Finding: [X] [citation]. This matters because: [Y]. Action: [Z]."
If you cannot complete all three parts, omit the finding.

RULE 5 (Battery is collective):
A COA strategy battery covers domains collectively across instruments.
Do NOT flag a domain as uncovered if another instrument in the battery measures it.
EORTC QLQ-C30 covers physical function, fatigue, pain, and emotional function —
do not report these as gaps if QLQ-C30 is in the battery.

RULE 6 (Web search is targeted):
Before searching, state what specific question you are trying to answer.
Priority sources: fda.gov, ema.europa.eu, clinicaltrials.gov, pubmed.ncbi.nlm.nih.gov,
proqolid.org, eortc.org, facit.org, euroqol.org, ispor.org.
For each instrument, search: "[instrument name] validation multiple myeloma" and
"[instrument name] recall period official documentation".
For each rejection, search: "FDA [drug name] medical review" or "EMA [drug name] medical review" to find the actual review document.

RULE 7 (Language — instrument-specific):
Search proqolid.org or the instrument developer website for actual validated language counts.
Report as: "[Instrument] has [N] validated translations per [source URL]. Languages available
for your trial sites: [list]. Action: [commission / sufficient / verify].

RULE 8 (Evidence gaps — strict restrictions):
The Gaps section lists only items where KG AND web search BOTH failed to find evidence.
PROHIBITED gap topics:
- Do NOT report "OS not measured by PRO" — overall survival is a clinical endpoint, not a PRO domain.
  PRO instruments measure how patients feel, not how long they live.
- Standard clinical endpoints (PFS, ORR, MRD, CR): these are not PRO domains.
- Do NOT report a domain as a gap if an instrument in the recommended battery covers it.
  EORTC QLQ-C30 covers: physical function, fatigue, pain, nausea, emotional function, role function.
  EORTC QLQ-MY20 covers: bone pain, disease symptoms, treatment side effects, future perspective.
  EQ-5D-5L covers: mobility, self-care, usual activities, pain/discomfort, anxiety/depression.
- Do NOT report a gap unless you actually searched for it and found nothing.
  State what query you searched: "Searched PubMed for '[query]' — no relevant results found."

RULE 9 (Competitor and comparator context):
For bispecific antibodies in MM, always explain WHY specific drugs are cited as comparators.
When referencing teclistamab, elranatamab, linvoseltamab, ciltacabtagene autoleucel, or
idecabtagene vicleucel, state explicitly:
- What mechanism they share with the trial drug (BCMA-targeting, T-cell engaging)
- Why their regulatory experience is directly relevant (same mechanism = same regulatory risks)
Example: "Teclistamab is a BCMA-targeting bispecific antibody — the same mechanism as this trial drug.
Its PRO rejection history [REJ-014] is therefore the most relevant precedent for this submission."

For the web search, specifically search: "[drug class] multiple myeloma PRO regulatory rejection"
and "[specific comparator drug] PRO endpoint FDA EMA review" to find the most relevant precedents.

OUTPUT STRUCTURE — follow exactly, writing as a senior colleague briefing a clinical team:

## COA Strategy for [Trial Description]

Open with 2-3 sentences summarising the strategy and the key challenge for this trial.
Example: "For this Phase 3 BCMA bispecific trial in RRMM, the recommended battery of three instruments
covers all FDA core domains while addressing the unique monitoring requirements of step-up dosing.
The primary regulatory challenge is the systematic pattern of null PRO results in MM bispecific trials,
which must be addressed through pre-specification and adequate powering."

## Recommended Battery

Table: Instrument | Domain | Score | Regulatory status | Key risk

Then for EACH instrument, one structured block:

**[Instrument name]** — [Domain role]
*What it measures:* [2 sentences on the instrument domains, cited from IR or web source]
*Why it is in this battery:* [The specific TPP claim or domain it covers, with KG precedent cited]
*Regulatory history:* [What FDA/EMA have accepted or rejected for this instrument in similar contexts, cited from RR or REJ records — with explicit statement of what the PRO endpoint received, not the drug]
*Main risk for this trial:* [One sentence on the most important risk]
*Action required:* [Concrete step the team must take before or during the trial]

## What Could Go Wrong — and How to Prevent It

For each risk, write: "[Risk] has caused PRO rejection before: [evidence with citation].
For this trial, this means [implication]. To prevent this: [specific mitigation with deadline]."

Only include risks that are genuinely relevant to this specific trial design.
If there are FDA-specific and EMA-specific risks, separate them clearly:
"FDA concern: ... EMA concern: ... Both agencies: ..."

## HTA Status

Short table: HTA body | Required instrument | Status | Action
Then one sentence per body on what the consequence of omission would be.

## What to Do Before the Trial Starts — Implementation Checklist

Numbered list, in chronological order of when each action must be taken.
Start each item with a timeframe: "Now:", "Before protocol finalisation:", "Before first patient:",
"During Cycle 1:", "At each assessment visit:"

## Gaps — What We Could Not Find

Only list items where the KG AND web search both failed to find evidence.
For each gap, state: what was searched, what was found, what remains unknown.
Do NOT list domains that are covered by instruments in the battery.

Glossary: {GLOSSARY_TEXT}"""

    # --- STEP J: Build Sonnet user prompt ---
    indication_for_search = indication or "this oncology indication"
    sonnet_user = f"""You are briefing a senior clinical scientist on the COA strategy for their trial.
Write as a knowledgeable colleague, not as a report generator.
Every claim must be cited. Every section must conclude with an action.
The battery has already been selected — your job is to justify it clearly and give the team
a concrete plan to execute it successfully.

TRIAL CONTEXT:
{json.dumps(context_json, indent=2)}

{kg_evidence_block}

{hta_block}

ORIGINAL USER QUERY: {user_text}

{"⚠️ KNOWLEDGE GRAPH OFFLINE — rely entirely on web search, state this in response." if error_status else ""}

Use your web search tool to supplement the KG evidence above. Prioritise these sources in order:
1. fda.gov — FDA PRO guidance documents and drug approval letters
2. ema.europa.eu — EMA EPARs and reflection papers
3. clinicaltrials.gov — recent Phase 3 {indication_for_search} trials with PRO endpoints
4. pubmed.ncbi.nlm.nih.gov — validation studies for recommended instruments
5. proqolid.org — instrument properties, translations, MCID values
6. ispor.org — ISPOR task force reports
7. nice.org.uk / icer.org — HTA guidance for {', '.join(context_json.get('hta_markets', []))}

Present a single integrated COA strategy following the output structure above.
Do not produce disconnected sections — weave the evidence into a coherent narrative."""
    
# Add drug-class-specific search instructions
    drug_class = context_json.get("drug_class", "").lower()
    if any(term in drug_class for term in ["bispecific", "car-t", "bcma", "gprc5d", "fcrh5"]):
        competitor_search_instruction = f"""
COMPETITOR ANALYSIS REQUIRED — {context_json.get('drug_class','')} in {indication}:
This trial uses a T-cell engaging bispecific antibody. Search specifically for:
1. All FDA/EMA approved BCMA-targeting agents in MM with PRO data: teclistamab (Tecvayli),
   elranatamab (Elrexfio), linvoseltamab — search "[drug name] PRO FDA EMA review rejection"
2. CAR-T precedents in MM (ciltacabtagene autoleucel, idecabtagene vicleucel) — same T-cell
   engagement mechanism, directly relevant regulatory history
3. For each drug found: state the mechanism, why it is relevant, what happened to PRO data

When citing these drugs, ALWAYS explain: "This is relevant because [drug] is a [mechanism],
the same class as the trial drug, meaning [specific shared regulatory risk]."
"""
        sonnet_user = sonnet_user.replace(
            "Use your web search tool to supplement",
            competitor_search_instruction + "\nUse your web search tool to supplement"
        )

    # --- STEP K: Call Sonnet ---
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10000,
            system=sonnet_system,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": sonnet_user}]
        )
        answer = " ".join(
            block.text for block in response.content
            if hasattr(block, "text") and block.text
        )
    except Exception as e:
        answer = f"Sonnet call failed: {e}"
        error_status = error_status or f"Sonnet error: {e}"
        logging.error(f"Sonnet failed: {e}")

    # --- STEP L: Build result dict ---
    result = {
        "answer": answer,
        "context_json": context_json,
        "top_scores": top_5,
        "all_scores": scored,
        "kg_raw_hits": kg_records,
        "reg_records": reg_records,
        "reg_rules": reg_rules,
        "inst_refs": inst_refs,
        "inst_regulatory_precedents": inst_regulatory_precedents,
        "battery_result": battery_result,
        "hta_context": {b: HTA_PREFERENCES[b] for b in context_json.get("hta_markets", []) if b in HTA_PREFERENCES},
        "error_status": error_status,
        "record_counts": {
            "instrument_records": len(kg_records),
            "regulatory_reviews": len(reg_records),
            "regulatory_rules": len(reg_rules),
            "instrument_refs": len(inst_refs),
            "scored_instruments": len(scored),
            "rejections_found": len([r for r in reg_records if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")])
        }
    }

    log_recommendation(user_text, result)
    return result


# =============================================================================
# STEP 7: EVALUATION LOGGING
# =============================================================================
def log_recommendation(user_text: str, result: dict) -> None:
    """Save every recommendation to a timestamped JSON log for evaluation."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        entry = {
            "timestamp": timestamp,
            "user_query": user_text,
            "indication": result.get("context_json", {}).get("indication", "unknown"),
            "phase": result.get("context_json", {}).get("phase", "unknown"),
            "assumptions_made": result.get("context_json", {}).get("assumptions_made", []),
            "battery": result.get("battery_result", {}).get("battery_names", []),
            "domain_gaps": result.get("battery_result", {}).get("gaps", []),
            "top_5_instruments": [
                {
                    "name": i["instrument_name"],
                    "score": i["scientific_score"],
                    "risk_level": i["risk_level"],
                    "operational_bonus": i["operational_bonus"]
                }
                for i in result.get("top_scores", [])
            ],
            "record_counts": result.get("record_counts", {}),
            "error_status": result.get("error_status"),
            "answer_length_chars": len(result.get("answer", "")),
            "answer": result.get("answer", "")
        }
        log_path = f"logs/recommendation_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        logging.info(f"Logged to {log_path}")
    except Exception as e:
        logging.error(f"Log failed: {e}")


# =============================================================================
# MAIN TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing imports...")
    from agent import (
        get_recommendation, build_battery, clean_mcid,
        clean_kg_narratives, KNOWN_LANGUAGE_COUNTS, ensure_full_stop,
        RECALL_PERIOD_UNKNOWN, INSTRUMENT_RECALL_PERIODS
    )
    print("All imports OK.")
    bfi = next((v for k,v in INSTRUMENT_RECALL_PERIODS.items() if "bfi" in k), RECALL_PERIOD_UNKNOWN)
    print(f"BFI recall period: {bfi} (expected: 1)")
    print(f"Unknown instrument sentinel: {RECALL_PERIOD_UNKNOWN} (expected: -1)")
    short, _ = clean_mcid("bfi total scale: 1.33 points pmc11398933 (2024) in brain/cns cancer patients")
    print(f"MCID clean: {short}")
    print(ensure_full_stop("Test sentence without stop"))