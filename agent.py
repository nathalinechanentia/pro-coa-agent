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

REGULATORY_CITATIONS = {
    "FDA PRO Guidance (2009)":
        "https://www.fda.gov/media/77832/download",
    "FDA (2024) Core Patient-Reported Outcomes in Cancer Clinical Trials":
        "https://www.fda.gov/media/149994/download",
    "FDA PFDD Guidance 1 (2018)":
        "https://www.fda.gov/media/139088/download",
    "FDA PFDD Guidance 2 (2022)":
        "https://www.fda.gov/media/131230/download",
    "FDA PFDD Guidance 3 (2025)":
        "https://www.fda.gov/media/159500/download",
    "FDA PFDD Guidance 4 (2023)":
        "https://www.fda.gov/media/166830/download",
    "EMA Reflection Paper on Patient Experience Data (2025)":
        "https://www.ema.europa.eu/en/documents/scientific-guideline/reflection-paper-patient-experience-data_en.pdf",
    "EMA Appendix 2 to the guideline on the evaluation of anticancer medicinal products in man":
        "https://www.ema.europa.eu/en/documents/other/appendix-2-guideline-evaluation-anticancer-medicinal-products-man_en.pdf",
    "HTA Guidance on outcomes for joint clinical assessments":
        "https://health.ec.europa.eu/document/download/a70a62c7-325c-401e-ba42-66174b656ab8_en?filename=hta_outcomes_jca_guidance_en.pdf"
}

def get_secret(key: str) -> str:
    """Works locally (.env) AND on Streamlit Cloud (st.secrets)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""

client = Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

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
# DOMAIN_SYNONYMS = {
#     "bone pain": ["pain", "nrs", "bpi", "musculoskeletal", "skeletal"],
#     "physical function": ["physical", "function", "activity", "mobility", "performance", "adl", "karnofsky"],
#     "fatigue": ["fatigue", "tiredness", "energy", "exhaustion", "asthenia", "vitality"],
#     "dyspnea": ["dyspnea", "breathlessness", "breathing", "respiratory", "shortness of breath"],
#     "cough": ["cough", "respiratory", "pulmonary"],
#     "pain": ["pain", "analgesic", "bpi", "nrs", "aches", "discomfort", "bone"],
#     "nausea": ["nausea", "vomiting", "gi", "gastrointestinal", "emesis"],
#     "urinary function": ["urinary", "urology", "bladder", "ipss", "micturition"],
#     "emotional function": ["emotional", "anxiety", "depression", "psychological", "mental", "hads", "phq"],
#     "appetite loss": ["appetite", "anorexia", "eating", "weight"],
#     "bowel function": ["bowel", "diarrhoea", "constipation", "gastrointestinal"],
#     "treatment tolerability": ["tolerability", "adverse", "toxicity", "ctcae", "symptom", "side effect", "crs", "cytokine release", "icans"],
#     "disease-related symptoms": ["bone pain", "disease symptoms", "mm symptoms",
#                                 "disease-specific", "symptom burden"],
#     "symptomatic adverse events": ["adverse events", "symptoms", "toxicity", "tolerability",
#                                 "side effects", "treatment side effects", "nausea",
#                                 "neuropathy", "fatigue"],
#     "side effect impact summary": ["side effects", "treatment impact", "toxicity burden",
#                                 "overall symptom burden"],
#     "role function":            ["physical function", "functioning", "daily activities",
#                                 "role functioning", "work", "activities"],
#     "physical functioning":     ["physical function", "functioning", "mobility", "activity"],
#     "peripheral neuropathy":    ["neuropathy", "cipn", "tingling", "numbness",
#                                 "sensory", "neuropathic pain"],
# "cytokine release syndrome (crs) symptoms": ["crs", "cytokine", "ctcae", "icans", "pro-ctcae", "tolerability", "adverse"],
#     "hrqol": ["hrqol", "quality of life", "health-related", "wellbeing", "function"],
#     "disease-specific symptoms": ["disease", "specific", "myeloma", "cancer-specific", "my20", "symptom"],
# }

DOMAIN_SYNONYMS = {
    "bone pain":              ["bpi", "bpi-sf", "nrs"],
    "physical function":      ["adl", "karnofsky", "ecog"],
    "fatigue":                ["mfsi", "bfi", "facit-fatigue"],
    "dyspnea":                ["breathlessness", "lcq"],
    "emotional function":     ["hads", "phq", "gad"],
    "treatment tolerability": ["pro-ctcae", "ctcae", "crs", "icans"],
    "hrqol":                  ["eq-5d", "sf-36", "qlq-c30"],
    "disease-specific symptoms": ["my20", "qlq-my20"],
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

def linkify_flag_citations(text: str) -> str:
    """
    Convert [Document Name] bracket-citations inside flag strings
    into clickable HTML anchor tags using REGULATORY_CITATIONS.
    Leaves unrecognised brackets untouched.
    """
    def _replace(match):
        inner = match.group(1).strip()
        url = REGULATORY_CITATIONS.get(inner)
        if not url:
            for key, val in REGULATORY_CITATIONS.items():
                prefix = key.split("(")[0].strip()
                if key in inner or inner.startswith(prefix):
                    url = val
                    break
        if url:
            return (
                f'<a href="{url}" target="_blank" '
                f'style="color:#185FA5;font-weight:600;text-decoration:none">'
                f'{inner}</a>'
            )
        return match.group(0)

    return re.sub(r'\[([^\]]+)\]', _replace, text)

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
HAIKU_SYSTEM_PROMPT = """You are extracting clinical trial parameters from natural language input.
The input may be a detailed research brief, a short question, or anything in between.
Read the ENTIRE input before extracting. Extract what is explicitly stated.

KEY EXTRACTION RULES:
1. Do NOT require specific keyword formats. Natural prose is valid.
   "reversible inhibitor of the 26S proteasome" → drug_class = "Proteasome Inhibitor"
   "late-line RRMM with several prior lines" → population_subtype = "Relapsed/Refractory", indication = "Multiple Myeloma"
   "spanning US, EU, and Asia" → geographic_footprint = "Global"
   "FDA and EMA submissions" → regulatory note (affects hta_markets)
   "NICE and ICER" → hta_markets = ["NICE", "ICER"]
   "bortezomib-like mechanism" → drug_class = "Proteasome Inhibitor"
   "proteasome inhibitor-based therapy" → drug_class = "Proteasome Inhibitor"
   "bispecific antibody" → drug_class = "Bispecific"
   "disease-related bone pain, fatigue, physical functioning" → add to core_domains_required AND additional_domains

2. assumptions_made: ONLY list fields you had to INFER because they were NOT in the text.
   If indication, phase, drug_class were all stated: assumptions_made = [].
   Do NOT list successful extractions as assumptions.

3. core_domains_required: FDA 2024 Core PRO Guidance for the indication PLUS any domains
   explicitly mentioned by the user.
   Multiple Myeloma core: disease-related symptoms (bone pain), physical function, fatigue,
                          symptomatic AEs, side effect impact summary, role function
   NSCLC core: dyspnoea, cough, chest pain, physical function, symptomatic AEs
   Default: physical function, fatigue, pain, symptomatic AEs

4. additional_domains: capture domains explicitly asked about beyond the standard core.
   Example: "peripheral neuropathy burden" → additional_domains = ["peripheral neuropathy"]

5. tpp_claims: extract from phrases like "we want to show/capture/demonstrate X".
   If not stated → default to ["treatment tolerability", "physical function maintenance"]
   and flag as inferred in assumptions_made.

INFERENCE RULES (apply ONLY if field NOT in text):
- phase missing → "Phase 3" (flag)
- geographic_footprint missing → Phase 3 = "Global" (flag)
- hta_markets missing → infer from footprint + regulatory agencies mentioned (flag)
- bispecific/CAR-T with no administration stated → "Step-up dosing" (flag)

Return ONLY valid JSON. No markdown. No explanation outside JSON."""


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
  "assumptions_made": ["each inference with reasoning"],
  "additional_domains": ["any domains explicitly requested beyond FDA core, e.g. peripheral neuropathy"]
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
        "trial_duration_cycles": "Unknown",
        "additional_domains": []
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
        get_secret("NEO4J_URI"),
        get_secret("NEO4J_USERNAME"),
        get_secret("NEO4J_PASSWORD")
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

def build_tier1_citation_index(indication: str, phase: str = "Phase 3") -> dict:
    """
    Build a minimal citation index for Tier 1/2 queries that have no prior
    strategy. Fetches KG records for the indication and maps TI-XXX / RR-XXX /
    REJ-XXX labels so Sonnet's answer can be linkified in app.py.
    """
    citation_index = {}
    if not indication or indication.lower() == "unknown":
        return citation_index

    try:
        kg_records = []
        for term in [indication][:3]:
            r = get_instruments_by_indication(term, phase, "")
            if r:
                kg_records.extend(r)

        for i, inst in enumerate(kg_records[:12], 1):
            label = f"TI-{i:03d}"
            nct   = str(inst.get("nct_id", ""))
            doi   = str(inst.get("publication_doi", ""))
            fda   = str(inst.get("fda_label_url", ""))
            ema   = str(inst.get("ema_label_url", ""))
            drug  = inst.get("drug_name", "")
            links = []
            if nct.startswith("NCT"):
                links.append({"label": f"{label} ClinicalTrials.gov",
                               "url": f"https://clinicaltrials.gov/study/{nct}"})
            if doi and doi not in ("nan", "None"):
                links.append({"label": f"{label} Publication",
                               "url": f"https://doi.org/{doi}"})
            if fda.startswith("http"):
                links.append({"label": "FDA label", "url": fda})
            if ema.startswith("http"):
                links.append({"label": "EMA label", "url": ema})
            if not links and drug:
                links.append({
                    "label": f"DailyMed {drug}",
                    "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm"
                           f"?query={drug.replace(' ', '+')}"
                })
            citation_index[label] = {
                "type": "trial_instrument",
                "instrument": inst.get("instrument_name", ""),
                "trial": inst.get("trial_name", "") or nct,
                "nct": nct, "drug": drug,
                "phase": inst.get("phase", ""),
                "key_finding": str(inst.get("key_finding", "") or ""),
                "links": links,
            }

        reg_records  = get_regulatory_evidence(indication, "FDA") or []
        reg_records += get_regulatory_evidence(indication, "EMA") or []
        non_rej = [r for r in reg_records if not r.get("rejection_reason_primary")]
        rej     = [r for r in reg_records if r.get("rejection_reason_primary")]

        for i, rr in enumerate(non_rej[:8], 1):
            label = f"RR-{i:03d}"
            fda   = str(rr.get("fda_label_url", ""))
            ema   = str(rr.get("ema_label_url", ""))
            drug  = rr.get("drug_name", "")
            links = []
            if fda.startswith("http"): links.append({"label": "FDA label", "url": fda})
            if ema.startswith("http"): links.append({"label": "EMA label", "url": ema})
            if not links:
                links.append({
                    "label": f"DailyMed {drug}",
                    "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm"
                           f"?query={drug.replace(' ', '+')}"
                })
            citation_index[label] = {
                "type": "regulatory_review",
                "drug": drug, "agency": rr.get("agency", ""),
                "decision": rr.get("decision", ""),
                "instruments_accepted": rr.get("instruments_accepted", ""),
                "links": links,
            }

        for i, rej_r in enumerate(rej[:8], 1):
            label = f"REJ-{i:03d}"
            fda   = str(rej_r.get("fda_label_url", ""))
            ema   = str(rej_r.get("ema_label_url", ""))
            drug  = rej_r.get("drug_name", "")
            links = []
            if fda.startswith("http"): links.append({"label": "FDA label", "url": fda})
            if ema.startswith("http"): links.append({"label": "EMA label", "url": ema})
            if not links:
                links.append({
                    "label": f"DailyMed {drug}",
                    "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm"
                           f"?query={drug.replace(' ', '+')}"
                })
            citation_index[label] = {
                "type": "rejection",
                "drug": drug, "agency": rej_r.get("agency", ""),
                "decision": rej_r.get("decision", ""),
                "primary_reason": rej_r.get("rejection_reason_primary", ""),
                "detailed_reason": str(rej_r.get("rejection_reason_detailed", "") or ""),
                "links": links,
            }

    except Exception as e:
        logging.warning(f"build_tier1_citation_index failed: {e}")

    return citation_index

# =============================================================================
# STEP 3: SCORING ENGINE
# =============================================================================

def score_evidence(context_json: dict, kg_records: list, instrument_metadata=None, raw_kg_records=None,) -> list:
    """
    Score each instrument on a 0-100 scientific scale plus operational bonuses.
    All penalties are replaced by a structured Risk Flag System.
    Scientific score is never deducted — flags carry severity independently.
    """
    if instrument_metadata is None:
        instrument_metadata = {}
    if raw_kg_records is None:
        raw_kg_records = kg_records

    indication        = _to_str(context_json.get("indication"))
    population        = str(context_json.get("population_subtype", "Symptomatic"))
    phase             = str(context_json.get("phase", "Phase 3"))
    administration    = str(context_json.get("administration", "Unknown"))
    tpp_claims        = [str(c).lower().replace("(inferred)", "").strip()
                         for c in context_json.get("tpp_claims", [])]
    core_domains      = [str(d).lower()
                         for d in context_json.get("core_domains_required", [])]
    geographic_footprint = str(context_json.get("geographic_footprint", "Global"))
    hta_markets       = context_json.get("hta_markets", [])
    drug_class        = _to_str(context_json.get("drug_class"))

    results = []
    instrument_metadata = instrument_metadata or {}

    for record in kg_records:
        instrument_name  = str(record.get("instrument_name", "Unknown"))
        instrument_lower = instrument_name.lower()

        # --- Domain resolution: Instrument node is authoritative, TI fields are fallback ---
        inst_node = instrument_metadata.get(instrument_name, {})
        node_domains = _to_str(inst_node.get("domains_measured", ""))
        node_developer = _to_str(inst_node.get("developer", ""))

        domain_search_parts = [
            node_domains,                                          # PRIMARY: Instrument node
            _to_str(record.get("instrument_domain")),             # FALLBACK: TrialInstrument fields
            _to_str(record.get("domains_measured")),
            _to_str(record.get("key_finding")),
            _to_str(record.get("subscale_results")),
            _to_str(record.get("instrument_subscales_assessed")),
            _to_str(record.get("strengths")),
        ]
        instrument_domains = " ".join(p for p in domain_search_parts if p)

        # Override developer_str with node data if richer
        if node_developer:
            developer_str = node_developer

        instrument_domains_list = [
            d.strip().lower()
            for d in re.split(r"[,;\s]+", instrument_domains)
            if d.strip()
        ]

        # --- Other record fields ---
        mode_options          = _to_str(record.get("mode_options"))
        source_documents      = _to_str(record.get("source_documents"))
        developer_str         = _to_str(inst_node.get("developer") or record.get("developer"))
        endpoint_role         = _to_str(record.get("endpoint_role"))
        prespecified          = _to_str(record.get("prespecified"))
        regulatory_acceptance = _to_str(inst_node.get("regulatory_acceptance") or record.get("regulatory_acceptance"))
        validation_status     = _to_str(inst_node.get("validation") or record.get("validation_status", ""))

        total_items_raw = str(record.get("total_items", ""))
        num_match = re.search(r"\d+", total_items_raw)
        total_items = int(num_match.group()) if num_match else 0

        recall_period = next(
            (days for key, days in INSTRUMENT_RECALL_PERIODS.items()
             if key in instrument_lower),
            RECALL_PERIOD_UNKNOWN
        )
        recall_period_key = next(
            (key for key in INSTRUMENT_RECALL_PERIODS if key in instrument_lower),
            None
        )

        known_lang = next(
            (count for key, count in KNOWN_LANGUAGE_COUNTS.items()
             if key in instrument_lower),
            None
        )
        if known_lang is not None:
            language_count = known_lang
        else:
            languages_val = record.get("languages", "")
            languages_str = str(languages_val).lower()
            if "85" in languages_str or "100" in languages_str or "all major" in languages_str:
                language_count = 100
            elif isinstance(languages_val, list):
                language_count = len([l for l in languages_val if l])
            else:
                language_count = len([l for l in languages_str.split() if l.strip()])

        raw_mcid     = record.get("mcid", "")
        mcid_str     = _to_str(raw_mcid)
        mcid_null    = {"none", "not established", "unknown", "nan", "na",
                        "tbd", "not reported", "pending", "null"}
        mcid_valid   = (
            mcid_str.strip() != "" and
            mcid_str.strip() not in mcid_null and
            not any(t in mcid_str for t in ["not established", "not reported",
                                             "unknown", "pending"])
        )
        mcid_display, _ = clean_mcid(raw_mcid)

        raw_score        = 0
        operational_bonus = 0
        flags            = []
        risk_level       = "LOW"

        # ═══════════════════════════════════════════════════════════════════
        # COMPONENT 1 — DOMAIN & CONTENT FIT  (0–40)
        #
        # Tier 1  +40  All core domains covered + validated in this or a
        #              comparable population
        # Tier 2  +25  All core domains covered, no population-specific
        #              validation
        # Tier 3  +10  Partial domain coverage
        # Tier 4    0  No meaningful overlap
        #
        # HTA utility instruments (EQ-5D, SF-6D) are exempt from the tier
        # system — they provide utility scores for cost-effectiveness
        # analysis, not primary disease measurement.
        #
        # Source: FDA (2021) Core Patient-Reported Outcomes in Cancer
        #         Clinical Trials
        # ═══════════════════════════════════════════════════════════════════
        matched_domains = []
        for domain in core_domains:
            domain_synonyms = DOMAIN_SYNONYMS.get(domain, [domain])
            all_terms = [domain] + domain_synonyms
            if any(term in instrument_domains for term in all_terms):
                matched_domains.append(domain)
        for claim in tpp_claims:
            if claim not in matched_domains:
                claim_synonyms = DOMAIN_SYNONYMS.get(claim, [claim])
                all_terms = [claim] + claim_synonyms
                if any(term in instrument_domains for term in all_terms):
                    matched_domains.append(claim)

        total_required     = len(core_domains)
        all_domains_covered = (total_required > 0 and
                                len(matched_domains) >= total_required)
        any_domain_covered  = len(matched_domains) > 0

        population_lower = population.lower()
        population_validated = (
            population_lower in validation_status.lower() or
            population_lower in _to_str(record.get("key_finding", "")).lower() or
            any(t in validation_status.lower()
                for t in ["validated in", "validated for",
                          "population-specific", "disease-specific"]) or
            indication.lower() in _to_str(record.get("disease_area", "")).lower()
        )

        HTA_UTILITY_INSTRUMENTS = ["eq-5d", "eq5d", "sf-6d", "sf-36"]
        is_hta_utility = any(h in instrument_lower
                              for h in HTA_UTILITY_INSTRUMENTS)

        if is_hta_utility:
            if any_domain_covered:
                raw_score += 10
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "Domain & Content Fit +10 (HTA utility instrument): "
                    f"{instrument_name} is a generic HRQoL measure providing utility "
                    "scores for cost-effectiveness analysis. Partial domain coverage "
                    "is expected by design — include alongside a disease-specific PRO "
                    "[FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials]."
                )))
            else:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "Domain & Content Fit 0 (HTA utility instrument): No core domain "
                    "overlap — include alongside a disease-specific PRO "
                    "[FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials]."
                )))
        elif all_domains_covered and population_validated:
            raw_score += 40
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Domain & Content Fit +40: All core domains covered and instrument "
                f"validated in {population} or a comparable population. "
                "Highest confidence tier — instrument is fit-for-purpose for this "
                "indication [FDA (2021) Core Patient-Reported Outcomes in Cancer "
                "Clinical Trials]."
            )))
        elif all_domains_covered:
            raw_score += 25
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Domain & Content Fit +25: All core domains covered but no "
                f"population-specific validation found for {population}. "
                "Consider commissioning a population-specific validation study "
                "[FDA PRO Guidance (2009) Section IV; FDA (2021) Core PRO Guidance]."
            )))
        elif any_domain_covered:
            n_matched = len(matched_domains)
            missing = [d for d in core_domains if d not in matched_domains]
            raw_score += 10
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"Domain & Content Fit +10: Partial domain coverage — {n_matched} of "
                f"{total_required} required domains matched "
                f"({', '.join(matched_domains)}). "
                f"Missing: {', '.join(missing)}. "
                "[FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials]."
            )))
        else:
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Domain & Content Fit 0: No meaningful overlap with required core "
                "domains. See Critical Domain Failure flag below "
                "[FDA (2021) Core Patient-Reported Outcomes in Cancer Clinical Trials]."
            )))

        raw_score = min(raw_score, 100)

        # ═══════════════════════════════════════════════════════════════════
        # COMPONENT 2 — REGULATORY ACCEPTANCE  (0–25)
        #
        # +25  FDA/EMA label-level acceptance in the same indication
        # +15  Label-level acceptance in a comparable indication or
        #      drug class
        #   0  No precedent
        #
        # Source: FDA PRO Guidance (2009) Section V;
        #         EMA Reflection Paper on PRO (2005)
        # ═══════════════════════════════════════════════════════════════════
        indication_synonyms = [
            s.lower()
            for s in context_json.get("indication_synonyms", [])
        ]
        same_indication_terms = [indication.lower()] + indication_synonyms

        has_label_acceptance = any(
            t in regulatory_acceptance
            for t in ["fda", "ema", "accepted", "approved", "label", "strong"]
        )
        same_indication_match = (
            any(term in regulatory_acceptance for term in same_indication_terms) or
            any(term in _to_str(record.get("disease_area", "")).lower()
                for term in same_indication_terms)
        )

        if has_label_acceptance and same_indication_match:
            raw_score += 25
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Regulatory Acceptance +25: FDA/EMA label-level acceptance documented "
                "in the same indication — highest regulatory confidence tier "
                "[FDA PRO Guidance (2009) Section V; EMA Reflection Paper on "
                "PRO (2005)]."
            )))
        elif has_label_acceptance:
            raw_score += 15
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Regulatory Acceptance +15: FDA/EMA label-level acceptance in a "
                "comparable indication or drug class — strong precedent but "
                "indication-specific validation is recommended "
                "[FDA PRO Guidance (2009) Section V; EMA Reflection Paper on "
                "PRO (2005)]."
            )))
        elif any(t in regulatory_acceptance
                  for t in ["moderate", "conditional", "exploratory"]):
            raw_score += 15
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Regulatory Acceptance +15: Conditional/moderate acceptance "
                "documented in a comparable indication or drug class "
                "[FDA PRO Guidance (2009) Section V]."
            )))
        else:
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Regulatory Acceptance 0: No FDA/EMA label-level precedent found. "
                "Prospective alignment meeting recommended before protocol lock "
                "[FDA PRO Guidance (2009) Section V; EMA Reflection Paper on "
                "PRO (2005)]."
            )))

        raw_score = min(raw_score, 100)
# ═══════════════════════════════════════════════════════════════════
        # COMPONENT 3 — VALIDATED MCID  (0–20, gated)
        #
        # Scored in two dimensions:
        #  (a) Method quality: anchor-based (+20) > distribution-based (+12)
        #      Anchor-based is preferred by FDA because it reflects patient
        #      perspective of change, not statistical distribution.
        #  (b) Population alignment: same indication > adjacent haematological
        #      > general oncology > unknown/non-oncology
        #      An MCID in breast cancer is not equivalent to one in RRMM.
        #
        # Gate: if NO validated MCID exists, score is hard-capped at 75.
        # Rationale: without MCID, responder analysis is impossible, which
        # limits label claim language to mean change statistics only.
        #
        # Source: FDA PRO Guidance (2009) Section V.C
        # ═══════════════════════════════════════════════════════════════════
        if mcid_valid:
            mcid_full_text = _to_str(raw_mcid).lower()

            # Dimension (a): method quality
            anchor_terms = [
                "anchor", "patient global", "pgic", "external criterion",
                "anchor-based", "anchor based", "clinician global",
                "meaningful to patients", "patient-meaningful"
            ]
            dist_terms = [
                "distribution", "sem", "standard error", "effect size",
                "half sd", "0.5 sd", "distribution-based", "distribution based",
                "responsiveness statistic"
            ]
            is_anchor_based = any(t in mcid_full_text for t in anchor_terms)
            is_dist_only    = (any(t in mcid_full_text for t in dist_terms)
                                and not is_anchor_based)
            mcid_method_pts = 12 if is_dist_only else 20
            mcid_method_str = "distribution-based" if is_dist_only else (
                "anchor-based" if is_anchor_based else "established"
            )

            # Dimension (b): population alignment — what population was MCID established in?
            # Check MCID text AND the instrument's disease_area field
            mcid_pop_text = mcid_full_text + " " + _to_str(record.get("disease_area",""))
            ind_lower     = indication.lower()
            ind_synonyms  = [s.lower() for s in context_json.get("indication_synonyms",[])]
            all_ind_terms = [ind_lower] + ind_synonyms

            if any(t in mcid_pop_text for t in all_ind_terms + ["myeloma","mm","rrmm"]):
                pop_tier = "same indication"
                pop_mult = 1.0   # full points
                pop_note = f"MCID established in {indication} or directly comparable population — highest confidence."
            elif any(t in mcid_pop_text for t in ["haematol","hematol","lymphoma","leukemia","blood cancer"]):
                pop_tier = "adjacent haematological malignancy"
                pop_mult = 0.75  # 75% of method points
                pop_note = f"MCID established in adjacent haematological malignancy — applicable but not indication-specific. Consider whether a {indication}-specific MCID study is feasible."
            elif any(t in mcid_pop_text for t in ["cancer","oncol","tumour","tumor","carcinoma"]):
                pop_tier = "general oncology"
                pop_mult = 0.5   # 50% of method points
                pop_note = f"MCID established in general oncology population — may not reflect {indication} patients' experience of meaningful change. Human review recommended."
            else:
                pop_tier = "unknown or non-oncology population"
                pop_mult = 0.35  # 35% of method points
                pop_note = f"MCID population unclear or from non-oncology setting — applicability to {indication} uncertain. Commission {indication}-specific MCID study."

            mcid_pts = max(1, round(mcid_method_pts * pop_mult))
            raw_score += mcid_pts

            flags.append(linkify_flag_citations(ensure_full_stop(
                f"Validated MCID +{mcid_pts} ({mcid_method_str}, {pop_tier}): "
                f"{mcid_display}. {pop_note} "
                f"[FDA PRO Guidance (2009) Section V.C]."
            )))

            # Distribution-based method warning
            if is_dist_only:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "⚠️ MCID METHOD NOTE: Distribution-based MCID only. "
                    "FDA PRO Guidance (2009) Section V.C explicitly prefers "
                    "anchor-based methods for responder analysis supporting "
                    "label claims. Distribution-based MCIDs may not be accepted "
                    "by FDA as the primary responder threshold."
                )))

        else:
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Validated MCID 0: No validated MCID found in KG for this instrument. "
                "Score capped at 75 — without MCID, responder analysis is impossible "
                "and label claim language is limited to mean change statistics. "
                "Verify via PROQOLID (proqolid.org) or instrument developer "
                "[FDA PRO Guidance (2009) Section V.C]."
            )))

        # Apply MCID gate: cap at 75 if no valid MCID
        if not mcid_valid:
            raw_score = min(raw_score, 75)

        raw_score = min(raw_score, 100)

        # ═══════════════════════════════════════════════════════════════════
        # COMPONENT 4 — MoA SENSITIVITY  (0–15)
        #
        # +15  Full coverage of mechanism-specific toxicity domains
        #      (≥65 % of mechanism domains captured)
        #  +8  Partial coverage
        #   0  None
        #
        # Source: FDA PFDD Guidance 1 (2017)
        # ═══════════════════════════════════════════════════════════════════
        MOA_KEYWORDS = {
            "bispecific":             ["cytokine release", "crs", "fatigue",
                                       "neurotoxicity", "icans", "infection"],
            "car-t":                  ["cytokine release", "crs", "fatigue",
                                       "neurotoxicity", "icans"],
            "proteasome inhibitor":   ["peripheral neuropathy", "neuropathy", "fatigue"],
            "ici":                    ["fatigue", "immune-related", "diarrhea",
                                       "endocrine", "colitis"],
            "cdk4/6":                 ["fatigue", "nausea", "neutropenia"],
            "antibody drug conjugate":["nausea", "fatigue", "neuropathy", "alopecia"],
            "bcma":                   ["fatigue", "infection", "crs",
                                       "neurotoxicity", "cytokine release"],
        }
        moa_required_domains = []
        moa_matched_domains  = []
        for class_key, tox_domains in MOA_KEYWORDS.items():
            if class_key in drug_class:
                moa_required_domains = tox_domains
                moa_matched_domains  = [t for t in tox_domains
                                         if t in instrument_domains]
                break

        if moa_required_domains:
            coverage_ratio = (len(moa_matched_domains) /
                               len(moa_required_domains))
            missing_moa = [d for d in moa_required_domains
                            if d not in moa_matched_domains]
            if coverage_ratio >= 0.65:
                raw_score += 15
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"MoA Sensitivity +15: Full coverage of mechanism-specific "
                    f"toxicity domains for {drug_class} "
                    f"({', '.join(moa_matched_domains)}) "
                    "[FDA PFDD Guidance 1 (2017)]."
                )))
            elif moa_matched_domains:
                raw_score += 8
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"MoA Sensitivity +8 (partial): Captures "
                    f"{len(moa_matched_domains)} of "
                    f"{len(moa_required_domains)} mechanism-specific domains "
                    f"({', '.join(moa_matched_domains)}). "
                    f"Missing: {', '.join(missing_moa)}. "
                    "[FDA PFDD Guidance 1 (2017)]."
                )))
            else:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"MoA Sensitivity 0: No mechanism-specific toxicity domains "
                    f"captured for {drug_class}. "
                    f"Key missing domains: {', '.join(moa_required_domains[:4])}. "
                    "[FDA PFDD Guidance 1 (2017)]."
                )))

        raw_score = min(raw_score, 100)

        # ═══════════════════════════════════════════════════════════════════
        # COMPONENT 4b — CHANGE DETECTION IN PRECEDENT TRIALS  (informational)
        #
        # This is NOT a scored component — it is a separate signal that COA
        # strategists use when reviewing comparator analysis. A high scientific
        # score on criteria 1-4 does not guarantee the instrument will detect
        # change in this trial context. KG evidence of actual change detected
        # in similar trials is the most direct evidence available.
        #
        # Adds +5 bonus if significant change was detected in ≥1 KG trial
        # for this instrument in the same or comparable indication.
        # This is a small bonus because detection in a different trial context
        # does not guarantee detection in the current one.
        #
        # Source: COA strategist practice — comparator analysis
        # ═══════════════════════════════════════════════════════════════════
        # Find all KG records for this specific instrument
        # (record is the current record; we need to check the full kg_records list
        #  but score_evidence only receives records not the full list)
        # Instead, use the current record's own significance/direction fields

        all_records_for_instrument = [
            r for r in raw_kg_records
            if r.get("instrument_name") == instrument_name
        ]

        # Find the best positive-change record (significant improvement in any trial)
        best_positive = next(
            (
                r for r in all_records_for_instrument
                if any(t in _to_str(r.get("significance", ""))
                    for t in ["significant", "p <", "p<", "p=0.0", "favours", "favor"])
                and any(t in _to_str(r.get("direction", ""))
                        for t in ["favour", "favor", "improvement", "better", "positive"])
            ),
            None
        )

        # Find the best null-change record (only relevant if no positive record exists)
        best_null = next(
            (
                r for r in all_records_for_instrument
                if any(t in _to_str(r.get("significance", ""))
                    for t in ["not significant", "no significant", "ns ", "p > 0", "p>0"])
            ),
            None
        ) if not best_positive else None

        change_detected = best_positive is not None
        change_null = best_null is not None

        significance_excerpt = ""
        precedent_trial_name = "a KG trial"

        if best_positive:
            significance_excerpt = str(best_positive.get("significance", "")).strip()[:80]
            precedent_trial_name = best_positive.get("trial_name", "") or "a KG trial"
        elif best_null:
            precedent_trial_name = best_null.get("trial_name", "") or "a KG trial"

        if change_detected:
            raw_score += 5
            detail = f" ({significance_excerpt})" if significance_excerpt else ""
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"Change Detected +5: KG record for {instrument_name} in "
                f"{precedent_trial_name} shows statistically significant improvement"
                f"{detail}. "
                "This is a comparator signal, not a guarantee for the current trial — "
                "trial design and population differences apply."
            )))
        elif change_null:
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"Change Not Detected (informational): {instrument_name} showed no "
                f"statistically significant change in {precedent_trial_name}. "
                "Review whether trial design, sample size, or patient population "
                "explains the null result before relying on this instrument for primary endpoints."
            )))


        # ═══════════════════════════════════════════════════════════════════
        # RISK FLAG SYSTEM
        #
        # Flags communicate specific, named risks independently of the score.
        # A CRITICAL flag means the instrument should not be ranked first
        # regardless of its numerical score.  No numeric deductions are made.
        #
        # 🔴 CRITICAL — instrument-disqualifying if unresolved
        # 🟠 HIGH     — materially limits label claim strength
        # 🟡 MODERATE — addressable risk requiring documented mitigation
        # ═══════════════════════════════════════════════════════════════════

        # ── 🔴 CRITICAL: CRITICAL DOMAIN FAILURE ──────────────────────────
        # Domain & Content Fit = 0 in a symptomatic/active-disease population.
        SYMPTOMATIC_TERMS = [
            "symptomatic", "relapsed", "refractory", "relapsed/refractory",
            "metastatic", "advanced", "progressive", "active disease",
            "previously treated", "heavily pretreated", "rrmm", "rrbc",
            "first-line", "second-line", "later-line", "newly diagnosed",
            "treatment-naive",
        ]
        is_symptomatic = any(term in population.lower()
                              for term in SYMPTOMATIC_TERMS)
        if (is_symptomatic and not is_hta_utility and
                not any_domain_covered and total_required > 0):
            risk_level = "CRITICAL"
            flags.append(linkify_flag_citations(ensure_full_stop(
                "🔴 CRITICAL FLAG — CRITICAL DOMAIN FAILURE: "
                "Domain & Content Fit = 0. Instrument has no meaningful overlap "
                "with FDA-required core domains for this indication. "
                "Per FDA (2021) 'Core Patient-Reported Outcomes in Cancer "
                "Clinical Trials', failure to measure core domains risks "
                "Refusal to File or PRO label claim rejection. "
                "This instrument should NOT be ranked first for this indication."
            )))

        # ── 🔴 CRITICAL: INSTRUMENT-ATTRIBUTED REJECTION ──────────────────
        # A prior CRL explicitly names the instrument as the cause of rejection.
        rejection_reason = (
            _to_str(record.get("rejection_reason_primary", "")) + " " +
            _to_str(record.get("rejection_reason_detailed", ""))
        )
        instrument_rejection_terms = [
            "instrument", "questionnaire", "content validity",
            "recall period", "cross-cultural", "linguistic validation",
            "not fit for purpose",
        ]
        if (any(t in rejection_reason.lower()
                 for t in instrument_rejection_terms) and
                any(t in rejection_reason.lower()
                     for t in ["reject", "refuse", "crl", "complete response"])):
            risk_level = "CRITICAL"
            flags.append(linkify_flag_citations(ensure_full_stop(
                "🔴 CRITICAL FLAG — INSTRUMENT-ATTRIBUTED REJECTION: "
                "Regulatory record indicates a prior CRL/rejection explicitly "
                "citing the instrument as the cause of failure (content validity, "
                "recall period, or cross-cultural validity concerns). "
                "This is distinct from trial design failures and carries the "
                "highest regulatory risk. Seek FDA alignment meeting before "
                "adopting this instrument."
            )))

        # ── 🔴 CRITICAL: RECALL INCOMPATIBILITY ───────────────────────────
        # Recall window cannot temporally capture the mechanism's key symptom
        # events (e.g. 7-day recall in bispecific step-up dosing).
        STEP_UP_ADMINS = ["step-up dosing", "weekly iv", "weekly"]
        is_step_up = any(a in administration.lower() for a in STEP_UP_ADMINS)
        if is_step_up:
            if recall_period == RECALL_PERIOD_UNKNOWN:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"RECALL PERIOD UNKNOWN: {instrument_name} recall period "
                    "not in reference database. "
                    f"For {administration}, FDA PFDD Guidance 3 (2025) requires "
                    "recall to match symptom fluctuation — CRS/ICANS events "
                    "occur within 24–72 hours of dosing. "
                    "Sonnet has been instructed to verify the official recall "
                    "period via web search."
                )))
            elif recall_period > 3:
                if risk_level != "CRITICAL":
                    risk_level = "CRITICAL"
                citation = (f"per {recall_period_key} validation"
                             if recall_period_key else "per published validation")
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"🔴 CRITICAL FLAG — RECALL INCOMPATIBILITY: "
                    f"{recall_period}-day recall window cannot capture "
                    f"{administration} mechanism's key symptom events "
                    f"({citation}). "
                    "CRS/ICANS events occur within 24–72 hours of dosing — "
                    f"a {recall_period}-day window structurally misses peak "
                    "symptom severity. "
                    "Per FDA PFDD Guidance 3 (2025), recall must match "
                    "symptom fluctuation pattern."
                )))
            else:
                source = recall_period_key or "published validation"
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"Recall period compatible: {instrument_name} has "
                    f"{recall_period}-day recall ({source}), compatible with "
                    f"{administration} [FDA PFDD Guidance 3 (2025)]."
                )))

        # ── 🟠 HIGH: NOT PRE-SPECIFIED IN SAP ─────────────────────────────
        # Protocol decision — not an instrument defect.
        has_explicit_record = (
            instrument_name != "Unknown" and
            (prespecified != "" or endpoint_role != "")
        )
        if (has_explicit_record and
                prespecified not in ["yes", "true", "1"] and
                endpoint_role in ["exploratory", "unknown"]):
            if risk_level not in ["CRITICAL"]:
                risk_level = "HIGH"
            flags.append(linkify_flag_citations(ensure_full_stop(
                "🟠 HIGH FLAG — NOT PRE-SPECIFIED IN SAP: KG record shows "
                "instrument was not pre-specified with alpha controlled. "
                "Results will be exploratory only — cannot support formal "
                "label claims. Note: this is a sponsor protocol decision, "
                "not an instrument defect. Pre-specify before first patient in "
                "[FDA PRO Guidance (2009) Section V; "
                "ICH E9 (1998) Section 2.2.5]."
            )))

        # ── 🟠 HIGH: ESTIMAND BURDEN ───────────────────────────────────────
        # Assessed at battery level (>50 total items across the battery).
        if (("phase 3" in phase.lower() or "phase iii" in phase.lower())
                and total_items > 50):
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "HIGH"
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"🟠 HIGH FLAG — ESTIMAND BURDEN: {total_items}-item "
                "instrument in Phase 3. "
                "Note: estimand burden is assessed at battery level "
                "(>50 total items across the battery). "
                "ICH E9(R1) Addendum (2019) requires Treatment Policy "
                "estimand — PRO collection must continue post-"
                "discontinuation. Battery-level completion rates are "
                "adversely affected at this item count. "
                "Consider subscale approach or shorter companion instrument."
            )))

        # ── 🟡 MODERATE: MODE EQUIVALENCE GAP ────────────────────────────
        # Trial uses eCOA; instrument paper-validated only;
        # no published equivalence study.
        paper_validated = any(
            t in validation_status.lower()
            for t in ["paper", "pen-and-paper", "paper-based", "paper version"]
        )
        is_ecoa_available = any(
            t in mode_options.lower()
            for t in ["ecoa", "electronic", "app", "tablet", "digital"]
        )
        no_equivalence_study = not any(
            t in validation_status.lower()
            for t in ["equivalence", "mode equivalence",
                      "electronic validation", "ecoa validation"]
        )
        if is_ecoa_available and paper_validated and no_equivalence_study:
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "MODERATE"
            flags.append(linkify_flag_citations(ensure_full_stop(
                "🟡 MODERATE FLAG — MODE EQUIVALENCE GAP: Trial uses eCOA "
                "but instrument was originally validated on paper only. "
                "No published mode equivalence study found. "
                "FDA and EMA require evidence that electronic scores are "
                "psychometrically equivalent to paper scores. "
                "Commission a mode equivalence study or identify published "
                "equivalence data before finalising eCOA configuration."
            )))

        # ── 🟡 MODERATE: ASYMPTOMATIC POPULATION MISMATCH ────────────────
        SYMPTOM_HEAVY = [
            "bpi", "bone pain", "nrs", "pain intensity", "symptom",
            "facit-fatigue", "mfsi", "brief fatigue", "pain catastrophizing",
            "nausea", "dyspnea", "appetite"
        ]
        is_symptom_heavy = any(
            s in instrument_lower or s in instrument_domains
            for s in SYMPTOM_HEAVY
        )
        if ("asymptomatic" in population.lower() or
                "smoldering" in population.lower()):
            if is_symptom_heavy:
                if risk_level not in ["CRITICAL", "HIGH"]:
                    risk_level = "MODERATE"
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "🟡 MODERATE FLAG — ASYMPTOMATIC POPULATION MISMATCH: "
                    "Symptom-heavy instrument applied to asymptomatic/"
                    "smoldering population. "
                    "Measuring symptoms the patient does not have causes "
                    "questionnaire fatigue "
                    "[FDA PRO Guidance (2009) Section IV.B; "
                    "FDA PFDD Guidance 2 (2018)]. "
                    "Consider HRQoL-focused instrument (EQ-5D-5L, FACT-G)."
                )))

        # ═══════════════════════════════════════════════════════════════════
        # OPERATIONAL BONUS  (additive, not capped by scientific score)
        #
        # eCOA Ready     0–40   Electronic/app-based administration available
        # Open Access    0–25   Developed by EORTC, NCI, FACIT, WHO, RAND,
        #                       PCORI, or NIH
        # Translation      -5   >0 but <50 validated translations (global/EU)
        # Translation     -10   No translation information available at all
        #
        # Source: FDA eCOA Guidance (2023); FDA PRO Guidance (2009) §IV.A;
        #         EMA Reflection Paper on PRO (2005)
        # ═══════════════════════════════════════════════════════════════════

        # ── eCOA Ready (0–40) ─────────────────────────────────────────────
        if any(t in mode_options
               for t in ["ecoa", "electronic", "app", "tablet", "digital"]):
            operational_bonus += 8
            flags.append(linkify_flag_citations(ensure_full_stop(
                "eCOA Ready +8: Electronic/app-based "
                "administration mode available — reduces transcription error "
                "and enables real-time monitoring "
                "[FDA eCOA Guidance (2023)]."
            )))

        # ── Open Access (0–25) ────────────────────────────────────────────
        OPEN_ACCESS_DEVS = [
            "eortc", "nci", "national cancer institute", "facit",
            "rand", "who", "world health organization", "nih", "pcori",
        ]
        if any(d in developer_str or d in source_documents or d in instrument_lower
               for d in OPEN_ACCESS_DEVS):
            operational_bonus += 5
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Open Access +5 (operational): Instrument developed by an "
                "open-access organisation (EORTC, NCI, FACIT, WHO, RAND, "
                "PCORI, or NIH) — no commercial licensing fees, publicly "
                "maintained, and freely available for trial use."
            )))

        # ── Translation coverage ──────────────────────────────────────────
        if geographic_footprint in ("Global", "EU"):
            geo          = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(
                               geographic_footprint,
                               GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
            key_langs    = geo["key_languages"]
            if language_count >= 50:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"Language coverage: {instrument_name} has approximately "
                    f"{language_count} validated translations — strong coverage "
                    f"for a {geographic_footprint} trial. "
                    "Verify specific language availability for trial sites "
                    "[FDA PRO Guidance (2009) Section IV.A]."
                )))
            elif language_count > 0:
                operational_bonus -= 5
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"Limited translation -5 (operational): {instrument_name} "
                    f"has approximately {language_count} validated translations. "
                    f"For a {geographic_footprint} trial, verify coverage for "
                    f"{', '.join(key_langs[:6])}. "
                    "Commission additional translations if needed "
                    "(typically 6–12 months) "
                    "[FDA PRO Guidance (2009) Section IV.A; "
                    "ISPOR ePRO Task Force (2009)]."
                )))
            else:
                operational_bonus -= 10
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"No translation data -10 (operational): No translation "
                    f"information available for {instrument_name}. "
                    "Sonnet instructed to verify via web search or PROQOLID. "
                    "Linguistically validated translations are required for "
                    "all trial languages "
                    "[FDA PRO Guidance (2009) Section IV.A]."
                )))

        # ═══════════════════════════════════════════════════════════════════
        # HTA ALIGNMENT NOTES  (informational — no score impact)
        #
        # Flags whether the instrument supports cost-utility analysis for
        # each HTA body in scope.  These are advisory notes, not risk flags.
        #
        # Source: NICE DSU Technical Support Document 2 (2019);
        #         ICER Value Assessment Framework (2020);
        #         EUnetHTA Methodological Guideline on HRQoL (2021)
        # ═══════════════════════════════════════════════════════════════════

        if "NICE" in hta_markets:
            if any(u in instrument_lower for u in ["eq-5d", "eq5d"]):
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "HTA Alignment — NICE: EQ-5D included — supports QALY "
                    "calculation for NICE cost-utility analysis. "
                    "UK reimbursement submission can proceed as planned "
                    "[NICE DSU Technical Support Document 2 (2019)]."
                )))
            else:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "HTA NOTE — NICE: This instrument alone cannot support "
                    "QALY-based cost-utility analysis. "
                    "EQ-5D-5L must be included alongside this instrument "
                    "for UK market access. Without it, NICE cannot calculate "
                    "a cost-per-QALY and will request additional data, "
                    "delaying reimbursement "
                    "[NICE DSU Technical Support Document 2 (2019)]."
                )))

        if "ICER" in hta_markets:
            if any(u in instrument_lower
                   for u in ["eq-5d", "eq5d", "sf-6d", "sf-36"]):
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "HTA Alignment — ICER: Utility-based measure included — "
                    "supports ICER cost-effectiveness analysis. "
                    "US value assessment submission can proceed as planned "
                    "[ICER Value Assessment Framework (2020)]."
                )))
            else:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "HTA NOTE — ICER: No utility-based measure detected. "
                    "ICER cost-effectiveness models require a preference-based "
                    "utility score (EQ-5D-5L, SF-6D). "
                    "Include EQ-5D-5L alongside this instrument to support "
                    "US value assessment submissions "
                    "[ICER Value Assessment Framework (2020)]."
                )))

        if "EUnetHTA" in hta_markets:
            if any(u in instrument_lower for u in ["eq-5d", "eq5d"]):
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "HTA Alignment — EUnetHTA: EQ-5D included — satisfies "
                    "EU Joint Clinical Assessment HRQoL data requirement. "
                    "Cross-country comparability is supported "
                    "[EUnetHTA Methodological Guideline on HRQoL (2021); "
                    "EU Regulation 2021/2282]."
                )))
            else:
                flags.append(linkify_flag_citations(ensure_full_stop(
                    "HTA NOTE — EUnetHTA: EU Joint Clinical Assessment under "
                    "Regulation 2021/2282 requires comparable HRQoL data "
                    "across member states. "
                    "EQ-5D-5L is strongly preferred for cross-country "
                    "comparability. Include alongside this instrument "
                    "[EUnetHTA Methodological Guideline on HRQoL (2021)]."
                )))

        # ═══════════════════════════════════════════════════════════════════
        # OVERLAP DETECTION  (data-driven from KG Domains_Measured field)
        #
        # Rather than hardcoding known pairs, compute overlap dynamically
        # from the instrument's own domain data vs other scored instruments.
        # Sonnet receives the overlap signal and contextualises it.
        # Threshold: ≥2 shared domains = worth flagging.
        # ═══════════════════════════════════════════════════════════════════
        this_domains = set(
            d.strip().lower()
            for part in [
                _to_str(record.get("domains_measured","")),
                _to_str(record.get("instrument_domain","")),
            ]
            for d in re.split(r"[,;/]", part)
            if d.strip() and len(d.strip()) > 3
        )
        if this_domains:
            overlap_notes = []
            for other in kg_records:
                other_name = str(other.get("instrument_name",""))
                if other_name == instrument_name or not other_name:
                    continue
                other_domains = set(
                    d.strip().lower()
                    for part in [
                        _to_str(other.get("domains_measured","")),
                        _to_str(other.get("instrument_domain","")),
                    ]
                    for d in re.split(r"[,;/]", part)
                    if d.strip() and len(d.strip()) > 3
                )
                shared = this_domains & other_domains
                if len(shared) >= 2:
                    overlap_notes.append(
                        f"{other_name} (shared: {', '.join(list(shared)[:3])})"
                    )
            if overlap_notes:
                # Deduplicate — only flag once per unique other instrument
                seen = set()
                unique_notes = []
                for n in overlap_notes:
                    key = n.split(" (shared")[0]
                    if key not in seen:
                        seen.add(key)
                        unique_notes.append(n)
                flags.append(ensure_full_stop(
                    f"⚠️ OVERLAP NOTE: {instrument_name} shares ≥2 domains with: "
                    f"{'; '.join(unique_notes[:3])}. "
                    "Including both instruments creates respondent burden without "
                    "additional regulatory value. Consider item library approach "
                    "or selecting one. Decision for the COA expert."
                ))

        # ═══════════════════════════════════════════════════════════════════
        # FINAL SCORES
        # scientific_score  — sum of the four components (0–100, no deductions)
        # final_adjusted_score — scientific + operational bonus
        # ═══════════════════════════════════════════════════════════════════
        scientific_score = raw_score   # penalties converted to flags; no deductions

        results.append({
            "instrument_name":    instrument_name,
            "scientific_score":   scientific_score,
            "raw_positive_score": raw_score,
            "operational_bonus":  operational_bonus,
            "final_adjusted_score": scientific_score + operational_bonus,
            "risk_level":         risk_level,
            "flags":              flags,
            "drug_name":          record.get("drug_name", ""),
            "trial_name":         record.get("trial_name", ""),
            "nct_id":             record.get("nct_id", ""),
            "phase":              record.get("phase", ""),
            "disease_area":       record.get("disease_area", ""),
            "patient_population": record.get("patient_population", ""),
            "proposition":        record.get("proposition", ""),
            "key_finding":        record.get("key_finding", ""),
            "compliance_rate":    record.get("compliance_rate", ""),
            "assessment_schedule":record.get("assessment_schedule", ""),
            "publication_doi":    record.get("publication_doi", ""),
            "publication_year":   record.get("publication_year", ""),
            "p_value":            record.get("p_value", ""),
            "effect_size":        record.get("effect_size", ""),
            "fda_label_url":      record.get("fda_label_url", ""),
            "ema_label_url":      record.get("ema_label_url", ""),
            "key_toxicities":     record.get("key_toxicities", ""),
            "validation_status":  record.get("validation_status", ""),
            "strengths":          record.get("strengths", ""),
            "limitations":        record.get("limitations", ""),
            "recall_period":      recall_period,
            "language_count":     language_count,
        })

    risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    results.sort(key=lambda x: (risk_order.get(x["risk_level"], 4),
                                 -x["scientific_score"]))
    return results

# =============================================================================
# STEP 4: BATTERY OPTIMIZER
# =============================================================================
def build_coverage_matrix(scored: list, context_json: dict, raw_kg_records: list, instrument_metadata = None,) -> dict:
    """
    Build a domain coverage matrix showing which instruments cover which FDA core domains.
    This replaces the battery optimizer — we show options, experts decide.

    Returns:
      domains: list of {domain, candidates, item_library_note, is_fda_core}
      comparator_trials: list of {trial_name, drug, drug_class, instruments, nct_id}
      hta_mandatory: list of instruments always required regardless of score
      item_library_applicable: bool — True if any trial used subscales/item library
      all_candidates: top 8 scored instruments with full data
    """
    if instrument_metadata is None:
        instrument_metadata = {}

    # Use the same keys as everywhere else in context_json / analyzer
    indication = _to_str(context_json.get("indication"))
    hta_markets = context_json.get("htamarkets", [])  # NOT "hta_markets"
    drug_class = _to_str(context_json.get("drugclass"))
    core_domains = [str(d).lower() for d in context_json.get("coredomainsrequired", [])]
    extra_domains = [str(d).lower() for d in context_json.get("additionaldomains", [])]
    all_domains = list(dict.fromkeys(core_domains + extra_domains))

    domain_coverage = []
    for domain in all_domains:
        synonyms = DOMAIN_SYNONYMS.get(domain, [domain])
        all_terms = [domain] + synonyms
        candidates = []

        for inst in scored:
            inst_name = inst["instrument_name"]
            inst_lower = inst_name.lower()
            inst_node = instrument_metadata.get(inst_name, {})
            node_domains = _to_str(inst_node.get("domains_measured", ""))

            # Keep search text tied to real metadata + scored explanations
            flags_text = " ".join(inst.get("flags", [])).lower()
            search_text = " ".join([inst_lower, node_domains, flags_text])

            if any(t in search_text for t in all_terms):
                kg_matches = [
                    r for r in raw_kg_records
                    if r.get("instrument_name") == inst_name
                ]

                change_detected = "Yes" if any(
                    any(t in _to_str(r.get("significance", "")) for t in ["significant", "p <", "p<", "favours", "favor"])
                    and any(t in _to_str(r.get("direction", "")) for t in ["favour", "favor", "improvement", "better", "positive"])
                    for r in kg_matches
                ) else ("No" if kg_matches else "NR")

                precedent = next(
                    (r for r in kg_matches if r.get("trial_name") or r.get("nctid")),
                    {}
                )

                candidates.append({
                    "instrument": inst_name,
                    "score": inst["scientific_score"],
                    "risk": inst["risk_level"],
                    "change_detected": change_detected,
                    "precedent_trial": precedent.get("trial_name", ""),
                    "precedent_nct": precedent.get("nctid", ""),
                    "prevalence": inst.get("_prevalence", 1),
                })

        candidates.sort(key=lambda x: (-x["score"], -x["prevalence"], x["instrument"]))

        item_library_note = ""
        if candidates:
            best_name = candidates[0]["instrument"]
            best_rec = next((s for s in scored if s["instrument_name"] == best_name), None)
            items_raw = best_rec.get("total_items", 0) if best_rec else 0
            try:
                n_items = int(items_raw) if items_raw else 0
            except Exception:
                n_items = 0

            subscale_used = any(
                r.get("subscale_results") or r.get("instrument_subscales_assessed")
                for r in raw_kg_records
                if r.get("instrument_name") == best_name
            )
            if n_items >= 30 and subscale_used:
                item_library_note = (
                    f"Note: Comparator trials have used subscale/item-library approaches with "
                    f"{best_name} rather than full administration."
                )

        domain_coverage.append({
            "domain": domain,
            "candidates": candidates,
            "item_library_note": item_library_note,
            "is_fda_core": domain in core_domains,
        })

    # HTA mandatory instruments (EQ-5D wildcards)
    htamandatory = []
    hta_wildcards = {
        "NICE": "eq-5d",
        "ICER": "eq-5d",
        "EUnetHTA": "eq-5d",
        "SMC": "eq-5d",
    }

    for market in hta_markets:
        wildcard = hta_wildcards.get(market)
        if wildcard:
            eq5d_in_scored = [i for i in scored if wildcard in i["instrument_name"].lower()]
            score_val = eq5d_in_scored[0]["scientific_score"] if eq5d_in_scored else "Not in KG"
            already_listed = any(
                wildcard in c["instrument"].lower()
                for d in domain_coverage for c in d["candidates"]
            )
            if not already_listed:
                htamandatory.append({
                    "instrument": "EQ-5D-5L",
                    "market": market,
                    "score": score_val,
                    "reason": "Required for cost-utility / QALY analysis."
                })

    # Comparator trials: use raw_kg_records (one row per trial×instrument)
    comparator_map = {}
    for r in raw_kg_records:
        trial_name = r.get("trial_name") or r.get("nctid")
        if not trial_name:
            continue
        if trial_name not in comparator_map:
            comparator_map[trial_name] = {
                "trial_name": trial_name,
                "drug": r.get("drug_name", ""),
                "drug_class": r.get("drugclassname") or r.get("diseaseclassification", ""),
                "nct_id": r.get("nctid", ""),
                "phase": r.get("phase", ""),
                "instruments": [],
            }
        comparator_map[trial_name]["instruments"].append({
            "name": r.get("instrument_name", ""),
            "role": r.get("endpoint_role") or r.get("proposition", ""),
            "significance": r.get("significance", ""),
            "direction": r.get("direction", ""),
            "prespecified": r.get("prespecified", ""),
            "subscales": r.get("instrument_subscales_assessed", ""),
        })

    comparators = list(comparator_map.values())
    comparators.sort(
        key=lambda x: (
            0 if any(term in _to_str(x["drug_class"]) for term in drug_class.split()) else 1,
            x["trial_name"]
        )
    )

    item_library_applicable = any(
        r.get("instrument_subscales_assessed") or r.get("subscale_results")
        for r in raw_kg_records
    )

    return {
        "domains": domain_coverage,
        "comparator_trials": comparators[:5],
        "hta_mandatory": htamandatory,
        "item_library_applicable": item_library_applicable,
        "all_candidates": scored[:8],
    }

def build_pro_measures_table(
    coverage: dict,
    inst_refs: list,
    rawkgrecords: list,
    contextjson: dict,
) -> list:
    """
    Build a PRO measures comparison table (Table 2) purely in Python.

    Returns a list of row dicts with keys:
      - trial: trial name
      - year: publication year (or "TBD"/"Not reported")
      - drug: drug name
      - drug_class: mechanism / class
      - pro_measures: "Instrument1 (n=30), Instrument2 (n=20)..."
      - calibrated_soa: "Yes" / "No" / "TBD"
      - total_items: int or None
      - est_time_min: float or None
    """

    def _norm(name: str) -> str:
        return (name or "").strip().lower()

    # Index instrument reference nodes by (shortname / instrumentname)
    ref_index = {}
    for ir in inst_refs or []:
        candidates = [
            ir.get("shortname"),
            ir.get("instrumentname"),
            ir.get("instrument_name"),
        ]
        for c in candidates:
            if c:
                ref_index[_norm(c)] = ir

    # Index raw KG rows by trial name / NCT for year lookup
    trial_years = {}
    for r in rawkgrecords or []:
        tname = r.get("trialname") or r.get("trial_name") or r.get("nctid") or r.get("nct_id")
        if not tname:
            continue
        year = r.get("publicationyear") or r.get("publication_year")
        if year and tname not in trial_years:
            trial_years[tname] = year

    rows = []

    # Comparator trials from coverage
    for trial in coverage.get("comparator_trials", []):
        tname = trial.get("trialname") or trial.get("trial_name") or trial.get("nctid") or trial.get("nct_id") or "Unknown trial"
        drug = trial.get("drug", "")
        dclass = trial.get("drugclass") or trial.get("drug_class") or ""
        year = trial_years.get(tname, "Not reported")

        pro_parts = []
        total_items = 0
        total_time = 0.0
        has_items = False
        has_time = False

        # Calibrated SOA = Yes if any instrument used subscales/item-library
        calibrated_soa = "No"
        for inst in trial.get("instruments", []):
            iname = inst.get("name") or ""
            norm = _norm(iname)
            ir = ref_index.get(norm)

            items = None
            admintime = None
            if ir:
                raw_items = ir.get("totalitems") or ir.get("total_items")
                raw_time = ir.get("admintime") or ir.get("admin_time")
                try:
                    items = int(raw_items) if raw_items not in (None, "", "nan") else None
                except Exception:
                    items = None
                try:
                    admintime = float(raw_time) if raw_time not in (None, "", "nan") else None
                except Exception:
                    admintime = None

            # Build display part like "QLQ-C30 (n=30)" or "QLQ-C30 (n=?)"
            if items is not None and items > 0:
                pro_parts.append(f"{iname} (n={items})")
                total_items += items
                has_items = True
            else:
                pro_parts.append(f"{iname} (n=?)")

            if admintime is not None and admintime > 0:
                total_time += admintime
                has_time = True

            # Subscales → calibrated SOA
            if inst.get("subscales") or inst.get("instrument_subscales_assessed"):
                calibrated_soa = "Yes"

        row = {
            "trial": tname,
            "year": year,
            "drug": drug,
            "drug_class": dclass,
            "pro_measures": ", ".join(pro_parts) if pro_parts else "Not recorded",
            "calibrated_soa": calibrated_soa,
            "total_items": total_items if has_items else None,
            "est_time_min": round(total_time, 1) if has_time else None,
        }
        rows.append(row)

    # Current trial row (Proposed)
    tname = "Current Trial (Proposed)"
    year = "TBD"  # To ensure AI doesn't randomly assign a year
    drug = f"Novel {contextjson.get('drugclass') or contextjson.get('drug_class') or 'regimen'}"
    dclass = contextjson.get("drugclass") or contextjson.get("drug_class") or ""

    rows.append({
        "trial": tname,
        "year": year,
        "drug": drug,
        "drug_class": dclass,
        "pro_measures": "TBD — expert decision required",
        "calibrated_soa": "TBD",
        "total_items": None,
        "est_time_min": None,
    })

    return rows

def build_gap_analysis(
    scored: list,
    instrument_meta: dict,
    reg_records: list,
    context_json: dict,
    top_n: int = 5,
) -> list:
    """
    Build a gap analysis table for the top-N instruments.

    Returns a list of dicts with keys:
      - instrument
      - content_validity
      - mcid_evidence
      - regulatory_acceptance
      - known_gaps
      - fit_for_purpose
      - score
      - risk_level
    """

    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _reg_hits(name: str) -> list:
        hits = []
        n = _norm(name)
        for r in reg_records or []:
            acc = r.get("instrumentsaccepted") or r.get("instruments_accepted")
            if acc and n in str(acc).lower():
                hits.append(f"{r.get('agency','')} {r.get('decision','')}")
        return hits

    rows = []
    for inst in (scored or [])[:top_n]:
        # Your score_evidence uses 'instrumentname' as key
        name = inst.get("instrumentname") or inst.get("instrument_name") or "Unknown"
        node = instrument_meta.get(name, {})

        # Content validity / validation
        val_status = (
            node.get("validation")
            or inst.get("validationstatus")
            or ""
        )
        val_status = str(val_status).strip()

        # MCID
        raw_mcid = node.get("mcid") or inst.get("mcid") or ""
        mcid_short = ""
        try:
            mcid_short, _ = clean_mcid(raw_mcid)
        except Exception:
            mcid_short = str(raw_mcid)[:80] if raw_mcid else ""
        mcid_display = mcid_short or "Not established / not reported"

        # Regulatory acceptance
        reg_node = node.get("regulatoryacceptance") or node.get("regulatory_acceptance") or ""
        reg_hits = _reg_hits(name)
        if reg_hits and reg_node:
            reg_text = f"{reg_node} | KG reviews: " + "; ".join(reg_hits)
        elif reg_hits:
            reg_text = "KG reviews: " + "; ".join(reg_hits)
        elif reg_node:
            reg_text = str(reg_node)
        else:
            reg_text = "No explicit regulatory precedent recorded in KG"

        strengths = (node.get("strengths") or inst.get("strengths") or "").strip()
        limitations = (node.get("limitations") or inst.get("limitations") or "").strip()
        known_gaps = limitations or "No specific limitations recorded in KG"

        score = inst.get("scientificscore", 0)
        risk = inst.get("risklevel", "LOW")

        # Simple, transparent fit-for-purpose tiering
        if score >= 65 and risk not in ("CRITICAL", "HIGH"):
            fit = "Likely fit-for-purpose in this context"
        elif score >= 40 and risk != "CRITICAL":
            fit = "Conditionally fit — gaps and/or risks need mitigation"
        else:
            fit = "Evidence gaps / risk flags — human review strongly recommended"

        rows.append({
            "instrument": name,
            "content_validity": val_status or "Not described",
            "mcid_evidence": mcid_display,
            "regulatory_acceptance": reg_text,
            "known_gaps": known_gaps[:250],
            "fit_for_purpose": fit,
            "score": score,
            "risk_level": risk,
        })

    return rows

def build_endpoint_positioning(
    raw_kg_records: list,
    scored: list,
    top_n: int = 5,
) -> list:
    """
    Summarise endpoint positioning for top-N instruments across the KG trial sample.

    Returns list of dicts:
      - instrument
      - primary_count
      - secondary_count
      - exploratory_count
      - other_count
      - comment
    """

    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    # Focus on the same instruments you highlight elsewhere (top-N by score)
    instrument_order = []
    for inst in (scored or [])[:top_n]:
        name = inst.get("instrumentname") or inst.get("instrument_name")
        if name and name not in instrument_order:
            instrument_order.append(name)

    counts = {
        name: {
            "Primary": 0,
            "Secondary": 0,
            "Exploratory": 0,
            "Other": 0,
        }
        for name in instrument_order
    }

    for r in raw_kg_records or []:
        name = r.get("instrumentname") or r.get("instrument_name")
        if name not in counts:
            continue
        role_raw = (
            r.get("pro_position")
            or r.get("endpointrole")
            or r.get("endpoint_role")
            or r.get("proposition")
            or ""
        )
        role = _norm(role_raw)
        if not role:
            continue

        if "primary" in role:
            key = "Primary"
        elif "secondary" in role:
            key = "Secondary"
        elif "explor" in role:
            key = "Exploratory"
        else:
            key = "Other"

        counts[name][key] += 1

    rows = []
    for name in instrument_order:
        c = counts[name]
        total = sum(c.values())
        if total == 0:
            comment = "No KG trials with explicit endpoint role for this instrument in this indication."
        else:
            dominant = max(c.items(), key=lambda kv: kv[1])[0]
            if c[dominant] == total:
                comment = f"Used exclusively as {dominant.lower()} endpoint in our KG trial sample."
            else:
                comment = (
                    f"Mixed endpoint roles in KG sample; most frequent: {dominant.lower()} "
                    f"({c[dominant]} of {total} trial-instrument records with explicit role)."
                )

        rows.append({
            "instrument": name,
            "primary_count": c["Primary"],
            "secondary_count": c["Secondary"],
            "exploratory_count": c["Exploratory"],
            "other_count": c["Other"],
            "comment": comment,
        })

    return rows


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

def build_competitor_profiles(indication: str, drug_class: str,
                               reg_records: list) -> list:
    """
    From all reg_records for this indication, use Haiku to identify
    relevant competitors, assess comparability, and generate PRO implications.
    Returns list of enriched competitor profile dicts.
    """
    all_drugs = list({r.get("drug_name", "") for r in reg_records
                      if r.get("drug_name", "")})
    if not all_drugs:
        return []

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=(
                "You are a pharmaceutical competitive intelligence analyst. "
                "From the candidate drugs provided, select ONLY those that are "
                "relevant competitors to the trial drug class. A relevant competitor "
                "shares the same mechanism, same molecular target, OR treats the same "
                "patient population in the same line of therapy. Exclude drugs with "
                "no mechanistic or population overlap.\n"
                "Return a JSON array ordered from most to least relevant. "
                "For each relevant drug include:\n"
                "  drug: exact drug name as provided\n"
                "  relevance: one sentence why this is a relevant competitor\n"
                "  mechanism: one sentence mechanism summary\n"
                "  pro_implication: the single most important PRO question "
                "this trial must answer differently from this competitor\n"
                "  comparability_required: true or false — true ONLY if this trial "
                "drug is a direct improvement over this competitor (same target, same "
                "patient population, same line of therapy, but better/different "
                "mechanism). If true, the trial MUST use the same PRO instruments as "
                "this competitor to allow regulatory contextual comparison.\n"
                "  comparability_reason: one sentence explaining why comparability "
                "is or is not required\n"
                "Return ONLY valid JSON array. No markdown."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Trial drug class: {drug_class}\n"
                    f"Indication: {indication}\n"
                    f"Candidate drugs: {', '.join(all_drugs)}"
                )
            }]
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "")
        enriched = json.loads(raw)
    except Exception as e:
        logging.warning(f"Competitor Haiku call failed: {e}")
        return []

    # Build a lookup from drug name → first reg_record for that drug
    reg_by_drug = {}
    for r in reg_records:
        drug = r.get("drug_name", "")
        if drug and drug not in reg_by_drug:
            reg_by_drug[drug] = r

    profiles = []
    for comp in enriched:
        drug = comp.get("drug", "")
        rec  = reg_by_drug.get(drug, {})
        fda  = str(rec.get("fda_label_url", ""))
        ema  = str(rec.get("ema_label_url", ""))
        profiles.append({
            "drug":                   drug,
            "agency":                 rec.get("agency", ""),
            "decision":               rec.get("decision", ""),
            "instruments":            str(rec.get("instruments_accepted", "") or ""),
            "claim_type":             str(rec.get("claim_type", "") or ""),
            "rejection":              str(rec.get("rejection_reason_primary", "") or ""),
            "fda_url":                fda,
            "ema_url":                ema,
            "relevance":              comp.get("relevance", ""),
            "mechanism":              comp.get("mechanism", ""),
            "pro_implication":        comp.get("pro_implication", ""),
            "comparability_required": comp.get("comparability_required", False),
            "comparability_reason":   comp.get("comparability_reason", ""),
        })

    logging.info(f"Competitor analysis: {len(profiles)} relevant competitors identified.")
    return profiles

# =============================================================================
# STEP 6: REASONER — MAIN ORCHESTRATOR
# =============================================================================
def get_recommendation(user_text: str) -> dict:
    """
    Step A: Haiku — extract trial context
    Step B: Neo4j — query instruments, regulatory reviews, rules
    Step C: Python — score instruments (0-100)
    Step D: Python — build coverage matrix
    Step E: Neo4j — instrument references + per-instrument regulatory precedent
    Step E.1: Haiku — competitor profile enrichment
    Step F: Haiku — clean KG narrative fields
    Step G: Python — build evidence block for Sonnet
    Step H: Python — build HTA context block
    Step I: Python — build Sonnet system prompt
    Step J: Python — build Sonnet user prompt
    Step K: Sonnet — synthesise recommendation with web search
    Step L: Return structured result dict
    """

    error_status = None
    inst_refs = []
    inst_regulatory_precedents = {}
    competitor_profiles = []

    # --- STEP A: Analyze trial context ---
    context_json = analyze_trial_context(user_text)
    indication = context_json.get("indication", "")
    synonyms = context_json.get("indication_synonyms") or [indication]
    phase = context_json.get("phase", "Phase 3")

    # --- STEP B: Query Knowledge Graph ---
    instrument_meta: dict = {}
    raw_kg_records: list = []
    kg_records: list = []
    reg_records: list = []
    reg_rules: list = []

    try:
        search_terms = list(dict.fromkeys([indication] + synonyms[:3]))

        # 1) Fetch ALL raw rows first: one row per trial × instrument
        for term in search_terms:
            rows = get_instruments_by_indication(indication=term, phase=phase, endpoint="")
            if rows:
                raw_kg_records.extend(rows)

        # 2) Count prevalence BEFORE deduplication
        prevalence_map: dict[str, int] = {}
        for r in raw_kg_records:
            name = str(r.get("instrument_name", "Unknown")).strip() or "Unknown"
            prevalence_map[name] = prevalence_map.get(name, 0) + 1

        # 3) Deduplicate for scoring only: keep the richest row per instrument
        def _nonempty_score(rec: dict) -> int:
            return sum(
                1 for v in rec.values()
                if v is not None and str(v).strip() not in ("", "nan", "None", "null")
            )

        deduped: dict[str, dict] = {}
        for r in raw_kg_records:
            name = str(r.get("instrument_name", "Unknown")).strip() or "Unknown"
            if name not in deduped or _nonempty_score(r) > _nonempty_score(deduped[name]):
                deduped[name] = dict(r)

        kg_records = list(deduped.values())

        # 4) Annotate representative rows with prevalence
        for r in kg_records:
            name = str(r.get("instrument_name", "Unknown")).strip() or "Unknown"
            r["_prevalence"] = prevalence_map.get(name, 1)

        # 5) Pre-fetch Instrument node metadata for all unique instruments
        unique_instrument_names = sorted({
            str(r.get("instrument_name", "")).strip()
            for r in raw_kg_records
            if str(r.get("instrument_name", "")).strip()
        })

        for name in unique_instrument_names:
            refs = get_instrument_reference(instrument_name=name)
            if refs:
                instrument_meta[name] = refs[0]

        # 6) Regulatory evidence: deduplicate by review_id
        all_reg = []
        for term in search_terms:
            rows = get_regulatory_evidence(indication=term, agency="")
            if rows:
                all_reg.extend(rows)

        seen_ids: set[str] = set()
        for r in all_reg:
            rid = r.get("review_id") or f"{r.get('drug_name','')}|{r.get('agency','')}|{r.get('decision','')}"
            if rid not in seen_ids:
                seen_ids.add(rid)
                reg_records.append(r)

        # # 7) Regulatory rules
        # reg_rules = get_regulatory_rules(
        #     indication=indication,
        #     lifecycle_stage="",
        #     decision_type=""
        # )

        logging.info(
            f"KG: {len(kg_records)} unique instruments from {len(raw_kg_records)} raw rows, "
            f"{len(reg_records)} reviews, {len(instrument_meta)} instrument nodes"
        )


    except Exception as e:
        error_status = f"Knowledge Graph offline: {e}"
        raw_kg_records = []
        kg_records = []
        instrument_meta = {}
        reg_records = []
        reg_rules = []
        logging.error(f"KG query failed: {e}")


    # Step B3 — retrieve applicable regulatory rules
    all_rules = get_regulatory_rules(
        indication=context_json.get("indication", ""),
        lifecycle_stage="Instrument_Selection",
        decision_type=""        
    )
    must_rules  = [r for r in all_rules if _to_str(r.get("decision_type")) == "must"]
    should_rules = [r for r in all_rules if _to_str(r.get("decision_type")) == "should"]

    reg_rules = all_rules

    logging.info(
        f"Regulatory rules: {len(reg_rules)} total "
        f"({len(must_rules)} MUST, {len(should_rules)} SHOULD)."
    )

    # --- STEP C: Score instruments ---
    scored = score_evidence(context_json, kg_records, instrument_metadata=instrument_meta, raw_kg_records=raw_kg_records)

    # --- STEP D: Build coverage matrix (replaces battery optimizer) ---
    coverage = build_coverage_matrix(
        scored,
        context_json,
        raw_kg_records,              
        instrument_metadata=instrument_meta
    ) if scored else {
        "domains": [],
        "comparator_trials": [],
        "hta_mandatory": [],
        "item_library_applicable": False,
        "all_candidates": []
    }
    top_5 = coverage["all_candidates"][:5]

    # --- STEP E: Fetch instrument refs + per-instrument regulatory precedent ---
    inst_refs = []
    inst_regulatory_precedents = {}
    if not error_status:
        try:
            # Build the set of instrument names we care about:
            #   - all top-5 scored instruments for this trial
            #   - every instrument used in comparator trials from the coverage matrix
            instrument_names = {inst["instrument_name"] for inst in top_5}
            for trial in coverage.get("comparator_trials", []):
                for inst in trial.get("instruments", []):
                    name = inst.get("name")
                    if name:
                        instrument_names.add(name)

            # Fetch instrument reference nodes and per-instrument regulatory evidence
            for name in sorted(instrument_names):
                refs = get_instrument_reference(instrument_name=name)
                if refs:
                    if isinstance(refs, list):
                        inst_refs.extend(refs)
                    else:
                        inst_refs.append(refs)

                precedents = get_regulatory_evidence_for_instrument(instrument_name=name)
                if precedents:
                    inst_regulatory_precedents[name] = precedents

        except Exception as e:
            logging.error(f"Instrument ref/precedent fetch failed: {e}")

    # --- STEP E.1: Competitor analysis ---
    competitor_profiles = []
    try:
        competitor_profiles = build_competitor_profiles(
            indication,
            context_json.get("drug_class", "Unknown"),
            reg_records
        )
    except Exception as e:
        logging.warning(f"Competitor analysis step failed: {e}")

     # --- STEP E.2 Build PRO measures comparison table in Python ---
    pro_measures_table = []
    try:
        pro_measures_table = build_pro_measures_table(
            coverage=coverage,
            inst_refs=inst_refs,
            rawkgrecords=raw_kg_records,
            contextjson=context_json,
        )
    except Exception as e:
        logging.error(f"build_pro_measures_table failed: {e}")

    # --- STEP F: Clean KG narrative fields ---
    if kg_records:
        try:
            kg_records = clean_kg_narratives(kg_records)
        except Exception as e:
            logging.warning(f"KG cleaning skipped: {e}")
    
    # --- GAP ANALYSIS TABLE ---
    gap_analysis = []
    try:
        gap_analysis = build_gap_analysis(
            scored=scored,
            instrument_meta=instrument_meta,
            reg_records=reg_records,
            context_json=context_json,
            top_n=5,
        )
    except Exception as e:
        logging.error(f"build_gap_analysis failed: {e}")
    
    # --- ENDPOINT POSITIONING TABLE ---
    endpoint_positioning = []
    try:
        endpoint_positioning = build_endpoint_positioning(
            raw_kg_records=raw_kg_records,
            scored=scored,
            top_n=5,
        )
    except Exception as e:
        logging.error(f"build_endpoint_positioning failed: {e}")

    # --- STEP G: Build structured evidence block for Sonnet ---
    citation_index = {}
    kg_block_lines = []

    if error_status:
        kg_block_lines.append(f"⚠️ KG OFFLINE — {error_status}")
        kg_block_lines.append(
            "IMPORTANT: The knowledge graph is offline. "
            "DO NOT generate Table 1 or Table 2 with placeholder data. "
            "Instead, write exactly this at the top of your response:\n"
            "'⚠️ Knowledge graph is offline. Table 1 and Table 2 cannot be generated "
            "without KG data. Please reconnect Neo4j and re-run the query.'\n"
            "Then proceed with web-search-only observations in the Key Observations section."
        )
    else:
        # === DOMAIN COVERAGE MATRIX ===
        kg_block_lines.append("=== DOMAIN COVERAGE MATRIX ===")
        kg_block_lines.append("Use this to populate Table 1 in your output.\n")
        for d in coverage["domains"]:
            candidates_str = ", ".join(
                f"{c['instrument']} (score {c['score']}, change: {c['change_detected']})"
                for c in d["candidates"][:3]
            ) or "No instrument found"
            kg_block_lines.append(f"Domain: {d['domain']}")
            kg_block_lines.append(f"  FDA core: {'Yes' if d['is_fda_core'] else 'No'}")
            kg_block_lines.append(f"  Candidate instruments: {candidates_str}")
            if d["item_library_note"]:
                kg_block_lines.append(f"  ⚠️ {d['item_library_note']}")

        # === HTA MANDATORY ===
        if coverage["hta_mandatory"]:
            kg_block_lines.append("\n=== HTA MANDATORY INSTRUMENTS ===")
            kg_block_lines.append("These must appear in Table 1 HTA row regardless of score.\n")
            for h in coverage["hta_mandatory"]:
                kg_block_lines.append(
                    f"  {h['instrument']} — required for {h['market']}. {h['reason']}"
                )

        # === COMPARATOR TRIALS ===
        kg_block_lines.append(f"\n=== COMPARATOR TRIALS ({len(coverage['comparator_trials'])} trials) ===")
        kg_block_lines.append("Use these to populate Table 2 (trial rows).\n")
        for i, trial in enumerate(coverage["comparator_trials"], 1):
            label = f"TI-{i:03d}"
            kg_block_lines.append(
                f"[{label}] Trial: {trial['trial_name']} | Drug: {trial['drug']} | "
                f"Phase: {trial['phase']}"
            )
            for inst in trial["instruments"][:6]:
                kg_block_lines.append(
                    f"  - {inst['name']} | Role: {inst['role']} | "
                    f"Significance: {inst['significance']} | "
                    f"Pre-specified: {inst['prespecified']}"
                    + (f" | Subscales used: {inst['subscales']}" if inst["subscales"] else "")
                )

        # === PRO ENDPOINT POSITIONING (derived from KG, not hardcoded) ===
        # Show the actual distribution of PRO endpoint positions in KG trials
        # for this indication so Sonnet can reason from data, not from a rule.
        # Use endpoint_positioning summary instead of recomputing from kg_records
        if endpoint_positioning:
            # Aggregate across instruments
            agg = {"Primary": 0, "Secondary": 0, "Exploratory": 0, "Other": 0}
            for row in endpoint_positioning:
                agg["Primary"]      += row.get("primary_count", 0)
                agg["Secondary"]    += row.get("secondary_count", 0)
                agg["Exploratory"]  += row.get("exploratory_count", 0)
                agg["Other"]        += row.get("other_count", 0)

            total_pos = sum(agg.values())
            if total_pos > 0:
                kg_block_lines.append("\n=== PRO ENDPOINT POSITIONING IN KG TRIALS ===")
                kg_block_lines.append(
                    f"Distribution of PRO endpoint positions across {total_pos} "
                    f"trial-instrument records for the top instruments:"
                )
                for key in ("Primary", "Secondary", "Exploratory", "Other"):
                    count = agg[key]
                    if count:
                        pct = int(100 * count / total_pos)
                        kg_block_lines.append(f"  {key}: {count} record(s) ({pct}%)")

                # Also highlight any instrument that has *ever* been primary
                primaries = [row["instrument"] for row in endpoint_positioning if row.get("primary_count", 0) > 0]
                if primaries:
                    kg_block_lines.append(
                        "Instruments with at least one primary PRO endpoint in the KG sample: "
                        + ", ".join(primaries)
                    )
                else:
                    kg_block_lines.append(
                        "No instruments in the KG sample were used as primary PRO endpoints."
                    )

                kg_block_lines.append(
                    "Use this data to reason about appropriate endpoint positioning "
                    "for the current trial. Do not apply a fixed rule."
                )

        # === INSTRUMENT SCORING (for Table 2 item counts) ===
        if inst_refs:
            kg_block_lines.append("INSTRUMENT REFERENCE DATA")
            # One IR-XXX line per instrument reference node (capped at 8 for token control)
            for i, ir in enumerate(inst_refs[:8], 1):
                short = ir.get("shortname") or ir.get("instrumentname") or ir.get("instrument_name") or ""
                items = (
                    ir.get("totalitems")
                    or ir.get("total_items")
                    or ""
                )
                admintime = (
                    ir.get("admintime")
                    or ir.get("admin_time")
                    or ""
                )
                mcid = ir.get("mcid") or "Not established"

                regacc = (
                    ir.get("regulatoryacceptance")
                    or ir.get("regulatory_acceptance")
                    or ""
                )

                kg_block_lines.append(
                    f"IR-{i:03d} {short} "
                    f"Items {items} "
                    f"Admin time {admintime} min "
                    f"MCID {mcid} "
                    f"Regulatory acceptance {regacc}"
                )


        # === COMPETITOR LANDSCAPE ===
        if competitor_profiles:
            kg_block_lines.append(f"\n=== COMPETITOR LANDSCAPE ===")
            for i, comp in enumerate(competitor_profiles[:6], 1):
                label = f"COMP-{i:03d}"
                kg_block_lines.append(
                    f"[{label}] {comp['drug']} | {comp['agency']} | {comp['decision']}"
                )
                kg_block_lines.append(f"  Instruments: {comp['instruments'] or 'Not recorded'}")
                kg_block_lines.append(f"  PRO outcome: {comp.get('pro_implication','')}")

        # === REGULATORY REVIEWS ===
        if reg_records:
            kg_block_lines.append(f"\n=== REGULATORY REVIEWS ({len(reg_records)} records) ===")
            for i, rr in enumerate(reg_records[:10], 1):
                kg_block_lines.append(
                    f"[RR-{i:03d}] {rr.get('agency','')} | {rr.get('drug_name','')} | "
                    f"Decision: {rr.get('decision','')} | "
                    f"Instruments accepted: {rr.get('instruments_accepted','')}"
                )

        # === REJECTIONS ===
        rejections = [r for r in reg_records if r.get("rejection_reason_primary")]
        if rejections:
            kg_block_lines.append(f"\n=== REJECTION RECORDS ({len(rejections)}) ===")
            for i, rr in enumerate(rejections, 1):
                kg_block_lines.append(
                    f"[REJ-{i:03d}] {rr.get('agency','')} | {rr.get('drug_name','')} | "
                    f"Primary reason: {rr.get('rejection_reason_primary','')}"
                )

        # === REGULATORY RULES ===
        if reg_rules:
            kg_block_lines.append(
                f"\n=== REGULATORY RULES ({len(reg_rules)}) ==="
            )
            kg_block_lines.append(
                f"Summary: {len(must_rules)} MUST rules, {len(should_rules)} SHOULD rules "
                f"for instrument selection in this indication."
            )
            for i, rule in enumerate(reg_rules[:8], 1):
                dtype = _to_str(rule.get("decision_type")).upper() or "UNSPECIFIED"
                kg_block_lines.append(
                    f"[RULE-{i:03d}] ({dtype}) {rule.get('source_document','')} | "
                    f"{rule.get('rule_text','')[:200]}"
                )


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

    glossary_section = f"\n\nGlossary:\n{GLOSSARY_TEXT}" if GLOSSARY_TEXT else ""

    # --- STEP I: Build Sonnet system prompt ---
    sonnet_system = f"""You are a COA specialist synthesising evidence for a senior COA expert who will make the final decisions. Your job is to present information clearly so the expert can decide — not to recommend or prescribe.

RULE 1 (Two tables are mandatory — this is the most important output):
You must produce exactly two markdown tables. Do not skip or combine them.

TABLE 1 — Domain Coverage Comparison (matches your supervisor's slide):
This table has exactly these columns, in this order:
| FDA Core Concept | Source | Current Trial Candidates | [Trial name 1] [year] | [Trial name 2] [year] | [Trial name 3] [year] |

Rows: One per FDA core domain + one HTA utility row
Columns:
- "FDA Core Concept": the domain name (e.g. "Disease-related bone pain")
- "Source": the regulatory document that defines this as core (e.g. "FDA 2024 Core PRO Guidance")
- "Current Trial Candidates": instruments from the Domain Coverage Matrix that cover this domain
- Each comparator trial column: the instrument used + ✅/⚠️/❌ + whether change was detected (Y/N/NR)
  Format each cell as: INSTRUMENT_NAME ✅ Change: Y [TI-001]
  If not collected: ❌ Not collected
  If general oncology only: ⚠️ General (QLQ-C30 only, no MM module)

WEB SEARCH FOR TABLE 1 GAPS:
For any comparator trial column where a domain shows "❌ Not collected":
Search "[trial name] [domain] PRO endpoint" to check whether the instrument was
used but not in the KG. If web search finds it was collected: update the cell.
If web search confirms it was not collected: add a footnote explaining why
(e.g. "Neuropathy not measured — PI class not primary focus of this trial").
This prevents the table from appearing sparse when data exists in literature.
Maximum 2 additional web searches for gap-filling. Prioritise the most important gaps.

Name each comparator column with the actual trial name from the KG [TI-XXX].
If KG has no comparator data, write: "No KG data for this indication" in those columns.
DO NOT write "[Comparator Trial 1]" as a placeholder — use the actual trial name or state no data.
DO NOT write "[Candidate instruments needed]" — use the candidates from the Domain Coverage Matrix
or write "No KG scoring data available" if KG is offline.

TABLE 2 — PRO Measures Comparison (matches your supervisor's slide):
This table has exactly these columns:
| Trial | Year | Drug | Drug class | PRO Measures (n items) | Calibrated SOA | Total items | Est. time |

Rows: One per comparator trial from KG + one row for "Current Trial (Proposed)"
Columns:
- "Trial": trial name [TI-XXX]
- "Year": publication year from KG
- "Drug": drug name from KG
- "Drug class": mechanism from KG
- "PRO Measures (n items)": list each instrument with item count in parentheses,
  e.g. "EORTC QLQ-C30 (30), QLQ-MY20 (20), EQ-5D-5L (5)"
  Use item counts from [IR-XXX] blocks. If not in KG, write "n=?"
- "Calibrated SOA": Yes if subscales were used per KG record, No if full instruments, TBD for current
- "Total items": sum of all instrument items per row
- "Est. time": sum of admin times in minutes (from IR-XXX admin_time fields)

For the "Current Trial (Proposed)" row:
- List the candidate instruments from the Domain Coverage Matrix
- Mark Calibrated SOA as "TBD — expert decision required"

CRITICAL: If KG is offline or returns no comparator data, write one sentence explaining this
and do NOT generate placeholder rows. A hollow table with "[Candidate instruments needed]"
is worse than no table.

RULE 2 (PRO endpoint positioning — use KG data):
The KG evidence block contains a section "PRO ENDPOINT POSITIONING IN KG TRIALS"
showing the actual distribution of primary/secondary/exploratory positions across
trials for this indication.

Use this data to state: "In [N] KG trials for this indication, [X]% used PRO as
[secondary/exploratory], [Y]% as primary. This suggests [your reasoning]."

Do NOT apply a hardcoded rule. Reason from the evidence.

If the KG shows 0 primary PRO endpoints, note this and explain why it matters.
If the KG shows any primary PRO endpoints, note the context (which drug, which trial).

Always recommend endpoint positioning based on both (a) the KG distribution and
(b) the regulatory rationale — not from a fixed rule.

RULE 3 (Citations):
Every factual claim needs a citation immediately after it.
KG records: [TI-001], [RR-001], [REJ-001], [IR-001], [RULE-001], [COMP-001]
Web sources: [Source Name](https://complete-url.com)
If you cannot find a source, write the claim as: "[Not found in KG or web search]"
Do not state facts from training memory without a web search confirming them.

RULE 4 (Item library note — add if applicable):
If the KG evidence block shows comparator trials used subscales rather than full instruments,
add a note: "Item library / calibrated SOA approach: [comparator trial] used [subscale/items]
rather than the full [instrument]. This reduces patient burden from [N] to approximately [M]
items per timepoint. Consider whether a similar approach is appropriate for this trial —
decision for the COA expert."

RULE 5 (Competitor context):
For each competitor in COMP-XXX, state mechanism relevance and PRO outcome in one sentence.
Connect every competitor finding to a specific candidate instrument decision.

RULE 6 (No hallucination):
Do not state statistics, trial results, or regulatory decisions from training memory.
Always search the web to verify, then cite the source URL.

RULE 7 (Regulatory rules — must cite when relevant):
The KG evidence block may contain a section "=== REGULATORY RULES ===" with [RULE-XXX] entries.
These are published FDA/ICH/EMA rules directly applicable to this indication and phase.

If RULE entries exist, you MUST cite at least one [RULE-XXX] in your output.
Specifically cite rules when discussing:
- Pre-specification and alpha control → cite the relevant RULE
- Estimand strategy → cite the relevant RULE
- Missing data requirements → cite the relevant RULE
- Testing hierarchy → cite the relevant RULE

If no RULE entries exist in the KG block, note:
"No indication-specific regulatory rules retrieved from KG — consult FDA PRO Guidance (2009) directly."

OUTPUT STRUCTURE (follow exactly):

## COA Measurement Strategy — [Indication] [Phase]

**In one sentence:** [what the trial is trying to show with PROs]
**Key challenge:** [the single biggest PRO design challenge for this trial]
**Recommended starting point:** [2-3 instruments the expert should seriously consider, citing KG evidence of change detection]
**Critical gap:** [one thing that will fail the strategy if not addressed — e.g. EQ-5D missing, neuropathy not covered]

## Table 1: Domain Coverage Comparison
[mandatory table — see RULE 1]

## Table 2: PRO Measures Comparison
[mandatory table — see RULE 1]

## Key Observations
[maximum 6 bullet points, each citing a source, each connecting to a specific table cell]
[include item library note if applicable — RULE 4]

## Comparator Analysis
For each competitor in the COMP-XXX blocks of the evidence:
- One row per comparator showing: drug | mechanism | PRO instruments used | outcome | implication for current trial
- If a comparator used an instrument that detected significant change: note this explicitly as evidence the instrument is sensitive in this context
- If a comparator used an instrument and found null results: note this as a calibration risk — does the trial design differ enough to expect different results?
- Conclude with: "Based on comparator evidence, [instrument X] has the strongest signal of change in this indication because [one-sentence reason from KG/web evidence]."
This section is mandatory when COMP-XXX entries exist in the evidence block.
If no COMP entries exist, write: "No same-mechanism comparator data available in KG — expert should review recent FDA and EMA medical reviews for [drug class] submissions."

## HTA Requirements
[one-line table: HTA Body | Required Instrument | In candidate list? | Action needed]

## What the Expert Needs to Decide
[numbered list — these are DECISIONS for the expert, not recommendations]
1. Which instruments from the candidate list to include
2. Whether item library / calibrated SOA approach is appropriate
3. Endpoint hierarchy positioning within the testing sequence
4. Assessment schedule aligned with dosing
5. Any additional domains not covered by candidates above

Glossary: {GLOSSARY_TEXT}"""

    # --- STEP J: Build Sonnet user prompt ---
    indication_for_search = indication or "this oncology indication"
    sonnet_user = f"""You are briefing a senior clinical scientist on the COA strategy for their trial.
Write as a knowledgeable colleague, not as a report generator.
Every claim must be cited. Every section must conclude with an action.
Present the evidence clearly so the senior COA expert can make the final
instrument selection. Your role is to organise the evidence — not to select
or pre-select instruments.
IMPORTANT: Total output must fit within 15000 tokens. Be concise.
Tables first. Key Observations: maximum 6 bullets, 2 sentences each.
What the Expert Needs to Decide: maximum 5 items, 1 sentence each.
Do not repeat information from tables in prose.

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
        # Try with extended thinking first (no web search)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20000,
            thinking={"type": "enabled", "budget_tokens": 3000},
            system=sonnet_system,
            tools=[],   # thinking mode: no tools (Sonnet reasons from KG only)
            messages=[{"role": "user", "content": sonnet_user}]
        )
        answer = " ".join(
            block.text for block in response.content
            if hasattr(block, "text") and block.text
            and getattr(block, "type", "") != "thinking"
        )
        logging.info("Sonnet answered with extended thinking (KG-only mode)")
    except Exception as thinking_err:
        logging.warning(f"Extended thinking failed ({thinking_err}), falling back to web search")
        # Fallback: standard call with web search, no thinking
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

    # --- STEP L: Build result dict ---
    result = {
        "answer": answer,
        "context_json": context_json,
        "top_scores": top_5,
        "all_scores": scored,
        "kg_raw_hits": kg_records,
        "reg_records": reg_records,
        "regulatory_records": reg_records,
        "reg_rules": reg_rules,
        "inst_refs": inst_refs,
        "inst_regulatory_precedents": inst_regulatory_precedents,
        "coverage": coverage,
        "hta_context": {b: HTA_PREFERENCES[b] for b in context_json.get("hta_markets", []) if b in HTA_PREFERENCES},
        "error_status": error_status,
        "record_counts": {
            "instrument_records": len(kg_records),
            "regulatory_reviews": len(reg_records),
            "regulatory_rules": len(reg_rules),
            "instrument_refs": len(inst_refs),
            "scored_instruments": len(scored),
            "rejections_found": len([r for r in reg_records if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")])
        },
        "citation_index":     citation_index,
        "scored_instruments": scored,
        "competitor_profiles": competitor_profiles, 
        "kg_evidence_block": kg_evidence_block,
        "pro_measures": pro_measures_table,
        "gap_analysis": gap_analysis,
        "endpoint_positioning": endpoint_positioning,
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
            "coverage_domains": [d["domain"] for d in result.get("coverage", {}).get("domains", [])],
            "domains_with_candidates": len([d for d in result.get("coverage", {}).get("domains", []) if d.get("candidates")]),
            "hta_mandatory": [h["instrument"] for h in result.get("coverage", {}).get("hta_mandatory", [])],
            "comparator_trials": [t["trial_name"] for t in result.get("coverage", {}).get("comparator_trials", [])[:3]],
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
        get_recommendation, clean_mcid,
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