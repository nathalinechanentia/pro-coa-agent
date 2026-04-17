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
from anthropic import Anthropic, tools
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
HAIKU_SYSTEM_PROMPT = """You are extracting clinical trial parameters from natural language.
The input may be a detailed research brief, a short question, or anything in between.
Read the ENTIRE input carefully before extracting. Extract what is explicitly stated.

EXTRACTION RULES:
1. Extract from natural prose. Do not require specific keyword formats.
   Examples of valid extractions:
   "bortezomib-like mechanism" → drug_class = "Proteasome Inhibitor"
   "late-line RRMM" → population_subtype = "Relapsed/Refractory", indication = "Multiple Myeloma"
   "spanning US, EU, and Asia" → geographic_footprint = "Global"
   "FDA and EMA submissions" → regulatory_agencies = ["FDA", "EMA"]
   "NICE and ICER" → hta_markets = ["NICE", "ICER"]
   "reversible inhibitor of the 26S proteasome" → drug_class = "Proteasome Inhibitor"
   "proteasome inhibitor-based therapy" → drug_class = "Proteasome Inhibitor"

2. For assumptions_made: ONLY list fields where you genuinely had to infer something
   NOT explicitly stated. Do NOT list fields you successfully extracted.
   If indication, phase, and drug class were all in the text, assumptions_made = [].

3. core_domains_required: use the FDA 2024 Core PRO Guidance domains for the indication,
   PLUS any additional domains explicitly mentioned by the user.
   Multiple Myeloma: disease-related symptoms (bone pain), physical function, fatigue,
                    symptomatic AEs, side effect impact summary, role function
   NSCLC: disease symptoms (dyspnoea, cough, chest pain), physical function,
          symptomatic AEs, role function
   Default: physical function, fatigue, pain, symptomatic AEs

4. additional_domains: capture any domains the user explicitly asked about that are
   beyond the standard FDA core set (e.g. "peripheral neuropathy burden" is additional).

5. tpp_claims: extract from "we want to demonstrate/capture/show/measure [X]" phrases.
   If not stated, default to ["treatment tolerability", "physical function maintenance"]
   and flag in assumptions_made.

Inference rules (only apply if field NOT in the text):
- phase missing → "Phase 3" (flag as inferred)
- geographic_footprint missing → Phase 3 = "Global" (flag as inferred)
- hta_markets missing → infer from footprint + regulatory agencies mentioned (flag as inferred)
- administration missing for bispecific/CAR-T → "Step-up dosing" (flag as inferred)

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
# def _get_conn():
#     return Neo4jConnection(
#         os.getenv("NEO4J_URI"),
#         os.getenv("NEO4J_USERNAME"),
#         os.getenv("NEO4J_PASSWORD")
#     )

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

def score_evidence(context_json: dict, kg_records: list) -> list:
    """
    Score each instrument on a 0-100 scientific scale plus operational bonuses.
    All penalties are replaced by a structured Risk Flag System.
    Scientific score is never deducted — flags carry severity independently.
    """

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

    for record in kg_records:
        instrument_name   = str(record.get("instrument_name", "Unknown"))
        instrument_lower  = instrument_name.lower()

        domain_search_parts = [
            _to_str(record.get("instrument_domain")),
            _to_str(record.get("domains_measured")),
            _to_str(record.get("key_finding")),
            _to_str(record.get("subscale_results")),
            _to_str(record.get("instrument_subscales_assessed")),
            _to_str(record.get("strengths")),
        ]
        instrument_domains = " ".join(
            [" ".join(BROAD_INSTRUMENTS.get(instrument_lower, []))] +
            domain_search_parts
        )
        instrument_domains_list = [
            d.strip().lower()
            for part in domain_search_parts
            for d in re.split(r"[,;]", part)
            if d.strip()
        ]

        mode_options         = _to_str(record.get("mode_options"))
        source_documents     = _to_str(record.get("source_documents"))
        developer_str        = _to_str(record.get("developer"))
        endpoint_role        = _to_str(record.get("endpoint_role"))
        prespecified         = _to_str(record.get("prespecified"))
        regulatory_acceptance = _to_str(record.get("regulatory_acceptance"))
        validation_status    = _to_str(record.get("validation_status", ""))

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
        # COMPONENT 3 — VALIDATED MCID  (0–20)
        #
        # +20  Anchor-based MCID
        # +12  Distribution-based MCID only
        #   0  No MCID
        #
        # Source: FDA PRO Guidance (2009) Section V.C
        # ═══════════════════════════════════════════════════════════════════
        if mcid_valid:
            mcid_full_text = _to_str(raw_mcid).lower()
            anchor_terms = [
                "anchor", "patient global", "pgic", "external criterion",
                "anchor-based", "anchor based", "clinician global"
            ]
            dist_terms = [
                "distribution", "sem", "standard error", "effect size",
                "half sd", "0.5 sd", "distribution-based", "distribution based"
            ]
            is_anchor_based = any(t in mcid_full_text for t in anchor_terms)
            is_dist_only    = (any(t in mcid_full_text for t in dist_terms)
                                and not is_anchor_based)

            if is_dist_only:
                raw_score += 12
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"Validated MCID +12 (distribution-based): MCID established "
                    f"({mcid_display}) using distribution-based methods only. "
                    "Anchor-based MCID is preferred by FDA for label claim "
                    "responder analyses [FDA PRO Guidance (2009) Section V.C]."
                )))
            else:
                raw_score += 20
                method = "anchor-based" if is_anchor_based else "established"
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"Validated MCID +20 ({method}): MCID established — "
                    f"{mcid_display}. Enables responder analysis required for "
                    "label claims [FDA PRO Guidance (2009) Section V.C]."
                )))
        else:
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Validated MCID 0: MCID not established — responder analysis "
                "impossible, limiting label claim language to mean change statistics "
                "[FDA PRO Guidance (2009) Section V.C]."
            )))

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
                    f"For {administration}, FDA PFDD Guidance 3 (2022) requires "
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
                    "Per FDA PFDD Guidance 3 (2022), recall must match "
                    "symptom fluctuation pattern."
                )))
            else:
                source = recall_period_key or "published validation"
                flags.append(linkify_flag_citations(ensure_full_stop(
                    f"Recall period compatible: {instrument_name} has "
                    f"{recall_period}-day recall ({source}), compatible with "
                    f"{administration} [FDA PFDD Guidance 3 (2022)]."
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
            operational_bonus += 40
            flags.append(linkify_flag_citations(ensure_full_stop(
                "eCOA Ready +40 (operational): Electronic/app-based "
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
            operational_bonus += 25
            flags.append(linkify_flag_citations(ensure_full_stop(
                "Open Access +25 (operational): Instrument developed by an "
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
def build_coverage_matrix(scored: list, context_json: dict, kg_records: list) -> dict:
    """
    Build a domain coverage matrix showing which instruments cover which FDA core domains.
    This replaces the battery optimizer — we show options, experts decide.

    Returns:
      domains: list of {domain, candidates, hta_required, item_library_note}
      comparator_trials: list of {trial_name, drug, drug_class, instruments, nct_id}
      hta_mandatory: list of instruments always required regardless of score
      item_library_applicable: bool — True if instrument has >30 items
      all_candidates: top 8 scored instruments with full data
    """
    indication   = _to_str(context_json.get("indication"))
    hta_markets  = context_json.get("hta_markets", [])
    drug_class   = _to_str(context_json.get("drug_class"))
    core_domains = [str(d).lower() for d in context_json.get("core_domains_required", [])]
    extra_domains = [str(d).lower() for d in context_json.get("additional_domains", [])]
    all_domains  = list(dict.fromkeys(core_domains + extra_domains))

    # Map each domain to the instruments that cover it (from scored list)
    domain_coverage = []
    for domain in all_domains:
        synonyms  = DOMAIN_SYNONYMS.get(domain, [domain])
        all_terms = [domain] + synonyms
        candidates = []
        for inst in scored:
            inst_lower = inst["instrument_name"].lower()
            broad_text = " ".join(BROAD_INSTRUMENTS.get(inst_lower, []))
            flags_text = " ".join(inst.get("flags", [])).lower()
            search_text = " ".join([inst_lower, broad_text, flags_text])
            if any(t in search_text for t in all_terms):
                # Check if any KG record shows change was detected
                kg_matches = [r for r in kg_records
                              if r.get("instrument_name") == inst["instrument_name"]]
                change_detected = "Yes" if any(
                    "significant" in _to_str(r.get("significance"))
                    and "favour" in _to_str(r.get("direction"))
                    for r in kg_matches
                ) else ("No" if kg_matches else "NR")

                candidates.append({
                    "instrument":      inst["instrument_name"],
                    "score":           inst["scientific_score"],
                    "risk":            inst["risk_level"],
                    "change_detected": change_detected,
                    "precedent_trial": kg_matches[0].get("trial_name", "") if kg_matches else "",
                    "precedent_nct":   kg_matches[0].get("nct_id", "") if kg_matches else "",
                })
        candidates.sort(key=lambda x: -x["score"])

        # Item library note: if best candidate has >30 items, suggest subscale approach
        item_library_note = ""
        for inst in scored:
            if candidates and inst["instrument_name"] == candidates[0]["instrument"]:
                items_raw = inst.get("raw_positive_score", 0)  # use as proxy
                # Check KG records for subscale usage
                subscale_used = any(
                    r.get("subscale_results") or r.get("instrument_subscales_assessed")
                    for r in kg_records
                    if r.get("instrument_name") == inst["instrument_name"]
                )
                if subscale_used:
                    item_library_note = (
                        f"Note: Comparator trials have used subscale/item-library "
                        f"approaches with {inst['instrument_name']} rather than full "
                        f"administration. Consider calibrated PRO SOA approach to reduce "
                        f"patient burden — discuss with COA expert."
                    )

        domain_coverage.append({
            "domain":           domain,
            "candidates":       candidates,
            "item_library_note": item_library_note,
            "is_fda_core":      domain in core_domains,
        })

    # HTA mandatory instruments (always required, separate from scoring)
    hta_mandatory = []
    hta_wildcards = {"NICE": "eq-5d", "ICER": "eq-5d", "EUnetHTA": "eq-5d", "SMC": "eq-5d"}
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
                hta_mandatory.append({
                    "instrument": "EQ-5D-5L",
                    "market":     market,
                    "score":      score_val,
                    "reason":     f"Required for {market} cost-utility (QALY) analysis. "
                                  f"Without EQ-5D, QALY calculation is impossible.",
                })

    # Comparator trials: KG trials in same drug class or indication
    comparator_map = {}
    for r in kg_records:
        trial_name = r.get("trial_name") or r.get("nct_id", "")
        if not trial_name:
            continue
        if trial_name not in comparator_map:
            comparator_map[trial_name] = {
                "trial_name":  trial_name,
                "drug":        r.get("drug_name", ""),
                "drug_class":  r.get("drug_class_name", "") or r.get("disease_classification", ""),
                "nct_id":      r.get("nct_id", ""),
                "phase":       r.get("phase", ""),
                "instruments": [],
            }
        comparator_map[trial_name]["instruments"].append({
            "name":          r.get("instrument_name", ""),
            "role":          r.get("endpoint_role", "") or r.get("pro_position", ""),
            "significance":  r.get("significance", ""),
            "direction":     r.get("direction", ""),
            "prespecified":  r.get("prespecified", ""),
            "subscales":     r.get("instrument_subscales_assessed", ""),
        })

    # Sort comparators: same drug class first
    comparators = list(comparator_map.values())
    comparators.sort(key=lambda x: (
        0 if any(term in _to_str(x["drug_class"]) for term in drug_class.split())
        else 1
    ))

    # Check if item library approach is applicable
    item_library_applicable = any(
        r.get("instrument_subscales_assessed") or r.get("subscale_results")
        for r in kg_records
    )

    return {
        "domains":                domain_coverage,
        "comparator_trials":      comparators[:5],
        "hta_mandatory":          hta_mandatory,
        "item_library_applicable": item_library_applicable,
        "all_candidates":         scored[:8],
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

    # --- STEP D: Build coverage matrix (replaces battery optimizer) ---
    coverage = build_coverage_matrix(scored, context_json, kg_records) if scored else {
        "domains": [], "comparator_trials": [], "hta_mandatory": [],
        "item_library_applicable": False, "all_candidates": []
    }
    top_5 = coverage["all_candidates"][:5]

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

    # --- STEP F: Clean KG narrative fields ---
    if kg_records:
        try:
            kg_records = clean_kg_narratives(kg_records)
        except Exception as e:
            logging.warning(f"KG cleaning skipped: {e}")

    # --- STEP G: Build evidence block for Sonnet ---
    citation_index = {}
    # --- STEP G: Build structured evidence block for Sonnet ---
    kg_block_lines = []

    if error_status:
        kg_block_lines.append(f"⚠️ KG OFFLINE — {error_status}")
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
        position_counts = {}
        for r in kg_records:
            pos = str(r.get("pro_position") or r.get("endpoint_role") or "Unknown").strip()
            if pos and pos.lower() not in ("none", "nan", ""):
                position_counts[pos] = position_counts.get(pos, 0) + 1

        if position_counts:
            total_pos = sum(position_counts.values())
            pos_lines = [
                f"  {pos}: {count} trial(s) ({int(100*count/total_pos)}%)"
                for pos, count in sorted(position_counts.items(),
                                         key=lambda x: -x[1])
            ]
            kg_block_lines.append("\n=== PRO ENDPOINT POSITIONING IN KG TRIALS ===")
            kg_block_lines.append(
                f"Distribution of PRO endpoint positions across {total_pos} "
                f"KG trial records for this indication:"
            )
            kg_block_lines.extend(pos_lines)
            kg_block_lines.append(
                "Use this data to reason about appropriate endpoint positioning "
                "for the current trial. Do not apply a rule — let the data speak."
            )

        # === INSTRUMENT SCORING (for Table 2 item counts) ===
        kg_block_lines.append(f"\n=== INSTRUMENT REFERENCE DATA ===")
        for i, ir in enumerate(inst_refs[:8], 1):
            kg_block_lines.append(
                f"[IR-{i:03d}] {ir.get('short_name','')} | "
                f"Items: {ir.get('total_items','?')} | "
                f"Admin time: {ir.get('admin_time','?')} min | "
                f"Recall: see INSTRUMENT_RECALL_PERIODS | "
                f"MCID: {ir.get('mcid','Not established')} | "
                f"Regulatory acceptance: {ir.get('regulatory_acceptance','')}"
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
            kg_block_lines.append(f"\n=== REGULATORY RULES ({len(reg_rules)}) ===")
            for i, rule in enumerate(reg_rules[:8], 1):
                kg_block_lines.append(
                    f"[RULE-{i:03d}] {rule.get('source_document','')} | "
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

    # --- STEP I: Build Sonnet system prompt ---
    sonnet_system = f"""You are a COA specialist synthesising evidence for a senior COA expert who will make the final decisions. Your job is to present information clearly so the expert can decide — not to recommend or prescribe.

RULE 1 (Two tables are mandatory — this is the most important output):
You must produce exactly two markdown tables. Do not skip or combine them.

TABLE 1 — Domain Coverage Comparison:
Columns: FDA Core Concept | Current Trial (Candidate Instruments) | [Comparator Trial 1] | [Comparator Trial 2] | [Comparator Trial 3]
Rows: Each FDA core domain for this indication + one row for HTA requirement
Use the comparator trials from the KG evidence block to name the columns.
For each cell: list the instrument used + whether significant change was detected (Y/N/NR).
Use ✅ (covered), ⚠️ (partial/general only), ❌ (not collected) for the comparator columns.
For the Current Trial column: list candidate instruments from the Domain Coverage Matrix above.
The HTA row MUST include EQ-5D-5L for NICE/ICER/EUnetHTA markets — always.
Cite each cell with the KG label: [TI-001] etc.

TABLE 2 — PRO Measures Comparison:
Columns: Trial | PRO Measures (n items each) | Calibrated SOA | Total items | Est. completion time
Rows: Current trial (candidate approach) + each comparator trial from KG
Use instrument item counts from the IR-XXX blocks in the evidence.
For "Calibrated SOA": Yes if subscales were used, No if full instruments, TBD for current trial.
Cite each row with [TI-XXX] for comparator trials.

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

OUTPUT STRUCTURE (follow exactly):

## COA Measurement Strategy — [Indication] [Phase]
[1-2 sentences: the key COA challenge for this trial]

## Table 1: Domain Coverage Comparison
[mandatory table — see RULE 1]

## Table 2: PRO Measures Comparison
[mandatory table — see RULE 1]

## Key Observations
[maximum 6 bullet points, each citing a source, each connecting to a specific table cell]
[include item library note if applicable — RULE 4]

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
        # Try with extended thinking first (no web search)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=15000,
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
        "kg_evidence_block": kg_evidence_block
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
            "hta_mandatory": [h["instrument"] for h in result.get("coverage", {}).get("hta_mandatory", [])],
            "comparator_trials": [t["trial_name"] for t in result.get("coverage", {}).get("comparator_trials", [])],
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