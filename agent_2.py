"""
AI Agent for PRO COA Instrument Recommendation
Combines Neo4j knowledge graph evidence with live web search
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from graph import Neo4jConnection
import re

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# ============================================================================
# CONSTANTS — INDICATION-SPECIFIC CORE DOMAINS
# ============================================================================
INDICATION_CORE_DOMAINS = {
    "multiple myeloma": ["bone pain", "physical function", "fatigue"],
    "mm": ["bone pain", "physical function", "fatigue"],
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

# ============================================================================
# CONSTANTS — HTA/PAYER INSTRUMENT PREFERENCES
# ============================================================================
HTA_PREFERENCES = {
        "NICE": {
            "required_instruments": ["EQ-5D"],
            "accepted_versions": ["EQ-5D-5L", "EQ-5D-3L"],
            "preferred_version": "EQ-5D-5L",
        "preferred_instruments": ["EQ-5D-5L"],
        "notes": "NICE requires EQ-5D for cost-utility analysis. Without it, QALY calculation is impossible and UK reimbursement is severely compromised.",
        "reference": "NICE DSU Technical Support Document 2 (2011, updated 2019)"
    },
    "ICER": {
        "required_instruments": [],
        "preferred_instruments": ["EQ-5D-5L", "SF-36", "SF-6D"],
        "notes": "ICER uses utility-based measures for cost-effectiveness analysis in US value assessments.",
        "reference": "ICER Value Assessment Framework (2020)"
    },
    "EUnetHTA": {
        "required_instruments": [],
        "preferred_instruments": ["EQ-5D-5L", "EORTC QLQ-C30"],
        "notes": "Joint Clinical Assessments under EU HTA Regulation (2022/282) increasingly require standardised PRO instruments for cross-country comparison.",
        "reference": "EU HTA Regulation 2021/2282, EUnetHTA 21 methodology guidelines"
    },
    "SMC": {
        "required_instruments": ["EQ-5D-5L"],
        "preferred_instruments": ["EQ-5D-5L"],
        "notes": "Scottish Medicines Consortium aligns with NICE on EQ-5D requirement.",
        "reference": "SMC Modifiers and PACE framework"
    }
}

# ============================================================================
# CONSTANTS — GEOGRAPHIC LANGUAGE REQUIREMENTS
# ============================================================================
GEOGRAPHIC_LANGUAGE_REQUIREMENTS = {
    "Global": {
        "min_languages": 20,
        "key_languages": ["English", "Spanish", "French", "German", "Italian", "Japanese", "Mandarin", "Portuguese", "Russian", "Korean"],
        "regulatory_note": "FDA requires linguistic validation per FDA PRO Guidance (2009) Section IV.A. Mode equivalence studies required when transitioning paper-validated instruments to eCOA.",
        "reference": "FDA PRO Guidance (2009); ISPOR ePRO Task Force Report (2009)"
    },
    "US-only": {
        "min_languages": 1,
        "key_languages": ["English"],
        "regulatory_note": "English validation sufficient for US-only trials.",
        "reference": "FDA PRO Guidance (2009)"
    },
    "EU": {
        "min_languages": 10,
        "key_languages": ["English", "French", "German", "Spanish", "Italian", "Dutch", "Polish", "Swedish", "Danish", "Finnish"],
        "regulatory_note": "EMA requires validated translations for each member state language where the trial is conducted.",
        "reference": "EMA Reflection Paper on PRO (2005)"
    }
}

# ============================================================================
# CONSTANTS — KNOWN LANGUAGE COUNTS
# ============================================================================
KNOWN_LANGUAGE_COUNTS = {
    "eq-5d": 100,
    "eq-5d-5l": 100,
    "eq-5d-3l": 100,
    "eortc qlq-c30": 85,
    "eortc qlq-lc13": 85,
    "eortc qlq-my20": 85,
    "eortc qlq-pr25": 85,
    "eortc qlq-hn35": 85,
    "fact-g": 85,
    "fact-p": 85,
    "fact-b": 85,
    "fact-l": 85,
    "facit-fatigue": 85,
    "promis": 85,
    "sf-36": 85,
    "sf-12": 85,
    "sf-6d": 85,
    "bpi": 85,
    "bpi-sf": 85,
    "nrs": 85,
    "vas": 85,
    "pro-ctcae": 85,
    "pgis": 85,
    "pgic": 85,
    "hads": 85,
    "gad-7": 85,
    "phq-9": 85,
    "ipss": 85,
    "default": 0
}

# ============================================================================
# INSTRUMENT RECALL PERIODS
# ============================================================================
INSTRUMENT_RECALL_PERIODS = {
    "eq-5d": 0,
    "eq-5d-5l": 0,
    "eq-5d-3l": 0,
    "bpi-sf": 1,
    "bpi": 1,
    "nrs": 1,
    "vas": 1,
    "pro-ctcae": 7,
    "fact-p": 7,
    "fact-g": 7,
    "fact-b": 7,
    "fact-l": 7,
    "facit-fatigue": 7,
    "eortc qlq-c30": 7,
    "eortc qlq-lc13": 7,
    "eortc qlq-my20": 7,
    "eortc qlq-pr25": 7,
    "promis": 7,
    "pgis": 1,
    "pgic": 1,
    "sf-36": 28,
    "sf-12": 28,
    "sf-6d": 28,
    "hads": 7,
    "gad-7": 14,
    "phq-9": 14,
    "ipss": 30,
    "eortc qlq-hn35": 7,
    "default": 7
}

# ============================================================================
# GLOSSARY LOADING
# ============================================================================
GLOSSARY_TEXT = ""
try:
    glossary_df = pd.read_csv("PRO_Terminology_Glossary.csv")
    glossary_df = glossary_df.sort_values("Importance_Rank") if "Importance_Rank" in glossary_df.columns else glossary_df
    rows = []
    for _, row in glossary_df.head(20).iterrows():
        rows.append(" | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val)))
    GLOSSARY_TEXT = "\n".join(rows)
except Exception as e:
    GLOSSARY_TEXT = "Glossary unavailable. Use standard COA terminology."
    logging.warning(f"Glossary load failed: {e}")

# ============================================================================
# HAIKU SYSTEM PROMPT
# ============================================================================
HAIKU_SYSTEM_PROMPT = """You are a clinical trial parameter extractor for oncology COA strategy. Parse the user input into a strict JSON object. For any missing field, infer the most likely 2026 industry standard based on context, flag it in 'assumptions_made', and explain your reasoning.

Core Domain Lookup (from FDA 2021 Core PRO Guidance — use this to populate core_domains_required):
- Multiple Myeloma: bone pain, physical function, fatigue
- NSCLC: dyspnea, cough, chest pain, physical function
- CRPC/Prostate Cancer: pain, urinary function, physical function
- Breast Cancer: fatigue, pain, physical function, emotional function
- Colorectal Cancer: nausea, appetite loss, bowel function, fatigue
- Ovarian Cancer: abdominal pain, bloating, fatigue, physical function
- Default (unknown oncology): physical function, fatigue, pain

Inference Rules (apply these if data is missing):
1. If tpp_claims is missing → output ["Inferred: Treatment Tolerability", "Inferred: Physical Function Maintenance"] and note in assumptions_made
2. If population_subtype is missing → default to "Symptomatic" and note it
3. If phase is missing → default to "Phase 3" and note it
4. If drug_class suggests Bispecific, CAR-T, or contains "bispecific" or "car-t" → infer administration as "Step-up dosing" and note it
5. If geographic_footprint is missing → infer from phase: Phase 3 = "Global", Phase 2 = "EU or US", Phase 1 = "US-only"
6. If hta_markets is missing → infer from geographic_footprint: Global = ["NICE", "ICER", "EUnetHTA"], EU = ["NICE", "EUnetHTA"], US-only = ["ICER"]

Return ONLY valid JSON. No markdown. No explanation outside the JSON."""


# ============================================================================
# STEP 1: THE ANALYZER
# ============================================================================
def analyze_trial_context(user_text: str) -> dict:
    """Extract trial parameters using Claude Haiku"""
    expected_format = """{
  "indication": "string — primary cancer type",
  "indication_synonyms": ["list of synonyms and abbreviations for KG search"],
  "population_subtype": "string — e.g., Relapsed/Refractory, First-Line, Smoldering, Maintenance, Neoadjuvant, Unknown",
  "phase": "Phase 1 | Phase 2 | Phase 3",
  "drug_class": "string e.g. Bispecific, Proteasome Inhibitor, ICI, CDK4/6 inhibitor",
  "administration": "Step-up dosing | Subcutaneous | IV | Oral | Unknown",
  "dosing_frequency": "Weekly | Biweekly | Monthly | Unknown",
  "tpp_claims": ["list of desired label claims"],
  "core_domains_required": ["combine indication-specific core domains from the lookup WITH any specific symptom/function domains required to prove the tpp_claims"],
  "geographic_footprint": "Global | EU | US-only | Unknown",
  "hta_markets": ["list of relevant HTA bodies: NICE, ICER, EUnetHTA, SMC"],
  "trial_duration_cycles": "number or Unknown",
  "assumptions_made": ["list of strings — each explaining one inference with reasoning"]
}"""
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=HAIKU_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Extract trial parameters from this input. Return only JSON:\n\n{user_text}\n\nExpected format:\n{expected_format}"
            }]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        context = json.loads(raw)
        logging.info(f"Analyzer output: {json.dumps(context)}")
        return context
    except json.JSONDecodeError as e:
        logging.error(f"Analyzer JSON parse failed: {e}. Raw: {raw}")
        return {
            "indication": "unknown",
            "indication_synonyms": [],
            "population_subtype": "Symptomatic",
            "phase": "Phase 3",
            "drug_class": "Unknown",
            "administration": "Unknown",
            "tpp_claims": ["Inferred: Treatment Tolerability", "Inferred: Physical Function Maintenance"],
            "core_domains_required": ["physical function", "fatigue", "pain"],
            "geographic_footprint": "Global",
            "hta_markets": ["NICE", "ICER", "EUnetHTA"],
            "assumptions_made": ["Full inference applied — analyzer failed to parse input"],
            "dosing_frequency": "Unknown",
            "trial_duration_cycles": "Unknown"
        }
    except Exception as e:
        logging.error(f"Analyzer call failed: {e}")
        return {
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
            "assumptions_made": [f"Analyzer API failed ({str(e)}) — all defaults applied"],
            "dosing_frequency": "Unknown",
            "trial_duration_cycles": "Unknown"
        }


# ============================================================================
# STEP 2: THE SCORING ENGINE
# ============================================================================
def score_evidence(context_json: dict, kg_records: list) -> list:
    """Score instruments based on trial context and KG evidence"""
    
    indication = str(context_json.get("indication", "")).lower()
    population = str(context_json.get("population_subtype", "Symptomatic"))
    phase = str(context_json.get("phase", "Phase 3"))
    administration = str(context_json.get("administration", "Unknown"))
    # Strip "Inferred: " prefix so inferred TPP claims can match instrument domains
    tpp_claims = [
        str(c).lower().replace("inferred: ", "").strip()
        for c in context_json.get("tpp_claims", [])
    ]
    core_domains = [str(d).lower() for d in context_json.get("core_domains_required", [])]
    geographic_footprint = str(context_json.get("geographic_footprint", "Global"))
    hta_markets = context_json.get("hta_markets", [])
    drug_class = str(context_json.get("drug_class", "")).lower()

    results = []

    for record in kg_records:
        instrument_name = str(record.get("instrument_name", "Unknown"))
        instrument_lower = instrument_name.lower()

        # Handle instrument_domain as list or pipe-separated string
        raw_domains = record.get("instrument_domain", [])
        if isinstance(raw_domains, list):
            instrument_domains_list = [str(d).strip().lower() for d in raw_domains if d]
        else:
            instrument_domains_list = [d.strip().lower() for d in str(raw_domains).split("|") if d.strip()]
        instrument_domains = " ".join(instrument_domains_list)

        # Handle mode_options as list or string
        raw_mode = record.get("mode_options", "")
        mode_options = " ".join(str(m) for m in raw_mode).lower() if isinstance(raw_mode, list) else str(raw_mode).lower()

        # Handle source_documents as list or string
        raw_source = record.get("source_documents", "")
        source_documents = " ".join(str(s) for s in raw_source).lower() if isinstance(raw_source, list) else str(raw_source).lower()

        endpoint_role = str(record.get("endpoint_role", "")).lower()
        prespecified = str(record.get("prespecified", "")).lower()
        significance = str(record.get("significance", "")).lower()
        mid_met = str(record.get("mid_met", "")).lower()
        direction = str(record.get("direction", "")).lower()
        regulatory_acceptance = str(record.get("regulatory_acceptance", "")).lower()
        fda_alignment = str(record.get("fda_alignment", "")).lower()
        drug_class_match = str(record.get("disease_classification", "")).lower()

        # total_items: extract first integer found anywhere in the value
        total_items_raw = str(record.get("total_items", ""))
        num_match = re.search(r'\d+', total_items_raw)
        total_items = int(num_match.group()) if num_match else 0

        # recall_period: look up by instrument name
        recall_period = next(
            (days for key, days in INSTRUMENT_RECALL_PERIODS.items() if key in instrument_lower),
            INSTRUMENT_RECALL_PERIODS["default"]
        )

        # language count: handle "85+", lists, or pipe-separated strings
        languages_val = record.get("languages", "")
        languages_str = str(languages_val).lower()
        if "85+" in languages_str or "100+" in languages_str or "all major" in languages_str:
            language_count = 100
        elif isinstance(languages_val, list):
            language_count = len([l for l in languages_val if l])
        else:
            language_count = len([l for l in languages_str.split("|") if l.strip()])

        raw_score = 0
        operational_bonus = 0
        flags = []

        # ── POSITIVE WEIGHTS (max 100) ──────────────────────────────

        # 1. TPP / Core Domain Fit (+35)
        tpp_match = False
        for claim in tpp_claims:
            for domain in instrument_domains_list:
                if claim in domain or domain in claim:
                    tpp_match = True
                    break
            if tpp_match:
                break
        if tpp_match or any(domain in instrument_domains for domain in core_domains):
            raw_score += 35
            flags.append("TPP/Core Fit (+35): Instrument domains align with TPP claims and FDA-defined core domains for this indication [FDA 2021 Core PRO Guidance]")

        # 2. Regulatory Trust (+25)
        if any(term in regulatory_acceptance for term in ["fda", "ema", "accepted", "approved", "strong"]):
            raw_score += 25
            flags.append("Regulatory Trust (+25): Instrument has documented FDA/EMA regulatory acceptance [FDA PRO Guidance 2009, Section V; EMA Reflection Paper 2005]")
        elif any(term in regulatory_acceptance for term in ["moderate", "conditional", "exploratory"]):
            raw_score += 12
            flags.append("Regulatory Trust (+12, partial): Moderate regulatory acceptance documented")

        # 3. Competitor / SoC Benchmark (+20)
        if drug_class and drug_class_match and drug_class in drug_class_match:
            raw_score += 20
            flags.append("Competitor Benchmark (+20): Instrument used in standard-of-care trials for this drug class — indicates regulatory familiarity [FDA PRO Guidance 2009, Section III.B]")
        elif record.get("trial_prevalence", "") and "high" in str(record.get("trial_prevalence", "")).lower():
            raw_score += 10
            flags.append("Competitor Benchmark (+10, partial): High trial prevalence across oncology but not specific to this drug class")

        # 4. MoA-Specific Sensitivity (+20)
        moa_keywords = {
            "bispecific": ["cytokine release", "crs", "fatigue", "neurotoxicity", "icans"],
            "car-t": ["cytokine release", "crs", "fatigue", "neurotoxicity"],
            "proteasome inhibitor": ["peripheral neuropathy", "neuropathy", "fatigue"],
            "ici": ["fatigue", "immune-related", "diarrhea", "endocrine"],
            "cdk4/6": ["fatigue", "nausea", "neutropenia"],
            "antibody drug conjugate": ["nausea", "fatigue", "neuropathy", "alopecia"]
        }
        for drug_class_key, toxicity_domains in moa_keywords.items():
            if drug_class_key in drug_class:
                if any(tox in instrument_domains for tox in toxicity_domains):
                    raw_score += 20
                    flags.append(f"MoA Sensitivity (+20): Instrument captures class-specific toxicity domains for {drug_class} [FDA PFDD Guidance 1, 2017 — patient experience must reflect drug mechanism]")
                    break

        # 5. Validated MCID Exists (+10)
        raw_mcid = record.get("mcid", "")
        mcid = " ".join(str(m) for m in raw_mcid).lower() if isinstance(raw_mcid, list) else str(raw_mcid).lower()
        mcid_null_terms = ["none", "not established", "unknown", "nan", "n/a", "tbd",
                           "not reported", "pending", "null", ""]
        mcid_valid = (
            mcid.strip() != "" and
            mcid.strip() not in mcid_null_terms and
            not any(term in mcid for term in ["not established", "not reported", "unknown", "pending"])
        )
        if mcid_valid:
            raw_score += 10
            flags.append(ensure_full_stop(f"Validated MCID (+10): MCID established ({mcid.strip()}) — enables responder analysis required for label claims [FDA PRO Guidance 2009, Section V.C]"))
        else:
            flags.append(ensure_full_stop("No Validated MCID (note): MCID not established — responder analysis impossible, limiting label claim language [FDA PRO Guidance 2009, Section V.C]"))

        # Cap at 100
        raw_score = min(raw_score, 100)

        # ── CONDITIONAL PENALTIES ─────────────────────────────────
        penalty_total = 0
        risk_level = "LOW"

        # PENALTY 1: Missing Core (-50)
        if population == "Symptomatic" and core_domains:
            # Broaden search: check instrument domains, key_finding, subscale_results, and strengths
            extended_search_text = " ".join([
                instrument_domains,
                str(record.get("key_finding", "")).lower(),
                str(record.get("subscale_results", "")).lower(),
                str(record.get("strengths", "")).lower(),
                str(record.get("domains", "")).lower(),  # from Instrument node
                str(record.get("instrument_subscales_assessed", "")).lower(),
            ])

            # Also: known broad instruments that implicitly cover all core domains
            BROAD_INSTRUMENTS = [
                "eortc qlq-c30", "eortc qlq-my20", "fact-g", "fact-p",
                "sf-36", "promis", "eq-5d"
            ]
            is_known_broad = any(b in instrument_lower for b in BROAD_INSTRUMENTS)

            # Domain synonym map: expand narrow search terms to catch variant storage
            DOMAIN_SYNONYMS = {
                "bone pain": ["pain", "bone", "analgesic", "bpi", "nrs", "aches"],
                "physical function": ["physical", "function", "activity", "mobility", "performance"],
                "fatigue": ["fatigue", "tiredness", "energy", "exhaustion", "asthenia"],
                "dyspnea": ["dyspnea", "breathlessness", "breathing", "respiratory"],
                "cough": ["cough", "respiratory"],
                "pain": ["pain", "analgesic", "bpi", "nrs", "aches", "discomfort"],
                "nausea": ["nausea", "vomiting", "gi", "gastrointestinal"],
                "urinary function": ["urinary", "urology", "bladder", "ipss"],
                "emotional function": ["emotional", "anxiety", "depression", "psychological", "mental"],
            }

            missing_cores = []
            for domain in core_domains:
                synonyms_to_check = DOMAIN_SYNONYMS.get(domain.lower(), [domain])
                synonyms_to_check = [domain] + synonyms_to_check
                found = (
                    is_known_broad or
                    any(syn in extended_search_text for syn in synonyms_to_check)
                )
                if not found:
                    missing_cores.append(domain)
            if len(missing_cores) >= len(core_domains) / 2:
                penalty_total += 50
                risk_level = "CRITICAL"
                flags.append(
                    f"MISSING CORE PENALTY (-50, CRITICAL): Instrument does not measure core domains "
                    f"{missing_cores} required for {indication} per FDA (2021) 'Core Patient-Reported "
                    f"Outcomes in Cancer Clinical Trials'. Risk: Refusal to File or rejection of PRO "
                    f"label claim. This instrument CANNOT be ranked #1 for this indication."
                )

        # PENALTY 2: Recall Bias (-40)
        step_up_admins = ["step-up dosing", "weekly iv", "weekly"]
        if any(a in administration.lower() for a in step_up_admins) and recall_period > 7:
            penalty_total += 40
            if risk_level != "CRITICAL":
                risk_level = "CRITICAL"
            flags.append(
                f"RECALL BIAS PENALTY (-40, CRITICAL): {recall_period}-day recall period incompatible "
                f"with {administration} schedule. CRS/ICANS events occur within 24-72 hours of dosing. "
                f"Per FDA PFDD Guidance 3 (2022), recall must match symptom fluctuation pattern. "
                f"Data would likely be characterised as exploratory-only by FDA, precluding label claims."
            )

        # PENALTY 3: Pre-specification / Alpha Control (-35)
        # Only penalise when there IS a record explicitly showing non-pre-specification.
        # Do not penalise instruments with no KG record (prespecified="" means no data, not "no").
        has_record = instrument_name != "Unknown" and (prespecified != "" or endpoint_role != "")
        if has_record and prespecified not in ["yes", "true", "1"] and endpoint_role in ["exploratory", "unknown"]:
            penalty_total += 35
            if risk_level not in ["CRITICAL"]:
                risk_level = "HIGH"
            flags.append(
                "PRE-SPECIFICATION PENALTY (-35, HIGH): Instrument not pre-specified in SAP with "
                "alpha controlled in testing hierarchy. Results will be exploratory only and cannot "
                "support formal label claims per FDA PRO Guidance (2009) Section V and ICH E9 (1998) "
                "Section 2.2.5. This is the most common avoidable cause of PRO data failing to reach "
                "the label — as documented in the abiraterone COU-AA-301/302 precedent in this KG."
            )

        # PENALTY 4: Estimand Burden (-30)
        if ("phase 3" in phase.lower() or "phase iii" in phase.lower()) and total_items > 30:
            penalty_total += 30
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "HIGH"
            flags.append(
                f"ESTIMAND BURDEN PENALTY (-30, HIGH): {total_items}-item instrument in Phase 3 trial. "
                f"ICH E9(R1) Addendum (2019) requires Treatment Policy estimand — PRO data collection "
                f"must continue post-discontinuation. Instruments >30 items show materially lower "
                f"completion rates in this setting, generating missing data patterns that complicate "
                f"analysis under ICH E9(R1) Section 3.2. Consider shorter companion instrument or "
                f"subscale scoring approach."
            )

        # PENALTY 5: No Validated MCID (-20)
        if not mcid_valid:
            penalty_total += 20
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "MODERATE"
            flags.append(
                "NO MCID PENALTY (-20, MODERATE): No validated Minimum Important Difference established. "
                "FDA PRO Guidance (2009) Section V.C requires MCID for responder analysis. Without it, "
                "label language is restricted to mean change statistics — weaker regulatory argument "
                "than responder-based language (e.g. 'X% of patients achieved clinically meaningful "
                "improvement')."
            )

        # PENALTY 6: Asymptomatic Burden (-20)
        symptom_heavy_instruments = ["bpi", "pain", "nrs", "vas", "bone pain", "symptom"]
        is_symptom_heavy = any(s in instrument_name.lower() or s in instrument_domains for s in symptom_heavy_instruments)
        if population == "Asymptomatic/Smoldering" and is_symptom_heavy:
            penalty_total += 20
            if risk_level not in ["CRITICAL", "HIGH"]:
                risk_level = "MODERATE"
            flags.append(
                "ASYMPTOMATIC BURDEN PENALTY (-20, MODERATE): Symptom-heavy instrument applied to "
                "Asymptomatic/Smoldering population. FDA PRO Guidance (2009) Section IV.B requires "
                "instruments to be 'minimally burdensome' — measuring symptoms the patient does not "
                "yet have creates questionnaire fatigue that may reduce completion of clinically "
                "relevant items. Consider replacing with HRQoL-focused instrument (EQ-5D-5L, FACT-G) "
                "that does not presuppose symptoms [FDA PFDD Guidance 2, 2018]."
            )

        # ── OPERATIONAL BONUSES ────────────────────────────────

        # eCOA Ready (+10)
        if any(term in mode_options for term in ["ecoa", "electronic", "app", "tablet", "digital"]):
            operational_bonus += 10
            flags.append(ensure_full_stop("eCOA Ready (+10 operational): Electronic mode supported — reduces transcription error and enables real-time monitoring per FDA eCOA Guidance (2023)"))

        # Open Access (+5) — check by known open-access developer
        OPEN_ACCESS_DEVELOPERS = [
            "eortc", "nci", "national cancer institute", "facit", "promis",
            "rand", "who", "world health organization", "nih", "pcori",
            "fact-g", "fact-b", "fact-p"
        ]
        developer_raw = record.get("developer", "")
        developer_str = " ".join(str(d) for d in developer_raw).lower() if isinstance(developer_raw, list) else str(developer_raw).lower()
        if any(dev in developer_str or dev in source_documents or dev in instrument_lower
               for dev in OPEN_ACCESS_DEVELOPERS):
            operational_bonus += 5
            flags.append(ensure_full_stop("Open Access (+5 operational): Instrument from open-access developer — no restrictive commercial licensing, reduces trial setup time"))

        # Translation Gap (-15)
        geo_req = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(geographic_footprint, GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
        required_languages = geo_req["min_languages"]
        if geographic_footprint in ["Global", "EU"] and language_count < required_languages:
            operational_bonus -= 15
            flags.append(ensure_full_stop(
                f"TRANSLATION GAP (-15 operational): {language_count} validated translations available, "
                f"{required_languages} required for {geographic_footprint} trial footprint. "
                f"FDA PRO Guidance (2009) Section IV.A requires linguistic validation for each language "
                f"used. Missing translations require costly prospective validation studies, adding "
                f"6-12 months to trial setup [ISPOR ePRO Task Force Report, 2009]."
            ))

        # HTA Alignment check
        if "NICE" in hta_markets:
            if "eq-5d" not in instrument_name.lower() and "eq5d" not in instrument_name.lower():
                flags.append(ensure_full_stop(
                    "HTA NOTE — NICE MARKET: This instrument alone cannot support QALY-based "
                    "cost-utility analysis required by NICE. EQ-5D-5L must be included alongside "
                    "this instrument in the COA battery for UK market access [NICE DSU TSD 2, 2019]. "
                    "Without EQ-5D, NICE submission will require mapping algorithm — a weaker approach."
                ))
            else:
                flags.append(ensure_full_stop("HTA Alignment (+note): EQ-5D included — supports QALY calculation for NICE cost-utility analysis [NICE DSU TSD 2, 2019]"))

        if "ICER" in hta_markets:
            utility_instruments = ["eq-5d", "eq5d", "sf-6d", "sf-36"]
            if any(u in instrument_name.lower() for u in utility_instruments):
                flags.append(ensure_full_stop("HTA Alignment (+note): Utility-based measure included — supports ICER cost-effectiveness analysis [ICER Value Assessment Framework, 2020]"))

        # ── FINAL SCORE CALCULATION ────────────────────────────────
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
            # Trial and drug provenance
            "drug_name": record.get("drug_name", ""),
            "trial_name": record.get("trial_name", ""),
            "nct_id": record.get("nct_id", ""),
            "phase": record.get("phase", ""),
            "disease_area": record.get("disease_area", ""),
            "patient_population": record.get("patient_population", ""),
            "pro_position": record.get("pro_position", ""),
            # Evidence quality fields
            "key_finding": record.get("key_finding", ""),
            "compliance_rate": record.get("compliance_rate", ""),
            "assessment_schedule": record.get("assessment_schedule", ""),
            "publication_doi": record.get("publication_doi", ""),
            "publication_year": record.get("publication_year", ""),
            "p_value": record.get("p_value", ""),
            "effect_size": record.get("effect_size", ""),
            # Source links for UI
            "fda_label_url": record.get("fda_label_url", ""),
            "ema_label_url": record.get("ema_label_url", ""),
            # Instrument reference fields
            "key_toxicities": record.get("key_toxicities", ""),
            "validation_status": record.get("validation_status", ""),
            "strengths": record.get("strengths", ""),
            "limitations": record.get("limitations", ""),
        })

    # Sort: Risk Level first, then score descending
    risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    results.sort(key=lambda x: (risk_order.get(x["risk_level"], 4), -x["scientific_score"]))

    return results


# ============================================================================
# HELPER FUNCTIONS FOR GRAPH QUERIES
# ============================================================================
def get_instruments_by_indication(indication="", phase="", endpoint=""):
    """Wrapper for Neo4j query"""
    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        conn = Neo4jConnection(neo4j_uri, neo4j_username, neo4j_password)
        try:
            indications = [indication] if indication else [""]
            return conn.get_instruments_by_indication(indications=indications, phase=phase, endpoint=endpoint)
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_instruments_by_indication failed: {e}")
        return []


def get_regulatory_evidence(indication="", agency=""):
    """Wrapper for Neo4j query"""
    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        conn = Neo4jConnection(neo4j_uri, neo4j_username, neo4j_password)
        try:
            indications = [indication] if indication else [""]
            return conn.get_regulatory_evidence(indications=indications, agency=agency)
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_regulatory_evidence failed: {e}")
        return []


def get_instrument_reference(instrument_name=""):
    """Wrapper for Neo4j query"""
    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        conn = Neo4jConnection(neo4j_uri, neo4j_username, neo4j_password)
        try:
            return conn.get_instrument_reference(instrument_name=instrument_name)
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_instrument_reference failed: {e}")
        return []


def get_regulatory_evidence_for_instrument(instrument_name=""):
    """Find regulatory reviews that mentioned a specific instrument by name."""
    try:
        conn = Neo4jConnection(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        try:
            return conn.get_regulatory_evidence_for_instrument(instrument_name=instrument_name)
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"get_regulatory_evidence_for_instrument failed: {e}")
        return []


# ============================================================================
# STEP 3: THE REASONER
# ============================================================================
def ensure_full_stop(text: str) -> str:
    """Ensure text ends with a full stop. If empty, return empty string."""
    if not text or text.strip() == "":
        return ""
    text = text.strip()
    if not text.endswith(('.', '!', '?')):
        return text + "."
    return text


def clean_mcid(raw_mcid: str) -> str:
    """Clean MCID values to standard format: 'X.X points on 0-10 scale'"""
    if not raw_mcid or str(raw_mcid).strip() in ["none", "not established", "unknown", "nan", "n/a", "tbd",
                                                 "not reported", "pending", "null", ""]:
        return ""

    mcid_str = str(raw_mcid).lower().strip()

    # Remove noise and extract numeric value
    mcid_str = re.sub(r'\[.*?\]', '', mcid_str)  # Remove bracketed text
    mcid_str = re.sub(r'\(.*?\)', '', mcid_str)  # Remove parentheses
    mcid_str = re.sub(r'\bpoints?\b', '', mcid_str)  # Remove 'point/points'
    mcid_str = re.sub(r'\bon\b', '', mcid_str)  # Remove 'on'
    mcid_str = re.sub(r'\bscale\b', '', mcid_str)  # Remove 'scale'
    mcid_str = re.sub(r'[^\d.]+', ' ', mcid_str)  # Keep only digits and dots
    mcid_str = mcid_str.strip()

    # Extract first number found
    num_match = re.search(r'\d+\.?\d*', mcid_str)
    if num_match:
        value = num_match.group()
        return f"{value} points on 0-10 scale"
    return ""


def clean_kg_narratives(records: list) -> list:
    """
    Use Claude Haiku to clean messy narrative fields in KG records.
    Processes batches of records to minimise API calls.
    Only cleans fields that contain narrative text: mcid, key_finding,
    regulatory_acceptance, strengths, limitations.
    """
    if not records:
        return records

    # Extract only the fields that need cleaning
    dirty_fields = []
    for i, r in enumerate(records):
        dirty_fields.append({
            "idx": i,
            "instrument_name": r.get("instrument_name", ""),
            "mcid": str(r.get("mcid", ""))[:300],
            "key_finding": str(r.get("key_finding", ""))[:300],
            "regulatory_acceptance": str(r.get("regulatory_acceptance", ""))[:200],
            "strengths": str(r.get("strengths", ""))[:200],
            "limitations": str(r.get("limitations", ""))[:200],
        })

    system = """You are a medical editor cleaning raw database records for a clinical trials tool.
For each record provided, return a JSON array of cleaned records.
Rules:
- Remove raw database IDs like pmc12345678, NCT numbers embedded in text, and arXiv IDs
- Fix obvious typos and grammatical errors
- Convert MCID values to clean numeric format: e.g. "1.33 points on 0-10 scale"
- Make key_finding a single clean sentence in active voice ending with a full stop
- Make regulatory_acceptance a clean 1-sentence summary ending with a full stop
- Make strengths and limitations each a clean 1-sentence summary ending with a full stop
- If a field is empty, null, or only contains noise, return an empty string ""
- Do NOT add information that was not in the original text
- Return ONLY a valid JSON array, no markdown, no explanation"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=system,
            messages=[{
                "role": "user",
                "content": f"Clean these KG records. Return only JSON array:\n{json.dumps(dirty_fields[:10])}"
            }]
        )
        raw = response.content[0].text.strip().replace("```json","").replace("```","").strip()
        cleaned_fields = json.loads(raw)

        # Apply cleaned fields back to original records
        cleaned_map = {cf["idx"]: cf for cf in cleaned_fields}
        result_records = []
        for i, record in enumerate(records):
            r = dict(record)
            if i in cleaned_map:
                cf = cleaned_map[i]
                if cf.get("mcid"): r["mcid"] = cf["mcid"]
                if cf.get("key_finding"): r["key_finding"] = cf["key_finding"]
                if cf.get("regulatory_acceptance"): r["regulatory_acceptance"] = cf["regulatory_acceptance"]
                if cf.get("strengths"): r["strengths"] = cf["strengths"]
                if cf.get("limitations"): r["limitations"] = cf["limitations"]
            result_records.append(r)
        logging.info(f"Cleaned {len(cleaned_fields)} KG records via Haiku")
        return result_records
    except Exception as e:
        logging.warning(f"KG cleaning failed, using raw records: {e}")
        return records  # Graceful fallback — raw records still work


def build_battery(context_json: dict, top_scores: list) -> dict:
    """Build a complete COA battery recommendation based on context and scored instruments"""
    indication = str(context_json.get("indication", "")).lower()
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
    drug_class = str(context_json.get("drug_class", "")).lower()

    # Step 1: Select primary instrument (highest score that meets core domain requirements)
    primary_instrument = None
    for inst in top_scores:
        if inst["risk_level"] != "CRITICAL" and inst["scientific_score"] >= 70:
            primary_instrument = inst
            break

    # Step 2: Check if HTA requirements are met
    hta_battery = []
    if "NICE" in hta_markets:
        # NICE requires EQ-5D for QALY calculation
        eq5d_found = any("eq-5d" in i["instrument_name"].lower() for i in top_scores)
        if not eq5d_found:
            hta_battery.append({
                "instrument_name": "EQ-5D-5L",
                "reason": "NICE HTA requirement — mandatory for UK cost-utility analysis [NICE DSU TSD 2, 2019]",
                "risk_level": "LOW",
                "scientific_score": 95  # High score for regulatory compliance
            })

    # Step 3: Add utility measure if not present (for ICER/EUnetHTA)
    utility_instruments = ["eq-5d", "eq5d", "sf-6d", "sf-36"]
    utility_found = any(any(u in i["instrument_name"].lower() for u in utility_instruments) for i in top_scores)
    if "ICER" in hta_markets and not utility_found:
        hta_battery.append({
            "instrument_name": "SF-6D",
            "reason": "ICER HTA requirement — utility measure for US cost-effectiveness analysis [ICER Value Assessment Framework, 2020]",
            "risk_level": "LOW",
            "scientific_score": 90
        })

    # Step 4: Build final battery
    battery = []
    if primary_instrument:
        battery.append({
            "instrument_name": primary_instrument["instrument_name"],
            "role": "Primary PRO",
            "scientific_score": primary_instrument["scientific_score"],
            "risk_level": primary_instrument["risk_level"],
            "flags": primary_instrument["flags"],
            "reason": "Highest scoring instrument meeting core domain requirements"
        })

    # Add HTA battery instruments
    for inst in hta_battery:
        battery.append({
            "instrument_name": inst["instrument_name"],
            "role": "HTA Compliance",
            "scientific_score": inst["scientific_score"],
            "risk_level": inst["risk_level"],
            "flags": [inst["reason"]],
            "reason": inst["reason"]
        })

    # If no primary instrument found, use broad instrument as fallback
    if not primary_instrument and top_scores:
        broad_instruments = ["eortc qlq-c30", "eortc qlq-my20", "fact-g", "fact-p", "sf-36", "promis", "eq-5d"]
        for inst in top_scores:
            if any(b in inst["instrument_name"].lower() for b in broad_instruments):
                battery.append({
                    "instrument_name": inst["instrument_name"],
                    "role": "Fallback Broad Instrument",
                    "scientific_score": inst["scientific_score"],
                    "risk_level": inst["risk_level"],
                    "flags": inst["flags"],
                    "reason": "Broad instrument covering multiple domains as fallback"
                })
                break

    return {
        "battery": battery,
        "hta_battery": hta_battery,
        "primary_instrument": primary_instrument,
        "hta_compliance": len(hta_battery) > 0,
        "fallback_used": not primary_instrument
    }


def get_recommendation(user_text: str) -> dict:
    
    error_status = None
    # Before the KG query block — initialise all collections
    kg_records = []
    reg_records = []
    inst_refs = []
    reg_rules = []
    
    # --- STEP 3A: Run the Analyzer ---
    context_json = analyze_trial_context(user_text)
    indication = context_json.get("indication", "")
    synonyms = context_json.get("indication_synonyms") or [indication]
    drug_class = context_json.get("drug_class", "")
    phase = context_json.get("phase", "Phase 3")
    agency = ""
    
    # --- STEP 3B: Query the Knowledge Graph ---
    try:
        for search_term in [indication] + synonyms[:2]:
            results = get_instruments_by_indication(
                indication=search_term, 
                phase=phase, 
                endpoint=""
            )
            kg_records.extend(results)
            
        # Search regulatory evidence with primary indication AND all synonyms
        all_search_terms = list(dict.fromkeys([indication] + synonyms[:3]))  # deduplicated
        all_reg_records = []
        for term in all_search_terms:
            term_records = get_regulatory_evidence(indication=term, agency=agency)
            all_reg_records.extend(term_records)
        # Deduplicate by review_id
        seen_ids = set()
        reg_records = []
        for r in all_reg_records:
            rid = r.get("review_id", "") or r.get("drug_name","") + r.get("agency","")
            if rid not in seen_ids:
                seen_ids.add(rid)
                reg_records.append(r)

        # Pass empty lifecycle_stage — actual values are Instrument_Selection, Protocol_Design etc., NOT "label claim"
        reg_rules = get_regulatory_rules(indication=indication, lifecycle_stage="", decision_type="")
        logging.info(f"Regulatory search terms used: {all_search_terms}")
        logging.info(f"KG returned {len(reg_records)} regulatory reviews after synonym expansion")
        kg_records = list({r.get("instrument_name", "Unknown"): r for r in kg_records}.values())
        logging.info(f"KG returned {len(kg_records)} instrument records, {len(reg_records)} regulatory reviews")
        logging.info(f"KG returned {len(reg_rules)} regulatory rules")
    except Exception as e:
        error_status = f"Knowledge Graph offline: {str(e)}"
        logging.error(f"Neo4j query failed: {e}")
        
    # --- STEP 3C: Score the instruments FIRST ---
    scored = score_evidence(context_json, kg_records) if kg_records else []
    top_5 = scored[:5]
    
    # --- STEP 3D: Fetch instrument reference data, clean narratives, get per-instrument regulatory precedent ---
    inst_refs = []
    inst_regulatory_precedents = {}

    if not error_status:
        try:
            for inst in top_5:
                inst_name = inst["instrument_name"]
                ref = get_instrument_reference(instrument_name=inst_name)
                if ref:
                    inst_refs.extend(ref if isinstance(ref, list) else [ref])
                inst_reviews = get_regulatory_evidence_for_instrument(instrument_name=inst_name)
                if inst_reviews:
                    inst_regulatory_precedents[inst_name] = inst_reviews
        except Exception as e:
            logging.error(f"Step 3D failed: {e}")

        # Clean messy KG narrative fields via Haiku
        try:
            if kg_records:
                kg_records = clean_kg_narratives(kg_records)
        except Exception as e:
            logging.warning(f"KG cleaning failed, using raw: {e}")
    except Exception as e:
        logging.error(f"Failed to fetch instrument references: {e}")

# --- STEP 3E: Format KG evidence block ---
kg_block_lines = []
if error_status:
    kg_block_lines.append(f"⚠️ KNOWLEDGE GRAPH OFFLINE — {error_status}")
    kg_block_lines.append("Relying entirely on web search and internal knowledge base.")
else:
    kg_block_lines.append(f"=== SCORED INSTRUMENT RANKING ({len(scored)} instruments evaluated) ===\n")
    for i, inst in enumerate(top_5, 1):
        kg_block_lines.append(
            f"RANK {i}: {inst['instrument_name']} | "
            f"Score: {inst['scientific_score']}/100 | "
            f"Operational bonus: {inst['operational_bonus']:+d} | "
            f"Risk Level: {inst['risk_level']}\n"
            f"  Flags: {' | '.join(inst['flags'][:3])}\n"
            f"  Trial: {inst.get('trial_name','')} ({inst.get('nct_id','')}) | "
            f"Phase: {inst.get('phase','')} | Drug: {inst.get('drug_name','')}\n"
        )
        
    kg_block_lines.append(f"\n=== REGULATORY REVIEWS ({len(reg_records)} records) ===\n")
    for i, rr in enumerate(reg_records[:10], 1):
        kg_block_lines.append(
            f"[RR-{i:03d}] {rr.get('agency','')} | {rr.get('drug_name','')} | "
            f"Decision: {rr.get('decision','')} | "
            f"Accepted: {rr.get('instruments_accepted','')} | "
            f"Claim type: {rr.get('claim_type','')}"
        )
        
    kg_block_lines.append(f"\n=== INSTRUMENT REFERENCE DATA ({len(inst_refs)} records) ===\n")
    for i, ir in enumerate(inst_refs[:10], 1):
        kg_block_lines.append(
            f"[IR-{i:03d}] {ir.get('short_name','')} ({ir.get('full_name','')}) | "
            f"Domains: {ir.get('domains','')} | MCID: {ir.get('mcid','')} | "
            f"Validation: {ir.get('validation','')} | "
            f"Regulatory acceptance: {ir.get('regulatory_acceptance','')}"
        )

    # Rejection reason analysis block
    rejections_with_data = [
        r for r in reg_records
        if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")
    ]
    kg_block_lines.append(
        f"\n=== REJECTION REASON ANALYSIS "
        f"({len(rejections_with_data)} reviews with rejection data) ===\n"
    )
    kg_block_lines.append(
        "These rejection reasons come from actual FDA/EMA medical review documents. "
        "Use them to identify risks in the current recommendation and cite [REJ-XXX] when doing so.\n"
    )
    for i, rr in enumerate(reg_records, 1):
        rejection_primary = rr.get("rejection_reason_primary", "")
        rejection_detailed = rr.get("rejection_reason_detailed", "")
        if rejection_primary or rejection_detailed:
            kg_block_lines.append(
                f"[REJ-{i:03d}] Drug: {rr.get('drug_name', '')} | "
                f"Agency: {rr.get('agency', '')} | "
                f"Decision: {rr.get('decision', '')}\n"
                f"  Primary rejection reasons: {rejection_primary}\n"
                f"  Detailed analysis: "
                f"{str(rejection_detailed)[:600] if rejection_detailed else 'Not recorded'}\n"
                f"  Missing data issues: {rr.get('missing_data_issue', '')}\n"
                f"  Alpha controlled: {rr.get('alpha_controlled', '')}\n"
                f"  Final approved label language: "
                f"{rr.get('label_language', 'Not specified')}\n"
            )

    # Published regulatory rules block
    if reg_rules:
        kg_block_lines.append(
            f"\n=== PUBLISHED REGULATORY RULES ({len(reg_rules)} rules) ===\n"
        )
        kg_block_lines.append(
            "Published FDA/ICH/EMA rules relevant to this indication. "
            "Cite [RULE-XXX] when explaining why a penalty applies or what mitigation is required.\n"
        )
        for i, rule in enumerate(reg_rules, 1):
            kg_block_lines.append(
                f"[RULE-{i:03d}] Source: {rule.get('source_document', '')} | "
                f"Section: {rule.get('section', '')} | "
                f"Stage: {rule.get('lifecycle_stage', '')} | "
                f"Type: {rule.get('decision_type', '')} | "
                f"Stakeholder: {rule.get('stakeholder', '')}\n"
                f"  Rule: {rule.get('rule_text', '')}\n"
                f"  Context: {rule.get('context', '')}\n"
            )
            
    kg_evidence_block = "\n".join(kg_block_lines)

    # --- STEP 3F: HTA context block ---
    hta_block_lines = ["\n=== HTA/PAYER CONTEXT ===\n"]
    for hta_body in context_json.get("hta_markets", []):
        if hta_body in HTA_PREFERENCES:
            h = HTA_PREFERENCES[hta_body]
            hta_block_lines.append(
                f"{hta_body}: Required instruments — {h['required_instruments']} | "
                f"Preferred — {h['preferred_instruments']} | "
                f"Note: {h['notes']} | Ref: {h['reference']}"
            )
    hta_block = "\n".join(hta_block_lines)

    # --- STEP 3G: Build Sonnet system prompt ---
    sonnet_system = f"""You are an Elite Lead COA Strategist supporting oncology drug development at a major pharmaceutical company. Your recommendations must be scientifically rigorous, regulatory-compliant, and fully transparent.

RULE 1 (Assumption Audit): If the 'assumptions_made' array in the context JSON is not empty, begin your response with a bolded section titled '## Strategy Context Audit'. List each assumption explicitly, state what the alternative could have been, and explain how the recommendation would change if the assumption were wrong. 
RULE 2 (Penalty Justification with Specific Citations): For each penalty flag applied to any instrument in the scored list, you must explain: (a) the exact regulatory document and section that triggered it, (b) the clinical consequence if ignored, (c) whether any mitigation strategy exists. 
RULE 3 (Three Implementation Pillars): You must explicitly address all three pillars in a dedicated section:
  - Pillar 1 — Assessment Windows & Recall Bias: Specify the correct recall period for each recommended instrument given the dosing schedule
  - Pillar 2 — Estimands (Phase 3): For each instrument, specify the estimand strategy and how missing post-discontinuation data will be handled per ICH E9(R1)
  - Pillar 3 — eCOA Migration: State whether each instrument has a validated electronic version and whether a mode equivalence study is required
RULE 4 (Technical Risk Warning): If a paper-only instrument is recommended because of its high scientific score, add a bolded section '## Technical Risk Warning' explaining the data integrity risks and FDA eCOA Guidance (2023) implications.
RULE 5 (HTA/Payer Alignment): Include a dedicated section '## HTA and Payer Alignment' that addresses: whether EQ-5D-5L is included for NICE markets, whether a utility-based measure supports cost-effectiveness analysis for ICER, and what the consequence is for market access if these are missing.
RULE 6 (Conflict Resolution): If KG evidence and web search evidence conflict, explicitly state the conflict, then prioritise KG Regulatory Review records (RR-XXX) for decision logic and explain why. Do not silently resolve conflicts.
RULE 7 (Full Citations): Cite all KG claims with their label [TI-XXX], [RR-XXX], or [IR-XXX]. Cite all web search claims with the full URL. Never make an uncited claim.
RULE 8 (Terminology): Use only the exact COA terminology from this glossary: {GLOSSARY_TEXT}
RULE 9 (Geographic Requirements): If the trial footprint is Global or EU, include a section '## Geographic and Linguistic Validation' addressing validated translation availability and mode equivalence requirements per FDA PRO Guidance (2009) Section IV.A.
RULE 10 (Rejection Pattern Analysis): The KG contains actual FDA/EMA rejection reasons from published medical reviews, labelled [REJ-XXX]. For each recommended instrument you must: (a) check whether any [REJ-XXX] record shows it being rejected for a related indication, (b) if yes, quote the rejection reason verbatim and explain whether the same risk applies to the current context, (c) if the same rejection reason pattern appears across multiple [REJ-XXX] records, flag it explicitly as a 'Systematic Regulatory Risk' — this means the agency has a consistent objection that will likely recur. Where [RULE-XXX] records exist, cite the specific rule text when explaining rejections or required mitigations. Include a dedicated output section: '## Rejection Risk Analysis — What Has Failed Before and Why'.

OUTPUT STRUCTURE (follow exactly):
## Strategy Context Audit [only if assumptions were made]
## Recommended COA Battery [ranked instruments with scores, citations, rationale]
## Regulatory Precedent [what FDA/EMA have accepted for this indication — from KG + web]
## HTA and Payer Alignment [EQ-5D for NICE, utility measures for ICER/EUnetHTA]
## Evidence from Literature [web search findings]
## Evidence Gaps [what is missing from both sources]
## Rejection Risk Analysis — What Has Failed Before and Why
## Implementation Notes [Three pillars: Assessment Windows, Estimands, eCOA Migration]
## Geographic and Linguistic Validation [if global/EU trial]
## Technical Risk Warning [if any paper-only instrument recommended]"""

    # --- STEP 3H: Build Sonnet user prompt ---
    sonnet_user = f"""TRIAL CONTEXT (extracted and inferred by Analyzer):
{json.dumps(context_json, indent=2)}

{kg_evidence_block}

{hta_block}

ORIGINAL USER QUERY: {user_text}

{"NOTE: Knowledge Graph is offline. Rely entirely on web search but explicitly state this in your response." if error_status else ""}

Using the scored instrument ranking, regulatory reviews, and instrument reference data above as your primary evidence base, use your web search tool to supplement with:
1. Current FDA/EMA guidance on PRO instruments for {indication or "this oncology indication"} — prioritise fda.gov, ema.europa.eu
2. ClinicalTrials.gov records for recent Phase 3 {indication or "oncology"} trials with PRO endpoints — prioritise clinicaltrials.gov
3. Published validation studies for the top-ranked instruments — prioritise pubmed.ncbi.nlm.nih.gov, nih.gov
4. ISPOR or PROQOLID entries for instrument properties — prioritise ispor.org
5. NICE/ICER/EUnetHTA guidance relevant to the HTA markets identified — prioritise nice.org.uk, icer.org

Synthesise all evidence into a complete COA strategy recommendation following the output structure in your system prompt."""

    # --- STEP 3I: Call Sonnet with Native Web Search Tool ---
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            system=sonnet_system,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": sonnet_user}]
        )
        answer = " ".join(
            block.text for block in response.content 
            if hasattr(block, "text") and block.text
        )
    except Exception as e:
        answer = f"Sonnet API call failed: {str(e)}"
        error_status = error_status or f"Sonnet error: {str(e)}"
        logging.error(f"Sonnet call failed: {e}")

    # --- STEP 3J: Build return dict ---
    result = {
        "answer": answer,
        "context_json": context_json,
        "top_scores": top_5,
        "kg_raw_hits": kg_records,
        "reg_records": reg_records,
        "reg_rules": reg_rules,
        "hta_context": {hta: HTA_PREFERENCES[hta] for hta in context_json.get("hta_markets", []) if hta in HTA_PREFERENCES},
        "error_status": error_status,
        "record_counts": {
            "instrument_records": len(kg_records),
            "regulatory_reviews": len(reg_records),
            "regulatory_rules": len(reg_rules),
            "instrument_refs": len(inst_refs),
            "scored_instruments": len(scored),
            "rejections_found": len([
                r for r in reg_records
                if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")
            ])
        }
    }
    
    # --- STEP 3K: Log for evaluation ---
    log_recommendation(user_text, result)
    
    return result


# ============================================================================
# EVALUATION LOGGING
# ============================================================================
def log_recommendation(user_text: str, result: dict) -> None:
    """Save every query and result to timestamped JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "timestamp": timestamp,
            "user_query": user_text,
            "indication": result.get("context_json", {}).get("indication", "unknown"),
            "phase": result.get("context_json", {}).get("phase", "unknown"),
            "assumptions_made": result.get("context_json", {}).get("assumptions_made", []),
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
            json.dump(log_entry, f, indent=2, default=str)
        logging.info(f"Recommendation logged to {log_path}")
    except Exception as e:
        logging.error(f"Failed to log recommendation: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("Agent module loaded successfully. Import test passed.")
