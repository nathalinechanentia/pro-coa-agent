"""
Architecture
  Step 1: Haiku  — extract trial context from free text
  Step 2: KG     — retrieve instrument + regulatory evidence from Neo4j
  Step 3: Python — score and rank instruments (100-pt scale)
  Step 4: Haiku  — map instrument subscales → canonical domains (cached)
  Step 5: Python — build structured evidence block with pre-numbered citations
  Step 6: Sonnet — synthesise evidence into full strategy + 5 tables
"""

import json
import logging
import os
import re
from collections import defaultdict
from typing import Optional
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from graph import Neo4jConnection

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

Path("logs").mkdir(exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-20250514"

# ── Regulatory citation reference URLs (for system prompt) ────────────────────
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

# ── Module-level caches (persist across requests in same server process) ──────
_subscale_cache: dict = {}   # instrument_name.lower() → {subscale: domain | None}
_lang_cache:     dict = {}   # instrument_name.lower() → {count, citation, warning}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_secret(key: str) -> str:
    """Read from .env (local) or Streamlit secrets (cloud)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, ""))
    except Exception:
        return ""

def _get_conn() -> Neo4jConnection:
    return Neo4jConnection(
        get_secret("NEO4J_URI"),
        get_secret("NEO4J_USERNAME"),
        get_secret("NEO4J_PASSWORD"),
    )


def _norm(s) -> str:
    """Normalise to lowercase stripped string."""
    return re.sub(r"\s+", " ", str(s or "").lower().strip())


def _s(v) -> str:
    """Stringify any value, joining lists with ' | '."""
    if v is None:
        return ""
    if isinstance(v, list):
        return " | ".join(str(x) for x in v if x is not None)
    return str(v).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DOMAIN TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

# Core domains required by FDA for oncology PRO trials.
# HTA Utility is added when NICE or ICER is an HTA market.
CORE_DOMAINS = [
    ("Disease-related Symptoms",          "FDA"),
    ("Symptomatic Adverse Events",        "FDA"),
    ("Overall Side Effect Impact Summary","FDA"),
    ("Physical Function",                 "FDA"),
    ("Role Function",                     "FDA"),
    ("Health Status (EQ‑5D)",              "NICE/ICER"),
]

# Static keyword map: substring → canonical domain.
# Applied before any Haiku call (free, instant).
_SUB2DOM: dict = {
    # Disease-related Symptoms
    "pain":                         "Disease-related Symptoms",
    "bpi":                          "Disease-related Symptoms",
    "bone pain":                    "Disease-related Symptoms",
    "ache":                         "Disease-related Symptoms",
    "analgesic":                    "Disease-related Symptoms",
    "fatigue":                      "Disease-related Symptoms",
    "energy":                       "Disease-related Symptoms",
    "tiredness":                    "Disease-related Symptoms",
    "bfi":                          "Disease-related Symptoms",
    "disease symptoms":             "Disease-related Symptoms",
    "myeloma":                      "Disease-related Symptoms",
    # Symptomatic Adverse Events
    "nausea":                       "Symptomatic Adverse Events",
    "vomiting":                     "Symptomatic Adverse Events",
    "peripheral neuropathy":        "Symptomatic Adverse Events",
    "alopecia":                     "Symptomatic Adverse Events",
    "dyspnea":                      "Symptomatic Adverse Events",
    "dyspnoea":                     "Symptomatic Adverse Events",
    "insomnia":                     "Symptomatic Adverse Events",
    "appetite":                     "Symptomatic Adverse Events",
    "diarrhea":                     "Symptomatic Adverse Events",
    "diarrhoea":                    "Symptomatic Adverse Events",
    "constipation":                 "Symptomatic Adverse Events",
    "sore mouth":                   "Symptomatic Adverse Events",
    "dysphagia":                    "Symptomatic Adverse Events",
    "cough":                        "Symptomatic Adverse Events",
    "hemoptysis":                   "Symptomatic Adverse Events",
    "blurred vision":               "Symptomatic Adverse Events",
    # Overall Side Effect Impact Summary
    "side effect":                  "Overall Side Effect Impact Summary",
    "adverse":                      "Overall Side Effect Impact Summary",
    "tolerab":                      "Overall Side Effect Impact Summary",
    "financial":                    "Overall Side Effect Impact Summary",
    "overall":                      "Overall Side Effect Impact Summary",
    "bother":                       "Overall Side Effect Impact Summary",
    # Physical Function
    "physical functioning":         "Physical Function",
    "physical well-being":          "Physical Function",
    "physical function":            "Physical Function",
    "physical impact":              "Physical Function",
    "mobility":                     "Physical Function",
    "usual activities":             "Physical Function",
    "self-care":                    "Physical Function",
    "functional well-being":        "Physical Function",
    "activities of daily living":   "Physical Function",
    "adl":                          "Physical Function",
    # Role Function
    "role functioning":             "Role Function",
    "role function":                "Role Function",
    "emotional functioning":        "Role Function",
    "cognitive functioning":        "Role Function",
    "social functioning":           "Role Function",
    "social/family":                "Role Function",
    "emotional well-being":         "Role Function",
    # Health Status
    "health utility":               "Health Status (EQ‑5D)",
    "eq-vas":                       "Health Status (EQ‑5D)",
    "eq vas":                       "Health Status (EQ‑5D)",
    "utility index":                "Health Status (EQ‑5D)",
    "anxiety/depression":           "Health Status (EQ‑5D)",
}

# Domain-specific keywords for change detection parsing
_DOMAIN_KWS: dict = {
    "Disease-related Symptoms":   ["pain", "bpi", "ache", "analgesic", "fatigue", "energy", "tiredness", "bfi", "disease"],
    "Symptomatic Adverse Events": ["nausea", "vomit", "neuropathy", "dyspnea", "insomnia",
                                   "appetite", "diarrhea", "constipation", "alopecia",
                                   "side effect", "adverse", "tolerab", "hemoptysis", "cough"],
    "Overall Side Effect Impact Summary": ["side effect", "adverse", "tolerab", "bother", "overall", "financial"],
    "Physical Function":          ["physical", "function", "mobility", "activit", "adl"],
    "Role Function":              ["role", "social", "emotional", "cognitive", "work", "daily"],
    "Health Status (EQ‑5D)":     ["utility", "eq-5d", "eq5d", "health utility", "qaly"],
}


def _static_subscale_to_domain(subscale: str) -> Optional[str]:
    sn = _norm(subscale)
    for kw, dom in _SUB2DOM.items():
        if kw in sn:
            return dom
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HAIKU HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_trial_context(text: str, api_key: str = "") -> dict:
    """
    Extract structured trial parameters from free text using Haiku.
    Returns dict with all fields app.py expects from context_json.
    """
    if not api_key:
        api_key = get_secret("ANTHROPIC_API_KEY") 
    client = Anthropic(api_key=api_key)

    default = {
        "indication":       "unknown",
        "phase":            "unknown",
        "drug_class":       "unknown",
        "population":       "general oncology",
        "population_subtype": "unknown",
        "hta_markets":      [],
        "geography":        [],
        "tpp_domains":      [],
        "administration":   "IV",
        "assumptions_made": [],
        "moa_aliases": [],
    }

    try:
        resp = client.messages.create(
            model=HAIKU,
            max_tokens=700,
            system = (
                "Extract clinical trial parameters from the input. Return ONLY valid JSON. "
                "Keys: indication (string), phase (e.g.'Phase 3', 'Phase 2', 'Phase 1/2', 'Phase 1'), "  
                "drug_class (e.g.'Proteasome Inhibitor'), "
                "population (e.g.'relapsed/refractory'), "
                "population_subtype (e.g.'Symptomatic' or 'Asymptomatic'), "
                "hta_markets (list, include 'FDA'/'EMA' if the user mentions label submissions to those agencies; "
                "include HTA body names such as 'NICE','ICER','EUnetHTA','SMC','CADTH','PBAC','HAS','G-BA','IQWiG' if mentioned; "
                "if the user refers to 'cost-effectiveness', 'QALY', 'reimbursement', 'market access', "
                "'European HTA', or 'Joint Clinical Assessment', add 'HTA' to the list), "
                "geography (list of regions, e.g.['US','EU','Asia-Pacific']), "
                "tpp_domains (list of symptom/function domains stated in TPP), "
                "administration (e.g.'IV weekly','oral daily'), "
                "moa_aliases (list of all known synonyms, generic drug names, and standard abbreviations "
                "for the extracted drug_class – e.g. for Proteasome Inhibitor: [bortezomib, carfilzomib, ixazomib, PI, 26S proteasome]), "  
                "assumptions_made (list ONLY the fields you had to INFER because they were NOT in the text, "
                "and that COULD change the strategy: indication, phase, drug_class, population_subtype, "
                "hta_markets, geography, or tpp_domains). "
                "Return ONLY a single, valid JSON object enclosed in curly braces. Do NOT wrap it in markdown fences. Do NOT add any text before or after the JSON object."
            ),
            messages=[{"role": "user", "content": text[:3500]}],
        )

        raw = resp.content[0].text.strip()
        logging.info(f"Haiku raw response: {raw[:500]}")

        # ── Robust JSON extraction (handles arrays, strings, nested objects) ──
        def _find_json_block(s: str) -> str:
            """Return the longest valid JSON substring starting at the first '{'."""
            # Remove leading text and any markdown fencing
            s = re.sub(r'^```(?:json)?\s*', '', s)
            s = re.sub(r'\s*```$', '', s)
            start = s.find('{')
            if start == -1:
                raise ValueError("No JSON object found in Haiku response")
            brace = 0
            bracket = 0
            in_string = False
            escape = False
            for i in range(start, len(s)):
                ch = s[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"' and not in_string:
                    in_string = True
                elif ch == '"' and in_string:
                    in_string = False
                if in_string:
                    continue
                if ch == '{': brace += 1
                elif ch == '}': brace -= 1
                elif ch == '[': bracket += 1
                elif ch == ']': bracket -= 1
                if brace == 0 and bracket == 0:
                    return s[start:i+1]
            raise ValueError("Unbalanced JSON in Haiku response")

        json_str = _find_json_block(raw)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Last resort: strip all non‑JSON characters
            cleaned = re.sub(r'^[^{\[]+', '', raw)
            cleaned = re.sub(r'[^}\]]+$', '', cleaned)
            parsed = json.loads(cleaned)

        # Ensure all default keys exist
        for k, v in default.items():
            parsed.setdefault(k, v)
        return parsed

    except Exception as e:
        logging.warning(f"analyze_trial_context failed: {e}")
        # ── String‑based fallback for critical fields ──
        fallback = dict(default)
        text_lower = text.lower()
        # Indication
        for kw, ind in {
            "multiple myeloma": "Multiple Myeloma", "myeloma": "Multiple Myeloma",
            "nsclc": "NSCLC", "non-small cell lung": "NSCLC",
            "prostate cancer": "Prostate Cancer", "mcrpc": "Prostate Cancer",
            "breast cancer": "Breast Cancer", "lung cancer": "Lung Cancer",
            "lymphoma": "Lymphoma", "myelofibrosis": "Myelofibrosis",
            "leukemia": "Leukemia", "glioblastoma": "Glioblastoma",
            "ovarian cancer": "Ovarian Cancer", "melanoma": "Melanoma",
        }.items():
            if kw in text_lower:
                fallback["indication"] = ind
                break
        # Phase
        ph = re.search(r"phase\s+([123])", text_lower)
        if ph:
            fallback["phase"] = f"Phase {ph.group(1)}"
        # also catch "Phase 1/2"
        if "phase 1/2" in text_lower:
            fallback["phase"] = "Phase 1/2"
        return fallback


def map_subscales_to_domains(instrument_name: str, subscale_text: str,
                             extra_domains: list = None,
                             api_key: str = "") -> dict:
    """
    Map pipe-delimited subscale names → canonical CORE_DOMAINS.
    Uses static keywords first; calls Haiku only for unmatched subscales.
    Cached per instrument_name (lowercase). Returns {subscale: domain | None}.
    """
    if not api_key:
        api_key = get_secret("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    ck = _norm(instrument_name)
    if ck in _subscale_cache:
        return _subscale_cache[ck]

    subs = [
        s.strip() for s in _s(subscale_text).split("|")
        if s.strip() and _norm(s) not in (
            "not reported", "not reportedspecified", "not reported/not specified", ""
        )
    ]
    if not subs:
        _subscale_cache[ck] = {}
        return {}

    result: dict = {}
    unmapped: list = []
    for s in subs:
        d = _static_subscale_to_domain(s)
        if d:
            result[s] = d
        else:
            unmapped.append(s)

    if unmapped:
        domain_list = [d for d, _ in CORE_DOMAINS] + (extra_domains if extra_domains else [])
        try:
            resp = client.messages.create(
                model=HAIKU,
                max_tokens=400,
                system=(
                    "Clinical outcome assessment expert. "
                    f"Map each subscale name to the closest domain from: {domain_list}. "
                    "If none fit, use null. "
                    'Return ONLY JSON: {"subscale_name": "domain_or_null"}.'
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Instrument: {instrument_name}\n"
                        f"Subscales to map: {json.dumps(unmapped)}"
                    ),
                }],
            )
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$", "",
                resp.content[0].text.strip(),
                flags=re.MULTILINE,
            )
            mapped = json.loads(raw)
            for s in unmapped:
                result[s] = mapped.get(s)  # None if no match
        except Exception as e:
            logging.warning(f"map_subscales_to_domains({instrument_name}): {e}")
            for s in unmapped:
                result[s] = None

    _subscale_cache[ck] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: KG RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

_IND_SYNONYMS: dict = {
    "multiple myeloma":      ["multiple myeloma", "myeloma", "MM", "RRMM", "NDMM"],
    "nsclc":                 ["NSCLC", "non-small cell lung cancer", "lung cancer"],
    "non-small cell":        ["NSCLC", "non-small cell lung cancer", "lung cancer"],
    "prostate":              ["prostate cancer", "mCRPC", "CRPC", "castration-resistant prostate"],
    "breast cancer":         ["breast cancer", "HER2", "metastatic breast"],
    "lymphoma":              ["lymphoma", "DLBCL", "follicular lymphoma", "NHL"],
    "myelofibrosis":         ["myelofibrosis", "MF", "primary myelofibrosis"],
    "cll":                   ["CLL", "chronic lymphocytic leukemia"],
    "aml":                   ["AML", "acute myeloid leukemia"],
    "gvhd":                  ["gvhd", "graft-versus-host disease", "chronic gvhd"],
}

def _get_synonyms(indication: str) -> list:
    ind = _norm(indication)
    for key, syns in _IND_SYNONYMS.items():
        if key in ind:
            return syns
    return [indication]


def get_kg_data(context: dict) -> tuple:
    """
    Query Neo4j for trial-instrument records, regulatory evidence, and rules.
    Returns (raw_records, reg_records, rules) — all lists of dicts.
    """
    indications = _get_synonyms(context.get("indication", "unknown"))
    phase       = context.get("phase", "")
    indication  = context.get("indication", "")

    conn = _get_conn()
    raw, reg, rules = [], [], []
    try:
        raw = conn.get_instruments_by_indication(indications=indications, phase="", endpoint="") or []
        reg   = conn.get_regulatory_evidence(indications=indications) or []
        # Fetch all regulatory rules, then keep only those relevant to strategy decisions
        all_rules = conn.get_regulatory_rules(indication="", lifecycle_stage="", decision_type="") or []
        STRATEGY_STAGES = {"Instrument_Selection", "Protocol_Design", "Concept_Selection"}
        rules = [r for r in all_rules if r.get("lifecycle_stage", "") in STRATEGY_STAGES]
        logging.info(
            f"KG: {len(raw)} instrument records | "
            f"{len(reg)} regulatory reviews | {len(rules)} rules"
        )
    except Exception as e:
        logging.error(f"KG retrieval failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return raw, reg, rules


def get_instruments_by_indication(indication: str = "", phase: str = "") -> list:
    """
    Public wrapper for app.py Tier 1/2 context building.
    Returns raw KG records for the given indication.
    """
    conn = _get_conn()
    try:
        syns = _get_synonyms(indication) if indication else [indication]
        return conn.get_instruments_by_indication(indications=syns, phase=phase) or []
    except Exception as e:
        logging.warning(f"get_instruments_by_indication({indication}): {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def score_instruments(raw_records: list, context: dict) -> list:
    """
    Score and rank unique instruments on a 100-point evidence scale.
    Returns sorted list (highest score first).
    Each entry includes raw_positive_score, operational_bonus, final_adjusted_score
    for the app.py scoring display panel.
    """
    by_inst: dict = defaultdict(list)
    for r in raw_records:
        name = _s(r.get("instrument_name") or "").strip()
        if name and _norm(name) not in ("not reported", "unknown", ""):
            by_inst[name].append(r)

    _role_pts  = {
        "primary": 20, "co-primary": 20,
        "key secondary": 16, "secondary": 13, "exploratory": 5,
    }
    _sig_pts = {
        "significant (p<0.001)": 20,
        "significant (p<0.05)":  15,
        "trend":                  7,
    }

    scored = []
    for name, records in by_inst.items():
        r0 = records[0]  # instrument-level fields

        # ── Prevalence (0–25) ──
        try:
            prev = min(int(float(_s(r0.get("trial_prevalence") or 0))), 100)
        except (ValueError, TypeError):
            prev = 0
        base_score = round(prev * 0.25)

        # ── Endpoint role (0–20) ──
        role_score, best_role = 0, "NR"
        for r in records:
            rv = _norm(_s(r.get("endpoint_role") or r.get("pro_position") or ""))
            for rk, rp in _role_pts.items():
                if rk in rv and rp > role_score:
                    role_score, best_role = rp, rv.title()

        # ── Significance (0–20) ──
        sig_score = 0
        for r in records:
            sv = _norm(_s(r.get("significance") or ""))
            for sk, sp in _sig_pts.items():
                if sk in sv and sp > sig_score:
                    sig_score = sp

        # ── Validation evidence (0–15) ──
        val_evidence = _s(r0.get("validation_evidence") or "")
        val_status   = _norm(_s(r0.get("validation_status") or ""))
        if val_evidence and val_evidence not in ("NR", "nan", "None", ""):
            val_score = 15
        elif "validated" in val_status:
            val_score = 10
        else:
            val_score = 0

        # ── Regulatory acceptance (0–15) ──
        reg_text = _norm(_s(r0.get("regulatory_acceptance") or ""))
        if any(kw in reg_text for kw in ["fda", "ema", "accepted", "approved", "label claim"]):
            reg_score = 15
        elif any(kw in reg_text for kw in ["reviewed", "supportive", "considered"]):
            reg_score = 8
        else:
            reg_score = 0

        # ── MCID (0–10) ──
        mcid_raw = _s(r0.get("mcid") or "")
        mid_met  = any(
            _norm(_s(r.get("mid_met") or "")) in ("yes", "met")
            for r in records
        )
        if mid_met:
            mcid_score = 10
        elif mcid_raw and mcid_raw not in ("NR", "nan", "None", ""):
            mcid_score = 5
        else:
            mcid_score = 0

        # ── FDA alignment (0–5) ──
        fda_raw = _norm(_s(r0.get("fda_alignment") or ""))
        if "fully" in fda_raw or "core" in fda_raw:
            fda_score = 5
        elif "partial" in fda_raw:
            fda_score = 3
        else:
            fda_score = 0

        raw_positive = base_score + role_score + sig_score + val_score + reg_score + mcid_score + fda_score

        # ── Operational flags ──
        flags: list = []
        mode_raw = _norm(_s(r0.get("mode_options") or ""))
        if "ecoa" in mode_raw or "electronic" in mode_raw:
            flags.append("eCOA ready +8")

        lang_raw  = _s(r0.get("languages") or "")
        if lang_raw and lang_raw not in ("NR", "nan", "None", ""):
            n_langs = len([l for l in lang_raw.split("|") if l.strip() and len(l.strip()) < 60])
            if n_langs >= 20:
                flags.append("Broad language coverage (≥20 languages) +5")
            elif n_langs > 0:
                flags.append(f"Limited translation ({n_langs} languages) (-5 operational)")
            else:
                flags.append("No translation data (-10 operational)")
        else:
            flags.append("No translation data (-10 operational)")

        op_bonus = 0
        if any("eCOA ready" in f for f in flags):
            op_bonus += 8
        if any("Broad language" in f for f in flags):
            op_bonus += 5
        if any("Limited translation" in f for f in flags):
            op_bonus -= 5
        if any("No translation data" in f for f in flags):
            op_bonus -= 10

        final_adj = min(max(0, raw_positive + op_bonus), 100)

        # ── Risk level ──
        if val_score == 0:
            risk = "CRITICAL"
            flags.append("⚠️ No validation evidence in KG — verify before use")
        elif final_adj >= 70:
            risk = "LOW"
        elif final_adj >= 45:
            risk = "MEDIUM"
        elif final_adj >= 25:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        # ── Best record (highest significance) for display ──
        best = max(
            records,
            key=lambda r: _sig_pts.get(_norm(_s(r.get("significance") or "")), 0),
        )

        scored.append({
            # Core fields (used by app.py display)
            "instrument_name":       name,
            "scientific_score":      final_adj,
            "raw_positive_score":    raw_positive,
            "operational_bonus":     op_bonus,
            "final_adjusted_score":  final_adj,   # final score after operational adjustment
            "risk_level":            risk,
            "flags":                 flags,
            "endpoint_role":         best_role,
            # Instrument metadata (used in evidence building + Tier 2)
            "best_trial":            _s(best.get("trial_name") or ""),
            "best_drug":             _s(best.get("drug_name") or ""),
            "nct_id":                _s(best.get("nct_id") or ""),
            "publication_doi":       _s(best.get("publication_doi") or ""), 
            "mcid":                  mcid_raw,
            "regulatory_acceptance": _s(r0.get("regulatory_acceptance") or ""),
            "validation_evidence":   val_evidence,
            "validation_status":     _s(r0.get("validation_status") or ""),
            "domains_measured":      _s(r0.get("domains_measured") or ""),
            "total_items":           r0.get("total_items"),
            "developer":             _s(r0.get("developer") or ""),
            "fda_url":               _s(best.get("fda_label_url") or ""),
            "ema_url":               _s(best.get("ema_label_url") or ""),
            "trial_count":           len(records),
            "records":               records,
        })

    scored.sort(key=lambda x: x["scientific_score"], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FIELD EXTRACTORS (pure Python, no API)
# ═══════════════════════════════════════════════════════════════════════════════

_FORMAL_CLAIM_RE = re.compile(
    r"\[Formal Claim\][^:]*:\s*\"([^\"]{10,300})\"", re.IGNORECASE
)
_NULL_RESULT_RE  = re.compile(r"\[Null Result\]", re.IGNORECASE)
_HR_PVAL_RE      = re.compile(
    r"(?:HR|hazard ratio)\s*[=:]?\s*([\d.]+)[^;,\n]{0,40}"
    r"(?:;|,)?\s*p\s*[<>=]?\s*([\d.]+)",
    re.IGNORECASE,
)
_PVAL_RE = re.compile(r"p\s*[<>=]\s*[\d.]+", re.IGNORECASE)


def extract_sap_language(pro_label_text: str, instrument_name: str = "") -> str:
    """
    Extract ≤10-word SAP endpoint language from [Formal Claim] tags
    in pro_label_language_final.  Returns '—' if nothing found.
    """
    if not pro_label_text:
        return "—"
    text = _s(pro_label_text)

    # Filter to segments mentioning this instrument (if name provided)
    if instrument_name:
        parts = [
            p.strip() for p in text.split("|")
            if "[Formal Claim]" in p
            or _norm(instrument_name) in _norm(p)
        ]
        text = " | ".join(parts) if parts else text

    m = _FORMAL_CLAIM_RE.search(text)
    if m:
        claim = m.group(1).strip()
        # Prefer endpoint-type phrases
        for pat in [
            r"(time to (?:deterioration|progression)[^\.;,]{0,50})",
            r"(proportion of patients[^\.;,]{0,50})",
            r"(significant(?:ly)? (?:delayed|improved|reduced|prolonged)[^\.;,]{0,50})",
        ]:
            mm = re.search(pat, claim, re.IGNORECASE)
            if mm:
                words = mm.group(1).split()
                return " ".join(words[:10]) + ("..." if len(words) > 10 else "")
        words = claim.split()
        return " ".join(words[:10]) + ("..." if len(words) > 10 else "")

    if _NULL_RESULT_RE.search(text):
        return "No significant endpoint — Null Result"

    return "—"


def extract_key_finding_short(key_finding_text: str) -> str:
    """
    Condense key_finding to HR + p-value only (≤15 words).
    Returns '—' for empty/not-reported values.
    """
    kf = _s(key_finding_text)
    if _norm(kf) in ("key findings not reported.", "not reported", ""):
        return "—"

    m = _HR_PVAL_RE.search(kf)
    if m:
        # Take a ~15-word window around the HR/p-value
        start = max(0, m.start() - 25)
        snippet = kf[start: m.end() + 20]
        words = snippet.split()
        return " ".join(words[:15]) + ("..." if len(words) > 15 else "")

    m2 = _PVAL_RE.search(kf)
    if m2:
        words = kf[: m2.end() + 10].split()
        return " ".join(words[:15]) + ("..." if len(words) > 15 else "")

    words = kf.split()
    return " ".join(words[:15]) + ("..." if len(words) > 15 else "")


# def infer_domain_change(
#     subscale_sig: str,
#     subscale_not_sig: str,
#     key_finding: str,
#     domain: str,
# ) -> str:
#     """
#     Determine whether a PRO domain showed change in a trial record.
#     Returns 'Yes', 'No', or 'NR'.
#     Priority: subscale_significant > subscale_not_significant > key_finding parsing.
#     """
#     kws = _DOMAIN_KWS.get(domain, [])
#     sig_text     = _norm(_s(subscale_sig     or ""))
#     not_sig_text = _norm(_s(subscale_not_sig or ""))
#     kf_text      = _norm(_s(key_finding      or ""))

#     for kw in kws:
#         if kw in sig_text:
#             return "Yes"

#     for kw in kws:
#         if kw in not_sig_text:
#             return "No"

#     dom_in_kf = any(kw in kf_text for kw in kws)
#     if dom_in_kf:
#         has_sig = bool(re.search(
#             r"(significant|p<0\.\d+|p=0\.0\d+|hr\s*[\d.]|delayed|improved|reduced|prolonged)",
#             kf_text,
#         ))
#         has_no  = bool(re.search(
#             r"no significant|not significant|no difference|no stat|null result",
#             kf_text,
#         ))
#         if has_sig and not has_no:
#             return "Yes"
#         if has_no:
#             return "No"

#     return "NR"

def infer_domain_change(
    subscale_sig: str,
    subscale_not_sig: str,
    key_finding: str,
    domain: str,
) -> str:
    """
    Returns 'Yes', 'No', or 'NR' by examining the pipe‑separated subscale
    significance lists, then falling back to key_finding text.
    """
    kws = _DOMAIN_KWS.get(domain, [domain.lower()])
    sig_list = [s.strip().lower() for s in _s(subscale_sig).split("|") if s.strip()]
    not_sig_list = [s.strip().lower() for s in _s(subscale_not_sig).split("|") if s.strip()]

    for kw in kws:
        if any(kw in sub for sub in sig_list):
            return "Yes"
        if any(kw in sub for sub in not_sig_list):
            return "No"

    # Fallback to key_finding text
    kf_text = _norm(_s(key_finding or ""))
    for kw in kws:
        if kw in kf_text:
            if re.search(r"(significant|p<0\.\d+|p=0\.0\d+|hr\s*[\d.]|delayed|improved|reduced|prolonged)", kf_text):
                return "Yes"
            if re.search(r"no significant|not significant|no difference|no stat|null result", kf_text):
                return "No"
            break
    return "NR"

def _web_search(query: str, instruction: str) -> Optional[dict]:
    """Run a Haiku web search and return parsed JSON, or None on failure."""
    client = Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

    try:
        resp = client.messages.create(
            model=HAIKU,
            max_tokens=1000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=(
                "You are a regulatory intelligence assistant. "
                "Search the web for the answer. "
                "Return ONLY valid JSON as instructed. No markdown."
            ),
            messages=[{
                "role": "user",
                "content": f"Search: {query}\n\nInstruction: {instruction}",
            }],
        )
        text = " ".join(
            b.text for b in resp.content if hasattr(b, "text") and b.text
        )
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        return json.loads(text)
    except Exception as e:
        logging.debug(f"_web_search({query[:40]}): {e}")
        return None


# def get_language_counts_top5(scored: list) -> dict:
#     """
#     Get language counts for the top 5 scored instruments only.
#     P1: KG languages field (already in scored records — free).
#     P2: Web search (Haiku) for instruments not in KG (capped to top 5).
#     All results cached in _lang_cache.
#     """
#     result: dict = {}
#     for inst in scored[:5]:
#         name = inst["instrument_name"]
#         ck   = _norm(name)

#         if ck in _lang_cache:
#             result[name] = _lang_cache[ck]
#             continue

#         # P1: languages field from the best KG record
#         lang_raw = ""
#         for r in inst.get("records", []):
#             lv = _s(r.get("languages") or "")
#             if lv and lv not in ("NR", "nan", "None", ""):
#                 lang_raw = lv
#                 break

#         if lang_raw:
#             langs = [l.strip() for l in lang_raw.split("|") if l.strip() and len(l.strip()) < 60]
#             entry = {
#                 "count":     len(langs) or None,
#                 "citation":  f"[KG: Instrument.languages — {name}]",
#                 "source_url": "",
#                 "warning":   None,
#             }
#             result[name] = entry
#             _lang_cache[ck] = entry
#             continue

#         # P2: Web search (capped — only for top 5 instruments not in KG)
#         ws = _web_search(
#             query=f"{name} PRO instrument validated language translations count",
#             instruction=(
#                 f"How many validated language translations exist for the PRO instrument "
#                 f"'{name}'? Return JSON: "
#                 '{"count": integer_or_null, "source_url": "string"}'
#             ),
#         )
#         if ws and ws.get("count") is not None:
#             entry = {
#                 "count":     ws["count"],
#                 "citation":  f"[Web: {ws.get('source_url', 'PROQOLID')}]",
#                 "source_url": ws.get("source_url", ""),
#                 "warning":   None,
#             }
#         else:
#             entry = {
#                 "count":     None,
#                 "citation":  "[Verify at PROQOLID or developer website]",
#                 "source_url": "",
#                 "warning":   f"⚠️ Translation data not confirmed for {name}",
#             }
#         result[name] = entry
#         _lang_cache[ck] = entry

#     return result

# ── Helper to fetch label language for a specific trial drug + instrument ──
def get_label_language_for_trial(drug_name: str, instrument_name: str,
                                 reg_records: list) -> str:
    """
    Return the pro_label_language_final text from a regulatory review that
    matches drug_name AND mentions instrument_name.
    Returns empty string if nothing found.
    """
    if not drug_name or not instrument_name or not reg_records:
        return ""
    drug_lower = _norm(drug_name)
    inst_lower = _norm(instrument_name)
    candidates = []
    for rr in reg_records:
        rr_drug = _norm(rr.get("drug_name", ""))
        if drug_lower in rr_drug or rr_drug in drug_lower:
            text = _s(rr.get("pro_label_language_final", ""))
            if text and inst_lower in text.lower():
                candidates.append(text)
    if not candidates:
        return ""
    # Prefer the first segment that contains a [Formal Claim]
    for c in candidates:
        if "[Formal Claim]" in c:
            return c[:1500]
    return candidates[0][:1500]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: DOMAIN COVERAGE MATRIX (Table 1 pre-computation)
# ═══════════════════════════════════════════════════════════════════════════════
def build_domain_coverage(
    scored: list,
    raw_records: list,
    context: dict,
    reg_records: list = None,
    user_text: str = "",
    api_key: str = "",         
) -> dict:
    """Pre-compute the domain coverage matrix for Table 1."""
    hta_markets = [_norm(m) for m in context.get("hta_markets", [])]

    hta_trigger_terms = {
        "nice", "icer", "eunethta", "smc", "cadth", "pbac",
        "has", "g-ba", "iqwig", "aifa", "jca", "eu hta",
        "ema", "cost-effectiveness", "qaly", "reimbursement",
        "market access", "european hta", "health technology assessment"
    }
    search_text = " ".join(hta_markets).lower() + " " + (user_text or "").lower()
    include_hta = any(term in search_text for term in hta_trigger_terms)

    top_inst_names = {s["instrument_name"] for s in scored[:8]}

    target_drug_class = _norm(context.get("drug_class", ""))
    moa_aliases = [_norm(a) for a in context.get("moa_aliases", []) if a]
    all_class_terms = [target_drug_class] + moa_aliases

    # If no aliases, use individual significant words from drug class
    if not moa_aliases:
        words = {w for w in re.split(r'[\s\-/]+', target_drug_class) if len(w) > 3}
        moa_aliases = list(words)
        all_class_terms = [target_drug_class] + moa_aliases

    # 1. Group all records by trial name
    trial_map: dict = {}
    for r in raw_records:
        tn = _s(r.get("trial_name") or "")
        if not tn:
            continue
        if tn not in trial_map:
            trial_map[tn] = {
                "trial_name": tn,
                "nct_id": _s(r.get("nct_id") or ""),
                "drug_name": _s(r.get("drug_name") or ""),
                "drug_class": _s(r.get("drug_class_name") or ""),
                "phase": _s(r.get("phase") or ""),
                "year": _s(r.get("publication_year") or "NR"),
                "instruments": set(),
                "has_item_lib": False,
                "_recs": [],
            }
        tr = trial_map[tn]
        tr["instruments"].add(_s(r.get("instrument_name") or ""))
        tr["has_item_lib"] = tr["has_item_lib"] or (
            r.get("instrument_subscales_assessed") and not r.get("total_items")
        )
        tr["_recs"].append(r)

    # 2. Score each trial by drug‑class relevance
    def trial_relevance_key(trial: dict) -> tuple:
        tc = _norm(trial["drug_class"])
        if any(term in tc or tc in term for term in all_class_terms if term):
            class_score = 0
        else:
            req_words = set(target_drug_class.split()) if target_drug_class else set()
            trial_words = set(tc.split())
            if req_words & trial_words:
                class_score = 1
            else:
                class_score = 2
        # Phase priority: lower is better
        phase_str = _norm(trial.get("phase", ""))
        if "phase 3" in phase_str:
            phase_score = 0
        elif "phase 2" in phase_str or "phase 1/2" in phase_str:
            phase_score = 1
        else:
            phase_score = 2
        return (class_score, phase_score)

    sorted_trials = sorted(trial_map.values(), key=trial_relevance_key)

    # 3. Take top 5 unique trials
    comp_names = []
    seen_names = set()
    for tr in sorted_trials:
        if tr["trial_name"] not in seen_names:
            seen_names.add(tr["trial_name"])
            comp_names.append(tr)
            if len(comp_names) >= 5:
                break

    # 4. Build comparator_trials list
    comparator_trials = []
    for i, tr in enumerate(comp_names, 1):
        comparator_trials.append({
            "trial_name": tr["trial_name"],
            "label": f"CT-{i:03d}",
            "nct_id": tr["nct_id"],
            "drug_name": tr["drug_name"],
            "drug_class": tr["drug_class"],
            "phase": tr["phase"],
            "year": tr["year"],
            "instruments": [{"name": nm} for nm in sorted(tr["instruments"]) if nm],
            "has_item_library": tr["has_item_lib"],
        })

    # 5. Index records by trial name
    by_trial: dict = defaultdict(list)
    for r in raw_records:
        tn = _s(r.get("trial_name") or "")
        if tn:
            by_trial[tn].append(r)

    # ── Extra TPP domains not already in CORE_DOMAINS ─────────────────
    existing_domains_list = [d for d, _ in CORE_DOMAINS]
    extra_domains = [
        d for d in context.get("tpp_domains", [])
        if d.lower() not in {ed.lower() for ed in existing_domains_list}
    ]

    # 6. Subscale maps for comparator‑trial instruments AND top‑8 instruments
    names_to_map = top_inst_names | {
        _s(r.get("instrument_name") or "")
        for r in raw_records
        if _s(r.get("trial_name") or "") in {tr["trial_name"] for tr in comp_names}
        and r.get("instrument_name")
    }
    for nm in names_to_map:
        if not nm:
            continue
        sub = next(
            (
                _s(r.get("instrument_subscales_assessed") or "")
                for r in raw_records
                if _s(r.get("instrument_name") or "") == nm
                and r.get("instrument_subscales_assessed")
            ),
            "",
        )
        if sub:
            map_subscales_to_domains(nm, sub, extra_domains, api_key=api_key)

    # 7. Build domain rows (core domains)
    item_library_note = (
        "Comparator trials used subscale or item-library approaches rather than full instruments. "
        "Consider whether a calibrated PRO Schedule of Assessments is appropriate."
        if any(ct["has_item_library"] for ct in comparator_trials)
        else None
    )

    domain_rows = []
    for domain, stakeholder in CORE_DOMAINS:
        if domain == "Health Status (EQ‑5D)" and not include_hta:
            continue

        is_fda_core = stakeholder == "FDA"
        candidates = []
        for inst in scored[:8]:
            nm  = inst["instrument_name"]
            sm  = _subscale_cache.get(_norm(nm), {})
            covers = any(v == domain for v in sm.values())
            if not covers:
                dm = _norm(_s(inst.get("domains_measured") or ""))
                covers = _norm(domain.split()[0]) in dm
            if covers:
                change = "NR"
                for r in inst.get("records", []):
                    c = infer_domain_change(
                        _s(r.get("subscale_significant") or ""),
                        _s(r.get("subscale_not_significant") or ""),
                        _s(r.get("key_finding") or ""),
                        domain,
                    )
                    if c == "Yes":
                        change = "Yes"
                        break
                    elif c == "No" and change == "NR":
                        change = "No"
                candidates.append({
                    "instrument":      nm,
                    "score":           inst["scientific_score"],
                    "change_detected": change,
                })

        trial_cells = []
        for ct in comparator_trials:
            tname  = ct["trial_name"]
            t_recs = by_trial.get(tname, [])
            cell   = None
            for r in t_recs:
                nm  = _s(r.get("instrument_name") or "")
                sm  = _subscale_cache.get(_norm(nm), {})
                covers = any(v == domain for v in sm.values())
                if not covers:
                    dm = _norm(_s(r.get("domains_measured") or ""))
                    covers = _norm(domain.split()[0]) in dm
                if covers:
                    change = infer_domain_change(
                        _s(r.get("subscale_significant") or ""),
                        _s(r.get("subscale_not_significant") or ""),
                        _s(r.get("key_finding") or ""),
                        domain,
                    )
                    cell = {
                        "instrument": nm,
                        "change":     change,
                        "label":      ct["label"],
                    }
                    break
            trial_cells.append({
                "trial_name":  tname,
                "trial_label": ct["label"],
                "cell":        cell,
            })

        domain_rows.append({
            "domain":           domain,
            "stakeholder":      stakeholder,
            "is_fda_core":      is_fda_core,
            "candidates":       candidates,
            "trials":           trial_cells,
            "item_library_note": item_library_note,
        })

    # 8. Extra TPP domains (only those not already mapped to a core domain)
    for domain in extra_domains:
        # Check static mapping and subscale cache
        dom_lower = domain.lower()
        mapped_core = None
        for static_key, core_dom in _SUB2DOM.items():
            if static_key in dom_lower or dom_lower in static_key:
                mapped_core = core_dom
                break
        if not mapped_core:
            for cache_entry in _subscale_cache.values():
                for sub, cd in cache_entry.items():
                    if cd and dom_lower in sub.lower():
                        mapped_core = cd
                        break
                if mapped_core:
                    break
        if mapped_core:
            continue  # already covered by core

        is_fda_core = True
        candidates = []
        for inst in scored[:8]:
            nm = inst["instrument_name"]
            sm = _subscale_cache.get(_norm(nm), {})
            covers = any(v == domain for v in sm.values())
            if not covers:
                dm = _norm(_s(inst.get("domains_measured") or ""))
                covers = _norm(domain.split()[0]) in dm
            if covers:
                change = "NR"
                for r in inst.get("records", []):
                    c = infer_domain_change(
                        _s(r.get("subscale_significant") or ""),
                        _s(r.get("subscale_not_significant") or ""),
                        _s(r.get("key_finding") or ""),
                        domain,
                    )
                    if c == "Yes":
                        change = "Yes"
                        break
                    elif c == "No" and change == "NR":
                        change = "No"
                candidates.append({
                    "instrument":      nm,
                    "score":           inst["scientific_score"],
                    "change_detected": change,
                })

        trial_cells = []
        for ct in comparator_trials:
            tname  = ct["trial_name"]
            t_recs = by_trial.get(tname, [])
            cell   = None
            for r in t_recs:
                nm = _s(r.get("instrument_name") or "")
                sm = _subscale_cache.get(_norm(nm), {})
                covers = any(v == domain for v in sm.values())
                if not covers:
                    dm = _norm(_s(r.get("domains_measured") or ""))
                    covers = _norm(domain.split()[0]) in dm
                if covers:
                    change = infer_domain_change(
                        _s(r.get("subscale_significant") or ""),
                        _s(r.get("subscale_not_significant") or ""),
                        _s(r.get("key_finding") or ""),
                        domain,
                    )
                    cell = {
                        "instrument": nm,
                        "change":     change,
                        "label":      ct["label"],
                    }
                    break
            trial_cells.append({
                "trial_name":  tname,
                "trial_label": ct["label"],
                "cell":        cell,
            })

        domain_rows.append({
            "domain":           domain,
            "stakeholder":      "FDA / TPP",
            "is_fda_core":      is_fda_core,
            "candidates":       candidates,
            "trials":           trial_cells,
            "item_library_note": item_library_note,
        })

    return {
        "domains":           domain_rows,
        "comparator_trials": comparator_trials,
        "item_library_applicable": any(ct["has_item_library"] for ct in comparator_trials),
        "hta_mandatory":     [
            {
                "instrument": "EQ-5D-5L",
                "market":     "NICE/ICER",
                "reason":     "Required for health utility assessment and cost-effectiveness modelling",
            }
        ] if include_hta else [],
    }

def _get_trial_assessment_schedules(trial_name: str, raw_records: list) -> str:
    """Return compact per-instrument assessment schedules for a given trial."""
    schedules = []
    for r in raw_records:
        tn = _s(r.get("trial_name") or "")
        if tn == trial_name:
            sched = _s(r.get("assessment_schedule") or "")
            inst = _s(r.get("instrument_name") or "")
            if sched and sched.lower() not in ("nr", "not reported", ""):
                schedules.append(f"{inst}: {sched}")
    if not schedules:
        return "NR"
    return " | ".join(schedules[:4])  

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: EVIDENCE BLOCK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_evidence_block(
    scored:      list,
    raw_records: list,
    reg_records: list,
    rules:       list,
    coverage:    dict,
    context:     dict,
) -> tuple:
    """
    Build the evidence text block for Sonnet + the citation_index dict.

    Returns (evidence_text: str, citation_index: dict, source_block: str).

    citation_index keys: TI-001, CT-001, RR-001, RULE-001
    Each value: {type, summary, links: [{label, url}], ...}

    source_block: pre-numbered source list injected into the Sonnet system prompt.
    This is the key fix for self-referential citations — Sonnet sees exactly which
    sources exist and must use only those labels.
    """
    lines:          list = []
    citation_index: dict = {}

    # ── 1. Trial context ──────────────────────────────────────────────────────
    lines.append("=== TRIAL CONTEXT ===")
    lines.append(f"Indication:     {context.get('indication','unknown')}")
    lines.append(f"Phase:          {context.get('phase','unknown')}")
    lines.append(f"Drug class:     {context.get('drug_class','unknown')}")
    lines.append(f"Population:     {context.get('population','unknown')}")
    lines.append(f"Administration: {context.get('administration','IV')}")
    lines.append(f"HTA markets:    {', '.join(context.get('hta_markets', [])) or 'Not specified'}")
    lines.append(f"Geography:      {', '.join(context.get('geography', [])) or 'Global'}")
    lines.append(f"TPP domains:    {', '.join(context.get('tpp_domains', [])) or 'Inferred from context'}")
    lines.append("")

    # ── 2. Regulatory rules ───────────────────────────────────────────────────
    if rules:
        lines.append("=== REGULATORY RULES ===")
        for i, r in enumerate(rules[:12], 1):
            label     = f"RULE-{i:03d}"
            rule_text = _s(r.get("rule_text") or "")[:300]
            src       = _s(r.get("source_document") or "")
            url       = _s(r.get("source_url") or "")
            lines.append(f"[{label}] {src}: {rule_text}")
            citation_index[label] = {
                "type":    "rule",
                "source":  src,
                "text":    rule_text[:100],
                "links":   [{"label": src, "url": url}] if url else [],
                "summary": f"{src}: {rule_text[:80]}",
            }
        lines.append("")

    # ── 3. Comparator trials (with label language) ──────
    comp_trials = coverage.get("comparator_trials", [])
    if comp_trials:
        lines.append("=== COMPARATOR TRIALS ===")
        for ct in comp_trials:
            label   = ct["label"]
            nct     = ct.get("nct_id", "")
            ct_url  = f"https://clinicaltrials.gov/study/{nct}" if nct else ""
            inst_names = ", ".join(i["name"] for i in ct.get("instruments", []) if i.get("name"))
            lines.append(
                f"[{label}] {ct['trial_name']} | {ct['drug_name']} ({ct['drug_class']}) | "
                f"{ct['phase']} | Year: {ct['year']} | NCT: {nct or 'NR'} | "
                f"Instruments: {inst_names or 'NR'}"
            )

            recs = [r for r in raw_records if _s(r.get("trial_name") or "") == ct["trial_name"]]

            # Collect all unique DOIs for this trial
            doi_set = set()
            for r in recs:
                raw_doi = _s(r.get("publication_doi") or "")
                for m in re.finditer(r'(10\.\d{4,}/[^\s]+)', raw_doi):
                    d = m.group(1).rstrip('.,;:')
                    doi_set.add(d)

            # Build links
            ct_links = [{"label": "ClinicalTrials.gov", "url": ct_url}] if ct_url else []
            for d in sorted(doi_set)[:5]:   # up to 5 DOIs
                ct_links.append({
                    "label": "Publication DOI",
                    "url": f"https://doi.org/{d}"
                })
                
            citation_index[label] = {
                "type":    "comparator_trial",
                "trial":   ct["trial_name"],
                "drug":    ct["drug_name"],
                "phase":   ct["phase"],
                "nct":     nct,                "links":   ct_links,
                "summary": f"{ct['trial_name']} ({ct['drug_name']}, {ct['phase']})",
            }

            sched_str = _get_trial_assessment_schedules(ct["trial_name"], raw_records)
            lines.append(f"  Assessment schedules: {sched_str}")

            for inst in ct.get("instruments", []):
                ll = get_label_language_for_trial(
                    ct["drug_name"], inst["name"], reg_records
                )
                if ll:
                    # truncate to avoid token blowout
                    lines.append(
                        f"  Label language ({inst['name']}): {ll[:600]}"
                    )
        lines.append("")

    # ── 4. Instrument evidence (top 8) ────────────────────
    lines.append("=== INSTRUMENT EVIDENCE (top 8 scored) ===")
    for i, inst in enumerate(scored[:8], 1):
        nm      = inst["instrument_name"]
        label   = f"TI-{i:03d}"
        records = inst.get("records", [])
        nct     = inst.get("nct_id", "")
        fda_url = inst.get("fda_url", "")
        ema_url = inst.get("ema_url", "")

        links = []
        if nct:
            links.append({"label": "ClinicalTrials.gov",
                          "url": f"https://clinicaltrials.gov/study/{nct}"})
        if fda_url:
            links.append({"label": "FDA label", "url": fda_url})
        if ema_url:
            links.append({"label": "EMA label", "url": ema_url})
        
        doi_raw = inst.get("publication_doi", "")
        for m in re.finditer(r'(10\.\d{4,}/[^\s]+)', doi_raw):
            d = m.group(1).rstrip('.,;:')
            links.append({
                "label": "Publication DOI",
                "url": f"https://doi.org/{d}"
            })   

        citation_index[label] = {
            "type":       "trial_instrument",
            "instrument": nm,
            "trial":      inst.get("best_trial", ""),
            "drug":       inst.get("best_drug", ""),
            "nct":        nct,
            "links":      links,
            "summary":    f"{nm} in {inst.get('best_trial','')} ({inst.get('best_drug','')})",
        }

        lines.append(
            f"[{label}] {nm} | Role: {inst.get('endpoint_role','NR')} | "
            f"n_trials: {inst.get('trial_count',0)}"
        )
        lines.append(f"  domains_measured:      {inst.get('domains_measured','NR')}")
        # lines.append(f"  validation_evidence:   {inst.get('validation_evidence','NR')[:200]}")
        # lines.append(f"  validation_status:     {inst.get('validation_status','NR')}")
        # lines.append(f"  mcid:                  {inst.get('mcid','NR')}")
        lines.append(f"  regulatory_acceptance: {inst.get('regulatory_acceptance','NR')[:200]}")
        # lines.append(f"  flags:                 {'; '.join(inst.get('flags', [])) or 'None'}")

        # Per-trial data (limit to 3 most relevant trials)
        for r in records[:3]:
            tname     = _s(r.get("trial_name") or "NR")
            drug      = _s(r.get("drug_name") or "NR")
            role      = _s(r.get("endpoint_role") or r.get("pro_position") or "NR")
            prespec   = _s(r.get("prespecified") or "NR")
            sig       = _s(r.get("significance") or "NR")
            pval      = _s(r.get("p_value") or "NR")
            sched     = _s(r.get("assessment_schedule") or "NR")
            subscales = _s(r.get("instrument_subscales_assessed") or "NR")
            kf_raw    = _s(r.get("key_finding") or "")
            items     = r.get("total_items")
            # --- SAP extraction: try to get label language from reg_records ---
            drug_for_label = _s(r.get("drug_name") or "")
            pro_label = get_label_language_for_trial(drug_for_label, nm, reg_records)
            sap_lang  = extract_sap_language(pro_label, nm)
            kf_short  = extract_key_finding_short(kf_raw)
            year = _s(r.get("publication_year") or "NR")

            lines.append(
                f"  --- Trial: {tname} ({drug}) | Role: {role} | "
                f"Prespec: {prespec} | Sig: {sig} | p: {pval}"
            )
            lines.append(f"      Assessment schedule:     {sched}")
            lines.append(f"      Subscales (trial-used):  {subscales[:250]}")
            lines.append(f"      Endpoint Language:   {sap_lang}")
            lines.append(f"      Key Finding (condensed): {kf_short}")
            lines.append(f"      Total items:             {items or 'NR'}")
            lines.append(f"      Publication year:          {year}")

        lines.append("")

    # ── 5. Regulatory reviews ─────────────────────────────────────────────────
    if reg_records:
        lines.append("=== REGULATORY REVIEWS ===")
        for i, r in enumerate(reg_records[:15], 1):
            rl      = f"RR-{i:03d}"
            drug    = _s(r.get("drug_name") or "")
            agency  = _s(r.get("agency") or "")
            dec     = _s(r.get("decision") or "")
            acc     = _s(r.get("instruments_accepted") or "")
            rej     = _s(r.get("rejection_reason_primary") or "")
            fda_url = _s(r.get("fda_label_url") or "")
            ema_url = _s(r.get("ema_label_url") or "")
            rl_links = []
            if fda_url:
                rl_links.append({"label": "FDA label", "url": fda_url})
            if ema_url:
                rl_links.append({"label": "EMA label", "url": ema_url})
            lines.append(f"[{rl}] {agency} | {drug} | {dec}")
            if acc:
                lines.append(f"  Accepted: {acc}")
            if rej:
                lines.append(f"  Rejection: {rej[:200]}")
            citation_index[rl] = {
                "type":     "reg_review",
                "agency":   agency,
                "drug":     drug,
                "decision": dec,
                "links":    rl_links,
                "summary":  f"{agency} {dec} — {drug} (accepted: {acc[:50]})",
            }
        lines.append("")

    # ── 6. Domain coverage matrix ─────────────────────────────────────────────
    lines.append("=== DOMAIN COVERAGE MATRIX (for Table 1) ===")
    for row in coverage.get("domains", []):
        dom   = row["domain"]
        sh    = row["stakeholder"]
        cands = ", ".join(c["instrument"] for c in row["candidates"]) or "No candidates"
        lines.append(f"Domain: {dom} | Stakeholder: {sh} | Candidates: {cands}")
        for tc in row["trials"]:
            tl   = tc["trial_label"]
            tn   = tc["trial_name"]
            cell = tc["cell"]
            if cell:
                lines.append(
                    f"  [{tl}] {tn}: {cell['instrument']} — Change: {cell['change']}"
                )
            else:
                lines.append(f"  [{tl}] {tn}: Not collected — no KG record")
    lines.append("")

    # ── 7. Language data ──────────────────────────────────────────────────────
    # if lang_counts:
    #     lines.append("=== LANGUAGE TRANSLATION DATA (Table 5) ===")
    #     for nm, lc in lang_counts.items():
    #         cnt     = lc.get("count")
    #         cit     = lc.get("citation", "")
    #         warn    = lc.get("warning", "")
    #         cnt_str = str(cnt) if cnt is not None else "Verify via PROQOLID"
    #         lines.append(f"{nm}: {cnt_str} validated translations | {cit}")
    #         if warn:
    #             lines.append(f"  ⚠️ {warn}")
    #     lines.append("")

    evidence_text = "\n".join(lines)

    ordered = (
        sorted(k for k in citation_index if k.startswith("RULE"))
        + sorted(k for k in citation_index if k.startswith("CT"))
        + sorted(k for k in citation_index if k.startswith("TI"))
        + sorted(k for k in citation_index if k.startswith("RR"))
    )
    src_lines = [
        f"  {lbl}: {citation_index[lbl]['summary']}"
        for lbl in ordered
    ]
    source_block = (
        "AVAILABLE KG SOURCES — use ONLY these labels for citations "
        "(e.g. [TI-001], [CT-002], [RULE-001], [RR-003]):\n"
        + "\n".join(src_lines)
    )

    return evidence_text, citation_index, source_block

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def get_recommendation(user_text: str, tables: set = None,
                       anthropic_api_key: str = "") -> dict:
    """
    Main entry point for Tier 3 strategy generation.

    tables: optional set of table names to build. None = all.
      Values: 'table1', 'table2', 'table3', 'table4', 'table5', 'sonnet'
      Example: get_recommendation(text, tables={'table1'}) — skip Sonnet call.

    Returns result dict consumed by app.py render_strategy_result.
    """
    if not anthropic_api_key:
        anthropic_api_key = get_secret("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=anthropic_api_key)

    _build = tables if tables is not None else {
        "table1", "table2", "table3", "table4", "table5", "sonnet"
    }

    # ── Initialise defaults ───────────────────────────────────────────────────
    error_status     = None
    answer           = ""
    raw_records      = []
    reg_records      = []
    rules            = []
    scored           = []
    coverage         = {
        "domains": [], "comparator_trials": [], "hta_mandatory": [],
    }
    kg_evidence_block = ""
    citation_index   = {}

    # ── Step 1: Extract trial context ─────────────────────────────────────────
    context_json = analyze_trial_context(user_text)
    indication   = context_json.get("indication", "unknown")
    logging.info(
        f"Context: {indication} | {context_json.get('phase')} | "
        f"{context_json.get('drug_class')}"
    )

    # ── Step 2: KG retrieval ──────────────────────────────────────────────────
    try:
        raw_records, reg_records, rules = get_kg_data(context_json)
        if not raw_records:
            error_status = "KG returned no records for this indication"
            logging.warning(error_status)
    except Exception as e:
        error_status = f"KG offline: {e}"
        logging.error(error_status)

    # ── Step 3: Score instruments ─────────────────────────────────────────────
    if raw_records:
        try:
            scored = score_instruments(raw_records, context_json)
            logging.info(
                f"Scored {len(scored)} instruments. Top: "
                f"{scored[0]['instrument_name']} ({scored[0]['scientific_score']}/100)"
                if scored else "No instruments scored"
            )
        except Exception as e:
            logging.error(f"Scoring failed: {e}")
            scored = []

    # ── Step 4: Domain coverage + subscale mapping (Table 1) ─────────────────
    if "table1" in _build and scored and not error_status:
        try:
            coverage = build_domain_coverage(scored, raw_records, context_json,
                                 reg_records, user_text,
                                 api_key=anthropic_api_key)
            logging.info(
                f"Coverage: {len(coverage['domains'])} domains | "
                f"{len(coverage['comparator_trials'])} comparator trials"
            )
        except Exception as e:
            logging.error(f"build_domain_coverage failed: {e}")

    # # ── Step 5: Language counts (top 5 only) ─────────────────────────────────
    # if "table5" in _build and scored:
    #     try:
    #         lang_counts = get_language_counts_top5(scored)
    #     except Exception as e:
    #         logging.warning(f"Language counts failed: {e}")

    # ── Step 6: Build evidence block ─────────────────────────────────────────
    try:
        kg_evidence_block, citation_index, source_block = build_evidence_block(
            scored, raw_records, reg_records, rules,
            coverage, context_json,
        )
    except Exception as e:
        logging.error(f"build_evidence_block failed: {e}")
        kg_evidence_block = f"Evidence build failed: {e}"
        source_block      = ""
        citation_index    = {}

    # ── Step 7: Sonnet synthesis ──────────────────────────────────────────────
    if "sonnet" in _build:
        hta_str = ", ".join(context_json.get("hta_markets", [])) or "None specified"
        geo_str = ", ".join(context_json.get("geography",   [])) or "Global"
        kg_url_refs = "\n".join(
            f"  - {name}: {url}" for name, url in REGULATORY_CITATIONS.items()
        )

        system_prompt = f"""You are a clinical outcome assessment (COA) expert generating a structured PRO measurement strategy.

{source_block}

═══════════════════════════════════════════════════════
CITATION RULES — MANDATORY
═══════════════════════════════════════════════════════

KG SOURCES
- Cite KG sources using their exact label: [TI-001], [CT-002], [RULE-001], [RR-003].
- The labels in the AVAILABLE KG SOURCES block are the complete list. Do not generate new KG labels.
- NEVER invent citation numbers like [7] referencing a table you wrote. Tables are not sources.

WEB‑SOURCED CITATIONS — VERIFICATION PROTOCOL
- Every web‑sourced fact must include a short, verbatim quote from the retrieved page that directly supports the claim.
- Format: "[exact quote](full‑URL)".
- The quote must be at least 5 words and copied exactly – no paraphrasing.
- After the quote you may add a brief summary, but the quote itself is mandatory.
- If the search result does not contain a passage that clearly states the asserted fact, discard that source and search again (counts toward your limit).
- If no valid source is found, write "No published evidence for this population — Need to verify" without any hyperlink.
- Never combine a statistic from your training data with a URL from a different study.
- If you cannot provide a full, clickable URL for a fact, do NOT create any citation marker.
  Simply write the fact as plain text. A missing citation is better than a dead or unverifiable one.

═══════════════════════════════════════════════════════
REGULATORY RULES — MANDATORY USE
═══════════════════════════════════════════════════════
- The evidence block contains a "REGULATORY RULES" section with [RULE-XXX] entries.
- You MUST cite at least one relevant [RULE-XXX] when discussing:
    * Pre‑specification and alpha control
    * Endpoint hierarchy and multiplicity
    * Instrument selection rationale
    * Missing data handling or estimand strategy
- When making a recommendation (e.g., "we suggest using EORTC QLQ‑C30"), ground it in both the comparator evidence [CT-XXX] AND the applicable regulatory rule [RULE-XXX].  
  Example: "Pre‑specifying pain as a key secondary endpoint is consistent with [RULE-002] and the precedent set in ASPIRE [CT-003]."
- If no RULE entries are present, note this in the "Key challenge" section and refer to the FDA PRO Guidance (2009) directly.

Regulatory reference URLs (for web citations if needed):
{kg_url_refs}

HTA CONTEXT: {hta_str}
GEOGRAPHY:   {geo_str}

─────────────────────────────────────────
OUTPUT — produce ALL sections below, in this exact order:

TABLE FORMATTING — MANDATORY FOR ALL TABLES
- Every table must have each row on its own line.  
- The header row, separator row (e.g., |---|---|), and every data row must each be on a separate line.  
- Never put multiple table rows on a single line, no matter how long the cells are.  
- After each data row, press Enter (newline).  
- Verify: every `|---|` separator line must be immediately below the header, and each data row below that.

## COA Measurement Strategy — {indication} {context_json.get('phase','')}

**In one sentence:** [single sentence summary — cite one KG source]
**Key challenge:** [the single biggest regulatory risk for this trial, citing at least one relevant [RULE-XXX] and explaining why it matters for this indication/drug class. If the KG contains no rules, state that and cite FDA PRO Guidance 2009 via web search.]
**Recommended starting point:** [2–3 instruments + reason grounded in both comparator usage [CT-XXX] and regulatory rules [RULE-XXX]. Explain why this combination satisfies the regulatory expectations for this trial design.]
**Critical gap:** [specific gap with citation]

## Table 1: Domain Coverage Comparison

The DOMAIN COVERAGE MATRIX in the evidence block defines the EXACT rows and cells for this table.  
You must **not** add, remove, rename, or reorder rows.  
Copy the matrix rows verbatim.

Columns: Concept | Key Stakeholder | Current Trial Candidates | [CT‑001 trial] | [CT‑002 trial] | ... (one column per comparator trial)

- **Concept** and **Key Stakeholder**: taken directly from the matrix row.
- **Current Trial Candidates**: the list of candidate instruments from the matrix row, separated by commas.  
  If the matrix shows “No candidates”, write “No candidates in KG — consult COA expert”.
- **Comparator trial cells**:  
   - If the matrix contains a pre‑computed cell for that trial and domain, render it exactly.  
     The cell shows the instrument name and a change status:  
     `Change: Y`  →  `✅ INSTRUMENT_NAME — Change: Y [CT‑XXX]`  
     `Change: No` →  `✅ INSTRUMENT_NAME — Change: N (NS) [CT‑XXX]`  
     `Change: NR` →  `✅ INSTRUMENT_NAME — Change: NR [CT‑XXX]`  
   - If the matrix has NO cell for that trial/domain, write: `❌ Not collected¹`
- Add the footnote exactly as below immediately after the table:
¹ Not collected in this trial — no KG record.  
² Y = statistically significant improvement detected; N (NS) = no statistically significant difference; NR = instrument collected but change result not reported in available sources.

## Table 2: PRO Measures Comparison

Columns: Trial | Year | Drug | Drug Class | PRO Measures (n items) | Assessment Schedule | Total Items

For each comparator trial row:
- "Trial": trial name and [CT-XXX] label.
- "Year": from the trial's Year field in the evidence block.
- "Drug" and "Drug class": from the evidence block.
- "PRO Measures (n items)": list each instrument with item count from "Total items:" field in INSTRUMENT EVIDENCE blocks. If not available, write "n=?".
- "Assessment Schedule": 
    * Use the "Assessment schedules:" line from the comparator trial's evidence block. That line is a pipe‑separated list of strings like `EORTC QLQ‑C30: Baseline; C1D1; C4D1; …`.
    * Rewrite into a single short, human‑readable sentence that captures the key timing pattern for each instrument. Preserve all timepoints but remove redundant phrasing.  
      Examples:  
      `EORTC QLQ‑C30: Baseline; C1D1; C4D1; C7D1; C10D1; then every 6 cycles until EOT`  
      `BPI‑SF: Screening; Q4W D1; EOT | FACT‑P: C1D1; C3D1; C5D1; C7D1; then every 3 cycles until EOT`
    * If the line shows "NR", write "NR — not reported in source [CT-XXX]". Do not invent a schedule.
- "Total items": sum of item counts; if any count is missing, mark "~" (e.g. "~64").

For the **Current Trial (Proposed)** row (always last):
- "Trial": "Current Trial (Proposed)"
- "Year": "—"
- "Drug": "Novel [drug class]"
- "Drug class": from the trial context
- "PRO Measures (n items)": list candidate instruments from Table 1 "Current Trial Candidates" column, with item counts from INSTRUMENT EVIDENCE (write "n=?" if missing)
- "Assessment Schedule": "TBD — expert decision required"
- "Total items": calculate from candidate instruments' counts; if missing, write "TBD"

## Table 3: Instrument Gap Analysis

Select up to 4 instruments:
- Primary: instruments used in >=2 comparator trials whose drug class matches the current trial.
- Secondary: instruments from the DOMAIN COVERAGE MATRIX covering the most core FDA domains.
- If fewer than 4 instruments are selected, add the most frequently used instrument(s) from the DOMAIN COVERAGE MATRIX not already selected.

Columns: Instrument | Content Validity | Psychometric Properties | MCID Evidence | Regulatory Acceptance | Known Gaps / Risks | Fit for Purpose

═══════════════════════════════════════════════════════
EVIDENCE RULES (applied to every cell)
═══════════════════════════════════════════════════════

1) **Sources & search rules**
    - Regulatory Acceptance: use the KG field `regulatory_acceptance` + [RR-XXX]/[REJ-XXX] citations.
    - For Content Validity, Psychometrics, and MCID: **web search is mandatory**. The evidence block contains no usable values for these fields.  
      Search specifically for the indication and population stated in the TRIAL CONTEXT.  
      Max 2 web searches per instrument; max 8 total.

2) **Web citation format — EXTRACT‑THEN‑CITE (hallucination‑proof)**
   - Open the web search result and locate a sentence or statistic that directly contains the instrument name AND the target population AND a numerical result (ICC, α, p‑value, n, etc.) or a specific validation statement.
   - Copy that EXACT sentence or phrase from the search result. Paste it verbatim into the cell. Do NOT summarise, paraphrase, or shorten it into your own words.
   - Immediately after the quoted text, add the markdown hyperlink: `[Author Year](full‑URL)`.
   - Your cell must follow this pattern exactly:
     `"exact phrase from the page that includes the statistic" [Author Year](full‑URL)`
   - The "exact phrase" must contain at least ONE of: a Cronbach's α, an ICC, a p‑value, a response rate, a sample size, or a specific validation term ("test‑retest", "convergent validity", "known‑groups", "content validity", "MCID", "MID").
   - If the search result does NOT contain a verifiable phrase with the instrument name and the target population, discard that source and search again.
   - If two searches fail to return a usable result, write "No published evidence for this population — Need to verify" with NO hyperlink.
   - **NEVER** combine a statistics from your training data with a URL from an unrelated study.
   - **NEVER** write a generic phrase in quotes ("Content validity established in RRMM patients") and attach a URL – the quoted text must be demonstrably FROM that URL.

3) **Handling named studies from the KG**
   - If the KG mentions a study (e.g., "Cocks 2007") without a URL, web‑search for the DOI and cite it as `[Cocks 2007](https://doi.org/...)`. If the search fails, write the study name in plain text without brackets.

4) **Missing evidence**
   - Write "No published evidence for this population — Need to verify" where no evidence is found. Never invent data.

5) **Post‑writing verification**
   - After writing each cell, silently read the quoted phrase and the linked URL. Ask: "Does this exact phrase appear in the search snippet for that URL?" If the answer is no, delete the hyperlink and write "No published evidence — Need to verify".

═══════════════════════════════════════════════════════
COLUMN CONTENT
═══════════════════════════════════════════════════════

- **Content Validity** – one verbatim quote from the source containing the instrument name, population, and a validity finding. Follow the EXTRACT‑THEN‑CITE rule above.  
  Example: "The EORTC QLQ-C30 and nine items from the EORTC QLQ-MY20 demonstrated content validity for use in smoldering multiple myeloma clinical trials" [Author Year](URL).

- **Psychometric Properties** – one verbatim quote from the source containing a specific statistic (ICC, α, etc.). Follow the EXTRACT‑THEN‑CITE rule above.  
  Example: "Cronbach's α >0.70 for all but one scale (cognitive functioning) and good item convergence (96%) and discrimination (78%) rates were confirmed for the QLQ-C30 and QLQ-MY20 in multiple myeloma patients" [Author Year](URL).

- **MCID Evidence** – one verbatim quote containing the MCID threshold, anchor type, and population. Follow the EXTRACT‑THEN‑CITE rule above.  
  Example: "The mean MCID value for the EQ-5D-5L was 0.075 for multiple myeloma patients based on anchor-based and distribution-based methods" [Author Year](URL).

- **Known Gaps / Risks** – one sentence grounded in the evidence you gathered.  
  This may include: missing population‑specific MCID, lack of content validity in the target population, absence of published psychometrics, translation limitations, or any other evidence gap.  
  If you mention translation limitations, you MUST use the validated‑translation count from the pre‑performed language searches.  
  If the pre‑search found ≥15 languages, do NOT cite translation as a gap.  
  Only flag translation as a gap if key languages for the trial footprint are missing.

- **Fit for Purpose** – synthesise all evidence into:
  ✅ Likely fit – content validity + MCID + psychometrics available.
  ⚠️ Conditionally fit – one key gap.
  ❓ Evidence gaps – multiple missing pieces.
  Format: `[verdict] — [one‑sentence justification citing sources]. KG scope: curated sample only.

## Table 4: Endpoint Positioning

One row per instrument × trial combination from the INSTRUMENT EVIDENCE blocks.
Columns: Instrument | Role | Prespec. | Endpoint Language | Subscales Used | Key Finding | Sig. | p/Effect | Trial (Drug) | Year

**Reminder for Table 4:**  
For every row where the evidence block shows "SAP Endpoint Language: —", you MUST extract a phrase from the Key Finding and write it in the table. Leave the cell as "—" only if the Key Finding is completely empty or contains no endpoint mention at all.

For each row you must populate the critical columns using the sources provided in the evidence block:

═══════════════════════════════════════════════════════
SOURCE RULES (applied strictly per row)
═══════════════════════════════════════════════════════

1) "Endpoint Language"
   - The "Endpoint Language" cell must contain a **short endpoint concept phrase**, NEVER a full sentence or a result summary.  
     Acceptable formats:  
       "Time to deterioration in [domain]"  
       "Time to pain progression"  
       "Pain palliation proportion (≥30% reduction in BPI‑SF)"  
       "Change from baseline in [domain]"  
       "Proportion of patients with ≥50% reduction in [scale]"  
   - Derive the concept from the richest source for this instrument and trial:
     * Primary source: [Formal Claim] tags in the Label language that mention this instrument.
     * Fallback: the "Key Finding (condensed)" text in the evidence block.
   - Extract the endpoint concept, not the numerical result.  
     **Examples**:  
       "Time to deterioration in Global Health Status and pain symptoms"  
       "Change from baseline in EQ‑VAS"  
       "Change from baseline in pain, fatigue, and physical functioning"
   - If the evidence block shows “SAP Endpoint Language: —” for this trial, derive the concept from the Key Finding line as above.  
     Only write “—” if no endpoint concept can be extracted.
   - Always append the [CT‑XXX] citation at the end of the cell.
   - **NEVER** output a long phrase like “Improved Global Health Status scores compared with Rd [[2]]”. That is a result, not an endpoint concept.

2) "Key Finding"
   - Extract the key statistical results from the richest source that belongs to THIS instrument in THIS trial:
     * Prefer the [Formal Claim] in the label language that names this instrument.
     * If no such claim exists, use the "Key Finding (condensed):" or the original key_finding text from the per‑trial block.
   - Format compactly: HR/OR, 95% CI, p‑value, and endpoint name if multiple endpoints are present. Example:
     "HR 0.79 (95% CI 0.67–0.93), p=0.005 (pain interference); HR 0.82 (95% CI 0.67–1.00), p=0.049 (mean pain intensity)"
   - NEVER truncate with "…". If the condensed text is cut off, restore the full statistical details from the original key_finding field. Use up to 300 characters to include complete results.

3) "Subscales Used"
   - Open the evidence block and locate the line that starts with “Subscales (trial‑used):” FOR THIS EXACT TRIAL AND INSTRUMENT. Copy that line verbatim into the cell. Do not use the instrument’s full subscale list from memory. If you cannot find that line, write “NR”.  
   - **Violation check**: Before writing Table 4, verify that every row’s subscales cell matches the corresponding trial‑used line. If a single row does not, rewrite it before proceeding.

4) Other columns
   - "Instrument", "Role", "Prespec.", "Sig.", "p/Effect": from the same per‑trial block.
   - "Trial (Drug)": trial name (drug name) with [CT‑XXX] label.
   - "Year": from the "Publication year:" line in the INSTRUMENT EVIDENCE per‑trial block. If "NR", write "—".

5) Sorting
   - Primary endpoints first, then Secondary, then Exploratory.
   - Within each role, sort by the instrument's score (from the INSTRUMENT SCORES block).

═══════════════════════════════════════════════════════
EXAMPLES (do NOT copy these into the table)
═══════════════════════════════════════════════════════

* If the label language for EORTC QLQ‑C30 contains:
  [Formal Claim] QLQ‑C30: "Time to deterioration in Global Health Status was prolonged (HR 0.67, p<0.001)"
  → Endpoint Language: "Time to deterioration in Global Health Status" [CT‑001]
  → Key Finding: "HR 0.67, p<0.001"

* If the label language for BPI‑SF contains no [Formal Claim], but the key_finding says:
  "BPI‑SF showed significantly delayed pain progression for pain interference (HR 0.79, p=0.005) and mean pain intensity (HR 0.82, p=0.049)"
  → Endpoint Language: "Time to pain progression for pain interference and mean pain intensity" [CT‑002]
  → Key Finding: "HR 0.79 (95% CI 0.67–0.93), p=0.005 (pain interference); HR 0.82 (95% CI 0.67–1.00), p=0.049 (mean pain intensity)"

* If a label language line mentions EQ‑5D‑5L but only contains a [Baseline Descriptor], and the key_finding contains no endpoint concept for EQ‑5D‑5L:
  → Endpoint Language: "—" [CT‑003]
  → Key Finding: (copy the best available numeric result, if any; otherwise "—")

CRITICAL: Never use a [Formal Claim] that refers to an instrument other than the one listed in the current row. If in doubt, fall back to the key_finding text for that instrument.

**FINAL CHECK for Table 4**  
Before writing Table 4, scan every row you are about to produce.  
If the "Endpoint Language" cell would contain "—", look again at the "Key Finding" column for that row and extract a short endpoint phrase (e.g., "Time to deterioration in Global Health Status", "Pain palliation proportion").  
If the Key Finding is not empty, you MUST place that derived phrase in the "Endpoint Language" cell and cite the [CT‑XXX] label.  
Leave "—" only if the Key Finding is completely empty or genuinely contains no endpoint concept.

**Pre‑writing check for Subscales Used**  
Verify each row: the "Subscales Used" column must match the trial‑specific `Subscales (trial-used):` line, not the instrument's general subscale description. If you cannot find a trial‑specific line, write "NR".

## Table 5: Language & Translation Readiness

**Before filling this table, you must perform language web searches. Do NOT print this instruction.**
- For each instrument selected for Table 3, perform exactly ONE web search using the instrument name and the phrase "validated translations".
  Example: if the instrument is EORTC QLQ‑C30, search "EORTC QLQ‑C30 validated translations".
- After that single search, fill the row immediately. Do NOT search again for the same instrument.
- Max 4 searches total (one per instrument). If fewer than 4 instruments are selected, search only for those.
- Use these search results in both Table 5 and in Table 3 (when discussing translation gaps).

List the same instruments selected for Table 3.
Columns: Instrument | Validated Translations (approx.) | Key Languages Covered | Gap / Action

**Column content**
- "Validated Translations": write the approximate number found plus the markdown hyperlink.  
  Example: "~110 languages [EORTC translations page](https://example.com)".  
  If the search result does not contain a number, write "Verify via PROQOLID or developer site".
  If you write only `[21]` the cell is incorrect. You must include the number and the reason (e.g., "~110 languages").
- "Key Languages Covered": list up to 6 languages that match the trial's geographic footprint (see TRIAL CONTEXT).  
  If the source does not list specific languages, write “Not listed in source — verify via developer website”.
- "Gap / Action": if any key language for the trial's footprint is absent, suggest "Commission [language] translation (6‑12 months)".  
  If all key languages are covered, write "No action".

## Key Observations
Exactly 6 bullet points. Each bullet must be on its OWN LINE (press Enter after each bullet).
Use the format:
- First bullet text [source]
- Second bullet text [source]
...
(Each line starts with a dash and a space. Do NOT write multiple bullets on one line.)
Each bullet must cite at least one source ([TI-XXX], [CT-XXX], [RULE-XXX], or web).

## Comparator Analysis
Write the entire section as one paragraph.  
Do NOT use bullet points, dashes, or numbered lists.  
Structure it as follows:  
  - Sentence 1: summarise [CT-001] (trial, instruments, outcome).  
  - Sentence 2: summarise [CT-002], and so on for each same‑class comparator trial.  
  - Final sentence: name the single instrument best supported by comparators for the current trial's TPP domain, with a citation.
Use clear, full sentences separated by periods.  
Max 2 web searches total — only if you need to confirm an instrument's sensitivity in a specific domain.

## HTA Requirements

If the trial context contains any HTA body or HTA indicator (NICE, ICER, EUnetHTA, SMC, CADTH, PBAC, HAS, G‑BA, IQWiG, EMA, HTA), produce the following table:

| HTA Body | Required Instrument & Version | Preferred Value Set | Current Battery Status | Risk if Omitted |
|---|---|---|---|---|

For each HTA body in scope:
- Required Instrument & Version: the exact instrument and version needed for cost‑utility analysis (e.g., "EQ‑5D‑5L").
- Preferred Value Set: the country‑specific tariff or value set that the body expects (e.g., "England 5L value set" for NICE, "US value set" for ICER). If not explicitly stated in web‑search or KG, write "Verify with local HTA guidelines".
- Current Battery Status: whether this instrument appears in the Domain Coverage Matrix candidates. Use: "In candidates" or "Not in current candidates — must add".
- Risk if Omitted: one sentence explaining the submission consequence (e.g., "QALY calculation impossible; NICE will issue a Request for Additional Data, delaying reimbursement by ≥12 months"). Cite the relevant HTA guidance or KG source.

If the trial context contains NONE of the above HTA indicators (e.g., an FDA‑only submission with no mention of cost‑effectiveness or HTA), do NOT generate a table. Instead, write: "No HTA bodies specified — HTA requirements not applicable for this submission."

## What the Expert Needs to Decide
List exactly 5 decisions. Each must:
- Reference a specific fact or gap from Tables 1‑5 or the Comparator Analysis.
- State the two most evidence‑supported options (from the data presented).
- Be written in one sentence that a clinical team could respond to directly.
- At least two decisions must cite a specific [RULE-XXX] that creates a binding constraint.

Examples of the format (do NOT copy verbatim; adapt to the current trial):
1. Whether to use EORTC QLQ‑C30 as the primary PRO given its use in 4 same‑class PI trials (Table 1, [CT-001]‑[CT-004]) but no population‑specific MCID for RRMM (Table 3).
2. Whether to add FACT/GOG‑NTX for neuropathy capture, given ENDEAVOR showed sensitivity (Table 4, [CT-004]) but only 9 languages available (Table 5); or rely on EORTC QLQ‑MY20 which has more domain overlap but weaker neuropathy evidence.
3. Endpoint hierarchy: pre‑specify pain response vs. physical function as the key secondary PRO, given A.R.R.O.W. and IKEMA showed mixed results (Table 1, [CT-001],[CT-005]).
4. Assessment frequency: adopt the IKEMA schedule (Baseline; C1D1; Q cycle D1) vs. a reduced schedule given the weekly IV administration burden (Table 2).
5. Translation strategy: commission immediate translations for Japanese and Mandarin if Asia‑Pacific is confirmed, given EORTC QLQ‑MY20 currently has only 1 language (Table 5).

The decisions must be ordered by urgency: the most critical decision first.

─────────────────────────────────────────
If KG is offline or returned no records, state this clearly at the top and do not fabricate data."""

        user_msg = (
            f"Trial description:\n{user_text}\n\n"
            f"KNOWLEDGE GRAPH EVIDENCE:\n{kg_evidence_block}"
        )

        try:
            resp = client.messages.create(
                model=SONNET,
                max_tokens=8000,
                system=system_prompt,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": user_msg}],
            )
            answer = " ".join(
                b.text
                for b in resp.content
                if hasattr(b, "text") and b.text
            )
            logging.info(f"Sonnet: {len(answer)} chars generated")
        except Exception as e:
            logging.error(f"Sonnet call failed: {e}")
            answer = (
                f"⚠️ Strategy generation failed: {e}\n\n"
                "Knowledge graph data was retrieved. Please retry."
            )
            error_status = str(e)

    result = {
        "answer":           answer,
        "top_scores":       scored,
        "all_scores":       scored,
        "kg_raw_hits":      raw_records,
        "reg_records":      reg_records,
        "reg_rules":        rules,
        "coverage":         coverage,
        "context_json":     context_json,
        "citation_index":   citation_index,
        "kg_evidence_block": kg_evidence_block,
        "record_counts": {
            "instrument_records":  len(raw_records),
            "regulatory_reviews":  len(reg_records),
            "regulatory_rules":    len(rules),
            "rejections_found":    len([r for r in reg_records if r.get("rejection_reason_primary")]),
            "all_scores":          len(scored),
            "kg_online":           error_status is None,
        },
        "error_status":     error_status,
        "gap_analysis":     [],
    }

    log_recommendation(user_text, result)   
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: HELPERS USED BY APP.PY
# ═══════════════════════════════════════════════════════════════════════════════

def linkify_flag_citations(text: str) -> str:
    """Replace [[LABEL]] with **[LABEL]** markdown. Used in sidebar display."""
    return re.sub(r"\[\[([^\]]+)\]\]", r"**[\1]**", text)


def build_tier1_citation_index(
    indication: str = "",
    phase: str = "",
    scored: list = None,
    raw_kg_records: list = None,
) -> dict:
    """
    Build citation_index for Tier 1/2 (factual/follow-up) answers.
    Uses scored instruments from a prior Tier 3 result if available.
    """
    ci: dict = {}
    if not scored:
        return ci

    for i, inst in enumerate(scored[:12], 1):
        label   = f"TI-{i:03d}"
        nct     = inst.get("nct_id", "")
        fda_url = inst.get("fda_url", "")
        ema_url = inst.get("ema_url", "")
        links   = []
        if nct:
            links.append({
                "label": "ClinicalTrials.gov",
                "url":   f"https://clinicaltrials.gov/study/{nct}",
            })
        if fda_url:
            links.append({"label": "FDA label", "url": fda_url})
        if ema_url:
            links.append({"label": "EMA label", "url": ema_url})

        ci[label] = {
            "type":       "trial_instrument",
            "instrument": inst["instrument_name"],
            "trial":      inst.get("best_trial", ""),
            "drug":       inst.get("best_drug", ""),
            "nct":        nct,
            "links":      links,
            "summary":    (
                f"{inst['instrument_name']} in {inst.get('best_trial','')} "
                f"({inst.get('best_drug','')})"
            ),
        }
    return ci

def log_recommendation(user_text: str, result: dict) -> None:
    """Save a timestamped JSON record of every Tier 3 strategy generation."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        coverage  = result.get("coverage", {})
        ctx       = result.get("context_json", {})
        top5 = [
            {
                "name": i["instrument_name"],
                "score": i["scientific_score"],
                "risk": i["risk_level"],
            }
            for i in result.get("top_scores", [])[:5]
        ]
        entry = {
            "timestamp":          timestamp,
            "user_query":         user_text,
            "indication":         ctx.get("indication", "unknown"),
            "phase":              ctx.get("phase", "unknown"),
            "drug_class":         ctx.get("drug_class", "unknown"),
            "assumptions_made":   ctx.get("assumptions_made", []),
            "coverage_domains":   [d["domain"] for d in coverage.get("domains", [])],
            "domains_covered":    len([d for d in coverage.get("domains", []) if d.get("candidates")]),
            "hta_mandatory":      [h["instrument"] for h in coverage.get("hta_mandatory", [])],
            "comparators":        [t["trial_name"] for t in coverage.get("comparator_trials", [])[:3]],
            "top_5_instruments":  top5,
            "record_counts":      result.get("record_counts", {}),
            "error_status":       result.get("error_status"),
            "answer_length":      len(result.get("answer", "")),
            "answer":             result.get("answer", ""),
        }
        path = f"logs/recommendation_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        logging.info(f"Log saved to {path}")
    except Exception as e:
        logging.error(f"Log write failed: {e}")