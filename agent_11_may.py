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

# ─────────────────────────────────────────────────────────────
# WEB SEARCH HELPER
# Used as Priority 3 fallback across all KG-first functions.
# Returns parsed JSON dict or None. Never raises — logs warnings.
# ─────────────────────────────────────────────────────────────

import re as _re
from typing import Optional

def _sonnet_web_search(query: str, instruction: str) -> Optional[dict]:
    """
    Fires a single Anthropic tool-use call with web_search enabled.
    Expects Sonnet to return a JSON object matching the schema
    described in `instruction`.

    Returns: parsed dict on success, None on any failure.
    Always logs the source URL when found.
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",        
            max_tokens=200,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
            system=(
                "You are a regulatory intelligence assistant. "
                "When asked a question, search the web for the most authoritative source "
                "(FDA.gov, EMA.europa.eu, NICE.org.uk, ICER, peer-reviewed journals, "
                "or official instrument developer sites). "
                "Always include the exact URL you retrieved the answer from. "
                f"{instruction}"
            ),
            messages=[{"role": "user", "content": query}],
        )

        # Walk response blocks — find the first text block with JSON
        for block in response.content:
            if hasattr(block, "text") and block.text:
                # Try direct JSON parse first
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    pass
                # Try extracting a JSON object substring
                match = _re.search(r'\{.*?\}', block.text, _re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        pass

        logging.warning(f"[_sonnet_web_search] No parseable JSON in response for query: {query!r}")
        return None

    except Exception as e:
        logging.warning(f"[_sonnet_web_search] Failed for query {query!r}: {e}")
        return None

def get_core_domains(indication, stakeholder, graph_client):
    """
    Returns core PRO domains for an indication and regulatory stakeholder.
    P1: KG RegulatoryRule nodes with decision_type = 'MUST' for the indication + stakeholder
    P3: Sonnet web search fallback — agency-aware
    P4: Agent inference — flagged as unverified
    
    stakeholder: comes from analyze_trial_context() in app.py
                 e.g. "FDA", "EMA", "NICE", "SMC"
    """
    # P1: KG — MUST rules for this indication and stakeholder
    kg_results = graph_client.get_regulatory_rules(
        indication=indication,
        decision_type="MUST"
    )

    # Filter by stakeholder if KG returned results
    if kg_results:
        stakeholder_results = [
            r for r in kg_results
            if not stakeholder or r.get("stakeholder", "").upper() == stakeholder.upper()
        ]
        # Fall back to all MUST results if stakeholder filter returns nothing
        filtered = stakeholder_results if stakeholder_results else kg_results

        return {
            "domains": [r["rule_text"] for r in filtered],
            "source_type": "KG",
            "citations": [
                f"[KG: {r.get('rule_id', 'RegulatoryRule')}] "
                f"{r.get('source_document', '')} {r.get('section', '')} "
                f"<{r.get('source_url', '')}>"
                for r in filtered
            ],
            "warning": None
        }

    # P3: Sonnet web search — agency-aware
    web_result = _sonnet_web_search(
        query=f"{stakeholder} recommended core PRO domains {indication} clinical trials guidance",
        instruction=(
            f"List the {stakeholder}-recommended core PRO domains for {indication} clinical trials. "
            "Cite the exact guidance document and URL. "
            "Return JSON: {\"domains\": [str], \"source_url\": str, \"source_title\": str, \"agency\": str}."
        )
    )
    if web_result and web_result.get("domains"):
        return {
            "domains": web_result["domains"],
            "source_type": "Web",
            "citations": [
                f"[Web: {web_result.get('source_url', 'unverified')}] "
                f"{web_result.get('source_title', '')} "
                f"({web_result.get('agency', stakeholder)})"
            ],
            "warning": None
        }

    # P4: Agent inference — flagged
    return {
        "domains": [],
        "source_type": "AgentInference",
        "citations": ["[Agent inference — manual verification required]"],
        "warning": f"⚠️ Core domains not found in KG or via web search for {stakeholder} / {indication}."
    }

def get_hta_preferences(markets, graph_client):
    """
    Returns HTA body PRO preferences for a list of markets.
    P1: KG RegulatoryRule nodes (stakeholder = body, decision_type = MUST/SHOULD)
    P3: Sonnet web search fallback
    P4: Agent inference — flagged as unverified

    markets: list of HTA body names from analyze_trial_context()
             e.g. ["NICE", "ICER", "EUnetHTA", "SMC"]
    Returns dict keyed by body name
    """
    result = {}

    for body in markets:
        # P1: KG query — MUST and SHOULD rules for this stakeholder
        kg_results = graph_client.get_regulatory_rules(
            indication="",
            lifecycle_stage="",
            decision_type=""
        )
        body_rules = [
            r for r in (kg_results or [])
            if r.get("stakeholder", "").upper() == body.upper()
        ]

        if body_rules:
            result[body] = {
                "notes": " | ".join(
                    f"{r.get('rule_text', '')} ({r.get('decision_type', '')})"
                    for r in body_rules
                ),
                "source_type": "KG",
                "source_url": body_rules[0].get("source_url", ""),
                "citation": " ".join(
                    f"[KG: {r.get('rule_id', body)}] "
                    f"{r.get('source_document', '')} {r.get('section', '')} "
                    f"<{r.get('source_url', '')}>"
                    for r in body_rules
                ),
                "warning": None
            }
            continue

        # P3: Sonnet web search
        web_result = _sonnet_web_search(
            query=f"{body} PRO instrument requirements patient-reported outcomes drug submission guidance",
            instruction=(
                f"What are the {body} requirements or preferences for PRO/COA instruments "
                "in drug regulatory submissions or HTA assessments? "
                "Cite the exact guidance document and URL. "
                "Return JSON: {\"notes\": str, \"source_url\": str, \"source_title\": str}."
            )
        )
        if web_result and web_result.get("notes"):
            result[body] = {
                "notes": web_result["notes"],
                "source_type": "Web",
                "source_url": web_result.get("source_url", ""),
                "citation": f"[Web: {web_result.get('source_url', 'unverified')}] {web_result.get('source_title', '')}",
                "warning": None
            }
            continue

        # P4: Agent inference — flagged
        result[body] = {
            "notes": f"No verified requirements found for {body}.",
            "source_type": "AgentInference",
            "source_url": "",
            "citation": "[Agent inference — verify manually]",
            "warning": f"⚠️ No KG or web source found for {body} PRO requirements."
        }

    return result

def get_language_requirements(geography, instruments, graph_client):
    """
    Returns language availability per instrument for a given geography.
    P1: KG Instrument.languages field (parsed, with VERIFY flag detection)
    P3: Sonnet web search to official developer site
    P4: Sonnet finds developer site URL — no hardcoded fallback, no invented thresholds

    NOTE: FDA/EMA publish NO numeric minimum language requirement.
    geography: from analyze_trial_context() e.g. "Global", "EU", "US-only"
    instruments: list of instrument name strings
    """
    result = {}

    for instr_name in instruments:
        if not instr_name or not str(instr_name).strip():
            continue

        # P1: KG — Instrument.languages field
        try:
            kg_ref = graph_client.get_instrument_reference(instr_name)
        except Exception as e:
            logging.warning(f"[get_language_requirements] KG query failed for {instr_name}: {e}")
            kg_ref = None

        if kg_ref:
            node = kg_ref[0] if isinstance(kg_ref, list) else kg_ref
            languages_raw = node.get("languages") if isinstance(node, dict) else None
            if languages_raw and str(languages_raw).strip():
                # Parse pipe-separated language names, skip prose sentences
                pipe_langs = [
                    l.strip() for l in str(languages_raw).split("|")
                    if l.strip()
                    and len(l.strip()) < 40
                    and not any(w in l.lower() for w in [
                        "verify", "estimated", "available", "licensed",
                        "no public", "translated for", "key languages include",
                        "through", "trial", "use in"
                    ])
                ]
                needs_verify = any(
                    w in str(languages_raw).upper()
                    for w in ["VERIFY", "ESTIMATED"]
                )
                result[instr_name] = {
                    "available_languages": pipe_langs,
                    "count": len(pipe_langs) if pipe_langs else None,
                    "geography": geography,
                    "source_type": "KG",
                    "source_url": node.get("source_url", "") if isinstance(node, dict) else "",
                    "citation": f"[KG: Instrument.languages — {instr_name}]",
                    "warning": (
                        f"⚠️ Language list for {instr_name} contains unverified estimates — "
                        f"confirm at developer site"
                        if needs_verify else None
                    )
                }
                continue

        # P3: Sonnet web search — find count + developer URL
        try:
            web_result = _sonnet_web_search(
                query=f"{instr_name} PRO instrument validated language translations available",
                instruction=(
                    f"How many validated language translations exist for the '{instr_name}' "
                    "patient-reported outcome instrument? "
                    "Find the official freely accessible developer site or translation registry. "
                    "Return JSON: {\"count\": int, \"languages\": [str], "
                    "\"source_url\": str, \"source_title\": str}."
                )
            )
        except Exception as e:
            logging.warning(f"[get_language_requirements] Web search failed for {instr_name}: {e}")
            web_result = None

        if web_result and web_result.get("count") is not None:
            result[instr_name] = {
                "available_languages": web_result.get("languages", []),
                "count": web_result["count"],
                "geography": geography,
                "source_type": "Web",
                "source_url": web_result.get("source_url", ""),
                "citation": (
                    f"[Web: {web_result.get('source_url', 'unverified')}] "
                    f"{web_result.get('source_title', '')}"
                ),
                "warning": None
            }
            continue

        # P4: Unverified — Sonnet finds developer site, no hardcoded URL
        try:
            dev_search = _sonnet_web_search(
                query=f"{instr_name} PRO instrument official developer website",
                instruction=(
                    f"Find the official freely accessible developer website for the "
                    f"'{instr_name}' PRO instrument where translation/language information is listed. "
                    "Return JSON: {\"source_url\": str, \"source_title\": str}."
                )
            )
        except Exception as e:
            logging.warning(f"[get_language_requirements] Developer URL search failed for {instr_name}: {e}")
            dev_search = None

        dev_url = dev_search.get("source_url", "") if dev_search else ""
        dev_title = dev_search.get("source_title", instr_name) if dev_search else instr_name

        result[instr_name] = {
            "available_languages": [],
            "count": None,
            "geography": geography,
            "source_type": "AgentInference",
            "source_url": dev_url,
            "citation": (
                f"[Agent inference — verify at {dev_title}]({dev_url})"
                if dev_url
                else "[Agent inference — developer site not found]"
            ),
            "warning": (
                f"⚠️ Translation coverage unverified for {instr_name} — "
                f"verify at developer site{f': {dev_url}' if dev_url else ''}"
            )
        }

    return result

def get_language_counts(instrument_names, graph_client):
    """
    Returns language count per instrument for the scoring engine.
    Simpler return shape than get_language_requirements() — count + citation only.
    P1: KG Instrument.languages
    P3: Sonnet web search
    P4: count=None — caller must NOT apply any scoring penalty on unverified data
    instrument_names: list of instrument name strings
    """
    result = {}

    for name in instrument_names:
        if not name or not str(name).strip():
            continue
        # Use cache — language counts don't change between runs
        _ck = name.strip().lower()
        if _ck in _lang_count_cache:
            result[name] = _lang_count_cache[_ck]
            continue

        # P1: KG
        try:
            kg_ref = graph_client.get_instrument_reference(name)
        except Exception as e:
            logging.warning(f"[get_language_counts] KG query failed for {name}: {e}")
            kg_ref = None

        if kg_ref:
            node = kg_ref[0] if isinstance(kg_ref, list) else kg_ref
            languages_raw = node.get("languages") if isinstance(node, dict) else None
            if languages_raw and str(languages_raw).strip():
                pipe_langs = [
                    l.strip() for l in str(languages_raw).split("|")
                    if l.strip()
                    and len(l.strip()) < 40
                    and not any(w in l.lower() for w in [
                        "verify", "estimated", "available", "licensed",
                        "no public", "translated for", "key languages include",
                        "through", "trial", "use in"
                    ])
                ]
                needs_verify = any(
                    w in str(languages_raw).upper()
                    for w in ["VERIFY", "ESTIMATED"]
                )
                result[name] = {
                    "count": len(pipe_langs) if pipe_langs else None,
                    "source_type": "KG",
                    "source_url": node.get("source_url", "") if isinstance(node, dict) else "",
                    "citation": f"[KG: Instrument.languages — {name}]",
                    "warning": (
                        f"⚠️ Language count for {name} unverified — confirm at developer site"
                        if needs_verify else None
                    )
                }
                _lang_count_cache[_ck] = result[name]
                continue

        # P3: Sonnet web search
        try:
            web_result = _sonnet_web_search(
                query=f"{name} PRO instrument validated language translations count",
                instruction=(
                    f"How many validated language translations exist for the '{name}' "
                    "PRO instrument? Find the official developer site. "
                    "Return JSON: {\"count\": int, \"source_url\": str, \"source_title\": str}."
                )
            )
        except Exception as e:
            logging.warning(f"[get_language_counts] Web search failed for {name}: {e}")
            web_result = None

        if web_result and web_result.get("count") is not None:
            result[name] = {
                "count": web_result["count"],
                "source_type": "Web",
                "source_url": web_result.get("source_url", ""),
                "citation": (
                    f"[Web: {web_result.get('source_url', 'unverified')}] "
                    f"{web_result.get('source_title', '')}"
                ),
                "warning": None
            }
            continue

        # P4: count=None — NO penalty applied by caller on unverified data
        try:
            dev_search = _sonnet_web_search(
                query=f"{name} PRO instrument official developer website",
                instruction=(
                    f"Find the official freely accessible developer website for the "
                    f"'{name}' PRO instrument. "
                    "Return JSON: {\"source_url\": str, \"source_title\": str}."
                )
            )
        except Exception as e:
            logging.warning(f"[get_language_counts] Developer URL search failed for {name}: {e}")
            dev_search = None

        dev_url = dev_search.get("source_url", "") if dev_search else ""
        dev_title = dev_search.get("source_title", name) if dev_search else name

        result[name] = {
            "count": None,
            "source_type": "AgentInference",
            "source_url": dev_url,
            "citation": (
                f"[Agent inference — verify at {dev_title}]({dev_url})"
                if dev_url
                else "[Agent inference — developer site not found]"
            ),
            "warning": (
                f"⚠️ Translation coverage unverified for {name} — "
                f"verify at developer site{f': {dev_url}' if dev_url else ''}"
            )
        }

    return result


def get_recall_period(instrument_name, graph_client):
    """
    Returns recall period in days for a single instrument.
    P1: KG Instrument node — recall_period / recall_window field
    P2: Local published reference values — scoped to this function, not a module constant
    P3: Sonnet web search — developer/validation study
    P4: Returns -1 sentinel — caller applies flag, no scoring penalty applied

    Returns: {"days": int or -1, "source_type": "KG"|"Static"|"Web"|"Unknown",
              "citation": str, "warning": str or None}
    """
    if not instrument_name or not str(instrument_name).strip():
        return {"days": -1, "source_type": "Unknown",
                "citation": "", "warning": "⚠️ No instrument name provided"}

    instr_lower = instrument_name.lower()

    # P1: KG — authoritative if populated
    try:
        kg_ref = graph_client.get_instrument_reference(instrument_name)
    except Exception as e:
        logging.warning(f"[get_recall_period] KG query failed for {instrument_name}: {e}")
        kg_ref = None

    if kg_ref:
        node = kg_ref[0] if isinstance(kg_ref, list) else kg_ref
        if isinstance(node, dict):
            recall_raw = node.get("recall_period") or node.get("recall_window")
            if recall_raw:
                num = re.search(r'\d+', str(recall_raw))
                if num:
                    return {"days": int(num.group()), "source_type": "KG",
                            "citation": f"[KG: Instrument.recall_period — {instrument_name}]",
                            "warning": None}

    # P2: Published reference values — scoped locally, not a module-level constant.
    # These are immutable validated facts from published instrument manuals.
    # Keeping them here avoids ~20-30 web search calls per query in the scoring loop.
    _published = {
        "eq-5d": 0, "eq-5d-5l": 0, "eq-5d-3l": 0,
        "bpi-sf": 1, "bpi": 1,
        "bfi": 1,
        "nrs": 1, "vas": 1,
        "pgis": 1, "pgic": 1,
        "pro-ctcae": 7,
        "fact-p": 7, "fact-g": 7, "fact-b": 7, "fact-l": 7, "facit-fatigue": 7,
        "eortc qlq-c30": 7, "eortc qlq-lc13": 7, "eortc qlq-my20": 7,
        "eortc qlq-pr25": 7, "eortc qlq-hn35": 7,
        "hads": 7,
        "gad-7": 14, "phq-9": 14,
        "sf-36": 28, "sf-12": 28,
    }
    match_key = next((k for k in _published if k in instr_lower), None)
    if match_key is not None:
        return {"days": _published[match_key], "source_type": "Static",
                "citation": f"[Reference: {match_key} — validated recall period]",
                "warning": None}

    # P3: Sonnet web search — for instruments not in KG or reference table
    try:
        web_result = _sonnet_web_search(
            query=f"{instrument_name} PRO instrument recall period days official",
            instruction=(
                f"What is the official recall period (in days) for the '{instrument_name}' "
                "patient-reported outcome instrument? "
                "Find the published validation study or developer documentation. "
                "Return JSON: {\"days\": int, \"source_url\": str, \"source_title\": str}."
            )
        )
    except Exception as e:
        logging.warning(f"[get_recall_period] Web search failed for {instrument_name}: {e}")
        web_result = None

    if web_result and web_result.get("days") is not None:
        return {"days": web_result["days"], "source_type": "Web",
                "citation": (f"[Web: {web_result.get('source_url', 'unverified')}] "
                             f"{web_result.get('source_title', '')}"),
                "warning": None}

    # P4: Unknown — caller applies flag, no penalty applied
    return {"days": -1, "source_type": "Unknown",
            "citation": "[Recall period not found — verify via developer documentation]",
            "warning": f"⚠️ Recall period unknown for {instrument_name}"}

def get_geographic_language_requirements(geographic_footprint: str, graph_client) -> dict:
    """
    Returns language requirements for a given geographic footprint.
    P1: KG — RegulatoryRule nodes with stakeholder matching footprint region
    P3: Sonnet web search fallback
    P4: Agent inference — flagged as unverified

    geographic_footprint: str from analyze_trial_context()
                          e.g. "Global", "EU", "US-only"
    Returns: {
        "min_languages": int or None,
        "key_languages": [str],
        "regulatory_note": str,
        "reference": str,
        "source_url": str,
        "citation": str,
        "warning": str or None
    }
    """
    if not geographic_footprint or geographic_footprint.strip().lower() == "unknown":
        return {
            "min_languages": None,
            "key_languages": [],
            "regulatory_note": "Geographic footprint unknown — language requirements cannot be determined.",
            "reference": "",
            "source_url": "",
            "citation": "[Agent inference — verify manually]",
            "warning": "⚠️ Geographic footprint not specified. Language requirements unverified."
        }

    footprint_lower = geographic_footprint.strip().lower()

    # P1: KG
    try:
        kg_results = graph_client.get_regulatory_rules(
            indication="", lifecycle_stage="", decision_type=""
        )
        region_rules = [
            r for r in (kg_results or [])
            if any(t in str(r.get("stakeholder", "")).lower()
                   for t in [footprint_lower, "fda", "ema"]
               )
            and any(t in str(r.get("rule_text", "")).lower()
                    for t in ["language", "translation", "linguistic"])
        ]
    except Exception as e:
        logging.warning(f"[get_geographic_language_requirements] KG failed: {e}")
        region_rules = []

    if region_rules:
        return {
            "min_languages": None,
            "key_languages": [],
            "regulatory_note": " | ".join(r.get("rule_text", "") for r in region_rules),
            "reference": region_rules[0].get("source_document", ""),
            "source_url": region_rules[0].get("source_url", ""),
            "citation": " ".join(
                f"[KG: {r.get('rule_id', 'RegulatoryRule')}] "
                f"{r.get('source_document', '')} <{r.get('source_url', '')}>"
                for r in region_rules
            ),
            "warning": None
        }

    # P3: Sonnet web search
    try:
        web_result = _sonnet_web_search(
            query=f"PRO linguistic validation language requirements {geographic_footprint} clinical trial FDA EMA",
            instruction=(
                f"What are the language translation and linguistic validation requirements "
                f"for PRO instruments in {geographic_footprint} clinical trials? "
                "Cite the exact FDA or EMA guidance document and URL. "
                "Return JSON: {\"min_languages\": int or null, \"key_languages\": [str], "
                "\"regulatory_note\": str, \"reference\": str, \"source_url\": str}."
            )
        )
    except Exception as e:
        logging.warning(f"[get_geographic_language_requirements] Web search failed: {e}")
        web_result = None

    if web_result and web_result.get("regulatory_note"):
        url = web_result.get("source_url", "")
        return {
            "min_languages": web_result.get("min_languages"),
            "key_languages": web_result.get("key_languages", []),
            "regulatory_note": web_result["regulatory_note"],
            "reference": web_result.get("reference", ""),
            "source_url": url,
            "citation": f"[Web: {url}]" if url else "[Web: unverified]",
            "warning": None
        }

    # P4: Agent inference
    return {
        "min_languages": None,
        "key_languages": [],
        "regulatory_note": f"No verified language requirements found for {geographic_footprint}.",
        "reference": "",
        "source_url": "",
        "citation": "[Agent inference — verify manually]",
        "warning": f"⚠️ No KG or web source found for {geographic_footprint} language requirements."
    }

def _get_domain_synonyms(domain: str) -> list:
    """
    Returns synonym list for a PRO domain keyword.
    Scoped locally — not a module-level constant.
    These are stable regulatory/clinical vocabulary mappings,
    not dynamic data, so no KG/web call is needed.
    """
    _synonyms = {
        "bone pain":              ["pain", "nrs", "bpi", "musculoskeletal", "skeletal"],
        "physical function":      ["physical", "function", "activity", "mobility",
                                   "performance", "adl", "karnofsky"],
        "fatigue":                ["fatigue", "tiredness", "energy", "exhaustion",
                                   "bfi", "brief fatigue", "facit-fatigue",
                                   "facit fatigue", "mfsi", "vitality", "asthenia"],
        "dyspnea":                ["dyspnea", "breathlessness", "breathing",
                                   "respiratory", "shortness of breath"],
        "cough":                  ["cough", "respiratory", "pulmonary"],
        "pain":                   ["pain", "bpi", "bpi-sf", "nrs", "worst pain",
                                   "pain intensity", "analgesic", "bone pain",
                                   "ache", "discomfort"],
        "nausea":                 ["nausea", "vomiting", "gi", "gastrointestinal", "emesis"],
        "urinary function":       ["urinary", "urology", "bladder", "ipss", "micturition"],
        "emotional function":     ["emotional", "anxiety", "depression",
                                   "psychological", "mental", "hads", "phq"],
        "appetite loss":          ["appetite", "anorexia", "eating", "weight"],
        "bowel function":         ["bowel", "diarrhoea", "constipation", "gastrointestinal"],
        "treatment tolerability": ["tolerability", "adverse", "toxicity", "ctcae",
                                   "symptom", "side effect", "crs",
                                   "cytokine release", "icans"],
        "disease-related symptoms": ["bone pain", "disease symptoms", "mm symptoms",
                                     "disease-specific", "symptom burden", "symptomatic"],
        "adverse events":         ["adverse events", "symptoms", "toxicity",
                                   "tolerability", "side effects", "treatment side effects",
                                   "nausea", "neuropathy", "fatigue"],
        "side effect impact summary": ["side effects", "treatment impact",
                                       "toxicity burden", "overall symptom burden",
                                       "tolerability", "adverse", "symptom"],
        "role function":          ["physical function", "functioning", "daily activities",
                                   "role functioning", "work", "activities", "function"],
        "physical functioning":   ["physical function", "functioning", "mobility",
                                   "activity"],
        "peripheral neuropathy":  ["neuropathy", "cipn", "tingling", "numbness",
                                   "sensory", "neuropathic pain"],
        "cytokine release syndrome": ["crs", "cytokine", "ctcae", "icans",
                                      "pro-ctcae", "tolerability", "adverse"],
        "hrqol":                  ["hrqol", "quality of life", "health-related",
                                   "wellbeing", "function"],
        "disease-specific symptoms": ["disease", "specific", "myeloma",
                                      "cancer-specific", "my20", "symptom"],
    }

    d = domain.strip().lower()

    # 1. Exact match
    if d in _synonyms:
        return _synonyms[d]

    # 2. Strip parentheticals and retry  e.g. "disease-related symptoms (bone pain)" → "disease-related symptoms"
    d_stripped = re.sub(r'\s*\(.*?\)', '', d).strip()
    if d_stripped in _synonyms:
        return _synonyms[d_stripped]

    # 3. Fuzzy: find the longest key that is a substring of d, or d is a substring of the key
    best_key, best_len = None, 0
    for key in _synonyms:
        if (key in d or d in key) and len(key) > best_len:
            best_key, best_len = key, len(key)
    if best_key:
        return _synonyms[best_key]

    return []

# Cache for subscale→domain maps — populated once per instrument per session
_subscale_map_cache: dict = {}
_domain_classify_cache: dict = {}
_lang_count_cache: dict = {}

def _is_nr(value) -> bool:
    """Returns True if a KG field is empty, null, or explicitly 'not reported'."""
    if value is None:
        return True
    s = str(value).strip().lower()
    return s in {"", "nan", "none", "null", "nr", "not reported",
                 "not reported/specified", "not specified", "not reportedspecified"}

def build_subscale_domain_map(instrument_name: str, all_assessed_values: list) -> dict:
    """
    Calls Haiku ONCE per instrument to map each subscale name → canonical domain.
    Returns dict like:
      {"fatigue intensity": "fatigue", "worst pain intensity": "pain",
       "physical functioning": "physical function", ...}
    Result is cached in _subscale_map_cache keyed by instrument_name.lower().
    Returns {} on failure — callers treat missing key as NOT_COLLECTED.
    """
    cache_key = (instrument_name or "").strip().lower()
    if not cache_key:
        return {}
    if cache_key in _subscale_map_cache:
        return _subscale_map_cache[cache_key]

    # Collect all unique subscale names from every KG record for this instrument
    all_subscales = set()
    for val in all_assessed_values:
        if _is_nr(val):
            continue
        for token in re.split(r'\|', str(val)):
            # Strip parenthetical qualifiers like "(single item)", "(composite)"
            clean = re.sub(r'\s*\(.*?\)', '', token).strip()
            if clean and clean.lower() not in {"total", "full scale",
                                                "full instrument", "all subscales"}:
                all_subscales.add(clean)

    if not all_subscales:
        _subscale_map_cache[cache_key] = {}
        return {}

    prompt = (
        f"Instrument: {instrument_name}\n"
        f"Subscale names: {', '.join(sorted(all_subscales))}\n\n"
        f"For each subscale, return a JSON object mapping the EXACT subscale name "
        f"(as given) to its single canonical clinical domain in lowercase.\n"
        f"Use only short domain names such as: fatigue, pain, physical function, "
        f"emotional function, role function, social function, nausea, dyspnea, "
        f"hrqol, appetite loss, peripheral neuropathy, constipation, diarrhea, "
        f"insomnia, disease symptoms, side effects, health utility, cognitive function.\n"
        f"Return ONLY a valid JSON object. No explanation, no markdown."
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        # Normalise: lowercase both keys and values
        result = {k.strip().lower(): v.strip().lower() for k, v in parsed.items()}
        _subscale_map_cache[cache_key] = result
        logging.info(f"build_subscale_domain_map: {instrument_name} → {len(result)} subscales mapped")
        return result
    except Exception as e:
        logging.warning(f"build_subscale_domain_map failed for '{instrument_name}': {e}")
        _subscale_map_cache[cache_key] = {}
        return {}
    
def resolve_domain_cell(
    domain: str,
    subscales_assessed_raw,       # r.get("instrument_subscales_assessed")
    subscale_results_raw,          # r.get("subscaleresults")
    subscale_significant_raw,      # r.get("subscalesignificant")
    subscale_not_significant_raw,  # r.get("subscalenotsignificant")
    subscale_map: dict,            # from build_subscale_domain_map()
) -> dict:
    """
    Pure Python. No API calls. Determines the epistemic state for one
    (domain, trial-instrument record) cell in Table 1.

    Returns a dict with keys:
      state:    "REPORTED"        — subscale collected AND outcome found in KG
                "COLLECTED_NR"   — subscale collected but no per-subscale result in KG
                "SUBSCALES_NR"   — instrument used but subscales_assessed is NR in KG
                "NOT_COLLECTED"  — subscale not in assessed list for this trial
      change:   "Yes" | "No" | "NR"
      subscale: the matching subscale name (str) or None
    """
    domain_lower = domain.strip().lower()

    # ── Step 1: Was this domain's subscale actually collected? ──────────────
    if _is_nr(subscales_assessed_raw):
        return {"state": "SUBSCALES_NR", "change": "NR", "subscale": None}

    # Parse the pipe-separated subscale list
    assessed_subscales = []
    for token in re.split(r'\|', str(subscales_assessed_raw)):
        clean = re.sub(r'\s*\(.*?\)', '', token).strip()  # strip "(single item)" etc
        if clean:
            assessed_subscales.append(clean)

    # Find which assessed subscales map to this domain
    matching_subscales = [
        s for s in assessed_subscales
        if subscale_map.get(s.lower()) == domain_lower
    ]

    if not matching_subscales:
        return {"state": "NOT_COLLECTED", "change": "NR", "subscale": None}

    # ── Step 2: Was a result reported for this domain? ──────────────────────
    # Use the most structured fields first: subscalesignificant and
    # subscalenotsignificant, then fall back to subscaleresults.
    sig_text    = _to_str(subscale_significant_raw)      # already lowercase from _to_str
    nonsig_text = _to_str(subscale_not_significant_raw)
    results_text = _to_str(subscale_results_raw)

    for subscale in matching_subscales:
        sl = subscale.lower()
        if sl in sig_text:
            return {"state": "REPORTED", "change": "Yes", "subscale": subscale}
        if sl in nonsig_text:
            return {"state": "REPORTED", "change": "No", "subscale": subscale}
        if sl in results_text:
            # In results text — determine direction from result text around this subscale
            # Extract a short window around the subscale mention
            idx = results_text.find(sl)
            window = results_text[max(0, idx - 20): idx + 80]
            if any(t in window for t in ["improved", "p<0", "p=0.0", "hr 0.", "significant"]):
                change = "Yes"
            elif any(t in window for t in ["ns", "not significant", "no significant"]):
                change = "No"
            else:
                change = "NR"
            return {"state": "REPORTED", "change": change, "subscale": subscale}

    # Subscale was collected but no per-subscale result found in any field
    return {"state": "COLLECTED_NR", "change": "NR", "subscale": matching_subscales[0]}

def build_trial_domain_matrix(
    domain_coverage: list,
    comparator_trials: list,
    raw_kg_records: list,
    instrument_meta: dict = None,
    subscale_maps: dict = None,) -> list:
    """
    Pre-computes Table 1 cells using only KG-derived data.
    
    Three states, all data-driven — no hardcoding:
    
    REPORTED           key_finding/significance for this record mentions this domain
    COLLECTED_NOT_REPORTED  instrument covers domain (from Instrument node), but 
                            no domain-specific outcome reported in KG
    NOT_COLLECTED      no KG record exists for this trial × instrument pair
    """

    # ── Step 1: Build (trial, instrument) → records lookup ──────────────────
    # Direct from raw_kg_records. One instrument can have multiple records
    # per trial (different endpoints). Keep all — merge change signal.
    from collections import defaultdict
    trial_inst_records: dict = defaultdict(lambda: defaultdict(list))
    for r in raw_kg_records:
        tname = r.get("trial_name") or r.get("nct_id", "")
        iname = str(r.get("instrument_name", "")).strip()
        if tname and iname:
            trial_inst_records[tname][iname].append(r)

    # ── Step 2: Build instrument → all_domains lookup ───────────────────────
    # Source of truth: Instrument node `domains` field (already enriched by
    # classify_instrument_domains_haiku in Step 5b of recommend()).
    # This tells us what the instrument IS CAPABLE of measuring — not what
    # any specific trial chose to report.
    inst_all_domains: dict = {}  # instrument_name → domain search string
    all_instrument_names = {
        str(r.get("instrument_name", "")).strip()
        for r in raw_kg_records if r.get("instrument_name")
    }
    for iname in all_instrument_names:
        ilower = iname.lower()
        _meta = instrument_meta or {}
        node = (
            _meta.get(ilower)
            or _meta.get(ilower.split()[-1] if ilower.split() else ilower)
            or {}
        )
        # Instrument.domains is authoritative; fall back to record fields
        node_domains = _to_str(node.get("domains", ""))
        inst_records = trial_inst_records_flat = [
            r for recs in trial_inst_records.values()
            for r in recs.get(iname, [])
        ]
        fallback = " ".join(filter(None, [
            _to_str(r.get("instrument_domain", "")) for r in inst_records] +
            [_to_str(r.get("domains_measured", "")) for r in inst_records] +
            [_to_str(r.get("key_finding", "")) for r in inst_records]
        ))
        inst_all_domains[iname] = node_domains or fallback or ilower

    # ── Step 3: For each (trial, instrument, record), does key_finding 
    #    mention this domain? This is the REPORTED check — purely data-driven.
    # We use the same synonym map already used throughout the codebase.
    def _key_finding_mentions_domain(records: list, domain: str) -> bool:
        domain_key = re.sub(r'[^a-z ]', '', domain.strip().lower())
        all_terms = list(dict.fromkeys(
            [domain.lower(), domain_key] + _get_domain_synonyms(domain_key)
        ))
        for r in records:
            text = " ".join(filter(None, [
                _to_str(r.get("key_finding", "")),
                _to_str(r.get("significance", "")),
                _to_str(r.get("subscale_results", "")),
                _to_str(r.get("instrument_subscales_assessed", "")),
            ]))
            if any(t in text for t in all_terms):
                return True
        return False

    def _get_change(records: list) -> str:
        for r in records:
            sig = _to_str(r.get("significance", ""))
            direction = _to_str(r.get("direction", ""))
            if (any(t in sig for t in ["significant", "p <", "p<", "favours", "favor"])
                    and any(t in direction for t in ["favour", "favor", "improvement", "better", "positive"])):
                return "Yes"
        for r in records:
            sig = _to_str(r.get("significance", ""))
            if any(t in sig for t in ["not significant", "no significant", "ns", "p > 0", "p>0"]):
                return "No"
        return "NR"

    def _inst_covers_domain(iname: str, domain: str) -> bool:
        """Does this instrument measure this domain at all, per Instrument.domains?"""
        domain_key = re.sub(r'[^a-z ]', '', domain.strip().lower())
        all_terms = list(dict.fromkeys(
            [domain.lower(), domain_key] + _get_domain_synonyms(domain_key)
        ))
        search_text = inst_all_domains.get(iname, iname.lower())
        return any(t in search_text for t in all_terms)

    # ── Step 4: Build matrix ────────────────────────────────────────────────
    matrix = []
    for domain_entry in domain_coverage:
        domain = domain_entry["domain"]
        is_fda_core = domain_entry["is_fda_core"]

        trial_cells: dict = {}
        for trial in comparator_trials:
            tname = trial["trial_name"]
            inst_map = trial_inst_records.get(tname, {})

            cells = []
            seen = set()
            for inst in trial.get("instruments", []):
                iname = str(inst.get("name", "")).strip()
                if not iname or iname in seen:
                    continue
                seen.add(iname)

                records = inst_map.get(iname, [])

                # Does this instrument cover this domain at all?
                if not _inst_covers_domain(iname, domain):
                    continue  # Instrument doesn't measure this domain — skip entirely

                # Does the KG actually report an outcome for this domain?
                if _key_finding_mentions_domain(records, domain):
                    cells.append({
                        "instrument": iname,
                        "change": _get_change(records),
                        "state": "REPORTED",
                    })
                else:
                    # Instrument covers domain by design, but KG has no 
                    # domain-specific outcome reported for this trial
                    cells.append({
                        "instrument": iname,
                        "change": "NR",
                        "state": "COLLECTED_NOT_REPORTED",
                    })

            trial_cells[tname] = cells if cells else None  # None = NOT_COLLECTED

        matrix.append({
            "domain": domain,
            "is_fda_core": is_fda_core,
            "trials": trial_cells,
        })

    return matrix

# =============================================================================
# CONSTANTS — INDICATION-SPECIFIC CORE DOMAINS
# Source: FDA (2021) "Core Patient-Reported Outcomes in Cancer Clinical Trials"
# =============================================================================
# INDICATION_CORE_DOMAINS = {
#     "multiple myeloma": ["bone pain", "physical function", "fatigue"],
#     "mm": ["bone pain", "physical function", "fatigue"],
#     "rrmm": ["bone pain", "physical function", "fatigue", "treatment tolerability"],
#     "nsclc": ["dyspnea", "cough", "chest pain", "physical function"],
#     "non-small cell lung": ["dyspnea", "cough", "chest pain", "physical function"],
#     "lung cancer": ["dyspnea", "cough", "physical function"],
#     "crpc": ["pain", "urinary function", "physical function"],
#     "prostate cancer": ["pain", "urinary function", "physical function"],
#     "metastatic castration-resistant": ["pain", "urinary function", "physical function"],
#     "breast cancer": ["fatigue", "pain", "physical function", "emotional function"],
#     "colorectal": ["nausea", "appetite loss", "bowel function", "fatigue"],
#     "crc": ["nausea", "appetite loss", "bowel function", "fatigue"],
#     "ovarian": ["abdominal pain", "bloating", "fatigue", "physical function"],
#     "lymphoma": ["fatigue", "night sweats", "physical function"],
#     "leukemia": ["fatigue", "physical function", "emotional function"],
#     "aml": ["fatigue", "physical function", "emotional function"],
#     "default": ["physical function", "fatigue", "pain"]
# }


# =============================================================================
# CONSTANTS — HTA/PAYER INSTRUMENT PREFERENCES
# Sources cited per entry
# =============================================================================
# HTA_PREFERENCES = {
#     "NICE": {
#         "required_instruments": ["EQ-5D"],   # Wildcard — EQ-5D-5L OR EQ-5D-3L satisfies this
#         "preferred_version": "EQ-5D-5L",
#         "accepted_versions": ["EQ-5D-5L", "EQ-5D-3L"],
#         "notes": (
#             "NICE requires a preference-based EQ-5D measure for cost-utility analysis. "
#             "EQ-5D-5L is preferred per NICE position statement (October 2019). "
#             "Without EQ-5D, QALY calculation is impossible and UK reimbursement is severely compromised."
#         ),
#         "reference": "NICE DSU Technical Support Document 2 (2011, updated 2019); NICE EQ-5D-5L position statement (2019)"
#     },
#     "ICER": {
#         "required_instruments": [],
#         "preferred_instruments": ["EQ-5D-5L", "SF-36", "SF-6D"],
#         "notes": (
#             "ICER uses utility-based measures for cost-effectiveness analysis in US value assessments. "
#             "EQ-5D-5L is strongly preferred for QALY calculation."
#         ),
#         "reference": "ICER Value Assessment Framework (2020)"
#     },
#     "EUnetHTA": {
#         "required_instruments": [],
#         "preferred_instruments": ["EQ-5D-5L", "EORTC QLQ-C30"],
#         "notes": (
#             "EU HTA Regulation 2021/2282 Joint Clinical Assessments increasingly require standardised "
#             "PRO instruments for cross-country comparison. EQ-5D-5L required for HTA utility analysis."
#         ),
#         "reference": "EU HTA Regulation 2021/2282; EUnetHTA 21 methodology guidelines"
#     },
#     "SMC": {
#         "required_instruments": ["EQ-5D"],
#         "preferred_version": "EQ-5D-5L",
#         "notes": "Scottish Medicines Consortium aligns with NICE on EQ-5D requirement.",
#         "reference": "SMC Modifiers and PACE framework"
#     }
# }


# =============================================================================
# CONSTANTS — GEOGRAPHIC LANGUAGE REQUIREMENTS
# IMPORTANT: FDA does NOT specify a minimum number of languages.
# Source: FDA PRO Guidance (2009) Section IV.A — requires linguistically validated
# translations for languages used in the trial. No numeric minimum is stated.
# Source: EMA Reflection Paper on PRO (2005) — requires translations for each
# EU member state language where the trial is conducted.
# =============================================================================
# GEOGRAPHIC_LANGUAGE_REQUIREMENTS = {
#     "Global": {
#         "min_languages": 15,
#         "key_languages": [
#             "English", "Spanish", "French", "German", "Italian",
#             "Japanese", "Mandarin", "Portuguese", "Russian", "Korean",
#             "Polish", "Dutch", "Swedish", "Turkish", "Arabic"
#         ],
#         "regulatory_note": (
#             "FDA PRO Guidance (2009) Section IV.A requires linguistically validated translations "
#             "for each language used in the trial. There is no FDA-specified minimum number of languages. "
#             "EMA requires translations for each EU member state language where the trial is conducted."
#         ),
#         "reference": "FDA PRO Guidance (2009) Section IV.A; EMA Reflection Paper on PRO (2005)"
#     },
#     "EU": {
#         "min_languages": 10,
#         "key_languages": [
#             "English", "French", "German", "Spanish", "Italian",
#             "Dutch", "Polish", "Swedish", "Danish", "Finnish",
#             "Czech", "Romanian", "Hungarian", "Portuguese", "Greek"
#         ],
#         "regulatory_note": (
#             "EMA requires validated translations for each member state language where the trial "
#             "is conducted per EMA Reflection Paper on PRO (2005). "
#             "EU HTA Regulation 2021/2282 requires standardised instruments for Joint Clinical Assessments."
#         ),
#         "reference": "EMA Reflection Paper on PRO (2005); EU HTA Regulation 2021/2282"
#     },
#     "US-only": {
#         "min_languages": 1,
#         "key_languages": ["English"],
#         "regulatory_note": (
#             "English linguistic validation required. "
#             "If trial population includes non-English speakers, additional translations required "
#             "per FDA PRO Guidance (2009) Section IV.A."
#         ),
#         "reference": "FDA PRO Guidance (2009) Section IV.A"
#     }
# }


# =============================================================================
# CONSTANTS — INSTRUMENT RECALL PERIODS (days)
# ONLY instruments with published, citable recall periods are listed.
# For instruments NOT listed: RECALL_PERIOD_UNKNOWN = -1 sentinel value.
# The recall bias penalty does NOT fire on unknown instruments —
# Sonnet is instructed to verify via web search.
# =============================================================================
RECALL_PERIOD_UNKNOWN = -1

# INSTRUMENT_RECALL_PERIODS = {
#     # EuroQol Group official documentation — "TODAY"
#     "eq-5d": 0,
#     "eq-5d-5l": 0,
#     "eq-5d-3l": 0,
#     # Cleeland & Ryan (1994) Pain 62(3):173-182 — "past 24 hours"
#     "bpi-sf": 1,
#     "bpi": 1,
#     # Mendoza et al. (1999) Cancer 85(5):1186-1196 — "right now" and "past 24 hours"
#     "bfi": 1,
#     # IMMPACT recommendations Dworkin et al. (2005) Pain 113(1-2):9-19 — "right now"
#     "nrs": 1,
#     "vas": 1,
#     # Clinician/patient global impression — current state or since last visit
#     "pgis": 1,
#     "pgic": 1,
#     # NCI PRO-CTCAE User Manual v1.0 — "past 7 days"
#     "pro-ctcae": 7,
#     # FACIT.org official documentation — "past 7 days"
#     "fact-p": 7,
#     "fact-g": 7,
#     "fact-b": 7,
#     "fact-l": 7,
#     "facit-fatigue": 7,
#     # EORTC Quality of Life Group manual — "during the past week"
#     "eortc qlq-c30": 7,
#     "eortc qlq-lc13": 7,
#     "eortc qlq-my20": 7,
#     "eortc qlq-pr25": 7,
#     "eortc qlq-hn35": 7,
#     # Zigmond & Snaith (1983) Acta Psychiatr Scand 67(6):361-370 — "past week"
#     "hads": 7,
#     # Spitzer et al. (2006) Arch Intern Med 166(10):1092-1097 — "last 2 weeks"
#     "gad-7": 14,
#     # Kroenke et al. (2001) J Gen Intern Med 16(9):606-613 — "last 2 weeks"
#     "phq-9": 14,
#     # Ware & Sherbourne (1992) Med Care 30(6):473-483 — "past 4 weeks"
#     "sf-36": 28,
#     "sf-12": 28,
# }

# =============================================================================
# CONSTANTS — KNOWN LANGUAGE COUNTS (approximate, for reporting)
# Sources: instrument developer documentation and published translations registries
# Used for REPORTING only — not for pass/fail thresholds
# =============================================================================
# KNOWN_LANGUAGE_COUNTS = {
#     "eq-5d": 150,     # EuroQol Group — 150+ validated translations
#     "eq-5d-5l": 150,
#     "eq-5d-3l": 150,
#     "eortc qlq-c30": 100,   # EORTC — 100+ languages
#     "eortc qlq-my20": 80,
#     "fact-g": 60,           # FACIT.org — 60+ languages
#     "fact-p": 60,
#     "fact-b": 60,
#     "bpi-sf": 40,           # MD Anderson — 40+ languages
#     "bpi": 40,
#     "pro-ctcae": 30,        # NCI — 30+ languages
#     "sf-36": 80,            # QualityMetric — 80+ languages
#     "sf-12": 80,
#     "bfi": 9,               # MD Anderson — approximately 9 languages (Mendoza 1999 + translations)
#     "hads": 30,             # Multiple translated versions available
#     "pgis": 15,
#     "pgic": 15,
# }


# =============================================================================
# DOMAIN SYNONYM MAP
# Allows broad instruments stored as "HRQoL" to match "physical function" etc.
# Source: FDA (2021) Core PRO Guidance domain definitions
# =============================================================================

DOMAIN_SYNONYMS = {
    "bone pain": ["pain", "nrs", "bpi", "musculoskeletal", "skeletal"],
    "physical function": ["physical", "function", "activity", "mobility", "performance", "adl", "karnofsky"],
    "fatigue": ["fatigue", "tiredness", "energy", "exhaustion","bfi", "brief fatigue", "facit-fatigue", "facit fatigue", "mfsi", "vitality", "asthenia",],
    "dyspnea": ["dyspnea", "breathlessness", "breathing", "respiratory", "shortness of breath"],
    "cough": ["cough", "respiratory", "pulmonary"],
    "pain": ["pain", "bpi", "bpi-sf", "nrs", "worst pain", "pain intensity",
        "analgesic", "bone pain", "ache", "discomfort"],
    "nausea": ["nausea", "vomiting", "gi", "gastrointestinal", "emesis"],
    "urinary function": ["urinary", "urology", "bladder", "ipss", "micturition"],
    "emotional function": ["emotional", "anxiety", "depression", "psychological", "mental", "hads", "phq"],
    "appetite loss": ["appetite", "anorexia", "eating", "weight"],
    "bowel function": ["bowel", "diarrhoea", "constipation", "gastrointestinal"],
    "treatment tolerability": ["tolerability", "adverse", "toxicity", "ctcae", "symptom", "side effect", "crs", "cytokine release", "icans"],
    "disease-related symptoms": ["bone pain", "disease symptoms", "mm symptoms", "disease-specific", "symptom burden"],
    "symptomatic adverse events": ["adverse events", "symptoms", "toxicity", "tolerability", "side effects", "treatment side effects", "nausea", "neuropathy", "fatigue"],
    "side effect impact summary": ["side effects", "treatment impact", "toxicity burden", "overall symptom burden", "tolerability", "adverse", "symptom"],
    "role function": ["physical function", "functioning", "daily activities", "role functioning", "work", "activities", "function"],
    "physical functioning": ["physical function", "functioning", "mobility", "activity"],
    "peripheral neuropathy": ["neuropathy", "cipn", "tingling", "numbness", "sensory", "neuropathic pain"],
    "cytokine release syndrome (crs) symptoms": ["crs", "cytokine", "ctcae", "icans", "pro-ctcae", "tolerability", "adverse"],
    "hrqol": ["hrqol", "quality of life", "health-related", "wellbeing", "function"],
    "disease-specific symptoms": ["disease", "specific", "myeloma", "cancer-specific", "my20", "symptom"],
}

_PRO_ENDPOINT_KEYWORDS = (
    "pro", "hrqol", "hqol", "patient-reported", "patient reported",
    "qol", "quality of life", "symptom", "pain", "fatigue", "function",
    "bpi", "fact-", "eortc", "eq-5d", "eq5d", "promis", "mfsi",
    "deterioration", "time to deterioration", "ttd",
    "bfi", "pgic", "mfsaf", "ppq", "tasq", "ctsq", "rasq",
)

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

def _is_calibrated_soa(record: dict) -> bool:
    """
    Returns True ONLY if there is explicit KG evidence that a subset of subscales
    (not the full instrument) was pre-specified in the trial SOA.
    Conservative: defaults to False when uncertain.
    """
    subs = str(record.get("instrument_subscales_assessed") or "").strip()
    if not subs:
        return False
    # Full instrument — not calibrated
    if any(t in subs.lower() for t in ("total", "full scale", "full instrument")):
        return False
    # Explicit KG boolean flag — trust it
    explicit_flag = record.get("calibrated_soa") or record.get("item_library_used")
    if explicit_flag is not None and str(explicit_flag).lower() in ("true", "yes", "1"):
        return True
    # Has specific subscale names listed (not total/full) — infer calibrated
    return True

def _norm(s: str) -> str:
    """
    Normalise text for fuzzy domain matching.
    Strips common grammatical suffixes so "functioning" matches "function",
    "symptoms" matches "symptom", etc.
    This is a linguistic operation only — no hardcoded clinical knowledge.
    """
    return re.sub(
        r'(ing|tion|ity|ness|ment|al|ed|s)\b',
        '',
        (s or "").lower()
    ).strip()


def _domain_matches_instrument(domain: str, instrument_text: str) -> bool:
    """
    Check whether a domain concept is covered by an instrument's text fields.
    Uses normalised word-level matching — no hardcoded synonym lists.
    Matches if ANY significant word (>3 chars) from the domain appears in
    the instrument text after normalisation.

    Quality depends entirely on the richness of domains_measured in the KG.
    """
    norm_inst = _norm(instrument_text)
    if not norm_inst:
        return False
    domain_words = [_norm(w) for w in domain.split() if len(w) > 3]
    return any(w and w in norm_inst for w in domain_words)


def _get_drug_toxicities(record: dict) -> list:
    """
    Extract mechanism-specific toxicity domains from the KG Drug node.
    Uses the key_toxicities field — no hardcoded toxicity lists.
    Returns list of normalised toxicity strings, or [] if not available.

    Source: Drug node key_toxicities field populated from drug labels
    and investigator brochures during KG construction.
    """
    raw = _to_str(record.get("key_toxicities", ""))
    if not raw or raw.lower() in ("nan", "none", ""):
        return []
    return [t.strip().lower() for t in re.split(r'[,;/]', raw) if t.strip()]

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

1b. moa_aliases: for every drug_class extracted, list all known synonyms, generic drug names,
and standard abbreviations. The goal is to match trial records stored with different naming.
Examples:
- drug_class = "Proteasome Inhibitor" → moa_aliases = ["PI", "bortezomib", "carfilzomib",
  "ixazomib", "marizomib", "oprozomib", "proteasome inhibition", "26S proteasome"]
- drug_class = "Bispecific" → moa_aliases = ["bispecific antibody", "T-cell redirecting",
  "teclistamab", "talquetamab", "elranatamab", "linvoseltamab"]
- drug_class = "CAR-T" → moa_aliases = ["CAR T-cell", "cilta-cel", "ide-cel",
  "ciltacabtagene", "idecabtagene", "bb2121"]
- drug_class = "CD38 monoclonal antibody" → moa_aliases = ["CD38 mAb", "anti-CD38",
  "daratumumab", "isatuximab", "CD38-targeted"]
- drug_class = "ADC" → moa_aliases = ["antibody-drug conjugate", "belantamab",
  "BCMA ADC", "mafodotin"]
If drug_class is "Unknown", set moa_aliases = [].

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
        "moa_aliases": ["all synonyms and drug names for the drug_class — e.g. for Proteasome Inhibitor: [bortezomib, carfilzomib, ixazomib, PI, 26S proteasome]"],
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

        # Never overrides what Haiku explicitly extracted — only fills gaps.
        ind_key = context.get("indication", "").lower()
        stakeholder = context.get("hta_markets", ["FDA"])[0] if context.get("hta_markets") else "FDA"

        domain_data = get_core_domains(
            indication=ind_key,
            stakeholder=stakeholder,
            graph_client=_get_conn()
        )
        existing = [d.lower() for d in context.get("core_domains_required", [])]
        for domain in domain_data["domains"]:
            if domain.lower() not in existing:
                context.setdefault("core_domains_required", []).append(domain)

        # Attach citation so it surfaces in the output
        context["core_domains_source"] = domain_data["source_type"]
        context["core_domains_citations"] = domain_data["citations"]
        if domain_data.get("warning"):
            context.setdefault("assumptions_made", []).append(domain_data["warning"])

        # Deduplicate core_domains_required — Haiku sometimes generates semantic
        # variants of the same concept ("side effect impact summary" and
        # "overall side effect impact summary measure" are the same FDA concept).
        raw_domains = context.get("core_domains_required", [])
        if raw_domains:
            seen_norms = []
            deduped_domains = []
            for d in raw_domains:
                d_norm = re.sub(r'\s+', ' ', d.lower().strip())
                is_dup = any(
                    d_norm in seen_norm or seen_norm in d_norm
                    for seen_norm in seen_norms
                )
                if not is_dup:
                    seen_norms.append(d_norm)
                    deduped_domains.append(d)
            context["core_domains_required"] = deduped_domains

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
    
def extract_pro_endpoints(raw_endpoints) -> str:
    """
    From a Trial node's trial_secondary_endpoints_raw (pipe-delimited string or list),
    extract only the entries that are PRO/HRQoL-related.
    Returns a clean pipe-delimited string, or '' if nothing found.
    """
    if not raw_endpoints:
        return ""
    if isinstance(raw_endpoints, list):
        entries = [str(e).strip() for e in raw_endpoints if e]
    else:
        entries = [e.strip() for e in str(raw_endpoints).split("|") if e.strip()]

    pro_entries = [
        e for e in entries
        if any(kw in e.lower() for kw in _PRO_ENDPOINT_KEYWORDS)
    ]
    return " | ".join(pro_entries)

# =============================================================================
# STEP 2b — HAIKU DOMAIN MICRO-CLASSIFIER
# Enriches Instrument nodes that have no "domains" field in the KG.
# Only called for thin nodes — not every instrument.
# =============================================================================
def classify_instrument_domains_haiku(instrument_name: str, context_hint: str = "") -> str:
    """
    Calls Haiku to infer an instrument's measured domains when
    inst_node.get("domains") is empty. Returns a space-separated
    domain keyword string (e.g. 'fatigue pain physical function').
    Falls back to "" on any failure — scoring is unaffected.
    """
    if not instrument_name or instrument_name.strip().lower() == "unknown":
        return ""
    _ck = instrument_name.strip().lower()
    if _ck in _domain_classify_cache:
        return _domain_classify_cache[_ck]

    prompt = (
        f"Instrument name: {instrument_name}\n"
        + (f"Context: {context_hint}\n" if context_hint else "")
        + "List every clinical domain this PRO/COA instrument measures. "
        "Return ONLY a comma-separated list of short domain keywords "
        "(e.g. fatigue, pain, physical function, nausea, emotional function). "
        "No explanations. No sentences. Just keywords."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip().lower()
        raw = re.sub(r"[.\n]+", ",", raw)
        domains = [d.strip() for d in re.split(r"[,;]+", raw) if d.strip()]
        result = " ".join(domains)
        _domain_classify_cache[_ck] = result
        return result
    except Exception as e:
        logging.warning(f"classify_instrument_domains_haiku({instrument_name}): {e}")
        _domain_classify_cache[_ck] = ""
        return ""
    
def build_tier1_citation_index(indication: str, phase: str = "Phase 3", scored: list = None, raw_kg_records: list = None) -> dict:
    """
    Build a minimal citation index for Tier 1/2 queries that have no prior
    strategy. Fetches KG records for the indication and maps TI-XXX / RR-XXX /
    REJ-XXX labels so Sonnet's answer can be linkified in app.py.
    """
    citation_index = {}
    if not indication or indication.lower() == "unknown":
        return citation_index
    
    kg_records = raw_kg_records or []

    try:
        # kg_records = []
        # for term in [indication][:3]:
        #     r = get_instruments_by_indication(term, phase, "")
        #     if r:
        #         kg_records.extend(r)

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

            inst_name = inst.get("instrument_name", "")         
            scored_match = next(
                (s for s in (scored or []) if s.get("instrument_name") == inst_name),
                None
            )
            citation_index[label] = {
                "type":          "trial_instrument",
                "instrument":    inst_name,
                "trial":         inst.get("trial_name", "") or nct,
                "nct":           nct,
                "drug":          drug,
                "phase":         inst.get("phase", ""),
                "key_finding":   str(inst.get("key_finding", "") or ""),
                "endpoint_role": inst.get("endpoint_role", "") or inst.get("pro_position", ""),
                "links":         links,
            }

        reg_records  = get_regulatory_evidence(indication, "FDA") or []
        reg_records += get_regulatory_evidence(indication, "EMA") or []

        non_rej = [r for r in reg_records if not r.get("rejection_reason_primary")]
        rej     = [r for r in reg_records if r.get("rejection_reason_primary")]

        raw_rules = get_regulatory_rules(indication="", lifecycle_stage="", decision_type="") or []
        _STRATEGY_STAGES = {"Instrument_Selection", "Protocol_Design", "Concept_Selection"}
        reg_rules = [r for r in raw_rules if r.get("lifecycle_stage", "") in _STRATEGY_STAGES]

        for i, rule in enumerate(reg_rules, 1):
            label = rule.get("rule_id") or f"RULE-{i:03d}"
            citation_index[label] = {
                "type":            "rule",
                "decision_type":   rule.get("decision_type", ""),
                "description":     rule.get("rule_text", "")[:200],
                "source":          rule.get("source_document", ""),
                "links": [{
                    "label": rule.get("source_document", ""),
                    "url":   REGULATORY_CITATIONS.get(rule.get("source_document", ""), ""),
                }],
            }

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

    # Add CT-XXX entries for comparator trials from the result
        # These are separate from TI-XXX (instrument-level) entries
        # CT labels match STEP G's labelling of coverage["comparator_trials"]
        # NOTE: this function is called before build_coverage_matrix runs,
        # so we can only add them if raw_kg_records has trial data
        trial_map = {}
        for r in (raw_kg_records or []):
            tname = r.get("trial_name") or r.get("nct_id", "")
            if tname and tname not in trial_map:
                trial_map[tname] = r
        for ct_i, (tname, trec) in enumerate(list(trial_map.items())[:5], 1):
            ct_label = f"CT-{ct_i:03d}"
            nct = str(trec.get("nct_id", ""))
            fda = str(trec.get("fda_label_url", ""))
            ema = str(trec.get("ema_label_url", ""))
            drug = trec.get("drug_name", "")
            ct_links = []
            if nct.startswith("NCT"):
                ct_links.append({"label": f"ClinicalTrials.gov", "url": f"https://clinicaltrials.gov/study/{nct}"})
            if fda.startswith("http"):
                ct_links.append({"label": "FDA label", "url": fda})
            if ema.startswith("http"):
                ct_links.append({"label": "EMA label", "url": ema})
            citation_index[ct_label] = {
                "type": "comparator_trial",
                "trial": tname,
                "nct": nct,
                "drug": drug,
                "phase": trec.get("phase", ""),
                "links": ct_links,
            }

    return citation_index

# =============================================================================
# STEP 3: SCORING ENGINE
# =============================================================================

def score_evidence(context_json: dict, kg_records: list, instrument_meta=None,
                   raw_kg_records=None, lang_counts=None, recall_periods=None) -> list:
    """
    Score each instrument on a 0-100 scientific scale plus operational bonuses.
    All penalties are replaced by a structured Risk Flag System.
    Scientific score is never deducted — flags carry severity independently.
    """
    if instrument_meta is None:
        instrument_meta = {}
    if raw_kg_records is None:
        raw_kg_records = kg_records
    if lang_counts is None:
        lang_counts = {}
    if recall_periods is None:
        recall_periods = {}

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
    instrument_meta = instrument_meta or {}

    for record in kg_records:
        instrument_name  = str(record.get("instrument_name", "Unknown"))
        instrument_lower = instrument_name.lower()

        # --- Domain resolution: Instrument node is authoritative, TI fields are fallback ---
        _k = str(instrument_name or "").strip().lower()
        inst_node = (instrument_meta.get(_k) or
                instrument_meta.get(_k.split()[-1] if _k.split() else _k) or
                {})
        node_domains = _to_str(inst_node.get("domains", ""))
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

        recall_info = recall_periods.get(instrument_name, recall_periods.get(instrument_lower, {}))
        recall_period = recall_info.get("days", RECALL_PERIOD_UNKNOWN)
        recall_period_key = recall_info.get("citation", None)

        lang_info = lang_counts.get(instrument_name) or lang_counts.get(instrument_lower)
        if lang_info and lang_info.get("count") is not None:
            language_count = lang_info["count"]
        else:
            # Fallback: parse directly from KG record field
            languages_val = record.get("languages", "")
            languages_str = str(languages_val).lower()
            if "85" in languages_str or "100" in languages_str or "all major" in languages_str:
                language_count = 100
            elif isinstance(languages_val, list):
                language_count = len([l for l in languages_val if l])
            else:
                language_count = len([l for l in languages_str.split("|") if l.strip()])

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
            all_terms = list(dict.fromkeys(
                [domain] + _get_domain_synonyms(domain)
            ))
            if any(term in instrument_domains for term in all_terms):
                matched_domains.append(domain)
        for claim in tpp_claims:
            if claim not in matched_domains:
                all_terms = list(dict.fromkeys(
                    [claim] + _get_domain_synonyms(claim)
                ))
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
                "Need to verify"
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
        # MOA_KEYWORDS = {
        #     "bispecific":             ["cytokine release", "crs", "fatigue",
        #                                "neurotoxicity", "icans", "infection"],
        #     "car-t":                  ["cytokine release", "crs", "fatigue",
        #                                "neurotoxicity", "icans"],
        #     "proteasome inhibitor":   ["peripheral neuropathy", "neuropathy", "fatigue"],
        #     "ici":                    ["fatigue", "immune-related", "diarrhea",
        #                                "endocrine", "colitis"],
        #     "cdk4/6":                 ["fatigue", "nausea", "neutropenia"],
        #     "antibody drug conjugate":["nausea", "fatigue", "neuropathy", "alopecia"],
        #     "bcma":                   ["fatigue", "infection", "crs",
        #                                "neurotoxicity", "cytokine release"],
        # }
        # moa_required_domains = []
        # moa_matched_domains  = []
        # for class_key, tox_domains in MOA_KEYWORDS.items():
        #     if class_key in drug_class:
        #         moa_required_domains = tox_domains
        #         moa_matched_domains  = [t for t in tox_domains
        #                                  if t in instrument_domains]
        #         break

        # if moa_required_domains:
        #     coverage_ratio = (len(moa_matched_domains) /
        #                        len(moa_required_domains))
        #     missing_moa = [d for d in moa_required_domains
        #                     if d not in moa_matched_domains]
        #     if coverage_ratio >= 0.65:
        #         raw_score += 15
        #         flags.append(linkify_flag_citations(ensure_full_stop(
        #             f"MoA Sensitivity +15: Full coverage of mechanism-specific "
        #             f"toxicity domains for {drug_class} "
        #             f"({', '.join(moa_matched_domains)}) "
        #             "[FDA PFDD Guidance 1 (2017)]."
        #         )))
        #     elif moa_matched_domains:
        #         raw_score += 8
        #         flags.append(linkify_flag_citations(ensure_full_stop(
        #             f"MoA Sensitivity +8 (partial): Captures "
        #             f"{len(moa_matched_domains)} of "
        #             f"{len(moa_required_domains)} mechanism-specific domains "
        #             f"({', '.join(moa_matched_domains)}). "
        #             f"Missing: {', '.join(missing_moa)}. "
        #             "[FDA PFDD Guidance 1 (2017)]."
        #         )))
        #     else:
        #         flags.append(linkify_flag_citations(ensure_full_stop(
        #             f"MoA Sensitivity 0: No mechanism-specific toxicity domains "
        #             f"captured for {drug_class}. "
        #             f"Key missing domains: {', '.join(moa_required_domains[:4])}. "
        #             "[FDA PFDD Guidance 1 (2017)]."
        #         )))

        # raw_score = min(raw_score, 100)

        # NEW — replace the MOA_KEYWORDS dict + for loop with these two lines:
        moa_required_domains = _get_drug_toxicities(record)
        moa_matched_domains  = [
            t for t in moa_required_domains
            if _domain_matches_instrument(t, instrument_domains)
        ]

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

        # Cap AFTER all additive components including change detection (+5)
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

        langinfo = lang_counts.get(instrument_name) or lang_counts.get(instrument_lower)
        if langinfo and langinfo.get('count') is not None:
            language_count = langinfo['count']
        else:
            language_count = 0 
            
        # ── Translation coverage ──────────────────────────────────────────
        if language_count >= 50:
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"Language coverage: {instrument_name} has approximately "
                f"{language_count} validated translations — strong coverage "
                f"for a {geographic_footprint} trial. Verify specific language "
                f"availability for trial sites — see Table 5 "
                f"[FDA PRO Guidance 2009 Section IV.A]."
            )))
        elif language_count > 0:
            operational_bonus -= 5
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"Limited translation (-5 operational): {instrument_name} "
                f"has approximately {language_count} validated translations. "
                f"For a {geographic_footprint} trial, verify site-specific language "
                f"coverage — see Table 5. Commission additional translations if needed "
                f"(typically 6–12 months) [FDA PRO Guidance 2009 Section IV.A] "
                f"[ISPOR ePRO Task Force 2009]."
            )))
        else:
            operational_bonus -= 10
            flags.append(linkify_flag_citations(ensure_full_stop(
                f"No translation data (-10 operational): No translation "
                f"information available for {instrument_name}. Sonnet instructed "
                f"to verify via web search — see Table 5. Linguistically validated "
                f"translations are required for all trial languages "
                f"[FDA PRO Guidance 2009 Section IV.A]."
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
            "pro_position":       record.get("pro_position", ""),
            "key_finding":        record.get("key_finding", ""),
            "compliance_rate":    record.get("compliance_rate", ""),
            "assessment_schedule":record.get("assessment_schedule", ""),
            "publication_doi":    record.get("publication_doi", ""),
            "publication_year":    record.get("publication_year", ""),
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
            "sap_endpoint":       extract_pro_endpoints(record.get("trial_secondary_endpoints_raw")),
        })

    risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    results.sort(key=lambda x: (risk_order.get(x["risk_level"], 4),
                                 -x["scientific_score"]))
    return results

# =============================================================================
# STEP 4: BATTERY OPTIMIZER
# =============================================================================
def drug_class_relevance(trial: dict, context_json: dict) -> tuple:
    """
    Returns a sort key tuple — lower = more relevant.
    Priority:
      0: explicit alias/class match
      1: partial word overlap
      2: same patient population line
      3: recency
      4: stable trial-name tie-break
    """
    trial_class = _to_str(
        trial.get("drug_class")
        or trial.get("drugclassname")
        or trial.get("diseaseclassification")
        or ""
    ).lower()

    trial_pop = _to_str(trial.get("patient_population", "")).lower()

    drug_class = _to_str(context_json.get("drug_class", "")).lower()
    moa_aliases = [
        _to_str(x).lower()
        for x in context_json.get("moa_aliases", [])
        if _to_str(x).strip()
    ]

    class_terms = [drug_class] + moa_aliases
    class_terms = [t.strip() for t in class_terms if t and t.strip() and t != "unknown"]

    # Tier 0: alias / exact substring match either direction
    if any(term in trial_class or trial_class in term for term in class_terms if trial_class):
        class_score = 0
    else:
        # Tier 1: meaningful word overlap across all class terms
        req_words = set()
        for term in class_terms:
            req_words.update(w for w in re.split(r"[\s\-/]+", term) if len(w) > 3)

        trial_words = {w for w in re.split(r"[\s\-/]+", trial_class) if len(w) > 3}
        class_score = 1 if (req_words & trial_words) else 2

    # Tier 2: patient population similarity
    req_pop = _to_str(context_json.get("population_subtype", "")).lower()
    if req_pop and trial_pop:
        if req_pop in trial_pop or trial_pop in req_pop:
            pop_score = 0
        else:
            req_pop_words = {w for w in re.split(r"[\s\-/]+", req_pop) if len(w) > 3}
            trial_pop_words = {w for w in re.split(r"[\s\-/]+", trial_pop) if len(w) > 3}
            pop_score = 1 if (req_pop_words & trial_pop_words) else 2
    else:
        pop_score = 1

    # # Tier 3: recency
    # year_raw = str(trial.get("publication_year") or trial.get("year") or "0")
    # try:
    #     year = int(year_raw[:4])
    #     recency_score = -year
    # except ValueError:
    #     recency_score = 0

    return (class_score, pop_score, trial.get("trial_name", ""))

def build_coverage_matrix(scored: list, context_json: dict, raw_kg_records: list, instrument_meta = None,) -> dict:
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
    if instrument_meta is None:
        instrument_meta = {}

    indication = _to_str(context_json.get("indication"))
    hta_markets = context_json.get("hta_markets", [])  # NOT "hta_markets"
    drug_class = _to_str(context_json.get("drug_class"))
    core_domains = [str(d).lower() for d in context_json.get("core_domains_required", [])]
    _raw_extra = context_json.get('additional_domains', [])
    extra_domains = [str(d).lower() for d in (_raw_extra if isinstance(_raw_extra, list) else []) if d]
    all_domains = list(dict.fromkeys(core_domains + extra_domains))

    domain_coverage = []
    for domain in all_domains:
        d_key = re.sub(r'\s*\(.*?\)', '', domain).strip().lower()
        paren_extras = [p.strip().lower() for p in re.findall(r'\(([^)]+)\)', domain.lower())]
        all_terms = list(dict.fromkeys(
            [domain.lower(), d_key] + _get_domain_synonyms(d_key) + paren_extras
        ))
        candidates = []


        # Build full candidate pool: scored instruments + any instrument in raw KG records
        all_candidate_names = {inst["instrument_name"] for inst in scored}
        for r in raw_kg_records:
            name = str(r.get("instrument_name", "")).strip()
            if name:
                all_candidate_names.add(name)

        for inst_name in sorted(all_candidate_names):
            inst_lower = inst_name.lower()
            inst_scored = next((s for s in scored if s["instrument_name"] == inst_name), None)
            inst_node = instrument_meta.get(inst_lower) or instrument_meta.get(inst_lower.split()[-1] if inst_lower.split() else inst_lower, {})
            
            node_domains = _to_str((inst_node or {}).get('domains', ''))
            inst_records = [r for r in raw_kg_records if r.get('instrument_name') == inst_name]
            search_text = ' '.join(filter(None, [
                inst_lower,
                node_domains,
                *[_to_str(r.get('instrument_domain', '')) for r in inst_records],
                *[_to_str(r.get('domains_measured', '')) for r in inst_records],
                *[_to_str(r.get('key_finding', '')) for r in inst_records],
                *[_to_str(r.get('instrument_subscales_assessed', '')) for r in inst_records],
                *[_to_str(r.get('strengths', '')) for r in inst_records],
            ]))


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
                    "score": inst_scored["scientific_score"] if inst_scored else 0,
                    "risk": inst_scored["risk_level"] if inst_scored else "LOW",
                    "change_detected": change_detected,
                    "precedent_trial": precedent.get("trial_name", ""),
                    "precedent_nct": precedent.get("nct_id", ""),
                    "prevalence": inst_scored.get("_prevalence", 1) if inst_scored else 1,
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

            subscale_records = [
                r for r in raw_kg_records
                if r.get("instrument_name") == best_name
                and _is_calibrated_soa(r)
            ]

            if subscale_records:
                all_subscales = set()
                for r in subscale_records:
                    raw = str(r.get("instrument_subscales_assessed") or "").strip()
                    for p in re.split(r"[,;|/]", raw):
                        p = p.strip()
                        if p and p.lower() not in ("total", "full scale", "full instrument", ""):
                            all_subscales.add(p)

                subscale_list = sorted(all_subscales)
                trial_count   = len(subscale_records)

                if subscale_list:
                    item_library_note = (
                        f"Calibrated SOA flag: {trial_count} comparator trial(s) used specific subscales "
                        f"rather than the full {best_name} — "
                        f"{', '.join(subscale_list[:5])}{'...' if len(subscale_list) > 5 else ''}. "
                        f"Full instrument has {n_items} items. "
                        f"Per-subscale item counts and standalone validation: need to verify "
                        f"before deciding on a calibrated SOA approach."
                    )
                else:
                    item_library_note = (
                        f"Calibrated SOA flag: {trial_count} comparator trial(s) used a partial "
                        f"administration of {best_name} (subscale names not recorded in KG). "
                        f"Full instrument has {n_items} items. Need to verify."
                    )
            else:
                item_library_note = None


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
                "drug_class": r.get("drug_class_name") or r.get("disease_classification", ""),
                "nct_id": r.get("nctid", ""),
                "phase": r.get("phase", ""),
                # "year": r.get("publication_year", ""),
                "instruments": [],
            }
        comparator_map[trial_name]["instruments"].append({
            "name": r.get("instrument_name", ""),
            "role": r.get("endpoint_role") or r.get("pro_position", ""),
            "significance": r.get("significance", ""),
            "direction": r.get("direction", ""),
            "prespecified": r.get("prespecified", ""),
            "subscales": r.get("instrument_subscales_assessed", ""),
            "sap_endpoint": extract_pro_endpoints(r.get("trial_secondary_endpoints_raw")),
        })

    comparators = list(comparator_map.values())

    # Annotate each comparator with cross-class flag before sorting
    for comp in comparators:
        comp_class = _to_str(comp.get("drug_class", "")).lower()
        dc_words = {w for w in drug_class.lower().split() if len(w) > 3}
        comp_words = {w for w in comp_class.split() if len(w) > 3}
        comp["_cross_class"] = bool(comp_class and drug_class) and not (
            drug_class.lower() in comp_class or comp_class in drug_class.lower()
            or bool(dc_words & comp_words)
        )

    comparators.sort(key=lambda x: drug_class_relevance(x, context_json))

    # Pre-build subscale→domain maps for every unique instrument (one Haiku call each)
    # all_instrument_names_for_maps = {
    #     str(r.get("instrument_name", "")).strip()
    #     for r in raw_kg_records if r.get("instrument_name")
    # }
    # subscale_maps: dict = {}
    # for iname in all_instrument_names_for_maps:
    #     if not iname:
    #         continue
    #     assessed_vals = [
    #         r.get("instrument_subscales_assessed")
    #         for r in raw_kg_records
    #         if str(r.get("instrument_name", "")).strip() == iname
    #     ]
    #     subscale_maps[iname.lower()] = build_subscale_domain_map(iname, assessed_vals)

    # Build subscale→domain maps ONLY for comparator trial instruments + top scored
    # (not all raw_kg_records — that could be 150+ instruments and 150 Haiku calls)
    comparator_inst_names = {
        str(inst.get("name", "")).strip()
        for trial in comparators[:5]
        for inst in trial.get("instruments", [])
        if inst.get("name")
    }
    top_scored_names = {
        str(s.get("instrument_name", "")).strip()
        for s in scored[:10]
        if s.get("instrument_name")
    }
    names_for_maps = (comparator_inst_names | top_scored_names) - {""}

    subscale_maps: dict = {}
    for iname in names_for_maps:
        assessed_vals = [
            r.get("instrument_subscales_assessed")
            for r in raw_kg_records
            if str(r.get("instrument_name", "")).strip() == iname
        ]
        subscale_maps[iname.lower()] = build_subscale_domain_map(iname, assessed_vals)

    # Build the pre-computed trial-domain matrix for Table 1
    trial_domain_matrix = build_trial_domain_matrix(
        domain_coverage=domain_coverage,
        comparator_trials=comparators[:5],  # same slice as returned below
        raw_kg_records=raw_kg_records,
        subscale_maps=subscale_maps,
    )

    item_library_applicable = any(_is_calibrated_soa(r) for r in raw_kg_records)

    # # ── ★ NEW: Build subscale→domain maps (one Haiku call per instrument) ──
    # all_instrument_names_for_maps = {
    #     str(r.get("instrument_name", "")).strip()
    #     for r in raw_kg_records if r.get("instrument_name")
    # }
    # subscale_maps: dict = {}
    # for iname in all_instrument_names_for_maps:
    #     if not iname:
    #         continue
    #     assessed_vals = [
    #         r.get("instrument_subscales_assessed")
    #         for r in raw_kg_records
    #         if str(r.get("instrument_name", "")).strip() == iname
    #     ]
    #     subscale_maps[iname.lower()] = build_subscale_domain_map(iname, assessed_vals)

    # # ── ★ NEW: Pre-compute Table 1 cells ────────────────────────────────────
    # trial_domain_matrix = build_trial_domain_matrix(
    #     domain_coverage=domain_coverage,
    #     comparator_trials=comparators[:5],
    #     raw_kg_records=raw_kg_records,
    #     subscale_maps=subscale_maps,
    # )

    # Diagnose 
    logging.info(f"Domain '{domain}' | allterms: {all_terms} | inst '{inst_name}' domains: '{all_domains[:100]}'")

    return {
        "domains": domain_coverage,
        "comparator_trials": comparators[:5],
        "hta_mandatory": htamandatory,
        "item_library_applicable": item_library_applicable,
        "all_candidates": scored[:8],
        "trial_domain_matrix":   trial_domain_matrix,
    }

def build_pro_measures_table(
    coverage: dict,
    inst_refs: list,
    raw_kg_records: list,
    context_json: dict,
) -> list:
    """
    Build a PRO measures comparison table (Table 2) purely in Python.

    Returns a list of row dicts with keys:
      - trial: trial name
      - drug: drug name
      - drug_class: mechanism / class
      - pro_measures: "Instrument1 (n=30), Instrument2 (n=20)..."
      - assessment_schedule: timepoints from KG (e.g. "C1D1, C2D1, then every 3 cycles") or "NR"
      - total_items: int or None
      - est_time_min: float or None
    """

    # def _norm(name: str) -> str:
    #     return (name or "").strip().lower()
    
    def _norm(name: str) -> str:
        return re.sub(r'[\s/\-_\.]+', '', (name or "").strip().lower())

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

    # # Index raw KG rows by trial name / NCT for year lookup
    # trial_years = {}
    # for r in rawkgrecords or []:
    #     tname = r.get("trialname") or r.get("trial_name") or r.get("nctid") or r.get("nct_id")
    #     if not tname:
    #         continue
    #     year = r.get("publicationyear") or r.get("publication_year")
    #     if year and tname not in trial_years:
    #         trial_years[tname] = year

    rows = []

    # Comparator trials from coverage
    for trial in coverage.get("comparator_trials", []):
        tname = trial.get("trialname") or trial.get("trial_name") or trial.get("nctid") or trial.get("nct_id") or "Unknown trial"
        drug = trial.get("drug", "")
        dclass = trial.get("drugclass") or trial.get("drug_class") or ""
        # year = trial_years.get(tname, "Not reported")

        pro_parts = []
        total_items = 0
        total_time = 0.0
        has_items = False
        has_time = False

        # schedule_vals = [
        #     r.get("assessment_schedule") or r.get("assessmentschedule") or ""
        #     for r in rawkgrecords or []
        #     if (r.get("trial_name") or r.get("trialname") or r.get("nct_id") or r.get("nctid")) == tname
        #     and (r.get("assessment_schedule") or r.get("assessmentschedule"))
        # ]
        # schedule_str = schedule_vals[0] if schedule_vals else "NR"

        # Populate instrument list from the trial's instrument records
        for inst in trial.get("instruments", []):
            inst_name = inst.get("name", "")
            if not inst_name:
                continue
            ref = ref_index.get(_norm(inst_name), {})
            items = ref.get("total_items") or ref.get("totalitems")
            admintime = ref.get("admin_time") or ref.get("admintime")
            label = inst_name
            if items and str(items).strip() not in ("", "NR", "nan", "None"):
                try:
                    n = int(str(items).strip())
                    label = f"{inst_name} (n={n})"
                    total_items += n
                    has_items = True
                except ValueError:
                    label = f"{inst_name} (n=?)"
            else:
                label = f"{inst_name} (n=?)"
            if admintime and str(admintime).strip() not in ("", "NR", "nan", "None"):
                try:
                    total_time += float(str(admintime).strip())
                    has_time = True
                except ValueError:
                    pass
            pro_parts.append(label)

        schedule_vals = [
            r.get("assessment_schedule") or r.get("assessmentschedule")
            for r in (raw_kg_records or [])
            if (r.get("trial_name") or r.get("nct_id") or "") == tname
        ]
        schedule_vals = [v for v in schedule_vals if v and str(v).strip()]
        schedule_str = schedule_vals[0] if schedule_vals else "NR"

        row = {
            "trial":          tname,
            "drug":           drug,
            "drugclass":      dclass,
            "pro_measures":   ", ".join(pro_parts) if pro_parts else "NR",
            "assessment_schedule": schedule_str,
            "total_items":    total_items if has_items else "NR",
            "est_time_min":   round(total_time, 1) if has_time else "NR",
        }

        rows.append(row)

    # Current trial row (Proposed)
    tname = "Current Trial (Proposed)"
    # year = "TBD"  # To ensure AI doesn't randomly assign a year
    drug = f"Novel {context_json.get('drugclass') or context_json.get('drug_class') or 'regimen'}"
    dclass = context_json.get("drugclass") or context_json.get("drug_class") or ""

    rows.append({
        "trial": tname,
        # "year": year,
        "drug": drug,
        "drug_class": dclass,
        "pro_measures": "TBD — expert decision required",
        "assessment_schedule": "TBD",
        "total_items": "TBD",
        "est_time_min": "TBD",
    })
    return rows

# def build_gap_analysis(
#     scored: list,
#     instrument_meta: dict,
#     reg_records: list,
#     context_json: dict,
#     top_n: int = 5,
# ) -> list:
#     """
#     Build a gap analysis table for the top-N instruments.

#     Returns a list of dicts with keys:
#       - instrument
#       - content_validity
#       - validation_evidence
#       - mcid_evidence
#       - regulatory_acceptance
#       - known_gaps
#       - fit_for_purpose
#       - score
#       - risk_level
#     """

#     def _norm(s: str) -> str:
#         return (s or "").strip().lower()

#     def _reg_hits(name: str) -> list:
#         n = _norm(name)
#         return [
#             f"{r.get('agency', '')} {r.get('decision', '')}"
#             for r in (reg_records or [])
#             if n in str(r.get("instruments_accepted") or r.get("instruments_reviewed", "")).lower()
#         ]

#     trial_population = _to_str(context_json.get("population_subtype", "")).strip() or "this population"

#     rows = []
#     for inst in (scored or [])[:top_n]:
#         name = inst.get("instrument_name") or inst.get("instrumentname") or "Unknown"
#         key  = name.strip().lower()
#         node = instrument_meta.get(key) or instrument_meta.get(name, {})

#         # --- Content validity ---
#         val_status = _to_str(
#             node.get("validation_evidence")
#             or node.get("validation")
#             or inst.get("validation_status")
#         ).strip() or "Not described"

#         validation_evidence = _to_str(
#             node.get("validation_evidence") or inst.get("validation_evidence")
#         ).strip()
#         validation_evidence = validation_evidence[:300] if validation_evidence else "NR"

#         # --- MCID ---
#         raw_mcid = node.get("mcid") or inst.get("mcid") or ""
#         try:
#             mcid_short, _ = clean_mcid(raw_mcid)
#         except Exception:
#             mcid_short = str(raw_mcid)[:80] if raw_mcid else ""
#         mcid_display = mcid_short or "Not established — verify via PROQOLID"

#         # --- Regulatory acceptance ---
#         reg_node = _to_str(node.get("regulatory_acceptance") or node.get("regulatoryAcceptance")).strip()
#         reg_hits = _reg_hits(name)
#         if reg_hits and reg_node:
#             reg_text = f"{reg_node} | KG reviews: " + "; ".join(reg_hits)
#         elif reg_hits:
#             reg_text = "KG reviews: " + "; ".join(reg_hits)
#         elif reg_node:
#             reg_text = reg_node
#         else:
#             reg_text = "No KG precedent found"

#         # --- Known gaps ---
#         limitations = _to_str(node.get("limitations") or inst.get("limitations")).strip()
#         known_gaps  = limitations[:250] if limitations else "No specific limitations recorded in KG"

#         # --- Fit for purpose with population clause ---
#         pop_raw = _to_str(
#             node.get("patient_population") or node.get("population")
#         ).strip()
#         validated_in = (
#             f"validated in {pop_raw}"
#             if pop_raw and pop_raw.lower() not in ("nr", "unknown", "")
#             else f"no {trial_population}-specific validation on record"
#         )

#         score = inst.get("scientific_score", 0)
#         risk  = inst.get("risk_level", "LOW")

#         if score >= 65 and risk not in ("CRITICAL", "HIGH"):
#             fit = f"Likely fit — {validated_in}"
#         elif score >= 40 and risk != "CRITICAL":
#             # Surface the most severe flag as the specific condition
#             top_flag = next(
#                 (
#                     f.split("—")[-1].strip()
#                     for f in inst.get("flags", [])
#                     if any(k in f.upper() for k in ("CRITICAL", "HIGH", "FLAG"))
#                 ),
#                 validated_in,
#             )
#             fit = f"Conditionally fit — {top_flag}"
#         else:
#             fit = f"Evidence gaps — {validated_in}"

#         rows.append({
#             "instrument":          name,
#             "content_validity":    val_status,
#             "validation_evidence": validation_evidence,
#             "mcid_evidence":       mcid_display,
#             "regulatory_acceptance": reg_text,
#             "known_gaps":          known_gaps,
#             "fit_for_purpose":     fit,
#             "score":               score,
#             "risk_level":          risk,
#         })

#     return rows

def build_gap_analysis(scored: list, instrument_meta: dict, reg_records: list, context_json: dict, top_n: int = 5) -> list:
    """
    Build a gap analysis table for the top-N instruments.
    Returns a list of dicts with keys:
      - instrument
      - content_validity       (from i.validation_status — was it validated and in what population?)
      - validation_evidence    (from i.validation_evidence — key psychometric stats, most population-specific citation)
      - regulatory_acceptance  (from RegulatoryReview nodes — truncated, with RR-XXX label)
      - known_gaps             (from i.limitations — single most important limitation)
      - fit_for_purpose        (computed tier + population clause)
      - score
      - risk_level
    """

    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _reg_hits(name: str) -> list:
        n = _norm(name)
        hits = []
        for idx, r in enumerate(reg_records or [], 1):
            if n in str(r.get("instruments_accepted") or r.get("instruments_reviewed", "")).lower():
                hits.append(
                    f"[RR-{idx:03d}] {r.get('agency', '')} {r.get('decision', '')} — {r.get('drug_name', '')}".strip()
                )
        return hits

    def _extract_validation_evidence(raw: str) -> str:
        """
        Extract the most population-specific psychometric stats from validation_evidence string.
        Returns a short formatted string like: ICC=0.68–0.89, α=0.73–0.89 (Cocks 2007, n=200 MM)
        Falls back to a cleaned short summary if no stats found.
        """
        if not raw or not raw.strip():
            return "NR"

        raw = re.sub(r'^\[|\]$', '', raw.strip())
        segments = [s.strip() for s in re.split(r'\|', raw) if s.strip()]

        # Prefer segments mentioning the indication population
        indication_terms = [
            _norm(context_json.get("indication", "")),
            *[_norm(s) for s in context_json.get("indication_synonyms", [])],
            "mm", "myeloma", "rrmm",
        ]
        pop_segments = [s for s in segments if any(t in s.lower() for t in indication_terms if t)]
        search_segments = pop_segments if pop_segments else segments

        # Extract stats from the best segments
        icc   = None
        alpha = None
        es    = None

        for seg in search_segments:
            if not icc:
                m = re.search(r'ICC[=\s]*([\d.]+(?:[-–][\d.]+)?)', seg, re.IGNORECASE)
                if m:
                    icc = m.group(1)
            if not alpha:
                m = re.search(r'alpha[=\s]*([\d.]+(?:[-–][\d.]+)?)', seg, re.IGNORECASE)
                if m:
                    alpha = m.group(1)
            if not es:
                m = re.search(r'\bES[=\s]*([\d.]+(?:[-–][\d.]+)?)', seg, re.IGNORECASE)
                if m:
                    es = m.group(1)

        parts = []
        if icc:
            parts.append(f"ICC={icc}")
        if alpha:
            parts.append(f"α={alpha}")
        if es:
            parts.append(f"ES={es}")

        # Extract the most population-specific author-year citation
        # Multi-word author support: van Andel, EuroQol, de Vet etc.
        citation_candidates = re.findall(r'([A-Z][A-Za-z]+(?:\s[A-Z][a-z]+)?\s\d{4})', raw)
        citation = citation_candidates[0] if citation_candidates else None

        # Prefer a citation from a population-matched segment
        for seg in pop_segments:
            c = re.findall(r'([A-Z][A-Za-z]+(?:\s[A-Z][a-z]+)?\s\d{4})', seg)
            if c:
                citation = c[0]
                break

        # Population hint
        pop_hint = None
        m = re.search(r'n=(\d+)\s*(MM|RRMM|cancer|MPN|NSCLC)?', raw, re.IGNORECASE)
        if m:
            pop_hint = f"n={m.group(1)}"
            if m.group(2):
                pop_hint += f" {m.group(2).upper()}"

        if parts:
            suffix = ""
            if citation:
                suffix = f" ({citation}"
                if pop_hint:
                    suffix += f", {pop_hint}"
                suffix += ")"
            return ", ".join(parts) + suffix

        # No stats — return a short cleaned fallback (validity keywords only)
        keywords = ['validity', 'reliability', 'responsiveness', 'confirmed', 'validated']
        useful = [s for s in segments if any(k in s.lower() for k in keywords)]
        fallback = "; ".join(useful[:2])
        return (fallback[:120] + "…") if len(fallback) > 120 else (fallback or "NR")

    trial_population = _to_str(context_json.get("population_subtype")).strip() or "this population"
    indication_terms = [
        _norm(context_json.get("indication", "")),
        *[_norm(s) for s in context_json.get("indication_synonyms", [])],
    ]

    rows = []
    for inst in (scored or [])[:top_n]:
        name  = inst.get("instrument_name") or inst.get("instrument_name") or "Unknown"
        key   = name.strip().lower()
        node  = instrument_meta.get(key) or instrument_meta.get(key.split()[-1] if key.split() else key, {})

        # --- Content validity ---
        val_status = _to_str(
            node.get("validation") or inst.get("validation_status")
        ).strip() or "Not described"
        content_validity = val_status.capitalize()

        # --- Validation evidence (psychometric stats) ---
        raw_val_evidence = _to_str(
            node.get("validationevidence") or node.get("validation_evidence") or inst.get("validation_evidence")
        ).strip()
        validation_evidence = _extract_validation_evidence(raw_val_evidence)

        # --- Regulatory acceptance (truncated, with RR-XXX tags handled by Sonnet) ---
        reg_node = _to_str(node.get("regulatory_acceptance") or node.get("regulatoryacceptance")).strip()
        reg_hits = _reg_hits(name)
        if reg_hits and reg_node:
            reg_text = f"{reg_node} [KG reviews: {'; '.join(reg_hits[:2])}]"
        elif reg_hits:
            reg_text = f"KG reviews: {'; '.join(reg_hits[:2])}"
        elif reg_node:
            reg_text = reg_node
        else:
            reg_text = "No KG precedent found"
        # Truncate cleanly at word boundary
        if len(reg_text) > 150:
            reg_text = reg_text[:150].rsplit(' ', 1)[0] + "…"
        regulatory_acceptance = reg_text.capitalize()

        # --- Known gaps (single most important limitation) ---
        limitations = _to_str(node.get("limitations") or inst.get("limitations")).strip()
        if limitations:
            # Take only the first sentence or first 200 chars
            first_sentence = re.split(r'(?<=[.!?])\s', limitations)[0]
            known_gaps = (first_sentence[:200].rsplit(' ', 1)[0] + "…") if len(first_sentence) > 200 else first_sentence
        else:
            known_gaps = "No specific limitations recorded in KG"
        known_gaps = known_gaps.capitalize()

        # --- Fit for purpose with population clause ---
        pop_raw = _to_str(node.get("patient_population") or node.get("population")).strip()
        validated_in = f"validated in {pop_raw}" if pop_raw and pop_raw.lower() not in ("nr", "unknown", "") \
                       else f"no {trial_population}-specific validation on record"
        score = inst.get("scientific_score", 0)
        risk  = inst.get("risk_level", "LOW")

        if score >= 65 and risk not in ("CRITICAL", "HIGH"):
            fit = f"Likely fit — {validated_in}"
        elif score >= 40 and risk != "CRITICAL":
            top_flag = next(
                (f.split("—")[-1].strip() for f in inst.get("flags", [])
                 if any(k in f.upper() for k in ("CRITICAL", "HIGH", "FLAG"))),
                validated_in
            )
            fit = f"Conditionally fit — {top_flag}"
        else:
            fit = f"Evidence gaps — {validated_in}"

        rows.append({
            "instrument":           name,
            "content_validity":     content_validity,
            "validation_evidence":  validation_evidence,
            "regulatory_acceptance": regulatory_acceptance,
            "known_gaps":           known_gaps,
            "fit_for_purpose":      fit,
            "score":                score,
            "risk_level":           risk,
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
        # role_raw = (
        #     r.get("pro_position")
        #     or r.get("endpointrole")
        #     or r.get("endpoint_role")
        #     or r.get("proposition")
        #     or ""
        # )
        role_raw = (r.get("endpoint_role") or r.get("pro_position") or 
            r.get("endpointrole") or r.get("role") or "")
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

# def build_pro_endpoint_table(raw_kg_records: list, scored: list, top_n: int = 8) -> list:
#     """
#     Build PRO endpoint SAP language table for top-N scored instruments.
#     One row per trial-instrument record. Replaces TABLE 4 (Endpoint Positioning).
#     Endpoint positioning counts are still computed by build_endpoint_positioning()
#     and passed to Sonnet for use in the Recommendation column of this table.
#     """
#     ROLE_ORDER = {"primary": 0, "secondary": 1, "exploratory": 2}
#     top_names = {
#         s["instrument_name"]
#         for s in (scored or [])[:top_n]
#         if s.get("instrument_name")
#     }
#     score_map = {
#         s["instrument_name"]: s.get("scientific_score", 0)
#         for s in (scored or [])
#     }
#     rows = []
#     for r in (raw_kg_records or []):
#         name = str(r.get("instrument_name") or "").strip()
#         if name not in top_names:
#             continue
#         role_raw = str(r.get("endpoint_role") or r.get("pro_position") or "").strip()
#         role_norm = role_raw.lower()
#         rows.append({
#             "instrument":    name,
#             "role":          role_raw or "NR",
#             "_role_order":   ROLE_ORDER.get(
#                                  next((k for k in ROLE_ORDER if k in role_norm), "other"), 3
#                              ),
#             "prespecified":  str(r.get("prespecified") or "NR").strip(),
#             "sap_endpoint":  extract_pro_endpoints(r.get("trial_secondary_endpoints_raw"))
#             "subscales":     str(r.get("instrument_subscales_assessed") or "").strip(),
#             "key_finding":   str(r.get("key_finding") or "").strip(),
#             "significance":  str(r.get("significance") or "").strip(),
#             "p_value":       str(r.get("p_value") or "").strip(),
#             "effect_size":   str(r.get("effect_size") or "").strip(),
#             "trial":         str(r.get("trial_name") or r.get("nct_id") or "").strip(),
#             "drug":          str(r.get("drug_name") or "").strip(),
#             # "year":          str(r.get("publication_year") or "").strip(),
#         })
#     rows.sort(key=lambda x: (
#         -score_map.get(x["instrument"], 0),
#         x["_role_order"],
#     ))
#     for r in rows:
#         r.pop("_role_order", None)
#     return rows

def extract_sap_precursor(raw_key_finding: str) -> str:
    """Extract analysis-rich text for SAP precedent (bulletproof version)."""
    if not raw_key_finding or raw_key_finding == "Key findings not reported.":
        return "—"
    
    # Prioritize sentences with analysis terms
    analysis_terms = {"hr ", "p<", "p=", "median ", "change from", "proportion"}
    sentences = re.split(r'[.;]', raw_key_finding)
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20 and any(term in sent.lower() for term in analysis_terms):
            return sent[:200].strip()
    
    # Fallback: first 200 chars
    return raw_key_finding[:200].strip() or "—"

def build_pro_endpoint_table(raw_kg_records: list, scored: list, top_n: int = 8) -> list:
    """
    Build PRO endpoint SAP language table for top-N scored instruments.
    SAP language = analysis-rich excerpt from key_finding_instrument (bulletproof).
    """
    ROLE_ORDER = {"primary": 0, "secondary": 1, "exploratory": 2}
    top_names = {
        s.get("instrument_name")
        for s in (scored or [])[:top_n]
        if s.get("instrument_name")
    }
    score_map = {
        s.get("instrument_name"): s.get("scientific_score", 0)
        for s in (scored or [])
    }
    
    rows = []
    for r in (raw_kg_records or []):
        name = str(r.get("instrument_name") or "").strip()
        if name not in top_names:
            continue
            
        role_raw = str(r.get("endpoint_role") or r.get("pro_position") or "").strip()
        role_norm = role_raw.lower()
        
        raw_kf = str(r.get("key_finding_instrument") or r.get("keyfinding") or "")
        sap_precursor = extract_sap_precursor(raw_kf)
        
        rows.append({
            "instrument":    name,
            "role":          role_raw or "NR",
            "_role_order":   ROLE_ORDER.get(
                                 next((k for k in ROLE_ORDER if k in role_norm), "other"), 3
                             ),
            "prespecified":  str(r.get("prespecified") or "NR").strip(),
            "sap_endpoint":  sap_precursor,  # Ready for table display
            "subscales":     str(r.get("instrument_subscales_assessed") or "").strip(),
            "key_finding":   raw_kf[:100].strip(),
            "significance":  str(r.get("significance") or "").strip(),
            "p_value":       str(r.get("pvalue") or "").strip(),
            "effect_size":   str(r.get("effectsize") or "").strip(),
            "trial":         str(r.get("trial_name") or r.get("nct_id") or "").strip(),
            "drug":          str(r.get("drug_name") or "").strip(),
        })
    
    rows.sort(key=lambda x: (
        -score_map.get(x["instrument"], 0),
        x["_role_order"],
    ))
    for r in rows:
        r.pop("_role_order", None)
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
    for i, r in enumerate(records[:5]):  # 
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
                               source_records: list) -> list:
    """
    From all source_records for this indication, use Haiku to identify
    relevant competitors, assess comparability, and generate PRO implications.
    Returns list of enriched competitor profile dicts.
    """
    all_drugs = list({r.get("drug_name", "") for r in source_records
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

    # Build lookup: drug name → first record for that drug
    reg_by_drug = {}
    for r in source_records:
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
            "agency":                 rec.get("agency", rec.get("drug_class_name", "")),
            "decision":               rec.get("decision", rec.get("significance", "")),
            "instruments":            str(rec.get("instruments_accepted", "") or
                                         rec.get("instrument_name", "") or ""),
            "claim_type":             str(rec.get("claim_type", "") or ""),
            "rejection":              str(rec.get("rejection_reason_primary", "") or ""),
            "fda_url":                fda,
            "ema_url":                ema,
            "trial_name":             rec.get("trial_name", ""),
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
    unique_instrument_names: list = []

    try:
        search_terms = list(dict.fromkeys([indication] + synonyms[:3]))
        raw_kg_records = []

        for term in search_terms:
            rows = get_instruments_by_indication(indication=term, phase=phase, endpoint="")
            raw_kg_records.extend(rows)

        # ADD dedupe (line ~1065):
        seen = set()
        deduped_raw = []
        for r in raw_kg_records:
            key = (r.get("instrument_name"), r.get("trial_name") or r.get("nct_id"))
            if key not in seen:
                seen.add(key)
                deduped_raw.append(r)
        raw_kg_records = deduped_raw  # Single source of truth

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
                key = name.strip().lower()                    # ← normalise to lowercase
                instrument_meta[key] = refs[0]
                # Also index by last token (abbreviation), e.g. "qlq-c30" from "eortc qlq-c30"
                tokens = key.split()
                if len(tokens) > 1:
                    abbrev = tokens[-1]                       # e.g. "qlq-c30"
                    if abbrev not in instrument_meta:
                        instrument_meta[abbrev] = refs[0]

        # 5b) Haiku domain enrichment — only for instruments with no KG domain data
        for name in unique_instrument_names:
            key = name.strip().lower()
            node = instrument_meta.get(key, {})
            if not _to_str(node.get("domains", "")).strip():   # only act on thin nodes
                hint_record = next(
                    (r for r in raw_kg_records if r.get("instrument_name") == name),
                    {}
                )
                context_hint = " ".join(filter(None, [
                    _to_str(hint_record.get("instrument_domain")),
                    _to_str(hint_record.get("domains_measured")),
                    _to_str(hint_record.get("key_finding", ""))[:120],
                ]))
                classified = classify_instrument_domains_haiku(name, context_hint)
                if classified:
                    if key not in instrument_meta:
                        instrument_meta[key] = {}
                    instrument_meta[key]["domains"] = classified
                    logging.info(f"Haiku domain enrichment: {name} → {classified}")

        # 6) Regulatory evidence: deduplicate by review_id
        all_reg = []
        for term in search_terms:
            rows = get_regulatory_evidence(indication=term, agency="")
            if rows:
                all_reg.extend(rows)

        reg_rules = get_regulatory_rules(indication="", lifecycle_stage="", decision_type="") or []
        STRATEGY_STAGES = {"Instrument_Selection", "Protocol_Design", "Concept_Selection"}
        reg_rules = [r for r in reg_rules if r.get("lifecycle_stage", "") in STRATEGY_STAGES]

        seen_ids: set[str] = set()
        for r in all_reg:
            rid = r.get("review_id") or f"{r.get('drug_name','')}|{r.get('agency','')}|{r.get('decision','')}"
            if rid not in seen_ids:
                seen_ids.add(rid)
                reg_records.append(r)

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

    must_rules   = [r for r in reg_rules if _to_str(r.get("decision_type")) == "must"]
    should_rules = [r for r in reg_rules if _to_str(r.get("decision_type")) == "should"]

    logging.info(
        f"Regulatory rules: {len(reg_rules)} total "
        f"({len(must_rules)} MUST, {len(should_rules)} SHOULD)."
    )

    # --- STEP B.5: Build recall_periods and lang_counts for scoring engine ---
    recall_periods: dict = {}
    _lang_conn = None
    try:
        _lang_conn = _get_conn()
        for name in unique_instrument_names:
            recall_periods[name] = get_recall_period(name, _lang_conn)
    except Exception as e:
        logging.warning(f"Recall period build failed: {e}")
        recall_periods = {}
    finally:
        if _lang_conn:
            try:
                _lang_conn.close()
            except Exception:
                pass

    # Cap language count web searches to top 20 unique instruments by KG prevalence.
    # The full unique_instrument_names can be 100-150 names; calling web search for
    # each one not in KG costs ~$0.008/instrument and dominates total run cost.
    _prevalence_order = sorted(
        unique_instrument_names,
        key=lambda n: sum(1 for r in raw_kg_records if r.get("instrument_name") == n),
        reverse=True
    )
    _lang_names_capped = _prevalence_order[:20]

    lang_counts: dict = {}
    _lang_conn2 = None
    try:
        _lang_conn2 = _get_conn()
        lang_counts = get_language_counts(
            instrument_names=_lang_names_capped,
            graph_client=_lang_conn2
        )
    except Exception as e:
        logging.warning(f"Language counts build failed: {e}")
        lang_counts = {}
    finally:
        if _lang_conn2:
            try:
                _lang_conn2.close()
            except Exception:
                pass

    # --- STEP C: Score instruments ---
    scored = score_evidence(context_json, kg_records, instrument_meta, raw_kg_records,
                        lang_counts=lang_counts, recall_periods=recall_periods)

    # if indication and indication.lower() != "unknown":
    #     citation_index = build_tier1_citation_index(indication, phase, scored, raw_kg_records)
    #     result["citation_index"] = citation_index
    # else:
    #     result["citation_index"] = {}

    # --- STEP D: Build coverage matrix (replaces battery optimizer) ---
    coverage = build_coverage_matrix(
        scored,
        context_json,
        raw_kg_records,              
        instrument_meta=instrument_meta
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
        # Use kg_records (TrialInstrument data) not reg_records (RegulatoryReview data).
        # kg_records contains the actual trial drugs (carfilzomib, ixazomib, bortezomib)
        # which ARE the PI competitors. reg_records contains the regulatory review drugs
        # (lenalidomide, cyclophosphamide) which are NOT mechanism-relevant.
        competitor_profiles = build_competitor_profiles(
            indication,
            context_json.get("drug_class", "Unknown"),
            kg_records          # ← was reg_records, now kg_records
        )
    except Exception as e:
        logging.warning(f"Competitor analysis step failed: {e}")

     # --- STEP E.2 Build PRO measures comparison table in Python ---
    pro_measures_table = []
    try:
        pro_measures_table = build_pro_measures_table(
            coverage=coverage,
            inst_refs=inst_refs,
            raw_kg_records=raw_kg_records,
            context_json=context_json,
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
    # Filter to instruments with score >= 40 OR used in comparator trials
    _comparator_inst_names = {
        i.get("name", "")
        for t in coverage.get("comparator_trials", [])
        for i in t.get("instruments", [])
        if i.get("name")
    }
    filtered_scores = [
        s for s in scored
        if s.get("scientific_score", 0) >= 40
        or s.get("instrument_name", "") in _comparator_inst_names
    ]
    dropped = [s["instrument_name"] for s in scored if s not in filtered_scores]
    if dropped:
        logging.info(f"Filtered out low-scoring instruments: {dropped}")

    pro_endpoint_table = []
    try:
        pro_endpoint_table = build_pro_endpoint_table(raw_kg_records, filtered_scores, top_n=5)
    except Exception as e:
        logging.error(f"build_pro_endpoint_table failed: {e}")
    
    gap_analysis = []
    try:
        gap_analysis = build_gap_analysis(
            scored=filtered_scores,
            instrument_meta=instrument_meta,
            reg_records=reg_records,
            context_json=context_json,
            top_n=8,
        )
    except Exception as e:
        logging.error(f"build_gap_analysis failed: {e}", exc_info=True)

    
    # --- ENDPOINT POSITIONING TABLE ---
    endpoint_positioning = []
    try:
        endpoint_positioning = build_endpoint_positioning(
            raw_kg_records=raw_kg_records,
            scored=filtered_scores,
            top_n=5,
        )
    except Exception as e:
        logging.error(f"build_endpoint_positioning failed: {e}", exc_info=True)

    # --- STEP F.5: Build lang_block for Sonnet evidence block (Table 5) ---
    try:
        lang_lines = ["LANGUAGE TRANSLATION DATA (use for Table 5 — Validated Translations column):"]
        for instr, info in lang_counts.items():
            count = info.get("count")
            citation = info.get("citation", "[unverified]")
            warning = info.get("warning", "")
            count_str = f"~{count} validated translations" if count else "count unverified — do not write a number"
            line = f"- {instr}: {count_str} | {citation}"
            if warning:
                line += f" | {warning}"
            lang_lines.append(line)
        lang_block = "\n".join(lang_lines)
    except Exception as e:
        logging.warning(f"lang_block build failed: {e}")
        lang_block = "LANGUAGE TRANSLATION DATA: Unavailable."

    # --- STEP F.7: HTA preferences ---
    try:
        conn = _get_conn()
        hta_data = get_hta_preferences(
            markets=context_json.get("hta_markets", []),
            graph_client=conn
        )
        conn.close()
    except Exception as e:
        logging.warning(f"HTA preferences build failed: {e}")
        hta_data = {}

    # --- STEP F.8: Geographic language requirements ---
    try:
        conn = _get_conn()
        geo_lang_reqs = get_geographic_language_requirements(
            geographic_footprint=context_json.get("geographic_footprint", "Global"),
            graph_client=conn
        )
        conn.close()
    except Exception as e:
        logging.warning(f"Geographic language requirements build failed: {e}")
        geo_lang_reqs = {}


    # --- STEPG: Build structured evidence block for Sonnet ---
    # Build citation index here, where it can actually be used
    if indication and indication.lower() != "unknown":
        citation_index = build_tier1_citation_index(indication, phase, scored, raw_kg_records)
    else:
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
        kg_block_lines.append("TABLE 1 — PRE-COMPUTED DOMAIN COVERAGE")
        kg_block_lines.append(
            "Each cell is pre-computed. Copy verbatim — do NOT infer, upgrade, or change any state."
        )
        kg_block_lines.append(
            "States: REPORTED=outcome in KG | COLLECTED_NR=collected but result NR in KG "
            "| SUBSCALES_NR=instrument used but which subscales unknown | NOT_COLLECTED=not in trial"
        )
        kg_block_lines.append("")

        # trial_names = [t.get("trial_name", "") for t in coverage.get("comparator_trials", [])]

        trial_names = [t.get("trial_name", "") for t in coverage.get("comparator_trials", [])]
        trial_label_map = {
            t.get("trial_name", ""): f"CT-{i:03d}"
            for i, t in enumerate(coverage.get("comparator_trials", []), 1)
        }

        for row in coverage.get("trial_domain_matrix", []):
            domain     = row["domain"]
            is_fda_core = row["is_fda_core"]
            fda_tag    = " [FDA CORE]" if is_fda_core else ""
            kg_block_lines.append(f"DOMAIN: {domain}{fda_tag}")

            # Current Trial Candidates — driven by Instrument.domains (capability)
            cand_entry = next((d for d in coverage["domains"] if d["domain"] == domain), None)
            if cand_entry:
                cands = ", ".join(c["instrument"] for c in cand_entry.get("candidates", [])[:3])
                kg_block_lines.append(f"  Current Trial Candidates: {cands or 'None scored'}")

            # Comparator trial cells
            for tname in trial_names:
                cells = row["trials"].get(tname)
                if not cells:
                    kg_block_lines.append(f"  [{tname}]: NOT_COLLECTED — no instrument covers this domain in trial")
                else:
                    for cell in cells:
                        iname    = cell["instrument"]
                        state    = cell["state"]
                        change   = cell["change"]
                        subscale = cell.get("subscale") or ""
                        sub_note = f" (subscale: {subscale})" if subscale else ""
                        
                        ct_label = trial_label_map.get(tname, tname)
                        if state == "REPORTED":
                            kg_block_lines.append(
                                f"  [{ct_label}] {iname}{sub_note}: REPORTED — Change: {change}"
                            )
                        elif state == "COLLECTED_NR":
                            kg_block_lines.append(
                                f"  [{ct_label}] {iname}{sub_note}: COLLECTED_NR — "
                                f"subscale collected but per-subscale outcome not in KG; verify SAP"
                            )
                        elif state == "SUBSCALES_NR":
                            kg_block_lines.append(
                                f"  [{ct_label}] {iname}: SUBSCALES_NR — "
                                f"instrument used but assessed subscales not recorded in KG; verify SAP"
                            )
                        elif state == "NOT_COLLECTED":
                            kg_block_lines.append(
                                f"  [{ct_label}] {iname}: NOT_COLLECTED — "
                                f"this domain's subscale not in trial's assessed subscale set"
                            )

                        # if state == "REPORTED":
                        #     kg_block_lines.append(
                        #         f"  [{tname}] {iname}{sub_note}: REPORTED — Change: {change}"
                        #     )
                        # elif state == "COLLECTED_NR":
                        #     kg_block_lines.append(
                        #         f"  [{tname}] {iname}{sub_note}: COLLECTED_NR — "
                        #         f"subscale collected but per-subscale outcome not in KG; verify SAP"
                        #     )
                        # elif state == "SUBSCALES_NR":
                        #     kg_block_lines.append(
                        #         f"  [{tname}] {iname}: SUBSCALES_NR — "
                        #         f"instrument used but assessed subscales not recorded in KG; verify SAP"
                        #     )
                        # elif state == "NOT_COLLECTED":
                        #     kg_block_lines.append(
                        #         f"  [{tname}] {iname}: NOT_COLLECTED — "
                        #         f"this domain's subscale not in trial's assessed subscale set"
                        #     )
            kg_block_lines.append("")

        # === COMPARATOR TRIALS ===
        kg_block_lines.append(f"\n=== COMPARATOR TRIALS ({len(coverage['comparator_trials'])} trials) ===")
        kg_block_lines.append("Use these to populate Table 2 (trial rows).\n")
        for i, trial in enumerate(coverage["comparator_trials"], 1):
            label = f"CT-{i:03d}"
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
            kg_block_lines.append("=== INSTRUMENT REFERENCE DATA ===")
            for i, ir in enumerate(inst_refs[:8], 1):
                short    = ir.get("shortname") or ir.get("instrument_name") or ""
                items    = ir.get("total_items") or ir.get("totalitems") or "NR"
                admintime = ir.get("admin_time") or ir.get("admintime") or "NR"
                mcid     = ir.get("mcid") or "Not established"
                regacc   = ir.get("regulatory_acceptance") or ir.get("regulatoryacceptance") or "NR"
                population = ir.get("patient_population") or ir.get("population") or "NR"   # ← ADD

                kg_block_lines.append(
                    f"IR-{i:03d} {short} | "
                    f"Items: {items} | "
                    f"Admin time: {admintime} min | "
                    f"MCID: {mcid} | "
                    f"Reg. acceptance: {regacc} | "
                    f"Validated population: {population}"             #
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

    kg_evidence_block = "\n".join(kg_block_lines) + "\n" + lang_block

    logging.info(f"KG evidence block: {len(kg_evidence_block.split())} words / "
             f"~{int(len(kg_evidence_block.split()) * 1.3)} tokens estimated")

    # --- STEP H: HTA context block ---
    # hta_data = get_hta_preferences(
    #     markets=context_json.get("hta_markets", []),
    #     graph_client=_get_conn()
    # )
    # hta_lines = ["\n=== HTA/PAYER CONTEXT ===\n"]
    # for body, prefs in hta_data.items():
    #     warning = f" {prefs['warning']}" if prefs.get("warning") else ""
    #     hta_lines.append(
    #         f"{body}: {prefs['notes']} "
    #         f"{prefs['citation']}"
    #         f"{warning}"
    #     )
    # hta_block = "\n".join(linkify_flag_citations(line) for line in hta_lines)

    # --- STEP H: HTA context block (uses hta_data already fetched in STEP F.7) ---
    hta_lines = ["\n=== HTA/PAYER CONTEXT ===\n"]
    for body, prefs in (hta_data or {}).items():
        warning = f" {prefs['warning']}" if prefs.get("warning") else ""
        hta_lines.append(
            f"{body}: {prefs['notes']} "
            f"{prefs['citation']}"
            f"{warning}"
        )
    hta_block = "\n".join(linkify_flag_citations(line) for line in hta_lines)


    score_lines = ["=== INSTRUMENT SCORES (Python-computed — Table 3 Fit for Purpose must use these exactly) ==="]
    for s in scored[:8]:
        tier = (
            "Likely fit" if s["scientific_score"] >= 65 and s["risk_level"] not in ("CRITICAL", "HIGH")
            else "Conditionally fit — verify gaps noted" if s["scientific_score"] >= 40 and s["risk_level"] != "CRITICAL"
            else "Evidence gaps — review required"
        )
        score_lines.append(
            f"  {s['instrument_name']}: score={s['scientific_score']}/100, "
            f"risk={s['risk_level']}, fit_tier=\"{tier}\""
        )
    kg_evidence_block += "\n" + "\n".join(score_lines)

    # gap_lines = ["=== GAP ANALYSIS (Python-computed — Table 3 cells must use these values verbatim) ==="]
    # for row in gap_analysis:
    #     gap_lines.append(
    #         f"  {row['instrument']}:"
    #         f" content_validity={row.get('content_validity', 'NR')!r},"
    #         f" mcid={row.get('mcid_evidence', 'NR')!r},"
    #         f" regulatory={row.get('regulatory_acceptance', 'NR')!r},"
    #         f" known_gaps={row.get('known_gaps', 'NR')!r},"
    #         f" fit={row.get('fit_for_purpose', 'NR')!r},"
    #         f" validation_evidence={row.get('validation_evidence', 'NR')!r}"
    #     )
    # kg_evidence_block += "\n" + "\n".join(gap_lines)

    gap_lines = ["=== GAP ANALYSIS (Python-computed — Table 3 cells must use these values verbatim) ==="]
    for row in gap_analysis:
        gap_lines.append(
            f"  {row['instrument']}:"
            f" content_validity={row.get('content_validity', 'NR')},"
            f" validation_evidence={row.get('validation_evidence', 'NR')},"
            f" regulatory={row.get('regulatory_acceptance', 'NR')},"
            f" known_gaps={row.get('known_gaps', 'NR')},"
            f" fit={row.get('fit_for_purpose', 'NR')}"
        )
    kg_evidence_block += "\n" + "\n".join(gap_lines)


    ep_lines = ["=== ENDPOINT POSITIONING (Python-computed from KG — Table 4 counts must use these exactly) ==="]
    for row in endpoint_positioning:
        ep_lines.append(
            f"  {row['instrument']}: "
            f"Primary={row.get('primary_count', 0)}, "
            f"Secondary={row.get('secondary_count', 0)}, "
            f"Exploratory={row.get('exploratory_count', 0)}, "
            f"comment={row.get('comment', '')!r}"
        )

    kg_evidence_block += "\n" + "\n".join(ep_lines)


    if pro_endpoint_table:
        ep_table_lines = ["\n=== PRO ENDPOINT SAP LANGUAGE — TABLE 4 DATA ==="]
        ep_table_lines.append(
            "Instrument | Role | Pre-spec | SAP Endpoint Language | Subscales | "
            "Key Finding | Sig. | p-value | Effect Size | Trial (Drug)"
        )
        for row in pro_endpoint_table[:5]:
            sap = row["sap_endpoint"][:180] if row["sap_endpoint"] else "—"
            kf  = row["key_finding"][:100]  if row["key_finding"]  else "—"
            ep_table_lines.append(
                f"{row['instrument']} | {row['role']} | {row['prespecified']} | "
                f"{sap} | {row['subscales'] or '—'} | {kf} | "
                f"{row['significance'] or '—'} | {row['p_value'] or '—'} | "
                f"{row['effect_size'] or '—'} | {row['trial']} ({row['drug']})"
            )
    
    kg_evidence_block += "\n" + "\n".join(ep_table_lines)

    # --- STEP I: Build Sonnet system prompt ---
    sonnet_system = f"""You are a COA (Clinical Outcome Assessment) specialist synthesising evidence for a senior COA expert who will make the final instrument selection decisions. Your role is to present evidence clearly and impartially — not to select or prescribe instruments. The expert decides; you organise the evidence.

IMPORTANT: Total output must fit within 23,000 tokens. Tables first. Prose concise. Do not repeat table content in prose sections.

=== PRO/COA TERMINOLOGY GLOSSARY ===
{GLOSSARY_TEXT}

═══════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════
RULE 1 — FIVE TABLES ARE MANDATORY (highest-priority constraint)
═══════════════════════════════════════════════════════════════
You must produce exactly five markdown tables in the order listed below.
Do not skip, merge, reorder, or combine them.
  Table 1 — Domain Coverage Comparison
  Table 2 — PRO Measures Comparison
  Table 3 — Instrument Gap Analysis
  Table 4 — Endpoint Positioning
  Table 5 — Language & Translation Readiness

Tables 1 and 2 draw on comparator trial evidence from the KG.
Tables 3, 4, and 5 use ONLY the instruments you recommend in your narrative — not the full scored list.

FORMATTING RULE (all five tables):
Use markdown pipe syntax: | col | col | with a header separator row |---|---|
Never use ASCII dashes, spaces, or any other method to draw tables.
Never use the pipe character | inside a table cell — use an em-dash (—) or slash (/) instead.

═══════════════════════════════════════════════════════════════
TABLE SPECIFICATIONS
═══════════════════════════════════════════════════════════════

════════════════════════════════════════════════
TABLE 1 — Domain Coverage Comparison
════════════════════════════════════════════════

STRUCTURE:
Render as a markdown table with EXACTLY these columns in this order:
| Concept | Key Stakeholder | Current Trial Candidates | [Trial Name 1] | [Trial Name 2] | [Trial Name 3]|

Column rules:
- "Concept": the FDA core domain name (e.g. "Disease-related bone pain", "Fatigue", "Physical functioning").
  Last row is always the HTA Utility row: "HTA Utility (QALY)" — this is EQ-5D.
- "Key Stakeholder": the regulatory document that defines this as a core domain
  (e.g. "FDA 2024 Core PRO Guidance", "EMA Reflection Paper 2005").
  For the HTA row: "NICE DSU TSD 2 / ICER Framework".
- "Current Trial Candidates": list the candidate instruments from the
  DOMAIN COVERAGE MATRIX for this domain. Never write "[Candidate instruments needed]".
  If no scored candidates exist, write "No candidates in KG — consult COA expert".
- Comparator columns: named with the ACTUAL trial name from KG, never "[Comparator Trial 1]".
  If no KG comparator data at all: write "No KG data for this indication" spanning all comparator columns.

CELL RENDERING — use the pre-computed state from TABLE 1 PRE-COMPUTED DATA:
  REPORTED + Yes    → ✅ INSTRUMENT_NAME — Change: Y [CT-XXX]
  REPORTED + No     → ✅ INSTRUMENT_NAME — Change: N (NS) [CT-XXX]
  REPORTED + NR     → ✅ INSTRUMENT_NAME — Change: NR [CT-XXX]
  COLLECTED_NR      → ⚠️ INSTRUMENT_NAME — outcome NR in KG²
  SUBSCALES_NR      → ⚠️ INSTRUMENT_NAME — subscales NR in KG²
  NOT_COLLECTED     → ❌ Not collected¹
  (no cell at all)  → ❌ Not collected¹

SPECIAL CELL FORMAT — general-only instrument warning:
  If the instrument is a general HRQoL tool (e.g. EORTC QLQ-C30) with no
  disease-specific module for the indication in question, append:
  ⚠️ General only (INSTRUMENT_NAME, no disease-specific module)

WEB SEARCH GAP-FILL (Table 1 only — max 2 searches total):
- Only trigger for cells showing ❌ Not collected¹.
- Search query format: "[trial name] [domain] PRO endpoint".
- If the instrument WAS used but missing from KG: update the cell and add source URL.
- If confirmed absent: keep ❌ Not collected and add a numbered footnote (see below).
- Prioritise the 2 most clinically important gaps only.

FOOTNOTES — strict rules:
¹ Appears after any ❌ Not collected cell.
  Text: "Not collected in this trial — no KG record."
  DO NOT add any explanation for WHY it was absent (e.g. "regulatory focus on
  survival", "CTCAE only"). If the reason is not in a specific KG field, omit it.
² Appears after any ⚠️ COLLECTED_NR or SUBSCALES_NR cell.
  Text: "Instrument used in this trial but per-subscale outcome data is incomplete
  in the KG. Confirm pre-specified subscales in the trial SAP."
A footnote number may ONLY be added when the pre-computed Python data shows that
state — never infer, never add footnotes for cells that have a REPORTED state.
DO NOT HALLUCINATE footnote reasons.

---

TABLE 2 — PRO Measures Comparison
Columns (exactly in this order):
| Trial | Drug | Drug class | PRO Measures (n items) | Assessment Schedule | Total items | Est. time (min) |

Rows: One per comparator trial from KG + one "Current Trial (Proposed)" row at the bottom.
- "Trial": trial name [TI-XXX]
- "Drug": drug name from KG
- "Drug class": mechanism class from KG
- "PRO Measures (n items)": list each instrument with item count, e.g. "EORTC QLQ-C30 (30), EQ-5D-5L (5)". Use counts from [IR-XXX] blocks; write n=? if not in KG
- "Assessment Schedule": frequency and timepoints from KG (e.g. "Cycles 1,2,4,6,9 then every 3 cycles"). Use the assessment_schedule value from the Python-computed row. Write "NR" if empty — do not invent a schedule.
- "Total items": sum of all instrument items for that row
- "Est. time (min)": sum of admin_time values from IR-XXX blocks for that row

For "Current Trial (Proposed)" row:
- "PRO Measures": list candidate instruments from the Domain Coverage Matrix
- "Assessment Schedule": "TBD — expert decision required"
- "Total items" and "Est. time": calculate from IR-XXX data for the candidates listed

STRICT RULE: For any cell where the Python layer has set the value to "NR", render exactly "NR". Do NOT substitute an estimate, guess, or any other value. NR means the data is not available in the knowledge graph. If KG returns no comparator data, do NOT generate placeholder rows.

---

TABLE 3 — Instrument Gap Analysis
Recommended instruments only. Do not include scored instruments you are not recommending.
Columns (exactly in this order):
| Instrument | Content Validity | Psychometric Properties | MCID Evidence | Regulatory Acceptance | Known Gaps / Risks | Fit for Purpose | 

- "Content Validity": one sentence from IR-XXX validation status or KG record. Use the validation field from the GAP ANALYSIS block. Cite [IR-XXX] if the IR block has a matching instrument.
- "MCID Evidence": numeric threshold and unit if known (e.g. "≥3 points, anchor-based, RRMM population"), else "Not established — Need to verify". se mcid from [IR-XXX] if available. Always include the numeric threshold.
- "Psychometric Properties (FDA 3)": ICC, Cronbach's alpha, responsiveness, or "NR" if not in KG.
  Use the validation_evidence field from the GAP ANALYSIS block. Do not fabricate values.
- Regulatory Acceptance corroborating evidence only — not the primary fit-for-purpose criterion.
  Use: "Accepted [Agency] — [drug]" citing RR-XXX, "Rejected — [reason]" citing REJ-XXX,
  or "No KG precedent" if neither exists. Do not weight this column above content validity
  or population fit when reasoning about the instrument's suitability.
- "Known Gaps / Risks": the single most important limitation in one sentence
- Fit for Purpose use the fit_tier from the INSTRUMENT SCORES block as your verdict,
  then add one population-specific clause. Format:
  "Likely fit — validated in [population e.g. RRMM frail patients]"
  "Conditionally fit — [specific gap e.g. no RRMM-specific MCID established]"
  "Evidence gaps — [specific missing evidence e.g. no content validity study in MM]"
  Never write just the tier label alone. The population clause is mandatory.

STRICT RULES: 
- The 'Fit for Purpose' column in Table 3 must use exactly the fit_tier value from the INSTRUMENT SCORES block above. 
Do not substitute your own assessment.
- Content Validity, MCID Evidence, Regulatory Acceptance, Known Gaps, and
Fit for Purpose columns in Table 3 must use values from the GAP ANALYSIS block verbatim.
Do not generate new text for these cells.

---

TABLE 4 — PRO Endpoint Language Reference
Recommended instruments only.
Columns (exactly in this order):
| Instrument | Role | Pre-spec. | Endpoint Language | Subscales | Key Finding | Sig. | p / Effect | Trial (Drug) |

- Use ONLY rows from "PRO ENDPOINT LANGUAGE — TABLE 4 DATA" section.
- "Trial (Drug)": match the trial name to [CT-XXX] from the COMPARATOR TRIALS block and append the label.
- "Endpoint Language": Quote verbatim (already truncated analysis excerpt). 
  This shows EXACT regulatory analysis precedent from prior trials.
  If "—", write "—". Add TI-XXX citation at end of cell.
- "Role": Primary / Secondary / Exploratory / NR (exactly as stored).
- "Pre-spec.": Yes / No / NR (exactly as stored).
- Sort: Primary first → Secondary → Exploratory; within role by score.
- KG positioning stats → Key Observations only, NOT table columns.

STRICT: Use sap_endpoint field EXACTLY as provided. No paraphrasing.

---

TABLE 5 — Language Translation Readiness
Recommended instruments only.
Columns (exactly in this order):
| Instrument | Validated Translations (approx.) | Key Languages Covered | Gap / Action |

SOURCES (WEB SEARCH these when KG says "unverified"):
1. "[instrument] validated languages" → EORTC/FACIT/EuroQol sites
2. "[instrument] linguistic validation" → PubMed/developer papers
3. Mapi Research Trust catalog (public listings)

- "Validated Translations": KG count OR web search result + URL. 
- "Validated Translations": use the count and citation from the "LANGUAGE TRANSLATION DATA" block exactly. If the source_type is "KG", 
cite [KG: Instrument.languages]. If "Web", cite the URL. If "AgentInference", write "Unverified — see warning".
- "Key Languages": Top 6 for trial footprint (EN, FR, DE, ES, IT, JP, CN, KR...)
- "Gap / Action": Missing lang → "Commission [lang] (6-12mo)"

EXAMPLES:
- EORTC QLQ-C30 → "110+ languages [eortc.org]" | EN/FR/DE/ES/IT/JP/CN | "No action"
- BFI → "45 languages [mdanderson.org]" | EN/FR/DE/ES | "Verify CN/KR coverage"

Max 2 searches. Prioritize top instruments.

CRITICAL: Tables 3, 4, and 5 must contain only the instruments you recommend in the narrative above them. Do not include instruments from the scoring system that you have not recommended.

RULE 2  PRO ENDPOINT POSITIONING — cite in Key Observations, NOT in any table column.
The KG block contains "PRO ENDPOINT POSITIONING IN KG TRIALS" with the distribution of
primary/secondary/exploratory positions across trials for this indication.
State: "In N KG trials, X were secondary, Y exploratory, Z primary."
Then reason: what does this tell us about regulatory precedent and approvability risk?
This goes in Key Observations — not in TABLE 4.
TABLE 4 uses the "PRO ENDPOINT SAP LANGUAGE — TABLE 4 DATA" block only.
Do NOT put Primary/Secondary/Exploratory counts into any TABLE 4 column.

RULE 3 — Citations: every factual claim must be cited immediately after the sentence
KG records: [TI-001] instrument records, [CT-001] comparator trials, [RR-001] regulatory reviews, [REJ-001] rejections, [IR-001] instrument refs, [RULE-001] rules, [COMP-001] competitorsWeb sources: [Source Name](https://complete-url.com)
If not found: "[Not found in KG or web search]"
Do not state trial results, statistics, or regulatory decisions from training memory
without a web search confirming them first.
STRICT PROHIBITION — circular citations:
Never cite your own output as a source. Specifically:
  [7] = "Table 2 comparison shows..." — this is NOT a valid citation.
  [8] = "Table 4 positioning data..." — this is NOT a valid citation.
  [9] = "Table 3 EORTC row shows..." — this is NOT a valid citation.
These are circular self-references. They add nothing and undermine trust.
A valid citation is either: a KG label [TI-XXX] / [IR-XXX] / [RR-XXX] / [RULE-XXX],
or a real external URL [Source Name](https://actual-url.com).
If you cannot find a real source for a claim, write the claim without a citation.
A missing citation is better than a fake one.

RULE 4 — Item library note: include when KG evidence supports it
If KG shows comparator trials used subscales rather than full instruments:
"Item library / calibrated SOA: [trial] used [subscale set] rather than the full [instrument],
reducing patient burden from [N] to ~[M] items per timepoint. Consider whether a similar
approach is appropriate for this trial — decision for the COA expert."
Include this note in the Key Observations section.

RULE 5 — Competitor context: connect every COMP-XXX finding to a specific instrument decision
For each COMP-XXX entry: state mechanism relevance and PRO outcome in one sentence.
Then explicitly link to a candidate instrument decision: "This supports/challenges [instrument X] because [reason from KG/web evidence]."

RULE 6 — No hallucination
Do not state statistics, trial results, or regulatory decisions from training memory.
Sonnet's training cutoff may not include the most recent FDA/EMA label decisions — always
web-search to verify before stating, then cite the source URL.

RULE 7 — Regulatory rules: must cite when relevant
The KG block may contain "=== REGULATORY RULES ===" with [RULE-XXX] entries —
published FDA/ICH/EMA rules applicable to this indication and phase.
If RULE entries exist: cite at least one [RULE-XXX] when discussing pre-specification,
alpha control, estimand strategy, missing data handling, or testing hierarchy.
If no RULE entries exist: note "No indication-specific regulatory rules retrieved from KG —
consult FDA PRO Guidance (2009) directly."

═══════════════════════════════════════════════════════════════
OUTPUT STRUCTURE — generate every section in this exact order
═══════════════════════════════════════════════════════════════

## COA Measurement Strategy — [Indication] [Phase]

**In one sentence:** [what this trial is trying to show with PROs]
**Key challenge:** [the single biggest PRO design challenge for this specific trial]
**Recommended starting point:** [2–3 instruments the expert should seriously consider, citing KG evidence of change detection and scores]
**Critical gap:** [the one issue that will fail the strategy if unaddressed — e.g. EQ-5D absent for NICE, neuropathy domain not covered for PI class]

## Table 1: Domain Coverage Comparison
[mandatory — five columns minimum — see TABLE 1 specification above]

## Table 2: PRO Measures Comparison
[mandatory — see TABLE 2 specification above]

## Table 3: Instrument Gap Analysis
[mandatory — recommended instruments only — see TABLE 3 specification above]

## Table 4: Endpoint Positioning
[mandatory — recommended instruments only — see TABLE 4 specification above]

## Table 5: Language & Translation Readiness
[mandatory — recommended instruments only — see TABLE 5 specification above]

## Key Observations
Maximum 6 bullet points. Each bullet must: (a) cite a source, (b) connect to a specific table cell by name (e.g. "Table 3, EORTC QLQ-MY20 row"), and (c) add information not already visible in the tables. Include item library note here if applicable (RULE 4).

## Comparator Analysis
For each COMP-XXX entry in the evidence block:
- One row per comparator: drug — mechanism — PRO instruments used — outcome — implication for current trial
- If a comparator detected significant change with an instrument: flag this as sensitivity evidence for that instrument
- If a comparator found null results: note this as a calibration risk and state whether trial design differences make the current trial more or less likely to detect change
Conclude: "Based on comparator evidence, [instrument X] has the strongest signal of change in this indication because [one-sentence reason from KG/web evidence]."
If no COMP entries exist: "No same-mechanism comparator data in KG — expert should review recent FDA and EMA medical reviews for [drug class] submissions."

## HTA Requirements
| HTA Body | Required Instrument | In Candidate List? | Action Needed |
|---|---|---|---|
[one row per HTA body in scope — use the HTA/PAYER CONTEXT block from the evidence]

## What the Expert Needs to Decide
These are open decisions — present options, not verdicts. The expert chooses.
1. Which instruments from the candidate list to include in the final battery
2. Whether item library / calibrated SOA approach is appropriate for this trial
3. Endpoint hierarchy: which instrument(s) to pre-specify for alpha-controlled testing
4. Assessment schedule aligned with dosing frequency and key clinical timepoints
5. Any additional domains not covered by the current candidate set

"""

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
CITATION RULE: Every factual claim — instrument properties, regulatory requirements,
MCID values, HTA requirements — must be followed by a citation label from the
KG evidence block (e.g. [TI-001], [RR-002], [RULE-001]) or a web search result.
Do NOT state regulatory requirements (e.g. "FDA requires...") without a citation label.
If no citation exists in the KG block, use web search and cite the URL.

"""
        sonnet_user = sonnet_user.replace(
            "Use your web search tool to supplement",
            competitor_search_instruction + "\nUse your web search tool to supplement"
        )

    # --- STEP K: Call Sonnet ---
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=15000,
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
    # try:
    #     response = client.messages.create(
    #         model="claude-sonnet-4-20250514",
    #         max_tokens=15000,                
    #         # thinking={"type": "enabled",
    #         #           "budget_tokens": 2000},  
    #         system=sonnet_system,
    #         tools=[],
    #         messages=[{"role": "user", "content": sonnet_user}]
    #     )
    #     answer = " ".join(
    #         block.text for block in response.content
    #         if hasattr(block, "text") and block.text
    #         and getattr(block, "type", "") != "thinking"
    #     )
    #     logging.info("Sonnet answered with extended thinking (KG-only mode)")
    # except Exception as thinking_err:
    #     logging.warning(f"Extended thinking failed ({thinking_err}), falling back to web search")
    #     # Fallback: standard call with web search, no thinking
    #     response = client.messages.create(
    #         model="claude-sonnet-4-20250514",
    #         max_tokens=10000,
    #         system=sonnet_system,
    #         tools=[{"type": "web_search_20250305", "name": "web_search"}],
    #         messages=[{"role": "user", "content": sonnet_user}]
    #     )
        # answer = " ".join(
        #     block.text for block in response.content
        #     if hasattr(block, "text") and block.text
        # )

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
        "coverage": coverage,
        "hta_context": hta_data,
        "error_status": error_status,
        "record_counts": {
            "instrument_records": len(kg_records),
            "regulatory_reviews": len(reg_records),
            "regulatory_rules": len(reg_rules),
            "instrument_refs": len(inst_refs),
            "scored_instruments": len(scored),
            "rejections_found": len([r for r in reg_records if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")])
        },
        "citation_index": citation_index,
        "competitor_profiles": competitor_profiles, 
        "kg_evidence_block": kg_evidence_block,
        "pro_measures": pro_measures_table,
        "gap_analysis": gap_analysis,
        "endpoint_positioning": endpoint_positioning,
        "pro_endpoint_table": pro_endpoint_table,
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
            "pro_endpoint_table_rows": len(result.get("pro_endpoint_table", [])),
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
# if __name__ == "__main__":
#     print("Testing imports...")
#     print("All imports OK.")

#     # Test recall period — uses local _published table inside get_recall_period()
#     # Pass None as graph_client to force P2 fallback (KG not available in test)
#     bfi_result = get_recall_period("bfi", None)
#     print(f"BFI recall period: {bfi_result['days']} (expected: 1) — source: {bfi_result['source_type']}")

#     unknown_result = get_recall_period("some_unknown_instrument_xyz", None)
#     print(f"Unknown instrument sentinel: {unknown_result['days']} (expected: -1) — source: {unknown_result['source_type']}")

#     # Test MCID cleaning
#     short, _ = clean_mcid("bfi total scale: 1.33 points pmc11398933 (2024) in brain/cns cancer patients")
#     print(f"MCID clean: {short}")

#     # Test ensure_full_stop
#     print(ensure_full_stop("Test sentence without stop"))

if __name__ == "__main__":
    # ========================================
    # TABLE 1 ZERO-COST TEST — VERIFIED SAFE
    # ========================================
    test_ctx = {
        "indication": "Multiple Myeloma",
        "coredomainsrequired": ["physical function", "fatigue", "pain", "disease-related symptoms"],
        "geographicfootprint": "Global",
        "phase": "Phase 3"
    }
    
    print("🔄 Fetching KG...")
    kg = get_instruments_by_indication(test_ctx["indication"])
    print(f"   KG records: {len(kg)}")
    
    print("🔄 Scoring...")
    scores = score_evidence(test_ctx, kg)
    print(f"   Scores: {len(scores)}")
    
    print("🔄 Building Table 1...")
    coverage = build_coverage_matrix(scores, test_ctx, kg)
    
    print("✅ TABLE 1 SUCCESS ($0 spent!)")
    print(f"• Domains: {len(coverage.get('domains', []))}")
    print(f"• Comparator trials: {len(coverage.get('comparatortrials', []))}")
    print(f"• HTA mandatory: {len(coverage.get('htamandatory', []))}")
    
    # Safe first domain previews
    matrix = coverage.get('trialdomainmatrix', [])
    if matrix:
        first = matrix[0]
        print("\n=== TABLE 1 PREVIEW (first domain) ===")
        print(f"Domain: {first.get('domain')}")
        print("Trial cells:")
        print(json.dumps(first.get('trials', {}), indent=2, default=str)[:800])
    else:
        print("• trialdomainmatrix: empty (normal if no comparator trials)")
    
    print("\n🎉 Ready to debug Table 1!")


