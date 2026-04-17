"""
PRO COA AI Agent — Chat Interface
University of Cambridge × Evinova (AstraZeneca)

Architecture:
  - Multi-turn chat with session history
  - Three-tier intent routing (Haiku classifier)
  - Full strategy pipeline OR direct Sonnet for simple queries
  - Inline citations with numbered references
  - Expandable scoring detail and evidence cards
"""

import streamlit as st
import json
import re
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

def get_secret(key: str) -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, "")   # st is already imported above
    except Exception:
        return ""
    
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="PRO COA Agent | Cambridge × Evinova",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Agent import ─────────────────────────────────────────────────────────────
try:
    from agent import (
        get_recommendation,
        analyze_trial_context,
        get_secret, 
        HTA_PREFERENCES,
        GEOGRAPHIC_LANGUAGE_REQUIREMENTS,
        KNOWN_LANGUAGE_COUNTS,
        get_instruments_by_indication,
        build_tier1_citation_index,
    )
    AGENT_AVAILABLE = True
    AGENT_ERROR = None
except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)
    # client = None

# ── KG known values (your actual stored strings — not synonyms, just
#    the exact values that appear in the KG so queries are precise) ──────────
KG_KNOWN_VALUES = {
    "multiple myeloma": ["Multiple Myeloma", "MM", "RRMM", "Myeloma",
                          "Relapsed/Refractory Multiple Myeloma"],
    "mm":               ["MM", "Multiple Myeloma", "Myeloma", "RRMM"],
    "rrmm":             ["RRMM", "MM", "Multiple Myeloma"],
    "nsclc":            ["NSCLC", "Non-Small Cell Lung Cancer", "Lung Cancer"],
    "non-small cell lung cancer": ["NSCLC", "Non-Small Cell Lung Cancer"],
    "crpc":             ["CRPC", "Prostate Cancer", "mCRPC",
                          "Metastatic Castration-Resistant Prostate Cancer"],
    "prostate cancer":  ["Prostate Cancer", "CRPC", "mCRPC"],
    "breast cancer":    ["Breast Cancer", "BC", "MBC", "Metastatic Breast Cancer"],
    "diffuse large b-cell lymphoma": ["DLBCL", "Lymphoma", "NHL"],
    "lymphoma":         ["Lymphoma", "NHL", "DLBCL", "CLL"],
    "cll":              ["CLL", "Lymphoma", "NHL"],
    "aml":              ["AML", "Leukemia", "Acute Myeloid Leukemia"],
    "colorectal cancer":["CRC", "Colorectal Cancer", "Colon Cancer"],
    "ovarian cancer":   ["Ovarian Cancer", "OC"],
}

# ── Neo4j health check ────────────────────────────────────────────────────────
def check_neo4j() -> str:
    try:
        from graph import Neo4jConnection
        _c = Neo4jConnection(
            get_secret("NEO4J_URI"),
            get_secret("NEO4J_USERNAME"),
            get_secret("NEO4J_PASSWORD")
        )
        _c.run_query("RETURN 1 AS ok")
        _c.close()
        return "connected"
    except Exception as _e:
        return str(_e)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""<style>
/* Chat bubbles */
.stChatMessage { padding: 0.5rem 0; }

/* Battery cards */
.battery-card {
    border-left: 4px solid #1D9E75;
    padding: 10px 14px;
    margin: 4px 0;
    background: #f0faf6;
    border-radius: 0 8px 8px 0;
}
.battery-hta {
    border-left: 4px solid #7F77DD;
    padding: 10px 14px;
    margin: 4px 0;
    background: #eeedfe;
    border-radius: 0 8px 8px 0;
}
.battery-role {
    font-size: 0.71rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #666;
    margin-bottom: 2px;
}

/* Risk badges */
.risk-critical { background:#fdf0f0; color:#791F1F; border:1px solid #F09595;
                  padding:1px 7px; border-radius:3px; font-size:0.77rem; font-weight:600 }
.risk-high     { background:#faeeda; color:#633806; border:1px solid #FAC775;
                  padding:1px 7px; border-radius:3px; font-size:0.77rem; font-weight:600 }
.risk-moderate { background:#fdf6e8; color:#854F0B; border:1px solid #EF9F27;
                  padding:1px 7px; border-radius:3px; font-size:0.77rem; font-weight:600 }
.risk-low      { background:#e1f5ee; color:#085041; border:1px solid #5DCAA5;
                  padding:1px 7px; border-radius:3px; font-size:0.77rem; font-weight:600 }

/* Flag lines in scoring detail */
.flag-penalty  { border-left:3px solid #E24B4A; padding:3px 10px; background:#fdf0f0;
                  margin:2px 0; border-radius:0 4px 4px 0; font-size:0.81rem }
.flag-bonus    { border-left:3px solid #1D9E75; padding:3px 10px; background:#e1f5ee;
                  margin:2px 0; border-radius:0 4px 4px 0; font-size:0.81rem }
.flag-info     { border-left:3px solid #378ADD; padding:3px 10px; background:#e6f1fb;
                  margin:2px 0; border-radius:0 4px 4px 0; font-size:0.81rem }
.flag-neutral  { border-left:3px solid #D3D1C7; padding:3px 10px; background:#f8f8f6;
                  margin:2px 0; border-radius:0 4px 4px 0; font-size:0.81rem }

/* Source links */
.source-row    { border:1px solid #e0e0dc; border-radius:5px; padding:5px 10px;
                  margin:2px 0; background:#fafaf8; font-size:0.82rem }
.source-row a  { color:#185FA5; text-decoration:none }

/* Assumption pills */
.assumption-pill { display:inline-block; background:#faeeda; border:1px solid #FAC775;
                    border-radius:4px; padding:2px 8px; margin:2px 3px;
                    font-size:0.80rem; color:#633806 }

/* Step tracker */
.step-complete { border-left:3px solid #1D9E75; padding:4px 10px; margin:2px 0;
                  background:#f0faf6; border-radius:0 5px 5px 0;
                  font-size:0.82rem; color:#085041 }
.step-running  { border-left:3px solid #EF9F27; padding:4px 10px; margin:2px 0;
                  background:#fdf6e8; border-radius:0 5px 5px 0;
                  font-size:0.82rem; color:#633806 }
.step-pending  { border-left:3px solid #D3D1C7; padding:4px 10px; margin:2px 0;
                  background:#f8f8f6; border-radius:0 5px 5px 0;
                  font-size:0.82rem; color:#888780 }
.step-error    { border-left:3px solid #E24B4A; padding:4px 10px; margin:2px 0;
                  background:#fdf0f0; border-radius:0 5px 5px 0;
                  font-size:0.82rem; color:#791F1F }

/* Citation superscripts rendered by Streamlit markdown */
sup a { color: #185FA5; font-size: 0.75em; text-decoration: none; font-weight: 600 }

/* Clarifying question box */
.clarify-box   { border:1px solid #9FE1CB; background:#f0faf6; border-radius:8px;
                  padding:12px 16px; margin:6px 0; font-size:0.90rem }
</style>""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def risk_badge(level: str) -> str:
    return f'<span class="risk-{level.lower()}">{level}</span>'


def render_steps(steps: list) -> str:
    icons = {"complete": "✓", "running": "⟳", "pending": "○", "error": "✗"}
    html = "<div style='margin:6px 0'>"
    for s in steps:
        icon = icons.get(s["status"], "○")
        detail = f" — {s['detail']}" if s.get("detail") else ""
        html += f'<div class="step-{s["status"]}">{icon} {s["label"]}{detail}</div>'
    return html + "</div>"

# ── Gap-filling prompt builder (sidebar only fills what chat didn't provide) ──
def build_structured_prompt(user_message: str, context: dict,
                             sb_indication: str, sb_phase: str,
                             sb_drug_class: str, sb_admin: str,
                             sb_population: str, sb_hta: list,
                             sb_footprint: str) -> str:
    """Append sidebar values only for fields Haiku could NOT extract from the message."""
    parts = [user_message.strip()]
    assumptions = " ".join(context.get("assumptions_made", [])).lower()
    if sb_indication and context.get("indication", "unknown") == "unknown":
        parts.append(f"Indication: {sb_indication}")
    if sb_phase and "phase" in assumptions:
        parts.append(f"Phase: {sb_phase}")
    if sb_drug_class and context.get("drug_class") in ["Unknown", "", None]:
        parts.append(f"Drug class: {sb_drug_class}")
    if sb_admin and context.get("administration") in ["Unknown", "", None]:
        parts.append(f"Administration: {sb_admin}")
    if sb_population and context.get("population_subtype") == "Symptomatic":
        parts.append(f"Population: {sb_population}")
    if sb_hta and not context.get("hta_markets"):
        parts.append(f"HTA markets: {', '.join(sb_hta)}")
    if sb_footprint and context.get("geographic_footprint") in ["Unknown", "", None]:
        parts.append(f"Geographic footprint: {sb_footprint}")
    return "\n".join(parts)


def build_clarification_question(context: dict) -> str:
    indication  = context.get("indication", "unknown")
    phase       = context.get("phase", "")
    drug_class  = context.get("drug_class", "")
    assumptions = " ".join(context.get("assumptions_made", [])).lower()

    if indication == "unknown":
        return (
            "What **indication** (cancer type) and **trial phase** is this for? "
            "(e.g. Multiple Myeloma, Phase 3)\n\n"
            "Otherwise I'll proceed assuming a Phase 3 solid tumour trial and flag it."
        )
    if drug_class in ["Unknown", "", None]:
        return (
            f"What **drug class or mechanism** is being studied in your "
            f"{indication} {phase} trial? "
            f"(e.g. bispecific antibody, proteasome inhibitor, CAR-T, ICI)\n\n"
            f"This determines which toxicity domains the instruments need to capture."
        )
    if "phase" in assumptions:
        return (
            f"What **phase** is your {indication} trial? (Phase 1 / 2 / 3)\n\n"
            f"Otherwise I'll proceed assuming **Phase 3**."
        )
    return (
        f"A few trial parameters are unclear. Could you confirm:\n"
        f"- **Indication:** {indication}\n"
        f"- **Phase:** {phase or 'not specified'}\n"
        f"- **Drug class:** {drug_class or 'not specified'}\n\n"
        f"Otherwise I'll proceed with these assumptions and flag them in the output."
    )

def _show_confirmation_card(ctx: dict):
    """
    Show confirmation before running pipeline.
    If nothing was inferred: show compact summary and proceed automatically.
    If inferences were made: show only the inferences and ask to confirm.
    """
    assumptions = ctx.get("assumptions_made", [])
    extracted_summary = (
        f"**{ctx.get('indication', '—')}** | {ctx.get('phase', '—')} | "
        f"{ctx.get('drug_class', '—')} | {ctx.get('population_subtype', '—')} | "
        f"Footprint: {ctx.get('geographic_footprint', '—')} | "
        f"HTA: {', '.join(ctx.get('hta_markets', []))}"
    )

    if not assumptions:
        st.markdown(
            '<div class="clarify-box">'
            '✅ <b>All parameters extracted from your message.</b><br>'
            f'{extracted_summary}<br><br>'
            '<i>Type <b>yes</b> to run the analysis, or add corrections first.</i>'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="clarify-box">'
            '<b>📋 Extracted from your message:</b><br>'
            f'{extracted_summary}<br><br>'
            '<b>⚠️ The following were inferred (not stated in your message):</b><br>'
            + "<br>".join(f"• {a}" for a in assumptions)
            + '<br><br><i>Type <b>yes</b> to proceed with these inferences, '
            'or correct them and resend.</i>'
            '</div>', unsafe_allow_html=True
        )

# def classify_flag(flag: str) -> str:
#     """Return CSS class for a scoring flag based on its content."""
#     f = flag.upper()
#     if any(x in f for x in [
#         "PENALTY", "CRITICAL", "MISSING CORE", "RECALL BIAS", "ESTIMAND",
#         "NO MCID PENALTY", "ASYMPTOMATIC BURDEN", "PRE-SPECIFICATION",
#         "LANGUAGE DATA UNAVAILABLE", "PRIOR REJECTION"
#     ]):
#         return "flag-penalty"
#     if any(x in f for x in [
#         "+35", "+25", "+20", "+15", "+10)", "+5)",
#         "VALIDATED MCID (+", "TPP/CORE FIT", "REGULATORY TRUST",
#         "COMPETITOR BENCH", "MOA SENSITIVITY", "ECOA READY", "OPEN ACCESS",
#         "HTA ALIGNMENT (+", "RECALL PERIOD COMPATIBLE", "CONTENT VALIDITY"
#     ]):
#         return "flag-bonus"
#     if any(x in f for x in [
#         "RECALL PERIOD COMPATIBLE", "LANGUAGE COVERAGE", "RECALL PERIOD UNKNOWN"
#     ]):
#         return "flag-info"
#     return "flag-neutral"

def classify_flag(flag_text: str) -> str:
    t = flag_text.upper()
    # Emoji-prefixed flags (new system)
    if "🔴" in flag_text or "CRITICAL FLAG" in t:
        return "CRITICAL"
    if "🟠" in flag_text or "HIGH FLAG" in t:
        return "HIGH"
    if "🟡" in flag_text or "MODERATE FLAG" in t:
        return "MODERATE"
    # Keyword fallback
    if any(k in t for k in [
        "CRITICAL", "RECALL INCOMPATIBILITY", "DOMAIN FAILURE",
        "INSTRUMENT-ATTRIBUTED REJECTION"
    ]):
        return "CRITICAL"
    if any(k in t for k in [
        "HIGH", "NOT PRE-SPECIFIED", "ESTIMAND BURDEN"
    ]):
        return "HIGH"
    if any(k in t for k in [
        "MODERATE", "MODE EQUIVALENCE", "ASYMPTOMATIC", "DISTRIBUTION-BASED",
        "TRANSLATION PENALTY"
    ]):
        return "MODERATE"
    return "INFO"

def extract_web_links(text: str) -> list:
    """
    Extracts all [anchor text](https://url) markdown links from Sonnet's answer.
    Deduplicates by URL, preserves first-appearance order.
    Returns [(anchor_text, url), ...]
    """
    raw  = re.findall(r'\[([^\]]+)\]\((https?://[^\)\s]+)\)', text)
    seen, out = set(), []
    for anchor, url in raw:
        if url not in seen:
            seen.add(url)
            out.append((anchor, url))
    return out

def render_kg_evidence_cards(result: dict):
    """
    Shows ALL KG records retrieved — independent of what Sonnet cited.
    Collapsible. Two sections: instrument records and regulatory reviews.
    """
    scored      = result.get("scored_instruments", [])
    reg_records = result.get("regulatory_records", [])

    if not scored and not reg_records:
        return

    with st.expander(
        f"🗄️ Full knowledge graph data — "
        f"{len(scored)} instrument records · {len(reg_records)} regulatory reviews",
        expanded=False
    ):
        if scored:
            st.markdown("#### Instrument trial records")
            st.caption(
                "Every instrument record retrieved from the KG for this indication. "
                "All NCT, DOI, FDA, and EMA columns are clickable links."
            )
            rows = []
            for i, inst in enumerate(scored[:15], 1):
                nct     = str(inst.get("nct_id", ""))
                doi     = str(inst.get("publication_doi", ""))
                fda_url = str(inst.get("fda_label_url", ""))
                ema_url = str(inst.get("ema_label_url", ""))
                kf      = str(inst.get("key_finding", "") or "")

                rows.append({
                    "Ref":          f"TI-{i:03d}",
                    "Instrument":   inst.get("instrument_name", ""),
                    "Drug":         inst.get("drug_name", ""),
                    "Trial":        inst.get("trial_name", "") or "—",
                    "Phase":        inst.get("phase", ""),
                    "Score":        f"{inst.get('scientific_score','—')}/100",
                    "Risk":         inst.get("risk_level", ""),
                    "Role":         inst.get("endpoint_role", ""),
                    "Key finding":  (kf[:100] + "…") if len(kf) > 100 else kf,
                    "NCT":          (f"https://clinicaltrials.gov/study/{nct}"
                                     if nct.startswith("NCT") else ""),
                    "DOI":          (f"https://doi.org/{doi}"
                                     if doi and doi not in ("nan","None","") else ""),
                    "FDA":          fda_url if fda_url.startswith("http") else "",
                    "EMA":          ema_url if ema_url.startswith("http") else "",
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "NCT": st.column_config.LinkColumn("NCT"),
                    "DOI": st.column_config.LinkColumn("DOI"),
                    "FDA": st.column_config.LinkColumn("FDA"),
                    "EMA": st.column_config.LinkColumn("EMA"),
                }
            )

        if reg_records:
            st.markdown("#### Regulatory review records")
            st.caption("FDA/EMA decisions with PRO outcomes from the KG.")

            non_rejections = [r for r in reg_records if not r.get("rejection_reason_primary")]
            rejections     = [r for r in reg_records if     r.get("rejection_reason_primary")]

            # ── Accepted reviews ──────────────────────────────────────────────
            for i, rr in enumerate(non_rejections[:15], 1):   # ← FIX 1: was reg_records
                icon    = "✅"
                ref     = f"RR-{i:03d}"
                fda_url = str(rr.get("fda_label_url", ""))
                ema_url = str(rr.get("ema_label_url", ""))
                drug    = rr.get("drug_name", "")

                with st.container():
                    c1, c2 = st.columns([1, 7])
                    with c1:
                        st.markdown(f"`{ref}` {icon}")
                    with c2:
                        st.markdown(
                            f"**{drug}** — {rr.get('agency','')} | "
                            f"**{rr.get('decision','')}**"
                        )
                        accepted = rr.get("instruments_accepted", "")
                        if accepted and str(accepted) not in ("nan","None",""):
                            st.markdown(f"✅ Accepted: `{accepted}`")
                        claim = rr.get("claim_type", "")
                        if claim and str(claim) not in ("nan","None",""):
                            st.markdown(f"🏷️ Claim type: {claim}")
                        label_lang = str(rr.get("label_language","") or "")
                        if label_lang and label_lang not in ("nan","None",""):
                            st.markdown(f"📄 *{label_lang[:250]}*")
                        links = []
                        if fda_url.startswith("http"):
                            links.append(f"[FDA label]({fda_url})")
                        if ema_url.startswith("http"):
                            links.append(f"[EMA label]({ema_url})")
                        if not links:
                            links.append(
                                f"[DailyMed](https://dailymed.nlm.nih.gov/"
                                f"dailymed/search.cfm?query={drug.replace(' ','+')})"
                            )
                        st.markdown(" · ".join(links))
                st.markdown(
                    "<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
                    unsafe_allow_html=True
                )

            # ── Rejections ────────────────────────────────────────────────────
            for i, rr in enumerate(rejections[:15], 1):
                icon    = "⚠️"                                 # ← FIX 2: removed duplicate lines
                ref     = f"REJ-{i:03d}"
                fda_url = str(rr.get("fda_label_url", ""))
                ema_url = str(rr.get("ema_label_url", ""))
                drug    = rr.get("drug_name", "")

                with st.container():
                    c1, c2 = st.columns([1, 7])
                    with c1:
                        st.markdown(f"`{ref}` {icon}")
                    with c2:
                        st.markdown(
                            f"**{drug}** — {rr.get('agency','')} | "
                            f"**{rr.get('decision','')}**"
                        )
                        accepted = rr.get("instruments_accepted", "")
                        if accepted and str(accepted) not in ("nan","None",""):
                            st.markdown(f"✅ Accepted: `{accepted}`")
                        claim = rr.get("claim_type", "")
                        if claim and str(claim) not in ("nan","None",""):
                            st.markdown(f"🏷️ Claim type: {claim}")
                        label_lang = str(rr.get("label_language","") or "")
                        if label_lang and label_lang not in ("nan","None",""):
                            st.markdown(f"📄 *{label_lang[:250]}*")
                        st.markdown(                            # ← FIX 3: always show, no if-check
                            f"❌ **Rejection reason:** "
                            f"{rr.get('rejection_reason_primary','')}"
                        )
                        detail = str(rr.get("rejection_reason_detailed","") or "")
                        if detail and detail not in ("nan","None",""):
                            st.markdown(f"&nbsp;&nbsp;Detail: {detail[:300]}")
                        links = []
                        if fda_url.startswith("http"):
                            links.append(f"[FDA label]({fda_url})")
                        if ema_url.startswith("http"):
                            links.append(f"[EMA label]({ema_url})")
                        if not links:
                            links.append(
                                f"[DailyMed](https://dailymed.nlm.nih.gov/"
                                f"dailymed/search.cfm?query={drug.replace(' ','+')})"
                            )
                        st.markdown(" · ".join(links))
                st.markdown(
                    "<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
                    unsafe_allow_html=True
                )

def audit_uncited_sentences(answer: str) -> list:
    """
    Finds sentences with factual-sounding content but no citation marker.
    Returns list of suspicious sentence strings.
    """
    # Strip all known citation patterns first
    cleaned = re.sub(
        r'\[[^\]]+\]\(https?://[^\)]+\)', '', answer)       # web links
    cleaned = re.sub(
        r'\[(TI|RR|REJ|IR|RULE|PREC)-\d+\]', '', cleaned)  # KG labels

    SKIP_PREFIXES = (
        "therefore", "this means", "as a result", "in summary",
        "the next", "note:", "action:", "now:", "before", "during",
        "at each", "##", "#", "---", "***",
    )

    FACTUAL_PATTERN = re.compile(
        r'\d|%|approved|rejected|required|validated|accepted|'
        r'trial|phase\s+[123]|FDA|EMA|NICE|ICER|instrument|'
        r'MCID|recall|language|translation|score|domain|'
        r'significant|p\s*[<=]|hazard|confidence',
        re.IGNORECASE
    )

    suspicious = []
    for sentence in re.split(r'(?<=[.!?])\s+', cleaned):
        sentence = sentence.strip()
        if len(sentence) < 60:
            continue
        if any(sentence.lower().startswith(p) for p in SKIP_PREFIXES):
            continue
        if FACTUAL_PATTERN.search(sentence):
            suspicious.append(sentence)

    return suspicious

def build_source_links_html(record: dict) -> str:
    """Build HTML source-link row from a KG record dict."""
    links = []
    nct   = record.get("nct_id", "")
    doi   = record.get("publication_doi", "")
    year  = record.get("publication_year", "")
    drug  = record.get("drug_name", "")
    trial = record.get("trial_name", "")
    fda   = record.get("fda_label_url", "")
    ema   = record.get("ema_label_url", "")

    if nct and str(nct).startswith("NCT"):
        links.append(
            f'<a href="https://clinicaltrials.gov/study/{nct}" target="_blank">'
            f'ClinicalTrials.gov: {trial or nct}</a>'
        )
    if doi:
        label = f"Publication ({year})" if year else f"DOI: {doi[:25]}…"
        links.append(f'<a href="https://doi.org/{doi}" target="_blank">{label}</a>')
    if str(fda).startswith("http"):
        links.append(f'<a href="{fda}" target="_blank">FDA label: {drug}</a>')
    elif drug:
        query = drug.replace(" ", "+")
        links.append(
            f'<a href="https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={query}" target="_blank">FDA DailyMed</a>'
        )
    if str(ema).startswith("http"):
        links.append(f'<a href="{ema}" target="_blank">EMA label: {drug}</a>')

    if not links:
        return ""
    return (
        '<div class="source-row">'
        + " &nbsp;·&nbsp; ".join(links)
        + "</div>"
    )


def linkify_and_number_citations(text: str, citation_index: dict) -> tuple:
    """
    Finds [TI-001], [RR-003], [REJ-012], [IR-001], [RULE-001], [PREC-1]
    in Sonnet's output. Replaces with numbered superscript HTML links.
    Returns (modified_html_text, references).
    references = [(label, display_number, info_dict), ...] in first-appearance order.
    """
    # Strip Sonnet's internal [13-4] number-dash-number indices only
    # text = re.sub(r'\[\d+-\d+\]', '', text)
    text = re.sub(r'\[\d+-\d+\]', '', text)  

    pattern = re.compile(r'(TI|RR|REJ|IR|RULE|PREC|COMP)-(\d{1,3})')
    ref_seen  = {}   # label → display number
    ref_order = []   # (label, info) in first-appearance order

    def _replace(m):
        prefix    = m.group(1)
        digits    = m.group(2)
        raw_label = f"{prefix}-{digits}"
        label     = f"{prefix}-{digits.zfill(3)}"   # normalise PREC-1 → PREC-001

        # Try padded first, then raw
        info = citation_index.get(label) or citation_index.get(raw_label) or {}

        if label not in ref_seen:
            ref_seen[label] = len(ref_seen) + 1
            ref_order.append((label, info))

        n    = ref_seen[label]
        links = info.get("links", [])
        url  = next(
            (l["url"] for l in links if l.get("url", "").startswith("http")),
            None
        )

        if url:
            return (
                f'<sup><a href="{url}" target="_blank" '
                f'style="color:#1D9E75;font-weight:bold;'
                f'text-decoration:none;font-size:0.8em;">[{n}]</a></sup>'
            )
        return (
            f'<sup style="color:#1D9E75;font-weight:bold;'
            f'font-size:0.8em;">[{n}]</sup>'
        )

    modified   = pattern.sub(_replace, text)
    references = [
        (label, ref_seen[label], info)
        for label, info in ref_order
    ]
    return modified, references

def render_unified_footnotes(references: list, web_links: list):
    """
    Single unified footnote panel.
    references = KG records Sonnet cited: [(label, num, info_dict), ...]
    web_links  = web URLs Sonnet used:    [(anchor_text, url), ...]
    Always expanded. Numbered in order of first appearance.
    """
    if not references and not web_links:
        st.caption("⚠️ No citations found in this answer.")
        return

    st.markdown("---")
    st.markdown("#### 📎 Sources")

    counter = 1

    # ── KG citations ──────────────────────────────────────────────────────
    for label, num, info in references:
        ctype = info.get("type", "")

        if ctype == "trial_instrument":
            trial_str = f"*{info['trial']}*" if info.get("trial") else ""
            nct_str   = f"`{info['nct']}`"   if info.get("nct", "").startswith("NCT") else ""
            header    = " · ".join(filter(None, [
                f"**{info.get('instrument','')}**",
                trial_str,
                nct_str,
                info.get("drug",""),
                info.get("phase",""),
            ]))
            st.markdown(f"**[{num}]** &nbsp; 🗄️ &nbsp; `{label}` · {header}")
            kf = info.get("key_finding", "")
            if kf and kf not in ("nan", "None", "", "—"):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"<span style='color:#555;font-size:0.9em'>"
                    f"Key finding: {kf[:200]}</span>",
                    unsafe_allow_html=True
                )
            score = info.get("score", "")
            risk  = info.get("risk", "")
            role  = info.get("endpoint_role", "")
            if any([score, risk, role]):
                meta = " · ".join(filter(None, [
                    f"Score {score}/100" if score else "",
                    f"Risk: {risk}"      if risk  else "",
                    f"Role: {role}"      if role  else "",
                ]))
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"<span style='color:#888;font-size:0.85em'>{meta}</span>",
                    unsafe_allow_html=True
                )

        elif ctype == "regulatory_review":
            st.markdown(
                f"**[{num}]** &nbsp; ✅ &nbsp; `{label}` · "
                f"**{info.get('agency','')}** review · "
                f"**{info.get('drug','')}** · "
                f"Decision: {info.get('decision','')}"
            )
            accepted = info.get("instruments_accepted", "")
            if accepted and accepted not in ("nan", "None", ""):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"<span style='color:#555;font-size:0.9em'>"
                    f"Instruments accepted: {accepted}</span>",
                    unsafe_allow_html=True
                )
            claim = info.get("claim_type", "")
            if claim and claim not in ("nan", "None", ""):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"<span style='color:#555;font-size:0.9em'>"
                    f"Claim type: {claim}</span>",
                    unsafe_allow_html=True
                )

        elif ctype == "rejection":
            st.markdown(
                f"**[{num}]** &nbsp; ⚠️ &nbsp; `{label}` · "
                f"**{info.get('agency','')}** · "
                f"**{info.get('drug','')}** · "
                f"Decision: {info.get('decision','')}"
            )
            reason = info.get("primary_reason", "")
            if reason:
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"<span style='color:#c0392b;font-size:0.9em'>"
                    f"❌ Rejection reason: {reason}</span>",
                    unsafe_allow_html=True
                )
            detail = info.get("detailed_reason", "")
            if detail and detail not in ("nan", "None", ""):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"<span style='color:#888;font-size:0.85em'>"
                    f"{detail[:250]}</span>",
                    unsafe_allow_html=True
                )

        elif ctype == "instrument_reference":
            st.markdown(
                f"**[{num}]** &nbsp; 📋 &nbsp; `{label}` · "
                f"**{info.get('instrument','')}** · "
                f"Domains: {info.get('domains','')} · "
                f"MCID: {info.get('mcid','—')}"
            )
        
        elif ctype == "precedent":
            accepted_str = "✅ Accepted" if info.get("accepted") else "🔍 Reviewed"
            st.markdown(
                f"**[{num}]** &nbsp; `{label}` — {info.get('instrument','')} · "
                f"{info.get('agency','')} · {info.get('drug','')} · {accepted_str}"
            )
            claim = info.get("claim_type","")
            if claim and str(claim) not in ("nan","None",""):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#555;font-size:0.9em'>"
                    f"🏷️ Claim type: {claim}</span>",
                    unsafe_allow_html=True
                )
        
        elif ctype == "competitor":
            comp_flag = "⚠️ COMPARABILITY REQUIRED" if info.get("comparability_required") else ""
            st.markdown(
                f"**[{num}]** &nbsp; `{label}` — **{info.get('drug','')}** · "
                f"{info.get('agency','')} · {info.get('decision','')} "
                f"{comp_flag}"
            )
            mech = info.get("mechanism", "")
            if mech:
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#555;font-size:0.9em'>"
                    f"⚙️ {mech}</span>",
                    unsafe_allow_html=True
                )
            instr = info.get("instruments", "")
            if instr and str(instr) not in ("nan", "None", ""):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#555;font-size:0.9em'>"
                    f"🎯 PROs used: {instr}</span>",
                    unsafe_allow_html=True
                )
            pq = info.get("pro_implication", "")
            if pq:
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#378ADD;font-size:0.9em'>"
                    f"❓ {pq}</span>",
                    unsafe_allow_html=True
                )
            if info.get("comparability_required"):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#c0392b;font-size:0.9em'>"
                    f"⚠️ {info.get('comparability_reason','')}</span>",
                    unsafe_allow_html=True
                )

        else:
            st.markdown(f"**[{num}]** &nbsp; `{label}`")

        # Clickable source links for this KG record
        links = info.get("links", [])
        if links:
            link_parts = [
                f"[{l['label']}]({l['url']})"
                for l in links
                if l.get('url', '').startswith('http')   # ← filter here
            ]
            if link_parts:                                # ← only render if any valid links remain
                st.markdown(
                    "&nbsp;&nbsp;&nbsp;&nbsp;"
                    + " &nbsp;·&nbsp; ".join(link_parts)
                )

        st.markdown(
            "<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
            unsafe_allow_html=True
        )
        counter += 1

    # ── Web citations ─────────────────────────────────────────────────────
    for anchor, url in web_links:
        st.markdown(f"**[{counter}]** &nbsp; 🌐 &nbsp; [{anchor}]({url})")
        st.markdown(
            "<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
            unsafe_allow_html=True
        )
        counter += 1


# =============================================================================
# INTENT ROUTER
# =============================================================================
ROUTER_SYSTEM = """You are a routing assistant for a PRO COA strategy AI agent.
Classify the user's message into exactly one of three tiers:

TIER1_FACTUAL — Any factual or informational question. Both the internal
  knowledge graph AND web search will be used to answer it.
  Examples:
    "What instruments are most common in MM trials?"
    "What is the EORTC QLQ-C30?"
    "What is MCID and why does it matter?"
    "List PRO tools used in NSCLC"

TIER2_FOLLOWUP — A follow-up answerable from the strategy already in
  the conversation. Only use if a prior strategy exists.
  Examples:
    "Why did you exclude BPI-SF?"
    "Can you explain the EQ-5D choice?"

TIER3_STRATEGY — A request for a full COA strategy or battery selection.
  Examples:
    "What PRO strategy for our Phase 3 BCMA bispecific trial?"
    "Which instruments should we include?"
    "Help me design the PRO plan for this trial"

Return ONLY valid JSON — no explanation, no markdown:
{"tier": "TIER1_FACTUAL|TIER2_FOLLOWUP|TIER3_STRATEGY",
 "reason": "one sentence",
 "missing_critical": [],
 "can_answer_from_history": true|false}"""

def classify_intent(user_message: str, has_prior_strategy: bool) -> dict:
    """
    Use Haiku to classify the user's intent into three tiers.
    Returns dict with keys: tier, reason, missing_critical, can_answer_from_history
    """
    if not client:
        return {"tier": "TIER_3_STRATEGY", "reason": "client unavailable",
                "missing_critical": [], "can_answer_from_history": False}
    try:
        context_note = (
            "A full PRO strategy has already been generated in this session."
            if has_prior_strategy
            else "No prior strategy has been generated in this session."
        )
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=ROUTER_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Context: {context_note}\n\n"
                    f"User message: {user_message}"
                )
            }]
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception:
        # Default to strategy if router fails
        return {"tier": "TIER_3_STRATEGY", "reason": "router error",
                "missing_critical": [], "can_answer_from_history": False}


# =============================================================================
# KG SYNONYM BUILDER
# Uses known KG values as Layer 1, Haiku-generated extras as Layer 2
# =============================================================================

def build_search_terms(indication: str) -> list[str]:
    """
    Build the list of search terms to pass to KG queries.
    Layer 1: exact KG values we know are stored (from KG_KNOWN_VALUES).
    Layer 2: Haiku generates additional synonyms for robustness.
    Returns deduplicated list.
    """
    ind_lower = indication.lower().strip()

    # Layer 1: known exact values
    layer1 = []
    for key, values in KG_KNOWN_VALUES.items():
        if key in ind_lower or ind_lower in key:
            layer1.extend(values)

    # If we found nothing in Layer 1, use the raw string plus Haiku
    if not layer1:
        layer1 = [indication]

    # Layer 2: Haiku generates up to 5 extra synonyms
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=(
                "Generate up to 5 common synonyms, abbreviations, and alternate spellings "
                "for this medical indication as used in clinical trial databases. "
                "Return ONLY a JSON array of strings. No explanation."
            ),
            messages=[{"role": "user", "content": indication}]
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "")
        layer2 = json.loads(raw)
        if not isinstance(layer2, list):
            layer2 = []
    except Exception:
        layer2 = []

    # Merge and deduplicate, preserving order
    seen = set()
    result = []
    for t in layer1 + layer2:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# =============================================================================
# DIRECT SONNET FOR FACTUAL QUERIES (Tier 1 and Tier 2)
# =============================================================================

# def answer_direct(
#     user_message: str,
#     history: list,
#     indication: str = "",
#     prior_result: dict = None
# ) -> str:
#     """
#     Answer a simple factual or follow-up question using Sonnet + web search.
#     Optionally enriched with KG lookup for the current indication.
#     """
#     # Try a quick KG lookup if we have an indication
#     kg_context = ""
#     if indication and AGENT_AVAILABLE:
#         try:
#             terms = build_search_terms(indication)
#             records = []
#             for term in terms[:3]:
#                 r = get_instruments_by_indication(indication=term, phase="")
#                 records.extend(r)
#             if records:
#                 names = list({r.get("instrument_name", "") for r in records
#                               if r.get("instrument_name")})
#                 kg_context = (
#                     f"\n\nKnowledge graph data for '{indication}': "
#                     f"instruments found across {len(records)} precedent records: "
#                     f"{', '.join(names[:15])}."
#                 )
#         except Exception:
#             pass

#     # If there's a prior strategy, summarise it for context
#     strategy_ctx = ""
#     if prior_result:
#         battery = prior_result.get("battery_result", {}).get("battery_names", [])
#         ctx     = prior_result.get("context_json", {})
#         strategy_ctx = (
#             f"\n\nThe most recent strategy recommendation was for: "
#             f"{ctx.get('indication', '')} {ctx.get('phase', '')} "
#             f"{ctx.get('drug_class', '')}. "
#             f"Recommended battery: {', '.join(battery)}."
#         )

#     # Build message list from history (last 6 turns max)
#     messages = []
#     for m in history[-6:]:
#         if m["role"] in ("user", "assistant"):
#             messages.append({
#                 "role": m["role"],
#                 "content": m["content"][:600]  # truncate long assistant turns
#             })
#     messages.append({
#         "role": "user",
#         "content": user_message + kg_context + strategy_ctx
#     })

#     try:
#         resp = client.messages.create(
#             model="claude-sonnet-4-20250514",
#             max_tokens=2000,
#             system=(
#                 "You are a knowledgeable COA and PRO specialist. "
#                 "Answer concisely and accurately. "
#                 "Cite every factual claim as a markdown hyperlink: "
#                 "[Source Name](https://full-url.com). "
#                 "If you don't know something, say so clearly."
#             ),
#             tools=[{"type": "web_search_20250305", "name": "web_search"}],
#             messages=messages
#         )
#         return " ".join(
#             b.text for b in resp.content if hasattr(b, "text") and b.text
#         )
#     except Exception as e:
#         return f"Could not process query: {e}"

# def answer_direct(user_message: str, history: list, indication: str = None, prior_result: dict = None):
#     """
#     Answer a simple factual or follow-up question using Sonnet + web search.
#     Optionally enriched with KG lookup for the current indication.
#     Yields text chunks as they are generated, token by token
#     """
#     kg_context = ""
#     if indication and AGENT_AVAILABLE:
#         try:
#             terms = build_search_terms(indication)
#             records = []
#             for term in terms[:3]:
#                 r = get_instruments_by_indication(indication=term, phase="")
#                 records.extend(r)
#             if records:
#                 names = list({r.get("instrument_name") for r in records if r.get("instrument_name")})
#                 kg_context = (
#                     f"\n\nKG data for {indication}: {len(records)} precedent records found. "
#                     f"Instruments: {', '.join(names[:15])}."
#                 )
#         except Exception:
#             pass

#     strategy_ctx = ""
#     if prior_result:
#         battery = prior_result.get("battery_result", {}).get("battery_names", [])
#         ctx = prior_result.get("context_json", {})
#         strategy_ctx = (
#             f"\n\nMost recent strategy was for {ctx.get('indication')} {ctx.get('phase')} "
#             f"{ctx.get('drug_class')}. Recommended battery: {', '.join(battery)}."
#         )

#     messages = []
#     for m in (history or [])[-6:]:
#         if m["role"] in ("user", "assistant"):
#             messages.append({"role": m["role"], "content": m["content"][:600]})
#     messages.append({"role": "user", "content": user_message + kg_context + strategy_ctx})

#     try:
#         with client.messages.stream(
#             model="claude-sonnet-4-20250514",
#             max_tokens=2000,
#             system=(
#                 "You are a knowledgeable COA and PRO specialist. "
#                 "Answer concisely and accurately. "
#                 "For EVERY factual claim, cite the source as a markdown hyperlink inline, "
#                 "e.g. [EORTC QLQ-C30 manual](https://www.eortc.org/...). "
#                 "If a claim comes from a KG record, say 'per KG evidence'. "
#                 "If you cannot find a URL, say 'source not verified'. "
#                 "Never state a fact without a citation."
#             ),
#             tools=[{"type": "web_search_20250305", "name": "web_search"}],
#             messages=messages,
#         ) as stream:
#             for text in stream.text_stream:
#                 yield text
#     except Exception as e:
#         yield f"Could not process query: {e}"

def answer_direct(user_message: str, history: list,
                  indication: str = None,
                  prior_result: dict = None,
                  citation_index: dict = None):
    KG_SCOPE = {
        "total_drugs":  36,
        "mm_drugs":     27,
        "total_trials": 131,
    }

    effective_citation_index = (
        prior_result.get("citation_index")
        if prior_result
        else (citation_index or {})
    )

    # Replace the entire kg_context building block with:
    kg_context = ""

    if prior_result and prior_result.get("kg_evidence_block"):
        # Case 1: follow-up after a strategy run — reuse the exact same
        # scored evidence block that generated the recommendation
        kg_context = (
            f"\n\nINTERNAL KG DATA (same scored evidence used in "
            f"the recommendation above):\n"
            + prior_result["kg_evidence_block"]
            + f"\nCite records using their exact [TI-XXX], [RR-XXX], "
            f"[REJ-XXX] labels as they appear in the block above."
        )
    elif indication and AGENT_AVAILABLE:
        try:
            terms   = build_search_terms(indication)
            records = []
            for term in terms[:3]:
                r = get_instruments_by_indication(indication=term, phase="")
                records.extend(r)
            # Deduplicate by instrument name, keep best record per instrument
            seen_insts = {}
            for r in records:
                name = r.get("instrument_name", "")
                if name and name not in seen_insts:
                    seen_insts[name] = r
            deduped = list(seen_insts.values())[:12]

            if deduped:
                lines    = []
                fresh_ci = dict(citation_index) if citation_index else {}

                for idx, r in enumerate(deduped, 1):
                    inst  = r.get("instrument_name", "")
                    trial = r.get("trial_name", "") or r.get("nct_id", "")
                    drug  = r.get("drug_name", "")
                    role  = r.get("endpoint_role", "") or r.get("pro_position", "")
                    sig   = r.get("significance", "")
                    kf    = r.get("key_finding", "")

                    # Find existing label in citation_index by instrument name match
                    label = next(
                        (k for k, v in fresh_ci.items()
                         if v.get("type") == "trial_instrument"
                         and v.get("instrument") == inst),
                        None
                    )

                    # If not found, create a new entry
                    if label is None:
                        label = f"TI-{idx:03d}"
                        nct = str(r.get("nct_id", ""))
                        doi = str(r.get("publication_doi", ""))
                        fda = str(r.get("fda_label_url", ""))
                        ema = str(r.get("ema_label_url", ""))
                        _links = []
                        if nct.startswith("NCT"):
                            _links.append({
                                "label": f"ClinicalTrials.gov: {trial or nct}",
                                "url":   f"https://clinicaltrials.gov/study/{nct}"
                            })
                        if doi and doi not in ("nan", "None", ""):
                            _links.append({
                                "label": "Publication",
                                "url":   f"https://doi.org/{doi}"
                            })
                        if fda.startswith("http"):
                            _links.append({"label": f"FDA label: {drug}", "url": fda})
                        elif drug:
                            _links.append({
                                "label": f"FDA DailyMed: {drug}",
                                "url":   f"https://dailymed.nlm.nih.gov/dailymed/search.cfm"
                                         f"?query={drug.replace(' ', '+')}"
                            })
                        if ema.startswith("http"):
                            _links.append({"label": f"EMA label: {drug}", "url": ema})
                        fresh_ci[label] = {
                            "type":         "trial_instrument",
                            "instrument":   inst,
                            "trial":        trial,
                            "nct":          str(r.get("nct_id", "")),
                            "drug":         drug,
                            "phase":        r.get("phase", ""),
                            "key_finding":  str(kf or ""),
                            "endpoint_role": role,
                            "links":        _links,
                        }

                    # Build line with label always present
                    detail_parts = [p for p in [role, sig] if p]
                    detail = " · ".join(detail_parts)
                    line = f"  [{label}] {inst} — {trial} ({drug})"
                    if detail:
                        line += f"  [{detail}]"
                    if kf and kf not in ("nan", "None", ""):
                        line += f"\n    Key finding: {kf[:120]}"
                    lines.append(line)

                # Mutate citation_index in-place so caller's reference is updated
                if isinstance(citation_index, dict):
                    citation_index.clear()
                    citation_index.update(fresh_ci)

                kg_context = (
                    f"\n\nINTERNAL KG DATA ({KG_SCOPE['mm_drugs']} MM drugs, "
                    f"{KG_SCOPE['total_trials']} trials total — curated sample, not a registry):\n"
                    + "\n".join(lines)
                    + "\n\n"
                    + "Each record above has a label like [TI-001]. "
                    + "MANDATORY: when you reference any of these trials or instruments "
                    + "from KG data, cite the label immediately after the sentence — "
                    + "e.g. 'EORTC QLQ-C30 was used as a secondary endpoint in "
                    + "TOURMALINE-MM1 [TI-003].' "
                    + "Do NOT use KG counts as field-wide prevalence. "
                    + "Use them as 'for example, in our trial sample...' "
                    + "You MUST include at least two [TI-XXX] citations in your answer "
                    + "if the records above are relevant to the question."
                )
        except Exception as _kg_err:
            kg_context = f"\n\n[KG unavailable: {_kg_err}]"

    strategy_ctx = ""
    if prior_result:
        battery = prior_result.get("battery_result", {}).get("battery_names", [])
        ctx     = prior_result.get("context_json", {})
        strategy_ctx = (
            f"\n\nMost recent strategy: {ctx.get('indication')} "
            f"{ctx.get('phase')} {ctx.get('drug_class')}. "
            f"Recommended battery: {', '.join(battery)}."
        )

    system_prompt = f"""You are a knowledgeable COA and PRO specialist. \
Write as a clinical colleague, not a report generator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 1 — WEB SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the web_search tool for: prevalence, landscape frequency, \
systematic reviews, guideline documents, validation studies, \
meta-analyses, and any field-wide claim.

Search PubMed, ClinicalTrials.gov, ISPOR, EORTC, fda.gov, \
ema.europa.eu, proqolid.org, ispor.org, nice.org.uk.

Cite every web-sourced sentence as a markdown hyperlink immediately \
after the sentence — the tool always returns a URL, so one must \
always exist:
  [Regnault 2021, Leukemia](https://doi.org/10.1038/...)
  [FDA PRO Guidance 2009](https://www.fda.gov/media/77832/download)

Anchor text format: Author Year, Journal/Source.
If web_search returns no result for a claim, do NOT state the fact. \
Never cite from training memory — always call web_search first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 2 — INTERNAL KNOWLEDGE GRAPH (KG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The KG covers {KG_SCOPE['mm_drugs']} MM drugs and \
{KG_SCOPE['total_trials']} trials. It is a curated sample, \
NOT a complete registry.

Use KG data only to say: \
"For example, in trials such as X and Y [TI-001]..."
Always flag KG data with: \
"In our trial sample of {KG_SCOPE['mm_drugs']} MM drugs..."
NEVER use KG frequency counts as field-wide prevalence figures.

Cite KG records with their exact label in square brackets \
immediately after the sentence:
  [TI-001], [RR-002], [REJ-003], [IR-004], [RULE-005], [PREC-6]

The KG records available for this query are listed below. \
Use only labels that appear in that list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO COMBINE BOTH SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Lead with the web-sourced landscape answer (cited with URL).
2. Follow with KG examples that illustrate or cross-check the finding \
   (cited with [TI-XXX] labels).
3. If web and KG agree: say so explicitly — it strengthens the claim.
4. If web and KG disagree: flag the discrepancy and explain why \
   (e.g. KG is recent approvals only; web source is older literature).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY CITATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every factual sentence must end with a citation — no exceptions.
- Statistics, trial counts, approval dates, MCID values, validation \
  dates, and prevalence figures all require citations.
- Cite sentence by sentence — never bundle facts under one citation \
  at the end of a paragraph.
- Never use parenthetical style (FDA, 2009) — it is invisible to \
  the reference renderer.
- If you cannot find a source for a claim, omit the claim entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KG RECORDS AVAILABLE FOR THIS QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kg_context}
"""

    messages = []
    for m in (history or [])[-6:]:
        if m["role"] in ("user", "assistanft"):
            messages.append({"role": m["role"], "content": m["content"][:600]})
    messages.append({
        "role":    "user",
        "content": user_message + (f"\n\n{strategy_ctx}" if strategy_ctx else "")
    })

    try:
        with client.messages.stream(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1000,
            system     = system_prompt,
            tools      = [{"type": "web_search_20250305", "name": "web_search"}],
            messages   = messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Could not process query: {e}"

# =============================================================================
# RENDER ONE ASSISTANT MESSAGE (strategy result)
# =============================================================================

def render_strategy_result(answer: str, result: dict, msg_idx: int) -> None:
    """
    Render a full strategy result inside a st.chat_message("assistant") block.
    Shows battery cards, linked recommendation text, scoring detail, and references.
    """
    ctx            = result.get("context_json", {})
    battery        = result.get("battery_result", {})
    citation_index = result.get("citation_index", {})
    top_scores     = result.get("top_scores", [])
    kg_records     = result.get("kg_raw_hits", [])
    counts         = result.get("record_counts", {})

    # ── Assumptions (collapsible) ────────────────────────────────────────────
    assumptions = ctx.get("assumptions_made", [])
    if assumptions:
        with st.expander(
            f"ℹ️ What the agent understood — {len(assumptions)} inference(s) made "
            f"(click to review and correct)",
            expanded=False
        ):
            st.caption(
                "If any inference is wrong, add the correction to your next message "
                "and the agent will update the strategy."
            )
            for a in assumptions:
                st.markdown(
                    f'<span class="assumption-pill">⚠️ {a}</span>',
                    unsafe_allow_html=True
                )
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Indication:** {ctx.get('indication', '—')}")
            c2.markdown(f"**Phase:** {ctx.get('phase', '—')}")
            c3.markdown(f"**Population:** {ctx.get('population_subtype', '—')}")
            c1.markdown(f"**Administration:** {ctx.get('administration', '—')}")
            c2.markdown(f"**Drug class:** {ctx.get('drug_class', '—')}")
            c3.markdown(f"**Footprint:** {ctx.get('geographic_footprint', '—')}")
            st.markdown(
                f"**Core domains:** "
                f"{', '.join(ctx.get('core_domains_required', []))}"
            )

    # ── Battery cards ────────────────────────────────────────────────────────
    battery_list = battery.get("battery", [])
    if battery_list:
        st.markdown(
            f"**Recommended COA Battery** — "
            f"covers {len(battery.get('covered_domains', []))} of "
            f"{len(ctx.get('core_domains_required', []))} required domains"
        )
        st.caption(
            "🟢 LOW · 🟡 MODERATE · 🟠 HIGH · 🔴 CRITICAL — "
            "Risk Level is independent of score. "
            "HTA-required instruments show LOW risk because the risk is in OMITTING them, "
            "not including them."
        )
        for b in battery_list:
            role   = b.get("battery_role", "")
            is_hta = "HTA Required" in role
            card   = "battery-hta" if is_hta else "battery-card"
            ecoa   = "📱" if any(
                "+10" in f and "ecoa" in f.lower() for f in b.get("flags", [])
            ) else "📄"

            # Operational breakdown
            op_parts = []
            for f in b.get("flags", []):
                fl = f.lower()
                if "ecoa ready (+10" in fl:
                    op_parts.append("+10 eCOA")
                if "open access (+5" in fl:
                    op_parts.append("+5 open access")
                if "language data unavailable" in fl:
                    op_parts.append("−10 lang")
                if "language coverage note" in fl:
                    op_parts.append("−5 lang")
            op_str = " · ".join(op_parts) if op_parts else ""

            st.markdown(
                f'<div class="{card}">'
                f'<div class="battery-role">{role}</div>'
                f'{ecoa} <b>{b["instrument_name"]}</b> &nbsp;'
                f'{risk_badge(b["risk_level"])} &nbsp;'
                f'<span style="font-size:0.83rem">'
                f'Score: {b["scientific_score"]}/100'
                + (f' · {op_str}' if op_str else '')
                + '</span>'
                + (f'<br><span style="font-size:0.80rem;color:#555">'
                   f'{b.get("battery_note","")}</span>'
                   if b.get("battery_note") else "")
                + '</div>',
                unsafe_allow_html=True
            )

        if battery.get("gaps"):
            st.warning(
                f"⚠️ Domain coverage gap: no instrument found for "
                f"**{', '.join(battery['gaps'])}**. "
                f"The Reasoner was instructed to search the web for instruments "
                f"covering these domains."
            )

    # ── Main recommendation text (citations linkified) ───────────────────────
    linked_text, references = linkify_and_number_citations(answer, citation_index)
    st.markdown(linked_text, unsafe_allow_html=True)

    web_links = extract_web_links(answer)
    render_unified_footnotes(references, web_links)
    render_kg_evidence_cards(result)

    suspicious = audit_uncited_sentences(answer)
    if suspicious:
        with st.expander(
            f"⚠️ {len(suspicious)} sentence(s) may lack citations — review recommended",
            expanded=False
        ):
            st.caption(
                "Heuristic check only — sentences with factual language but no detected "
                "citation marker. May be false positives if citation is embedded inline."
            )
            for s in suspicious:
                st.markdown(
                    f"<div style='background:#fff8e6;padding:8px 12px;"
                    f"border-left:3px solid #EF9F27;margin:4px 0;"
                    f"border-radius:4px;font-size:0.9em'>{s[:300]}</div>",
                    unsafe_allow_html=True
                )

    # ── Scoring detail (collapsible) ──────────────────────────────────────────
    if top_scores:
        with st.expander(
            "📊 Scoring detail — how each instrument was evaluated",
            expanded=False
        ):
            st.caption(
                "🟩 Green = points earned · 🟥 Red = penalty applied · "
                "🟦 Blue = informational (no score change) · ⬜ Grey = neutral note"
            )
            with st.expander("How the scoring works", expanded=False):
                st.markdown("""
**Regulatory Fit Score (0–100)** — measures how well an instrument fits THIS specific
trial, based on the four documented causes of PRO label claim failure
(eClinicalMedicine 2023 analysis of FDA approvals 2017–2022).

| Criterion | Max | Regulatory basis |
|---|---|---|
| Content Validity — Population Match | +20 | FDA PRO Guidance (2009) Section IV — highest stated priority |
| TPP / Core Domain Fit | +35 | FDA (2021) Core PRO Cancer Guidance |
| Regulatory Acceptance | +20 | FDA PRO Guidance (2009) Section V |
| Validated MCID | +15 (gated) | FDA PRO Guidance (2009) Section V.C — no MCID = hard cap at 75 |
| MoA-Specific Sensitivity | +10 | FDA PFDD Guidance 1 (2017) |

**Penalties** reduce the score when the instrument has a specific regulatory risk for
this trial context. Risk Level is set independently — so an instrument at score 0
still shows WHY it failed.

**Operational flags** (eCOA, languages, HTA) are shown separately — they are
practical barriers, not regulatory criteria, and mixing them into the score
would be misleading.
                """)

            for inst in top_scores:
                in_battery = inst["instrument_name"] in battery.get("battery_names", [])
                label_sfx  = " 🟢 In recommended battery" if in_battery else ""
                with st.expander(
                    f"{inst['instrument_name']}{label_sfx} — "
                    f"Score: {inst['scientific_score']}/100 — "
                    f"Risk: {inst['risk_level']}",
                    expanded=False
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Score",     f"{inst['scientific_score']}/100")
                    c2.metric("Positive",  f"+{inst['raw_positive_score']}")
                    c4.metric("Op. bonus", f"{inst['operational_bonus']:+d}")

                    st.markdown(risk_badge(inst["risk_level"]), unsafe_allow_html=True)

                    # Operational breakdown
                    op_detail = []
                    for f in inst.get("flags", []):
                        fl = f.lower()
                        if "ecoa ready (+10" in fl:
                            op_detail.append("eCOA: +10")
                        if "open access (+5" in fl:
                            op_detail.append("Open access: +5")
                        if "translation gap (-15" in fl:
                            op_detail.append("Translation gap: −15")
                        if "language coverage note (-5" in fl:
                            op_detail.append("Language note: −5")
                        if "language data unavailable (-10" in fl:
                            op_detail.append("Lang unavailable: −10")
                    if op_detail:
                        st.caption(
                            "Operational: "
                            + " · ".join(op_detail)
                            + f" = Net {inst['operational_bonus']:+d}"
                        )

                    st.markdown("**Score breakdown:**")
                    for flag in inst.get("flags", []):
                        css = classify_flag(flag)
                        st.markdown(
                            f'<div class="{css}">{flag}</div>',
                            unsafe_allow_html=True
                        )

                    # KG precedent records for this instrument
                    matching = [r for r in kg_records
                                if r.get("instrument_name") == inst["instrument_name"]]
                    if matching:
                        st.markdown("**Precedent records from knowledge graph:**")
                        st.caption(
                            "These trials show where this instrument has been used before. "
                            "They are evidence of regulatory familiarity — "
                            "not properties of the instrument itself."
                        )
                        for rec in matching[:2]:
                            st.markdown(
                                f'<div style="background:#f5f5f3;border-radius:4px;'
                                f'padding:6px 10px;margin:3px 0;font-size:0.82rem">'
                                f'<b>{rec.get("trial_name", "")} '
                                f'({rec.get("nct_id", "")})</b> — '
                                f'{rec.get("drug_name", "")} · '
                                f'{rec.get("phase", "")} · '
                                f'Role: {rec.get("endpoint_role", "") or rec.get("pro_position", "")}'
                                + (f'<br>Finding: {rec.get("key_finding", "")}'
                                   if rec.get("key_finding") else "")
                                + '</div>',
                                unsafe_allow_html=True
                            )
                            html = build_source_links_html(rec)
                            if html:
                                st.markdown(html, unsafe_allow_html=True)

    # ── Language status per battery instrument ────────────────────────────────
    if battery_list and ctx.get("geographic_footprint"):
        with st.expander(
            "🌐 Language and translation status per instrument", expanded=False
        ):
            footprint = ctx.get("geographic_footprint", "Global")
            geo       = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(
                footprint, GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"]
            )
            key_langs = geo.get("key_languages", [])
            reg_note  = geo.get("regulatory_note", "")
            ref       = geo.get("reference", "")

            for b in battery_list:
                inst_lower = b["instrument_name"].lower()
                lang_count = next(
                    (v for k, v in KNOWN_LANGUAGE_COUNTS.items() if k in inst_lower),
                    0
                )
                if lang_count == 0:
                    for rec in kg_records:
                        if rec.get("instrument_name") == b["instrument_name"]:
                            raw = rec.get("languages", "")
                            if isinstance(raw, list):
                                lang_count = len([x for x in raw if x])
                            elif raw:
                                lang_count = len(
                                    [x for x in str(raw).split("|") if x.strip()]
                                )
                            break

                if lang_count >= 50:
                    icon = "✅"
                    note = f"~{lang_count} validated translations — strong global coverage."
                elif lang_count >= 15:
                    icon = "ℹ️"
                    note = (
                        f"~{lang_count} validated translations. "
                        f"Verify coverage for: {', '.join(key_langs[:6])}."
                    )
                elif lang_count > 0:
                    icon = "⚠️"
                    note = (
                        f"~{lang_count} translations found. "
                        f"Commission additional translations for trial sites. "
                        f"Typically 6–12 months [ISPOR ePRO Task Force, 2009]."
                    )
                else:
                    icon = "⚠️"
                    note = (
                        "Language data unavailable. "
                        "Verify via PROQOLID (proqolid.org) or instrument developer."
                    )

                st.markdown(
                    f'<div style="border:1px solid #AFA9EC;background:#eeedfe;'
                    f'border-radius:6px;padding:8px 12px;margin:4px 0;font-size:0.85rem">'
                    f'{icon} <b>{b["instrument_name"]}</b><br>'
                    f'<span>{note}<br>'
                    f'<small>{reg_note} ({ref})</small></span></div>',
                    unsafe_allow_html=True
                )

    # ── Footer metadata ───────────────────────────────────────────────────────
    st.caption(
        f"KG: {counts.get('instrument_records', 0)} instrument records · "
        f"{counts.get('regulatory_reviews', 0)} regulatory reviews · "
        f"{counts.get('regulatory_rules', 0)} rules · "
        f"{counts.get('rejections_found', 0)} rejection records"
    )


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("🏥 PRO COA Agent")
    st.caption("University of Cambridge × Evinova (AstraZeneca)")
    st.divider()

    st.subheader("⚙️ Trial Context")
    st.caption(
        "Optional. Fill in once to provide context across your whole session. "
        "Leave blank and describe your trial in the chat instead — "
        "the agent will ask if it needs more information."
    )
    sb_indication  = st.text_input("Indication",   placeholder="e.g. Multiple Myeloma")
    sb_phase       = st.selectbox("Phase",         ["", "Phase 3", "Phase 2", "Phase 1"])
    sb_drug_class  = st.text_input("Drug class",   placeholder="e.g. Bispecific, PI, ICI")
    sb_admin       = st.selectbox(
        "Administration",
        ["", "Step-up dosing", "IV", "Subcutaneous", "Oral", "Weekly IV"]
    )
    sb_population  = st.text_input(
        "Population", placeholder="e.g. RRMM ≥3 prior lines, Newly Diagnosed"
    )
    sb_hta         = st.multiselect(
        "HTA markets", ["NICE", "ICER", "EUnetHTA", "SMC"], default=["NICE", "ICER"]
    )
    sb_footprint   = st.selectbox(
        "Geographic footprint", ["", "Global", "EU", "US-only"]
    )

    st.divider()
    st.subheader("Knowledge base")
    c1, c2, c3 = st.columns(3)
    c1.metric("Drugs",   "36")
    c2.metric("Trials",  "131")
    c3.metric("Instr.",  "193")
    st.metric("Reviews", "68")

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages          = []
        st.session_state.results           = {}
        st.session_state.last_strategy_idx = None
        st.rerun()

    # ── DOWNLOAD CONVERSATION ─────────────────────────────
    st.divider()
    if st.session_state.get("messages"):
        lines = ["PRO COA AI Agent — Conversation Export",
                f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 60]
        for msg in st.session_state.messages:
            role = "YOU" if msg["role"] == "user" else "AGENT"
            lines.append(f"\n[{role}]\n{msg['content']}")
        if st.session_state.get("results"):
            lines += ["\n", "=" * 60, "FULL STRATEGY DATA (JSON)"]
            for idx, result in st.session_state.results.items():
                lines.append(f"\n--- Result {idx} ---")
                lines.append(json.dumps(result, indent=2, default=str))
        st.download_button(
            label="⬇ Download conversation",
            data="\n".join(lines),
            file_name=f"coa_session_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download the full conversation (all tiers) as a plain-text file.",
        )
    else:
        st.caption("No conversation to download yet.")

    st.divider()
    # ── Neo4j status ──────────────────────────────────────────────────────────
    neo4j_status = check_neo4j()
    if neo4j_status == "connected":
        st.success("🟢 Neo4j connected")
    else:
        st.error("🔴 Neo4j unavailable")
        st.caption("Resume at console.neo4j.io before running a strategy query.")

    st.divider()

    # # ── Download conversation ─────────────────────────────────────────────────
    # if st.session_state.get("messages"):
    #     lines = [
    #         "PRO COA AI Agent — Conversation Export",
    #         f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    #         "=" * 60, ""
    #     ]
    #     for msg in st.session_state.messages:
    #         role = "YOU" if msg["role"] == "user" else "AGENT"
    #         lines += [f"[{role}]", msg["content"], ""]
    #     if st.session_state.get("results"):
    #         lines += ["", "=" * 60, "FULL STRATEGY DATA (JSON)", ""]
    #         for idx, result in st.session_state.results.items():
    #             lines += [f"--- Result {idx} ---",
    #                       json.dumps(result, indent=2), ""]
    #     st.download_button(
    #         label="⬇️ Download conversation",
    #         data="\n".join(lines),
    #         file_name=f"coa_session_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
    #         mime="text/plain",
    #         use_container_width=True
    #     )

    with st.expander("💡 Example questions", expanded=False):
        examples = [
            "What PRO instruments are most used in MM trials?",
            "What is the EORTC QLQ-C30?",
            "What is MCID and why does it matter?",
            (
                "We are running a Phase 3 BCMA bispecific (teclistamab-like) "
                "in RRMM ≥3 prior lines. Step-up dosing Cycle 1. "
                "Global trial, NICE and ICER submission. "
                "TPP: treatment tolerability and physical function. "
                "What PRO strategy do we need?"
            ),
        ]
        for ex in examples:
            if st.button(ex[:60] + ("…" if len(ex) > 60 else ""), key=f"ex_{ex[:20]}"):
                st.session_state._pending_input = ex

    st.caption(
        "Project 2025gsk2 — Dept. of Chemical Engineering & Biotechnology, "
        "University of Cambridge"
    )


# =============================================================================
# MAIN AREA — HEADER
# =============================================================================
st.title("PRO COA Agent")
st.caption(
    "Ask any COA question or describe your trial for a full strategy recommendation. "
    "You can ask follow-up questions, correct assumptions, or request clarification."
)

if not AGENT_AVAILABLE:
    st.error(f"Agent failed to load: {AGENT_ERROR}. Check your .env file.")
    st.stop()


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================
if "messages"          not in st.session_state:
    st.session_state.messages          = []
if "results"           not in st.session_state:
    st.session_state.results           = {}   # msg_idx → result dict
if "last_strategy_idx" not in st.session_state:
    st.session_state.last_strategy_idx = None

# =============================================================================
# REPLAY CONVERSATION HISTORY
# =============================================================================
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            result = st.session_state.results.get(i)
            if result:
                render_strategy_result(msg["content"], result, i)
            else:
                # Simple factual answer — render plain markdown
                prior_index = (
                    st.session_state.results
                    .get(st.session_state.last_strategy_idx, {})
                    .get("citation_index", {})
                )
                linked, refs = linkify_and_number_citations(
                    msg["content"], prior_index
                )
                st.markdown(linked, unsafe_allow_html=True)
                web_links = extract_web_links(msg["content"])
                render_unified_footnotes(refs, web_links)


# =============================================================================
# CHAT INPUT
# =============================================================================
# Handle example button injection
_pending = st.session_state.pop("_pending_input", None)
prompt   = st.chat_input("Ask a COA question or describe your trial…") or _pending

if prompt:

    # ── Build sidebar context string ─────────────────────────────────────────
    sidebar_parts = []
    if sb_indication:  sidebar_parts.append(f"Indication: {sb_indication}")
    if sb_phase:       sidebar_parts.append(f"Phase: {sb_phase}")
    if sb_drug_class:  sidebar_parts.append(f"Drug class: {sb_drug_class}")
    if sb_admin:       sidebar_parts.append(f"Administration: {sb_admin}")
    if sb_population:  sidebar_parts.append(f"Population: {sb_population}")
    if sb_hta:         sidebar_parts.append(f"HTA markets: {', '.join(sb_hta)}")
    if sb_footprint:   sidebar_parts.append(f"Geographic footprint: {sb_footprint}")
    sidebar_ctx = (
        "\n\nAdditional context from sidebar:\n" + "\n".join(sidebar_parts)
        if sidebar_parts else ""
    )

    # ── Show user message ─────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Route the query ───────────────────────────────────────────────────────
    has_prior = st.session_state.last_strategy_idx is not None
    routing   = classify_intent(prompt, has_prior)
    tier      = routing.get("tier", "TIER_3_STRATEGY")

    with st.chat_message("assistant"):

        # ── TIER 1 / TIER 2: Direct Sonnet answer ────────────────────────────
        # if tier in ("TIER_1_FACTUAL", "TIER_2_FOLLOWUP"):
        #     prior_result = (
        #         st.session_state.results.get(st.session_state.last_strategy_idx)
        #         if has_prior else None
        #     )
        #     indication = (
        #         sb_indication
        #         or (prior_result.get("context_json", {}).get("indication", "")
        #             if prior_result else "")
        #     )
        #     with st.spinner("Searching…"):
        #         answer = answer_direct(
        #             user_message=prompt + sidebar_ctx,
        #             history=st.session_state.messages[:-1],
        #             indication=indication,
        #             prior_result=prior_result
        #         )

        #     # Linkify against prior citation index if available
        #     ci = (
        #         prior_result.get("citation_index", {})
        #         if prior_result else {}
        #     )
        #     linked, refs = linkify_and_number_citations(answer, ci)
        #     st.markdown(linked)
        #     render_reference_list(refs, ci)

        #     msg_idx = len(st.session_state.messages)
        #     st.session_state.messages.append({"role": "assistant", "content": answer})

        # ── TIER 1 / TIER 2: Direct Sonnet answer ────────────────────────────
        if tier in ("TIER1_FACTUAL", "TIER2_FOLLOWUP"):
            prior_result = (st.session_state.results.get(st.session_state.last_strategy_idx)
                            if has_prior else None)
            indication = (
                sb_indication
                or (prior_result.get("context_json", {}).get("indication") if prior_result else None)
            )

            # If still no indication, try to extract it from the user's message.
            # First: quick keyword match against known KG values (zero latency).
            # Second: Haiku extraction as fallback for unrecognised indications.
            if not indication:
                msg_lower = (prompt + " " + sidebar_ctx).lower()
                indication = next(
                    (key for key in KG_KNOWN_VALUES if key in msg_lower),
                    None
                )
            if not indication and AGENT_AVAILABLE:
                try:
                    _resp = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=200,
                        system=(
                            "Extract the cancer indication from this query. "
                            "Return only the indication name (e.g. 'Multiple Myeloma', "
                            "'NSCLC', 'Prostate Cancer'). "
                            "If none is mentioned, return the single word: none"
                        ),
                        messages=[{"role": "user", "content": prompt}]
                    )
                    _extracted = _resp.content[0].text.strip()
                    if _extracted.lower() != "none" and len(_extracted) < 60:
                        indication = _extracted
                except Exception:
                    pass

            # Build citation index: reuse prior strategy's if available,
            # otherwise build a fresh one from the KG for this indication.
            prior_index = (
                st.session_state.results
                .get(st.session_state.last_strategy_idx, {})
                .get("citation_index", {})
            )
            if not prior_index and indication and AGENT_AVAILABLE:
                try:
                    prior_index = build_tier1_citation_index(
                        indication=indication, phase=""
                    )
                except Exception:
                    prior_index = {}

            # If a prior strategy exists, reuse its citation index (it has richer data).
            # Otherwise build a fresh one from the KG for this specific indication.
            prior_index = (
                st.session_state.results
                .get(st.session_state.last_strategy_idx, {})
                .get("citation_index", {})
            )
            if not prior_index and indication and AGENT_AVAILABLE:
                # Build a fresh citation index so KG records get [TI-XXX] labels
                # that Sonnet can actually cite
                try:
                    prior_index = build_tier1_citation_index(
                        indication=indication, phase=""
                    )
                except Exception as _ci_err:
                    prior_index = {}

            full_response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in answer_direct(
                    user_message   = prompt + sidebar_ctx,
                    history        = st.session_state.messages[:-1],
                    indication     = indication,
                    prior_result   = prior_result,
                    citation_index = prior_index,       # ← now defined above
                ):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                linked, refs = linkify_and_number_citations(full_response, prior_index)
                web_links    = extract_web_links(full_response)
                render_unified_footnotes(refs, web_links)

            msg_idx = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citation_index": prior_index    # now contains both old + new labels
            })
            # Cache the tier1 citation index so subsequent follow-ups can reuse it
            if "tier1_citation_index" not in st.session_state:
                st.session_state.tier1_citation_index = {}
            st.session_state.tier1_citation_index.update(prior_index)

        # ── TIER 3: Full strategy pipeline ────────────────────────────────────
        else:
            # Check if clarifying questions needed
            missing = routing.get("missing_critical", [])
            if missing and not sidebar_ctx:
                # Ask one focused clarifying question before running pipeline
                question_map = {
                    "phase":               (
                        "What trial phase is this? "
                        "(Phase 1 / 2 / 3) — this affects the estimand burden "
                        "and pre-specification requirements."
                    ),
                    "geographic_footprint": (
                        "Is this a global trial seeking both FDA and EMA approval, "
                        "EU-only, or US-only? "
                        "This affects which HTA instruments are mandatory and "
                        "what translation coverage is needed."
                    ),
                    "drug_class":          (
                        "What is the drug mechanism? "
                        "(e.g. bispecific antibody, proteasome inhibitor, ICI, ADC) "
                        "This determines which mechanism-specific toxicities "
                        "the PRO battery needs to capture."
                    ),
                    "indication":          (
                        "What cancer type / indication is this trial for?"
                    ),
                }
                top_missing = missing[0].lower()
                question = next(
                    (q for k, q in question_map.items() if k in top_missing),
                    f"Could you clarify: {missing[0]}?"
                )
                st.markdown(
                    f'<div class="clarify-box">💬 Before I build the full strategy, '
                    f'one thing would significantly affect the recommendation:<br><br>'
                    f'<b>{question}</b><br><br>'
                    f'<small>You can also fill this in the sidebar and re-send your '
                    f'message, or continue and I will make a reasonable inference '
                    f'and flag it as an assumption.</small></div>',
                    unsafe_allow_html=True
                )
                answer   = f"[Clarification requested: {question}]"
                msg_idx  = len(st.session_state.messages)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

#             else:
#                 # Run full pipeline
#                 full_prompt = build_structured_prompt(
#                 prompt, sb_indication, sb_phase, sb_drug_class,
#                 sb_admin, sb_population, sb_hta, sb_footprint
# )

#                 # Live step tracker
#                 steps = [
#                     {"label": "Analyzing trial context",          "status": "running"},
#                     {"label": "Querying knowledge graph",          "status": "pending"},
#                     {"label": "Scoring and battery optimisation",  "status": "pending"},
#                     {"label": "Synthesising recommendation",       "status": "pending"},
#                 ]
#                 step_ph = st.empty()
#                 step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

#                 result = get_recommendation(full_prompt)

            else:
                # Step 1 — extract context from raw chat message only
                _raw_ctx = analyze_trial_context(prompt)

                # Step 2 — build enriched prompt (sidebar fills gaps only)
                full_prompt = build_structured_prompt(
                    prompt, _raw_ctx,
                    sb_indication, sb_phase, sb_drug_class,
                    sb_admin, sb_population, sb_hta, sb_footprint
                )

                # Step 3 — re-extract context from enriched prompt
                _pre_ctx     = analyze_trial_context(full_prompt)
                _indication  = _pre_ctx.get("indication", "unknown")
                _drug_class  = _pre_ctx.get("drug_class", "Unknown")
                _assumptions = _pre_ctx.get("assumptions_made", [])

                # Step 4 — pre-check: ask clarifying question if still too vague
                _needs_clarify = (
                    _indication == "unknown" or
                    (len(_assumptions) >= 4 and _drug_class in ["Unknown", "", None])
                )
                if _needs_clarify:
                    _question = build_clarification_question(_pre_ctx)
                    _clarify_html = (
                        f'<div class="clarify-box">'
                        f'Before I build the full strategy, one thing would significantly '
                        f'affect the recommendation:<br><br>'
                        f'<b>{_question}</b><br><br>'
                        f'<small>You can also fill this in the sidebar and re-send your '
                        f'message, or continue and I will make a reasonable inference '
                        f'and flag it as an assumption.</small></div>'
                    )
                    st.markdown(_clarify_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": _question}
                    )
                    st.stop()

                # Step 5 — full pipeline with live step tracker
                steps = [
                    {"label": "Analyzing trial context",         "status": "running"},
                    {"label": "Querying knowledge graph",         "status": "pending"},
                    {"label": "Scoring and battery optimisation", "status": "pending"},
                    {"label": "Synthesising recommendation",      "status": "pending"},
                ]
                step_ph = st.empty()
                step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

                result = get_recommendation(full_prompt)

                # Update step statuses
                steps[0]["status"] = "complete"
                steps[0]["detail"] = (
                    f"{result.get('context_json', {}).get('indication', '')} · "
                    f"{result.get('context_json', {}).get('phase', '')}"
                )
                steps[1]["status"] = (
                    "error"
                    if result.get("error_status") and
                    "offline" in str(result.get("error_status", ""))
                    else "complete"
                )
                counts = result.get("record_counts", {})
                steps[1]["detail"] = (
                    f"{counts.get('instrument_records', 0)} instruments · "
                    f"{counts.get('regulatory_reviews', 0)} reviews"
                )
                steps[2]["status"] = "complete"
                steps[2]["detail"] = (
                    f"{counts.get('scored_instruments', 0)} scored · "
                    f"battery: "
                    f"{', '.join(result.get('battery_result', {}).get('battery_names', []))}"
                )
                steps[3]["status"] = (
                    "complete" if result.get("answer") else "error"
                )
                steps[3]["detail"] = f"{len(result.get('answer', ''))} chars"
                step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

                step_ph.empty()

                if result.get("error_status"):
                    st.warning(f"Notice: {result['error_status']}")

                answer  = result.get("answer", "No recommendation generated.")
                msg_idx = len(st.session_state.messages)

                st.session_state.results[msg_idx]  = result
                st.session_state.last_strategy_idx = msg_idx
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                render_strategy_result(answer, result, msg_idx)