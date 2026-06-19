"""
Chat Interface

Architecture:
  - Multi-turn chat with session history
  - Three-tier intent routing (Haiku classifier)
  - Full strategy pipeline OR direct Sonnet for simple queries
  - Inline citations with numbered references
  - Expandable scoring detail and evidence cards
"""

import streamlit as st
import io
import json
import re
import os
import pandas as pd
from io import BytesIO
from datetime import datetime
from anthropic import Anthropic

# ═══════════════════════════════════════════════════════════════════════════════
# 0. GLOBAL SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def get_secret(key: str) -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

client = Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

st.set_page_config(
    page_title="PRO COA Agent | Cambridge × Evinova",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import agent functions (only those actually used)
from agent import (
    get_recommendation,
    analyze_trial_context,
    get_instruments_by_indication,
    build_tier1_citation_index,
)

AGENT_AVAILABLE = True   # The import succeeded; errors are caught inside the agent
AGENT_ERROR = None

# Known indication synonyms for quick lookup
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

@st.cache_data(ttl=30)
def check_neo4j() -> str:
    try:
        from graph import Neo4jConnection
        conn = Neo4jConnection(
            get_secret("NEO4J_URI"),
            get_secret("NEO4J_USERNAME"),
            get_secret("NEO4J_PASSWORD"),
        )
        conn.run_query("RETURN 1 AS ok")
        conn.close()
        return "connected"
    except Exception as e:
        return str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSS
# ═══════════════════════════════════════════════════════════════════════════════

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

/* Flag lines */
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

/* Citation superscripts */
sup a { color: #185FA5; font-size: 0.75em; text-decoration: none; font-weight: 600 }

/* Clarifying question box */
.clarify-box   { border:1px solid #9FE1CB; background:#f0faf6; border-radius:8px;
                  padding:12px 16px; margin:6px 0; font-size:0.90rem }
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PURE UTILITY FUNCTIONS (no Streamlit calls)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_flag(flag_text: str) -> str:
    t = flag_text.upper()
    if "🔴" in flag_text or "CRITICAL FLAG" in t:
        return "CRITICAL"
    if "🟠" in flag_text or "HIGH FLAG" in t:
        return "HIGH"
    if "🟡" in flag_text or "MODERATE FLAG" in t:
        return "MODERATE"
    if any(k in t for k in ["CRITICAL", "RECALL INCOMPATIBILITY", "DOMAIN FAILURE",
                            "INSTRUMENT-ATTRIBUTED REJECTION"]):
        return "CRITICAL"
    if any(k in t for k in ["HIGH", "NOT PRE-SPECIFIED", "ESTIMAND BURDEN"]):
        return "HIGH"
    if any(k in t for k in ["MODERATE", "MODE EQUIVALENCE", "ASYMPTOMATIC",
                            "DISTRIBUTION-BASED", "TRANSLATION PENALTY"]):
        return "MODERATE"
    return "INFO"

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

def extract_web_links(text: str) -> list:
    """Extract [anchor](url) markdown links, dedup by URL."""
    raw = re.findall(r'\[([^\]]+)\]\((https?://[^\)\s]+)\)', text)
    seen, out = set(), []
    for anchor, url in raw:
        if url not in seen:
            seen.add(url)
            out.append((anchor, url))
    return out

def convert_web_citations_to_numbers(text: str, start_n: int = 1) -> tuple:
    """Convert markdown links to numbered superscript HTML."""
    pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)\s]+)\)')
    seen_urls: dict = {}
    footnotes: list = []

    def replace(match):
        anchor = match.group(1)
        url    = match.group(2)
        if url not in seen_urls:
            n = len(seen_urls) + start_n
            seen_urls[url] = n
            footnotes.append((n, anchor, url))
        n = seen_urls[url]
        return (
            f'<sup><a href="{url}" target="_blank" '
            f'style="color:#185FA5;font-weight:bold;text-decoration:none;'
            f'font-size:0.8em">[{n}]</a></sup>'
        )
    return pattern.sub(replace, text), footnotes

def linkify_and_number_citations(text: str, citation_index: dict, start_n: int = 1) -> tuple:
    """Replace [TI-001], [CT-002], etc. with numbered superscript HTML links."""
    text = re.sub(r'\[\d+-\d+\]', '', text)   # strip [13-4] style indices
    pattern = re.compile(r'(TI|CT|RR|REJ|IR|RULE|PREC|COMP)-(\d{1,3})')
    ref_seen: dict = {}
    ref_order: list = []

    def _replace(m):
        prefix    = m.group(1)
        digits    = m.group(2)
        raw_label = f"{prefix}-{digits}"
        label     = f"{prefix}-{digits.zfill(3)}"
        info = citation_index.get(label) or citation_index.get(raw_label) or {}
        if label not in ref_seen:
            ref_seen[label] = len(ref_seen) + start_n
            ref_order.append((label, info))
        n = ref_seen[label]
        links = info.get("links", [])
        url = next((l["url"] for l in links if l.get("url", "").startswith("http")), None)
        if url:
            return (
                f'<sup><a href="{url}" target="_blank" '
                f'style="color:#1D9E75;font-weight:bold;text-decoration:none;'
                f'font-size:0.8em">[{n}]</a></sup>'
            )
        return (
            f'<sup style="color:#1D9E75;font-weight:bold;'
            f'font-size:0.8em">[{n}]</sup>'
        )
    modified   = pattern.sub(_replace, text)
    references = [(label, ref_seen[label], info) for label, info in ref_order]
    return modified, references

def _markdown_table_to_df(text: str, heading_keyword: str) -> pd.DataFrame:
    """Extract first markdown table after a heading containing `heading_keyword`."""
    lines = text.split("\n")
    table_lines, found_heading, collecting = [], False, False
    for line in lines:
        s = line.strip()
        if not found_heading and heading_keyword.lower() in s.lower():
            found_heading = True
            continue
        if found_heading:
            if s.startswith("|"):
                collecting = True
                table_lines.append(s)
            elif collecting:
                break
    if len(table_lines) < 3:
        return None

    def parse_row(r):
        # Remove HTML superscript tags and their content
        cell_text = re.sub(r'<sup[^>]*>.*?</sup>', '', r, flags=re.DOTALL)
        # Remove KG citation labels: [TI-001], [CT-002], [RR-003], [REJ-004], [RULE-005]
        cell_text = re.sub(r'\[(TI|CT|RR|REJ|RULE)-\d{3}\]', '', cell_text)
        # Remove any remaining bracketed numbers like [1], [[1]], [1,2], etc.
        cell_text = re.sub(r'\[{1,2}\d+(?:,\d+)*\]{1,2}', '', cell_text)
        return [c.strip() for c in cell_text.split("|") if c.strip()]

    headers = parse_row(table_lines[0])
    rows = []
    for line in table_lines[2:]:
        cells = parse_row(line)
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return pd.DataFrame(rows) if rows else None

def build_source_links_html(record: dict) -> str:
    """HTML source-link row for a KG record."""
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
            f'<a href="https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={query}" target="_blank">'
            f'FDA DailyMed</a>'
        )
    if str(ema).startswith("http"):
        links.append(f'<a href="{ema}" target="_blank">EMA label: {drug}</a>')
    if not links:
        return ""
    return '<div class="source-row">' + " &nbsp;·&nbsp; ".join(links) + "</div>"

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRIAL CONTEXT HELPERS (for sidebar gap-filling)
# ═══════════════════════════════════════════════════════════════════════════════

def build_structured_prompt(
    user_message: str,
    context: dict,
    sb_indication: str,
    sb_phase: str,
    sb_drug_class: str,
    sb_admin: str,
    sb_population: str,
    sb_hta: list,
    sb_footprint: str,
) -> str:
    """Append sidebar values only for fields Haiku could not extract."""
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
        f"I will proceed with the following inferences: "
        f"**{indication}** · **{phase or 'Phase 3'}** · **{drug_class or 'unknown drug class'}**. "
        f"Correct anything in your next message if needed."
    )

def _show_confirmation_card(ctx: dict) -> None:
    """Informational card – shown just before the pipeline runs."""
    assumptions = ctx.get("assumptions_made", [])
    fields = " &nbsp;·&nbsp; ".join(filter(None, [
        f"<b>{ctx.get('indication', '—')}</b>",
        ctx.get('phase', '—'),
        ctx.get('drug_class', '—'),
        ctx.get('population_subtype', ''),
        f"Footprint: {ctx.get('geographic_footprint', '—')}" if ctx.get('geographic_footprint') else "",
        f"HTA: {', '.join(ctx.get('hta_markets', []))}" if ctx.get('hta_markets') else "",
    ]))
    if not assumptions:
        html = (
            '<div class="clarify-box">'
            '✅ <b>All parameters extracted from your message.</b><br>'
            f'{fields}'
            '</div>'
        )
    else:
        inferences = "<br>".join(f"&nbsp;&nbsp;⚠️ {a}" for a in assumptions)
        html = (
            '<div class="clarify-box">'
            '<b>📋 Parameters extracted from your message:</b><br>'
            f'{fields}<br><br>'
            '<b>Inferences made (not stated in your message):</b><br>'
            f'{inferences}<br><br>'
            '<small>If anything is wrong, send a correction in your next message '
            'and the analysis will update.</small>'
            '</div>'
        )
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RENDERING FUNCTIONS (used by `render_strategy_result`)
# ═══════════════════════════════════════════════════════════════════════════════

def render_unified_footnotes(references: list, web_links: list):
    """Show KG + web citations in a combined footnote panel."""
    if not references and not web_links:
        st.caption("⚠️ No citations found in this answer.")
        return

    st.markdown("---")
    st.markdown("#### 📎 Sources")

    for label, num, info in references:
        ctype = info.get("type", "")
        if ctype == "trial_instrument":
            trial_str = f"*{info['trial']}*" if info.get("trial") else ""
            nct_str   = f"`{info['nct']}`" if info.get("nct", "").startswith("NCT") else ""
            header    = " · ".join(filter(None, [
                f"**{info.get('instrument','')}**",
                trial_str, nct_str,
                info.get("drug",""), info.get("phase",""),
            ]))
            st.markdown(f"**[{num}]** &nbsp; 🗄️ &nbsp; `{label}` · {header}")
            kf = info.get("key_finding", "")
            if kf and kf not in ("nan", "None", "", "—"):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#555;font-size:0.9em'>"
                    f"Key finding: {kf[:200]}</span>", unsafe_allow_html=True
                )
            meta_parts = filter(None, [
                f"Score {info['score']}/100" if info.get("score") else "",
                f"Risk: {info['risk']}"      if info.get("risk")  else "",
                f"Role: {info['endpoint_role']}" if info.get("endpoint_role") else "",
            ])
            meta = " · ".join(meta_parts)
            if meta:
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#888;font-size:0.85em'>"
                    f"{meta}</span>", unsafe_allow_html=True
                )

        elif ctype == "comparator_trial":
            nct_str = f"`{info['nct']}`" if info.get("nct","").startswith("NCT") else ""
            st.markdown(
                f"**[{num}]** &nbsp; 🗂️ &nbsp; `{label}` · "
                f"**{info.get('trial','')}** · {info.get('drug','')} · "
                f"{info.get('phase','')} {nct_str}"
            )

        elif ctype == "regulatory_review":
            st.markdown(
                f"**[{num}]** &nbsp; ✅ &nbsp; `{label}` · "
                f"**{info.get('agency','')}** review · **{info.get('drug','')}** · "
                f"Decision: {info.get('decision','')}"
            )
        elif ctype == "rejection":
            st.markdown(
                f"**[{num}]** &nbsp; ⚠️ &nbsp; `{label}` · "
                f"**{info.get('agency','')}** · **{info.get('drug','')}** · "
                f"Decision: {info.get('decision','')}"
            )
        else:
            st.markdown(f"**[{num}]** &nbsp; `{label}`")

        links = info.get("links", [])
        valid_links = [l for l in links if l.get("url","").startswith("http")]
        if valid_links:
            st.markdown(
                "&nbsp;&nbsp;&nbsp;&nbsp;"
                + " &nbsp;·&nbsp; ".join(
                    f"[{l['label']}]({l['url']})" for l in valid_links
                )
            )
        st.markdown(
            "<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
            unsafe_allow_html=True,
        )

    for item in web_links:
        if len(item) == 3:
            num, anchor, url = item
        else:
            anchor, url = item
            num = "?"
        st.markdown(f"**[{num}]** &nbsp; 🌐 &nbsp; [{anchor}]({url})")
        st.markdown(
            "<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
            unsafe_allow_html=True,
        )

def render_kg_evidence_cards(result: dict):
    """Collapsible panel showing all raw KG records (instrument + regulatory)."""
    kg_records   = result.get("kg_raw_hits", [])
    reg_records = result.get("reg_records", [])
    if not kg_records and not reg_records:
        return

    with st.expander(
        f"🗄️ Full knowledge graph data — "
        f"{len(kg_records)} instrument records · {len(reg_records)} regulatory reviews",
        expanded=False,
    ):
        if kg_records:
            st.markdown("#### Instrument trial records")
            st.caption(
                "Every instrument record retrieved from the KG for this indication. "
                "All NCT, DOI, FDA, and EMA columns are clickable links."
            )
            # We use the raw trial-instrument records, not the scored summaries
            kg_records = result.get("kg_raw_hits", [])
            
            _na_vals = {"", "not applicable", "nan", "none", "n/a", "na", "tbd"}
            def _is_useful(rec):
                return any(
                    str(rec.get(k, "") or "").strip().lower() not in _na_vals
                    for k in ("trial_name", "drug_name", "key_finding", "significance")
                )
            useful_records = [r for r in kg_records if _is_useful(r)]
                        
            ci = result.get("citation_index", {})
            inst_to_label = {
                v.get("instrument", ""): k
                for k, v in ci.items()
                if v.get("type") == "trial_instrument"
            }

            # Sort by the numeric part of the Ref label, then by instrument name
            def _ref_sort_key(rec):
                inst = rec.get("instrument_name", "")
                label = inst_to_label.get(inst, "ZZ-999")
                # Extract digits from label like "TI-003" → 3; fallback to 0
                nums = re.findall(r'\d+', label)
                return (int(nums[0]) if nums else 0, inst)

            useful_records.sort(key=_ref_sort_key)
            
            rows = []
            for i, rec in enumerate(useful_records[:30], 1):   # show up to 30
                nct     = str(rec.get("nct_id", ""))
                doi     = str(rec.get("publication_doi", ""))
                fda_url = str(rec.get("fda_label_url", ""))
                ema_url = str(rec.get("ema_label_url", ""))
                kf      = str(rec.get("key_finding", "") or "")
                inst_name = rec.get("instrument_name", "")
                ref_label = inst_to_label.get(inst_name, "")
                rows.append({
                    "Ref":         ref_label,
                    "Instrument":  inst_name,
                    "Drug":        rec.get("drug_name", ""),
                    "Trial":       rec.get("trial_name", "") or "",
                    "Phase":       rec.get("phase", ""),
                    "Role":        rec.get("endpoint_role", "") or rec.get("pro_position", ""),
                    "Key finding": kf[:100] if len(kf) > 100 else kf,
                    "Year":        rec.get("publication_year", ""),
                    "P-value":     rec.get("p_value", ""),
                    "Effect size": rec.get("effect_size", ""),
                    "NCT": f"https://clinicaltrials.gov/study/{nct}" if nct.startswith("NCT") else "",
                    "DOI": f"https://doi.org/{doi}" if doi and doi not in ("nan", "None", "") else "",
                    "FDA": fda_url if fda_url.startswith("http") else "",
                    "EMA": ema_url if ema_url.startswith("http") else "",
                })

            df_instr = pd.DataFrame(rows)
            st.dataframe(
                df_instr,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "NCT": st.column_config.LinkColumn("NCT"),
                    "DOI": st.column_config.LinkColumn("DOI"),
                    "FDA": st.column_config.LinkColumn("FDA"),
                    "EMA": st.column_config.LinkColumn("EMA"),
                },
            )
            csv_instr = df_instr.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download instrument evidence as CSV",
                data=csv_instr,
                file_name="instrument_evidence_records.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if reg_records:
            st.markdown("#### Regulatory review records")
            st.caption("FDA/EMA decisions with PRO outcomes from the KG.")

            non_rejections = [r for r in reg_records if not r.get("rejection_reason_primary")]
            # Filter out placeholder reasons that are not real rejections
            _rej_placeholder = {"not applicable", "not reported", "not available", "none", "", "nan"}
            rejections = [
                r for r in reg_records
                if r.get("rejection_reason_primary")
                and str(r.get("rejection_reason_primary")).strip().lower() not in _rej_placeholder
            ]

            review_rows = []
            for i, rr in enumerate(non_rejections[:15], 1):
                review_rows.append({
                    "Ref": f"RR-{i:03d}",
                    "Drug": rr.get("drug_name", ""),
                    "Agency": rr.get("agency", ""),
                    "Decision": rr.get("decision", ""),
                    "Accepted instruments": rr.get("instruments_accepted", ""),
                    "Claim type": rr.get("claim_type", ""),
                    "Label language": rr.get("label_language", ""),
                    "FDA": str(rr.get("fda_label_url", "")) if str(rr.get("fda_label_url", "")).startswith("http") else "",
                    "EMA": str(rr.get("ema_label_url", "")) if str(rr.get("ema_label_url", "")).startswith("http") else "",
                })
            for i, rr in enumerate(rejections[:15], 1):
                review_rows.append({
                    "Ref": f"REJ-{i:03d}",
                    "Drug": rr.get("drug_name", ""),
                    "Agency": rr.get("agency", ""),
                    "Decision": rr.get("decision", ""),
                    "Accepted instruments": rr.get("instruments_accepted", ""),
                    "Claim type": rr.get("claim_type", ""),
                    "Label language": rr.get("label_language", ""),
                    "Primary rejection reason": rr.get("rejection_reason_primary", ""),
                    "Detailed rejection reason": rr.get("rejection_reason_detailed", ""),
                    "FDA": str(rr.get("fda_label_url", "")) if str(rr.get("fda_label_url", "")).startswith("http") else "",
                    "EMA": str(rr.get("ema_label_url", "")) if str(rr.get("ema_label_url", "")).startswith("http") else "",
                })

            if review_rows:
                df_reviews = pd.DataFrame(review_rows)
                st.download_button(
                    "Download regulatory review evidence as CSV",
                    data=df_reviews.to_csv(index=False).encode("utf-8"),
                    file_name="regulatory_review_records.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Simple rendering of reviews (accepted + rejected)
            for i, rr in enumerate(non_rejections[:15], 1):
                icon = "✅"
                fda_url = str(rr.get("fda_label_url", ""))
                ema_url = str(rr.get("ema_label_url", ""))
                drug    = rr.get("drug_name", "")
                with st.container():
                    c1, c2 = st.columns([1, 7])
                    with c1:
                        st.markdown(f"`RR-{i:03d}` {icon}")
                    with c2:
                        st.markdown(f"**{drug}** — {rr.get('agency','')} | **{rr.get('decision','')}**")
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
                                f"[DailyMed](https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(' ','+')})"
                            )
                        st.markdown(" · ".join(links))
                st.markdown("<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
                            unsafe_allow_html=True)

            for i, rr in enumerate(rejections[:15], 1):
                icon = "⚠️"
                fda_url = str(rr.get("fda_label_url", ""))
                ema_url = str(rr.get("ema_label_url", ""))
                drug    = rr.get("drug_name", "")
                with st.container():
                    c1, c2 = st.columns([1, 7])
                    with c1:
                        st.markdown(f"`REJ-{i:03d}` {icon}")
                    with c2:
                        st.markdown(f"**{drug}** — {rr.get('agency','')} | **{rr.get('decision','')}**")
                        accepted = rr.get("instruments_accepted", "")
                        if accepted and str(accepted) not in ("nan","None",""):
                            st.markdown(f"✅ Accepted: `{accepted}`")
                        claim = rr.get("claim_type", "")
                        if claim and str(claim) not in ("nan","None",""):
                            st.markdown(f"🏷️ Claim type: {claim}")
                        label_lang = str(rr.get("label_language","") or "")
                        if label_lang and label_lang not in ("nan","None",""):
                            st.markdown(f"📄 *{label_lang[:250]}*")
                        st.markdown(f"❌ **Rejection reason:** {rr.get('rejection_reason_primary','')}")
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
                                f"[DailyMed](https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(' ','+')})"
                            )
                        st.markdown(" · ".join(links))
                st.markdown("<hr style='margin:6px 0;border:none;border-top:1px solid #eee'>",
                            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TIER 1 / TIER 2 ANSWERING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def build_search_terms(indication: str) -> list[str]:
    ind_lower = indication.lower().strip()
    layer1 = []
    for key, values in KG_KNOWN_VALUES.items():
        if key in ind_lower or ind_lower in key:
            layer1.extend(values)
    if not layer1:
        layer1 = [indication]
    layer2 = []
    if not layer1 or layer1 == [indication]:
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system=(
                    "Generate up to 5 common synonyms, abbreviations, and alternate "
                    "spellings for this medical indication as used in clinical trial "
                    "databases. Return ONLY a JSON array of strings. No explanation."
                ),
                messages=[{"role": "user", "content": indication}]
            )
            raw = resp.content[0].text.strip().replace("```json","").replace("```","")
            layer2 = json.loads(raw) if isinstance(json.loads(raw), list) else []
        except Exception:
            layer2 = []
    seen = set()
    result = []
    for t in layer1 + layer2:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result

def answer_direct(user_message: str, history: list,
                  indication: str = None,
                  prior_result: dict = None,
                  citation_index: dict = None):
    KG_SCOPE = {"total_drugs": 36, "mm_drugs": 27, "total_trials": 131}
    effective_citation_index = (
        prior_result.get("citation_index") if prior_result else (citation_index or {})
    )

    kg_context = ""
    if prior_result and prior_result.get("kg_evidence_block"):
        scored = prior_result.get("top_scores", [])[:8]
        score_lines = [
            f"  [{i+1}] {s['instrument_name']}: score={s['scientific_score']}/100, "
            f"risk={s['risk_level']}, flags={len(s.get('flags', []))}"
            for i, s in enumerate(scored)
        ]
        ci = prior_result.get("citation_index", {})
        ci_lines = [
            f"  [{label}] {info.get('instrument', info.get('drug', label))}"
            f" — {info.get('trial', info.get('source', ''))}"
            for label, info in list(ci.items())[:15]
        ]
        kg_context = (
            "\n\nPRIOR STRATEGY CONTEXT:\n"
            "Scored instruments (top 8):\n" + "\n".join(score_lines) +
            "\n\nAvailable citation labels:\n" + "\n".join(ci_lines) +
            "\nCite using [TI-XXX], [CT-XXX], [RR-XXX], [REJ-XXX] labels."
        )
    elif indication and AGENT_AVAILABLE:
        try:
            terms = build_search_terms(indication)
            records = []
            for term in terms[:3]:
                r = get_instruments_by_indication(indication=term, phase="")
                records.extend(r)
            seen_insts = {}
            for r in records:
                name = r.get("instrument_name", "")
                if name and name not in seen_insts:
                    seen_insts[name] = r
            deduped = list(seen_insts.values())[:12]

            if deduped:
                lines = []
                fresh_ci = dict(citation_index) if citation_index else {}
                for idx, r in enumerate(deduped, 1):
                    inst = r.get("instrument_name", "")
                    trial = r.get("trial_name", "") or r.get("nct_id", "")
                    drug = r.get("drug_name", "")
                    role = r.get("endpoint_role", "") or r.get("pro_position", "")
                    sig = r.get("significance", "")
                    kf = r.get("key_finding", "")
                    label = next(
                        (k for k, v in fresh_ci.items()
                         if v.get("type") == "trial_instrument" and v.get("instrument") == inst),
                        None,
                    )
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
                                "url": f"https://clinicaltrials.gov/study/{nct}"
                            })
                        if doi and doi not in ("nan", "None", ""):
                            _links.append({"label": "Publication", "url": f"https://doi.org/{doi}"})
                        if fda.startswith("http"):
                            _links.append({"label": f"FDA label: {drug}", "url": fda})
                        elif drug:
                            _links.append({
                                "label": f"FDA DailyMed: {drug}",
                                "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(' ', '+')}"
                            })
                        if ema.startswith("http"):
                            _links.append({"label": f"EMA label: {drug}", "url": ema})
                        fresh_ci[label] = {
                            "type": "trial_instrument",
                            "instrument": inst,
                            "trial": trial,
                            "nct": str(r.get("nct_id", "")),
                            "drug": drug,
                            "phase": r.get("phase", ""),
                            "key_finding": str(kf or ""),
                            "endpoint_role": role,
                            "links": _links,
                        }
                    detail_parts = [p for p in [role, sig] if p]
                    detail = " · ".join(detail_parts)
                    line = f"  [{label}] {inst} — {trial} ({drug})"
                    if detail:
                        line += f"  [{detail}]"
                    if kf and kf not in ("nan", "None", ""):
                        line += f"\n    Key finding: {kf[:120]}"
                    lines.append(line)
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
                    + "from KG data, cite the label immediately after the sentence..."
                )
        except Exception as _kg_err:
            kg_context = f"\n\n[KG unavailable: {_kg_err}]"

    strategy_ctx = ""
    if prior_result:
        ctx = prior_result.get("context_json", {})
        top_candidates = [
            s["instrument_name"]
            for s in prior_result.get("top_scores", [])[:5]
            if s.get("scientific_score", 0) >= 40
        ]
        strategy_ctx = (
            f"\n\nMost recent strategy: {ctx.get('indication')} "
            f"{ctx.get('phase')} {ctx.get('drug_class')}. "
            f"Top scored candidates: {', '.join(top_candidates) or 'none above threshold'}."
        )

    # ── Base system prompt (used for both Tier 1 and Tier 2) ──
    base_system = f"""You are a knowledgeable COA and PRO specialist. \
Write as a clinical colleague, not a report generator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 1 — WEB SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the web_search tool for: prevalence, landscape frequency, \
systematic reviews, guideline documents, validation studies, \
meta-analyses, and any field-wide claim.

Search PubMed, ClinicalTrials.gov, ISPOR, EORTC, fda.gov, \
ema.europa.eu, proqolid.org, ispor.org, nice.org.uk.

SEARCH LIMIT: Maximum 5 web searches total per answer. \
Choose the 5 most important queries. After 5 searches, write the answer. \
Do not search the same source twice. \
Total answer must be under 2000 words — be concise and direct.

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

    # ── TIER 2 FOLLOW‑UP RULES (prepended only when a prior strategy exists) ──
    if prior_result:
        follow_up_rules = (
            "IMPORTANT — TIER 2 FOLLOW‑UP RULES:\n"
            "- You are answering a specific follow‑up question about a COA strategy that was already generated.\n"
            "- Answer in 3–5 concise sentences. Do NOT reproduce any table. Do NOT propose a new strategy.\n"
            "- Cite only the relevant evidence from the prior strategy using KG labels and web‑source hyperlinks.\n"
            "- If the question can be answered directly from the previously generated tables, do so WITHOUT any new web searches.\n"
            "- Only perform web searches if the question explicitly asks for new information not already in the strategy.\n\n"
        )
        system_prompt = follow_up_rules + base_system
    else:
        system_prompt = base_system

    messages = []
    for m in (history or [])[-6:]:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"][:600]})
    messages.append({
        "role": "user",
        "content": user_message + (f"\n\n{strategy_ctx}" if strategy_ctx else "") +
                   (f"\n\nIMPORTANT: You MUST cite at least 2 KG records using [TI-XXX] labels..."
                    if kg_context else "")
    })
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            system=system_prompt,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Could not process query: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN RENDERING FUNCTION FOR TIER 3 RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_strategy_result(answer: str, result: dict, msg_idx: int) -> None:
    """Render a full strategy result (tables, evidence cards, etc.)."""
    ctx            = result.get("context_json", {})
    coverage       = result.get("coverage", {})
    citation_index = result.get("citation_index", {})
    top_scores     = result.get("top_scores", [])
    kg_records     = result.get("kg_raw_hits", [])
    counts         = result.get("record_counts", {})

    # Assumptions
    assumptions = ctx.get("assumptions_made", [])
    if assumptions:
        with st.expander(
            f"ℹ️ What the agent understood — {len(assumptions)} inference(s) made",
            expanded=False,
        ):
            st.caption("If any inference is wrong, add the correction to your next message...")
            for a in assumptions:
                st.markdown(f'<span class="assumption-pill">⚠️ {a}</span>', unsafe_allow_html=True)
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Indication:** {ctx.get('indication','—')}")
            c2.markdown(f"**Phase:** {ctx.get('phase','—')}")
            c3.markdown(f"**Population:** {ctx.get('population_subtype','—')}")
            c1.markdown(f"**Administration:** {ctx.get('administration','—')}")
            c2.markdown(f"**Drug class:** {ctx.get('drug_class','—')}")
            c3.markdown(f"**Footprint:** {ctx.get('geographic_footprint','—')}")

    # HTA warnings
    hta_mandatory = coverage.get("hta_mandatory", [])
    if hta_mandatory:
        for h in hta_mandatory:
            st.warning(f"⚠️ **HTA Requirement:** {h['instrument']} must be added for **{h['market']}** — {h['reason']}")

    if coverage.get("item_library_applicable"):
        st.info(
            "ℹ️ **Item library note:** Comparator trials in the knowledge graph used "
            "subscale or item-library approaches rather than full instruments..."
        )

    comparator_trials = coverage.get("comparator_trials", [])
    if comparator_trials:
        with st.expander(
            f"ℹ️ Why were these {len(comparator_trials)} comparator trials chosen?",
            expanded=False
        ):
            drug_class = ctx.get("drug_class", "")
            moa_aliases = ctx.get("moa_aliases", [])
            # Build a set of all terms that would indicate "same drug class"
            all_terms = [drug_class.strip().lower()] if drug_class else []
            all_terms += [a.strip().lower() for a in moa_aliases if a and a.strip()]
            st.caption(
                "Trials ranked by: (1) same drug class as current trial, "
                "(2) overlapping mechanism keywords."
            )
            for comp in comparator_trials:
                comp_class = comp.get("drug_class", "")          # <--- defined first
                comp_class_lower = comp_class.strip().lower()
                # Check if any of our terms appears in the trial's drug class or vice versa
                is_same = False
                if all_terms:
                    is_same = any(
                        t in comp_class_lower or comp_class_lower in t
                        for t in all_terms
                    )
                tag = "🟢 Same drug class" if is_same else "🟡 Different mechanism"
                inst_names = ", ".join(
                    i["name"] for i in comp.get("instruments", []) if i.get("name")
                ) or "No instruments recorded"
                st.markdown(
                    f"**{comp['trial_name']}** &nbsp; {tag} &nbsp; "
                    f"({comp_class} · {comp.get('phase', '')})<br>"
                    f"<span style='color:#555;font-size:0.9em'>"
                    f"Instruments: {inst_names}</span>",
                    unsafe_allow_html=True,
                )
            if all(not is_same for comp in comparator_trials):
                st.warning("No same‑class comparator trials found in the KG. Showing the most relevant trials from the same indication.")

    # # Get all drug class terms from context
    # drug_class = ctx.get("drug_class", "")
    # moa_aliases = ctx.get("moa_aliases", [])
    # all_terms = [drug_class.lower()] + [a.lower() for a in moa_aliases]
    # # check if any term is a substring of the trial's class, or vice versa
    # comp_class_lower = comp_class.lower()
    # is_same = any(t in comp_class_lower or comp_class_lower in t for t in all_terms if t)

    domain_rows = coverage.get("domains", [])
    if domain_rows:
        with st.expander(f"📋 Domain coverage overview", expanded=False):
            for d in domain_rows:
                top = d["candidates"][0] if d["candidates"] else None
                icon = "✅" if top else "❌"
                core_tag = " *(FDA core)*" if d.get("is_fda_core") else ""
                if top:
                    st.markdown(f"{icon} **{d['domain']}{core_tag}** — top candidate: {top['instrument']} (score {top['score']})")
                else:
                    st.markdown(f"{icon} **{d['domain']}{core_tag}** — no candidate found")

    reg_rules = result.get("reg_rules", [])
    if reg_rules:
        with st.expander(f"Regulatory rules for instrument selection ({len(reg_rules)})", expanded=False):
            rows = []
            for i, r in enumerate(reg_rules, 1):
                rows.append({
                    "Label": f"RULE-{i:03d}",
                    "Decision type": (r.get("decision_type") or "").upper(),
                    "Source document": r.get("source_document", ""),
                    "Rule text": (r.get("rule_text", "") or "")[:300],
                })
            df_rules = pd.DataFrame(rows)
            st.dataframe(df_rules, use_container_width=True)
            st.download_button("Download regulatory rules as CSV",
                               data=df_rules.to_csv(index=False).encode("utf-8"),
                               file_name="regulatory_rules.csv", mime="text/csv",
                               use_container_width=True)

    # Main text with citations
    linked_text, references = linkify_and_number_citations(answer, citation_index)
    final_text, web_footnotes = convert_web_citations_to_numbers(linked_text, start_n=len(references) + 1)
    st.markdown(final_text, unsafe_allow_html=True)

    # Download buttons for individual tables
    table_exports = [
        ("Table 1 – Domain Coverage",          "Domain Coverage Comparison",   "domain_coverage_table1.csv"),
        ("Table 2 – PRO Measures",             "PRO Measures Comparison",      "pro_measures_table2.csv"),
        ("Table 3 – Instrument Gap Analysis",  "Instrument Gap Analysis",      "gap_analysis_table3.csv"),
        ("Table 4 – Endpoint Positioning",     "Endpoint Positioning",         "pro_endpoint_table4.csv"),
        ("Table 5 – Language & Translation",   "Language & Translation",       "language_table5.csv"),
    ]
    row1_cols = st.columns(3)
    row2_cols = st.columns(2)
    all_cols = row1_cols + row2_cols
    for col, (label, keyword, fname) in zip(all_cols, table_exports):
        df_extracted = _markdown_table_to_df(final_text, keyword)
        if df_extracted is not None and not df_extracted.empty:
            with col:
                st.download_button(f"⬇ Download {label}",
                                   data=df_extracted.to_csv(index=False).encode("utf-8"),
                                   file_name=fname, mime="text/csv",
                                   use_container_width=True)

    # Excel download for all tables
    if any(_markdown_table_to_df(final_text, kw) is not None for _, kw, _ in table_exports):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for label, kw, _ in table_exports:
                df = _markdown_table_to_df(final_text, kw)
                if df is not None and not df.empty:
                    sheet_name = label[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        st.download_button("⬇ Download all tables (Excel)",
                           data=output.getvalue(),
                           file_name="coa_tables.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    render_unified_footnotes(references, web_footnotes)
    render_kg_evidence_cards(result)

    # if top_scores:
    #     with st.expander("📊 Evidence context — instrument scoring detail", expanded=False):
    #         st.caption("This shows how instruments were ranked before Sonnet's synthesis. "
    #            "The agent's final recommendations may differ — see the narrative above.")

    #         with st.expander("How the scoring works", expanded=False):
    #             st.markdown("""
    #         **Regulatory Fit Score (0–100)** — measures how well an instrument fits THIS specific
    #         trial, based on the four documented causes of PRO label claim failure
    #         (eClinicalMedicine 2023 analysis of FDA approvals 2017–2022).

    #         | Criterion | Max | Regulatory basis |
    #         |---|---|---|
    #         | Content Validity — Population Match | +20 | FDA PRO Guidance (2009) Section IV — highest stated priority |
    #         | TPP / Core Domain Fit | +35 | FDA (2021) Core PRO Cancer Guidance |
    #         | Regulatory Acceptance | +20 | FDA PRO Guidance (2009) Section V |
    #         | Validated MCID | +15 (gated) | FDA PRO Guidance (2009) Section V.C — no MCID = hard cap at 75 |
    #         | MoA-Specific Sensitivity | +10 | FDA PFDD Guidance 1 (2017) |

    #         **Penalties** reduce the score when the instrument has a specific regulatory risk for
    #         this trial context. Risk Level is set independently — so an instrument at score 0
    #         still shows WHY it failed.

    #         **Operational flags** (eCOA, languages, HTA) are shown separately — they are
    #         practical barriers, not regulatory criteria, and mixing them into the score
    #         would be misleading.
    #                         """)

    #         # Build set of candidate instruments from coverage domains for label
    #         coverage_candidates = {
    #             c["instrument"]
    #             for d in coverage.get("domains", [])
    #             for c in d.get("candidates", [])
    #         }
    #         for inst in top_scores:
    #             in_coverage = inst["instrument_name"] in coverage_candidates
    #             label_sfx   = " 🟢 Top candidate" if in_coverage else ""
    #             with st.expander(
    #                 f"{inst['instrument_name']}{label_sfx} — "
    #                 f"Score: {inst['scientific_score']}/100 — "
    #                 f"Risk: {inst['risk_level']}",
    #                 expanded=False
    #             ):
    #                 c1, c2, c3, c4 = st.columns(4)
    #                 c1.metric("Score",        f"{inst['scientific_score']}/100")
    #                 c2.metric("Positive pts", f"+{inst['raw_positive_score']}")
    #                 c3.metric("Adj. score",   f"{inst['final_adjusted_score']:+d}")
    #                 c4.metric("Op. bonus",    f"{inst['operational_bonus']:+d}")

    #                 st.markdown(risk_badge(inst["risk_level"]), unsafe_allow_html=True)

    #                 # Operational breakdown
    #                 op_detail = []
    #                 for f in inst.get("flags", []):
    #                     fl = f.lower()
    #                     if "ecoa ready +8" in fl:
    #                         op_detail.append("eCOA: +8")
    #                     if "open access +5" in fl:
    #                         op_detail.append("Open access: +5")
    #                     if "limited translation (-5 operational)" in fl:
    #                         op_detail.append("Translation: −5")
    #                     if "no translation data (-10 operational)" in fl:
    #                         op_detail.append("No translation data: −10")
    #                 if op_detail:
    #                     st.caption(
    #                         "Operational: "
    #                         + " · ".join(op_detail)
    #                         + f" = Net {inst['operational_bonus']:+d}"
    #                     )

    #                 st.markdown("**Score breakdown:**")
    #                 for flag in inst.get("flags", []):
    #                     css = classify_flag(flag)
    #                     st.markdown(
    #                         f'<div class="{css}">{flag}</div>',
    #                         unsafe_allow_html=True
    #                     )

    #                 # KG precedent records for this instrument
    #                 matching = [r for r in kg_records
    #                             if r.get("instrument_name") == inst["instrument_name"]]
    #                 if matching:
    #                     st.markdown("**Precedent records from knowledge graph:**")
    #                     st.caption(
    #                         "These trials show where this instrument has been used before. "
    #                         "They are evidence of regulatory familiarity — "
    #                         "not properties of the instrument itself."
    #                     )
    #                     for rec in matching[:2]:
    #                         st.markdown(
    #                             f'<div style="background:#f5f5f3;border-radius:4px;'
    #                             f'padding:6px 10px;margin:3px 0;font-size:0.82rem">'
    #                             f'<b>{rec.get("trial_name", "")} '
    #                             f'({rec.get("nct_id", "")})</b> — '
    #                             f'{rec.get("drug_name", "")} · '
    #                             f'{rec.get("phase", "")} · '
    #                             f'Role: {rec.get("endpoint_role", "") or rec.get("pro_position", "")}'
    #                             + (f'<br>Finding: {rec.get("key_finding", "")}'
    #                                if rec.get("key_finding") else "")
    #                             + '</div>',
    #                             unsafe_allow_html=True
    #                         )
    #                         html = build_source_links_html(rec)
    #                         if html:
    #                             st.markdown(html, unsafe_allow_html=True)

    st.caption(
        f"KG: {counts.get('instrument_records', 0)} instrument records · "
        f"{counts.get('regulatory_reviews', 0)} regulatory reviews · "
        f"{counts.get('regulatory_rules', 0)} rules · "
        f"{counts.get('rejections_found', 0)} rejection records"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 7. INTENT ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

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
    if not client:
        return {"tier": "TIER3_STRATEGY", "reason": "client unavailable",
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
                "content": f"Context: {context_note}\n\nUser message: {user_message}"
            }]
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception:
        return {"tier": "TIER3_STRATEGY", "reason": "router error",
                "missing_critical": [], "can_answer_from_history": False}

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    neo4j_status = check_neo4j()
    if neo4j_status == "connected":
        st.success("🟢 Neo4j connected")
    else:
        st.error("🔴 Neo4j unavailable")
        st.caption("Resume at console.neo4j.io before running a strategy query.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drugs", "36")
    c2.metric("Trials", "131")
    c3.metric("Instr.", "193")
    c4.metric("Reviews", "68")

    st.divider()

    with st.expander("⚙️ Trial Context", expanded=False):
        sb_indication = st.text_input("Indication", placeholder="e.g. Multiple Myeloma")
        sb_phase      = st.selectbox("Phase", ["", "Phase 3", "Phase 2", "Phase 1"])
        sb_drug_class = st.text_input("Drug class", placeholder="e.g. Bispecific, PI, ICI")
        sb_admin      = st.selectbox("Administration", ["", "Step-up dosing", "IV", "Subcutaneous", "Oral", "Weekly IV"])
        sb_population = st.text_input("Population", placeholder="e.g. RRMM ≥3 prior lines, Newly Diagnosed")
        sb_hta        = st.multiselect("HTA markets", ["NICE", "ICER", "EUnetHTA", "SMC"], default=["NICE", "ICER"])
        sb_footprint  = st.selectbox("Geographic footprint", ["", "Global", "EU", "US-only"])

    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.results = {}
        st.session_state.last_strategy_idx = None
        st.rerun()

    if st.session_state.get("messages"):
        lines = [
            "PRO COA AI Agent — Conversation Export",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
        ]
        for i, msg in enumerate(st.session_state.messages):
            role = "YOU" if msg["role"] == "user" else "AGENT"
            lines.append(f"\n[{role}]\n{msg['content']}")
            if msg["role"] == "assistant" and i in st.session_state.get("results", {}):
                lines.append(
                    "\n--- FULL STRATEGY DATA (JSON) ---\n"
                    + json.dumps(st.session_state.results[i], indent=2, default=str)
                )
        st.download_button(
            label="⬇ Download conversation",
            data="\n".join(lines),
            file_name=f"coa_session_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_conv_main",
            help="Download the full conversation (all tiers) as a plain-text file.",
        )
    else:
        st.caption("No conversation to download yet.")

    st.divider()

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

    st.caption("Project 2025gsk2 · University of Cambridge")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN AREA & CHAT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

st.title("PRO COA Agent")
with st.expander("ℹ️ About the data in this tool", expanded=False):
    st.markdown(
        "The primary publication DOI (shown in citations) is the main trial publication. "
        "However, most KG data, including instrument details, subscale outcomes, PRO findings, "
        "and regulatory review text, was curated from multiple sources: "
        "ClinicalTrials.gov, FDA/EMA labels, EPARs, and additional publications such as "
        "PRO‑specific analyses. "
        "The full list of publications for each trial is available on its ClinicalTrials.gov record.\n\n"
        "**Regulatory rules** are curated from the following publicly available guidance documents:\n"
        "- [FDA PRO Guidance (2009)](https://www.fda.gov/media/77832/download)\n"
        "- [FDA Core PRO Cancer (2024)](https://www.fda.gov/media/149994/download)\n"
        "- [FDA PFDD Guidance 1 (2018)](https://www.fda.gov/media/139088/download)\n"
        "- [FDA PFDD Guidance 2 (2022)](https://www.fda.gov/media/131230/download)\n"
        "- [FDA PFDD Guidance 3 (2025)](https://www.fda.gov/media/159500/download)\n"
        "- [FDA PFDD Guidance 4 (2023)](https://www.fda.gov/media/166830/download)\n"
        "- [EMA Reflection Paper on Patient Experience Data (2025)]"
        "(https://www.ema.europa.eu/en/documents/scientific-guideline/reflection-paper-patient-experience-data_en.pdf)\n"
        "- [EMA Appendix 2 to the guideline on the evaluation of anticancer medicinal products in man]"
        "(https://www.ema.europa.eu/en/documents/other/appendix-2-guideline-evaluation-anticancer-medicinal-products-man_en.pdf)\n"
        "- [HTA Guidance on outcomes for joint clinical assessments]"
        "(https://health.ec.europa.eu/document/download/a70a62c7-325c-401e-ba42-66174b656ab8_en?filename=hta_outcomes_jca_guidance_en.pdf)"
    )
st.caption(
    "Ask any COA question or describe your trial for a full strategy recommendation. "
    "You can ask follow-up questions, correct assumptions, or request clarification."
)

if not AGENT_AVAILABLE:
    st.error(f"Agent failed to load: {AGENT_ERROR}. Check your .env file.")
    st.stop()

if "messages"          not in st.session_state:
    st.session_state.messages = []
if "results"           not in st.session_state:
    st.session_state.results = {}
if "last_strategy_idx" not in st.session_state:
    st.session_state.last_strategy_idx = None

# Replay conversation
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            result = st.session_state.results.get(i)
            if result:
                render_strategy_result(msg["content"], result, i)
            else:
                prior_index = (
                    st.session_state.results
                    .get(st.session_state.last_strategy_idx, {})
                    .get("citation_index", {})
                )
                linked, refs = linkify_and_number_citations(msg["content"], prior_index)
                st.markdown(linked, unsafe_allow_html=True)
                web_links = extract_web_links(msg["content"])
                render_unified_footnotes(refs, web_links)

_pending = st.session_state.pop("_pending_input", None)
prompt   = st.chat_input("Ask a COA question or describe your trial…") or _pending

if prompt:
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

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    has_prior = st.session_state.last_strategy_idx is not None
    routing   = classify_intent(prompt, has_prior)
    tier      = routing.get("tier", "TIER3_STRATEGY")

    with st.chat_message("assistant"):
        if tier in ("TIER1_FACTUAL", "TIER2_FOLLOWUP"):
            prior_result = (
                st.session_state.results.get(st.session_state.last_strategy_idx)
                if (has_prior and tier == "TIER2_FOLLOWUP")
                else None
            )
            indication = (
                sb_indication
                or (prior_result.get("context_json", {}).get("indication") if prior_result else None)
            )
            if not indication:
                msg_lower = (prompt + " " + sidebar_ctx).lower()
                indication = next((key for key in KG_KNOWN_VALUES if key in msg_lower), None)
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

            prior_index = (
                st.session_state.results
                .get(st.session_state.last_strategy_idx, {})
                .get("citation_index", {})
            )
            if not prior_index and indication and AGENT_AVAILABLE:
                try:
                    prior_index = build_tier1_citation_index(indication=indication, phase="")
                except Exception:
                    prior_index = {}

            full_response = ""
            placeholder = st.empty()
            for chunk in answer_direct(
                user_message=prompt + sidebar_ctx,
                history=st.session_state.messages[:-1],
                indication=indication,
                prior_result=prior_result,
                citation_index=prior_index,
            ):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            linked, refs = linkify_and_number_citations(full_response, prior_index)
            final, web_fns = convert_web_citations_to_numbers(linked, start_n=len(refs) + 1)
            placeholder.markdown(final, unsafe_allow_html=True)
            render_unified_footnotes(refs, web_fns)

            msg_idx = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citation_index": prior_index,
            })
            if "tier1_citation_index" not in st.session_state:
                st.session_state.tier1_citation_index = {}
            st.session_state.tier1_citation_index.update(prior_index)

        else:   # TIER3_STRATEGY
            _raw_ctx = analyze_trial_context(prompt, api_key=get_secret("ANTHROPIC_API_KEY"))
            full_prompt = build_structured_prompt(
                prompt, _raw_ctx,
                sb_indication, sb_phase, sb_drug_class,
                sb_admin, sb_population, sb_hta, sb_footprint,
            )
            if full_prompt.strip() != prompt.strip():
                _raw_ctx = analyze_trial_context(prompt, api_key=get_secret("ANTHROPIC_API_KEY"))
            else:
                _ctx = _raw_ctx
            _indication = _ctx.get("indication", "unknown")
            _drug_class = _ctx.get("drug_class", "Unknown")

            if _indication == "unknown" or (_drug_class in ["Unknown", "", None] and not sb_drug_class):
                _question = build_clarification_question(_ctx)
                st.markdown(f'<div class="clarify-box">💬 {_question}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": _question})
            else:
                _show_confirmation_card(_ctx)

                steps = [
                    {"label": "Analyzing trial context",               "status": "running"},
                    {"label": "Querying knowledge graph",               "status": "pending"},
                    {"label": "Scoring instruments and coverage analysis", "status": "pending"},
                    {"label": "Synthesising recommendation",            "status": "pending"},
                ]
                step_ph = st.empty()
                step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

                _neo4j_live = check_neo4j()
                if _neo4j_live != "connected":
                    st.error(
                        "🔴 **Neo4j is not connected.** "
                        "The strategy recommendation requires the knowledge graph. "
                        "Please reconnect at [console.neo4j.io](https://console.neo4j.io) "
                        "and try again. "
                        f"Error: {_neo4j_live}"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ Knowledge graph offline — cannot generate strategy. Please reconnect Neo4j."
                    })
                    st.stop()

                result = get_recommendation(full_prompt,
                            anthropic_api_key=get_secret("ANTHROPIC_API_KEY"), disable_web_search=False)

                steps[0] = {**steps[0], "status": "complete",
                            "detail": f"{_ctx.get('indication','')} · {_ctx.get('phase','')}"}
                steps[1] = {**steps[1],
                            "status": "error" if result.get("error_status") and
                            "offline" in str(result.get("error_status","")) else "complete",
                            "detail": f"{result.get('record_counts',{}).get('instrument_records',0)} instruments · "
                                      f"{result.get('record_counts',{}).get('regulatory_reviews',0)} reviews"}
                _cov = result.get("coverage", {})
                _nd = len([d for d in _cov.get("domains", []) if d.get("candidates")])
                _ntot = len(_cov.get("domains", []))
                steps[2] = {**steps[2], "status": "complete",
                            "detail": f"{result.get('record_counts',{}).get('all_scores',0)} scored · "
                                      f"{_nd}/{_ntot} domains covered"}
                steps[3] = {**steps[3],
                            "status": "complete" if result.get("answer") else "error",
                            "detail": f"{len(result.get('answer',''))} chars"}
                step_ph.markdown(render_steps(steps), unsafe_allow_html=True)
                step_ph.empty()

                if result.get("error_status"):
                    st.warning(f"Notice: {result['error_status']}")

                answer = result.get("answer", "No recommendation generated.")
                msg_idx = len(st.session_state.messages)
                st.session_state.results[msg_idx] = result
                st.session_state.last_strategy_idx = msg_idx
                st.session_state.messages.append({"role": "assistant", "content": answer})
                render_strategy_result(answer, result, msg_idx)
