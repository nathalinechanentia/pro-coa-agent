import streamlit as st
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import io

st.set_page_config(
    page_title="PRO COA AI Agent | Cambridge × Evinova",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from agent import (
        get_recommendation, HTA_PREFERENCES,
        GEOGRAPHIC_LANGUAGE_REQUIREMENTS, KNOWN_LANGUAGE_COUNTS
    )
    AGENT_AVAILABLE = True
    AGENT_ERROR = None
except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

# CSS
st.markdown("""<style>
.step-complete{border-left:3px solid #1D9E75;padding:5px 12px;margin:2px 0;background:#f0faf6;border-radius:0 5px 5px 0;font-size:0.83rem;color:#085041}
.step-running{border-left:3px solid #EF9F27;padding:5px 12px;margin:2px 0;background:#fdf6e8;border-radius:0 5px 5px 0;font-size:0.83rem;color:#633806}
.step-pending{border-left:3px solid #D3D1C7;padding:5px 12px;margin:2px 0;background:#f8f8f6;border-radius:0 5px 5px 0;font-size:0.83rem;color:#888780}
.step-error{border-left:3px solid #E24B4A;padding:5px 12px;margin:2px 0;background:#fdf0f0;border-radius:0 5px 5px 0;font-size:0.83rem;color:#791F1F}
.risk-critical{background:#fdf0f0;color:#791F1F;border:1px solid #F09595;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.risk-high{background:#faeeda;color:#633806;border:1px solid #FAC775;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.risk-moderate{background:#fdf6e8;color:#854F0B;border:1px solid #EF9F27;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.risk-low{background:#e1f5ee;color:#085041;border:1px solid #5DCAA5;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.score-bar-outer{background:#e8e8e4;border-radius:4px;height:8px;width:100%;margin:3px 0}
.score-bar-inner{height:8px;border-radius:4px}
.flag-penalty{border-left:3px solid #E24B4A;padding:3px 10px;background:#fdf0f0;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.81rem}
.flag-bonus{border-left:3px solid #1D9E75;padding:3px 10px;background:#e1f5ee;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.81rem}
.flag-geo{border-left:3px solid #7F77DD;padding:3px 10px;background:#eeedfe;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.81rem}
.flag-neutral{border-left:3px solid #D3D1C7;padding:3px 10px;background:#f8f8f6;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.81rem}
.flag-info{border-left:3px solid #378ADD;padding:3px 10px;background:#e6f1fb;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.81rem}
.source-card{border:1px solid #D3D1C7;border-radius:6px;padding:8px 12px;margin:3px 0;background:#fafaf8;font-size:0.84rem}
.source-card a{color:#185FA5;text-decoration:none}
.assumption-box{border:1px solid #FAC775;background:#faeeda;border-radius:6px;padding:8px 12px;margin:5px 0;font-size:0.87rem}
.battery-card{border:1px solid #9FE1CB;background:#f0faf6;border-radius:8px;padding:12px 16px;margin:6px 0}
.battery-role{font-size:0.75rem;color:#0F6E56;font-weight:500;text-transform:uppercase;letter-spacing:0.05em}
.gap-warning{border:1px solid #F09595;background:#fdf0f0;border-radius:6px;padding:8px 12px;margin:5px 0;font-size:0.87rem}
.precedent-card{border:1px solid #B5D4F4;background:#f0f7fd;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:0.83rem}
.geo-card{border:1px solid #AFA9EC;background:#eeedfe;border-radius:6px;padding:10px 14px;margin:5px 0;font-size:0.85rem}
</style>""", unsafe_allow_html=True)

# HELPER FUNCTIONS
def render_step(status, label, detail=""):
    icon = {"complete":"✓","running":"⟳","pending":"○","error":"✗"}.get(status,"○")
    d = f" — {detail}" if detail else ""
    return f'<div class="step-{status}">{icon} {label}{d}</div>'

def render_steps(steps):
    return "<div style='margin:8px 0'>" + "".join(
        render_step(s["status"], s["label"], s.get("detail","")) for s in steps
    ) + "</div>"

def score_bar(score):
    color = "#1D9E75" if score >= 60 else "#EF9F27" if score >= 30 else "#E24B4A"
    return f'<div class="score-bar-outer"><div class="score-bar-inner" style="width:{score}%;background:{color}"></div></div>'

def risk_badge(level):
    return f'<span class="risk-{level.lower()}">{level}</span>'

def classify_flag(flag):
    f = flag.upper()
    # Penalty flags (red)
    if any(x in f for x in ["PENALTY","CRITICAL","MISSING CORE","RECALL BIAS","ESTIMAND",
                             "NO MCID PENALTY","TRANSLATION GAP","HTA NOTE","ASYMPTOMATIC BURDEN",
                             "PRE-SPECIFICATION", "LANGUAGE DATA UNAVAILABLE"]):
        return "flag-penalty"
    # Positive scoring flags (green)
    if any(x in f for x in ["+35","+25","+20","+10 OPERATIONAL","+5 OPERATIONAL","+10)",
                             "VALIDATED MCID (+","TPP/CORE FIT","REGULATORY TRUST",
                             "COMPETITOR BENCH","MOA SENSITIVITY","ECOA READY","OPEN ACCESS",
                             "HTA ALIGNMENT (+", "LANGUAGE COVERAGE: EQ-5D","LANGUAGE COVERAGE: EORTC",
                             "RECALL PERIOD COMPATIBLE"]):
        return "flag-bonus"
    # Informational / compatible (blue-grey — neutral information, not a penalty or a score)
    if any(x in f for x in ["RECALL PERIOD COMPATIBLE","LANGUAGE COVERAGE","HTA NOTE — NICE MARKET",
                             "RECALL PERIOD UNKNOWN", "LANGUAGE COVERAGE NOTE"]):
        return "flag-info"
    if any(x in f for x in ["TRANSLATION GAP","GEO","LINGUISTIC"]):
        return "flag-geo"
    return "flag-neutral"

def build_source_links(record):
    """Build clickable HTML source links from KG record. Framed as precedent links."""
    links = []
    fda_url = record.get("fda_label_url", "")
    ema_url = record.get("ema_label_url", "")
    nct = record.get("nct_id", "")
    doi = record.get("publication_doi", "")
    year = record.get("publication_year", "")
    drug = record.get("drug_name", "")
    trial = record.get("trial_name", "")

    if nct and str(nct).startswith("NCT"):
        links.append(f'<a href="https://clinicaltrials.gov/study/{nct}" target="_blank">ClinicalTrials.gov: {trial or nct}</a>')
    if doi:
        links.append(f'<a href="https://doi.org/{doi}" target="_blank">Publication ({year}): {doi[:30]}...</a>' if len(str(doi)) > 30 else f'<a href="https://doi.org/{doi}" target="_blank">Publication ({year})</a>')
    if fda_url and str(fda_url).startswith("http"):
        links.append(f'<a href="{fda_url}" target="_blank">FDA label: {drug}</a>')
    elif drug:
        links.append(f'<a href="https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(" ","+")}" target="_blank">FDA DailyMed: {drug}</a>')
    if ema_url and str(ema_url).startswith("http"):
        links.append(f'<a href="{ema_url}" target="_blank">EMA label: {drug}</a>')
    return links

def linkify_citations(text: str, citation_index: dict) -> str:
    """
    Replace KG citation labels [TI-001], [RR-003] etc. with markdown hyperlinks
    that open the primary source URL. The label text is kept so the reference is visible.
    Web citations [Source](url) are already markdown and render as-is.
    """
    import re
    pattern = re.compile(r'\[(TI|RR|REJ|IR|RULE|PREC)-(\d{3})\]')

    def replace_label(match):
        label = f"{match.group(1)}-{match.group(2)}"
        data = citation_index.get(label, {})
        links = data.get("links", [])

        # Find best URL: prefer ClinicalTrials or DOI over label URL
        best_url = None
        for l in links:
            url = l.get("url", "")
            if "clinicaltrials.gov" in url or "doi.org" in url:
                best_url = url
                break
        if not best_url:
            best_url = next(
                (l["url"] for l in links if l.get("url","").startswith("http")),
                None
            )

        if best_url:
            # Add instrument/drug info in the link text as context
            context = data.get("instrument") or data.get("drug") or data.get("primary_reason","")[:40]
            if context:
                return f"[[{label}: {context}]]({best_url})"
            return f"[[{label}]]({best_url})"
        else:
            # No URL — render as inline code so it is visually distinct but not broken
            context = data.get("instrument") or data.get("drug") or ""
            if context:
                return f"`[{label}: {context}]`"
            return f"`[{label}]`"

    # Also fix the bad [number-number] citation format Sonnet sometimes uses
    # Replace [13-4], [45-6] etc. with nothing — these are internal search indices not citations
    bad_pattern = re.compile(r'\[\d+-\d+\]')
    text = bad_pattern.sub('', text)

    return pattern.sub(replace_label, text)

# def get_instrument_language_status(instrument_name: str, kg_language_count: int, footprint: str) -> dict:
#     inst_lower = instrument_name.lower()
#     known = next(
#         (v for k, v in KNOWN_LANGUAGE_COUNTS.items() if k in inst_lower),
#         kg_language_count
#     )
#     geo = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(footprint, GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
#     key_langs = geo.get("key_languages", [])
#     return {
#         "count": known,
#         "key_languages": key_langs,
#         "footprint": footprint,
#         "regulatory_note": geo.get("regulatory_note", ""),
#         "reference": geo.get("reference", ""),
#     }

def get_instrument_language_status(instrument_name: str, kg_language_count: int, footprint: str) -> dict:
    inst_lower = instrument_name.lower()
    known = next(
        (v for k, v in KNOWN_LANGUAGE_COUNTS.items() if k in inst_lower),
        kg_language_count
    )
    geo = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(footprint, GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
    return {
        "count": known,
        "key_languages": geo.get("key_languages", []),
        "regulatory_note": geo.get("regulatory_note", ""),
        "reference": geo.get("reference", ""),
        "sufficient_note": (
            "Strong coverage." if known >= 50
            else "Verify coverage for trial site languages."
            if known >= 15
            else "Commission translations — contact instrument developer."
            if known > 0
            else "Language data unavailable — verify via PROQOLID."
        )
    }

# SIDEBAR
st.sidebar.title("🏥 COA Agent")
st.sidebar.caption("Cambridge × Evinova (AstraZeneca)")
st.sidebar.divider()

st.sidebar.subheader("Trial Parameters")
st.sidebar.caption("Fill in what you know. Leave blank to let the agent infer.")

indication = st.sidebar.text_input("Indication", placeholder="e.g. Multiple Myeloma, NSCLC, CRPC")
phase = st.sidebar.selectbox("Phase", ["Phase 3","Phase 2","Phase 1","Phase 4"])
drug_class = st.sidebar.text_input("Drug Class", placeholder="e.g. Bispecific, Proteasome Inhibitor, ICI")
administration = st.sidebar.selectbox(
    "Administration",
    ["Unknown / Infer","Step-up dosing","IV","Subcutaneous","Oral","Weekly IV"],
    help="Step-up dosing → Recall Bias penalty check."
)
population = st.sidebar.text_input(
    "Patient Population",
    placeholder="e.g. Relapsed/Refractory, Newly Diagnosed, Smoldering",
    help="Type freely — e.g. RRMM, 3L+, asymptomatic. The agent will interpret this."
)

st.sidebar.divider()
st.sidebar.subheader("Market Scope")

hta_markets = st.sidebar.multiselect("HTA Markets", ["NICE","ICER","EUnetHTA","SMC"], default=["NICE","ICER"])
geographic_footprint = st.sidebar.selectbox("Geographic Footprint", ["Global","EU","US-only","Unknown / Infer"])

st.sidebar.divider()
st.sidebar.subheader("Knowledge Base")
c1,c2,c3 = st.sidebar.columns(3)
c1.metric("Drugs","36"); c2.metric("Trials","131"); c3.metric("Instruments","193")
st.sidebar.metric("Regulatory Reviews","68")

st.sidebar.divider()
show_eval = st.sidebar.toggle("Evaluation tab", value=True)
show_reasoning = st.sidebar.toggle("Agent reasoning trace", value=False)
st.sidebar.caption("Project 2025gsk2 — Dept. of Chemical Engineering & Biotechnology, University of Cambridge")

# MAIN HEADER AND QUERY INPUT
st.title("PRO COA AI Agent")
st.markdown("**Evidence-based COA instrument selection for oncology clinical trials**")
st.caption("University of Cambridge × Evinova (AstraZeneca)")

if not AGENT_AVAILABLE:
    st.error(f"Agent failed to load: {AGENT_ERROR}. Check your .env file and Neo4j connection.")
    st.stop()

st.divider()
st.subheader("Describe your trial")

user_query = st.text_area(
    "Trial description",
    height=130,
    placeholder=(
        "Describe the trial in plain language. The more detail you include, the more precise the recommendation.\n\n"
        "Include: indication, patient population, drug mechanism, phase, desired label claims, "
        "and any specific concerns (e.g. step-up dosing CRS risk, asymptomatic population, global submission).\n\n"
        "Example: Phase 3 BCMA bispecific antibody in RRMM (≥3 prior lines). "
        "Step-up dosing Cycle 1. TPP claims: treatment tolerability and physical function. "
        "Global submission, NICE/ICER HTA required."
    )
)

run_button = st.button("Generate COA Strategy", type="primary")

# AGENT EXECUTION
if run_button:
    if not user_query.strip():
        st.warning("Please describe your trial before running.")
        st.stop()

    parts = []
    if indication: parts.append(f"Indication: {indication}")
    if drug_class: parts.append(f"Drug class: {drug_class}")
    if administration != "Unknown / Infer": parts.append(f"Administration: {administration}")
    if population: parts.append(f"Patient population: {population}")
    if geographic_footprint != "Unknown / Infer": parts.append(f"Geographic footprint: {geographic_footprint}")
    if hta_markets: parts.append(f"HTA markets: {', '.join(hta_markets)}")
    if phase: parts.append(f"Phase: {phase}")
    sidebar_context = ("\n\nAdditional parameters provided by user:\n" + "\n".join(parts)) if parts else ""
    full_query = user_query + sidebar_context

    steps = [
        {"label": "Step 1: Analyzer — extracting trial context and parameters", "status": "pending"},
        {"label": "Step 2: Knowledge Graph — querying trials, reviews, regulatory rules", "status": "pending"},
        {"label": "Step 3: Scoring Engine — 100-point regulatory scale + battery optimiser", "status": "pending"},
        {"label": "Step 4: Reasoner — synthesising evidence + live web search", "status": "pending"},
        {"label": "Step 5: Logging to evaluation dataset", "status": "pending"},
    ]
    step_ph = st.empty()
    step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

    steps[0]["status"] = "running"
    steps[0]["detail"] = "Haiku parsing query..."
    step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

    result = get_recommendation(full_query)
    ctx = result.get("context_json", {})
    counts = result.get("record_counts", {})
    battery = result.get("battery_result", {})

    steps[0] = {"label": steps[0]["label"], "status": "complete",
                "detail": f"{ctx.get('indication','?')} | {ctx.get('phase','?')} | {len(ctx.get('assumptions_made',[]))} assumption(s)"}
    steps[1] = {"label": steps[1]["label"],
                "status": "error" if result.get("error_status") and "offline" in str(result.get("error_status","")) else "complete",
                "detail": f"{counts.get('instrument_records',0)} instruments | {counts.get('regulatory_reviews',0)} reviews | {counts.get('regulatory_rules',0)} rules"}
    steps[2] = {"label": steps[2]["label"], "status": "complete",
                "detail": f"{counts.get('scored_instruments',0)} scored | battery: {', '.join(battery.get('battery_names',[])[:3])}"}
    steps[3] = {"label": steps[3]["label"],
                "status": "complete" if result.get("answer") else "error",
                "detail": f"{len(result.get('answer',''))} chars"}
    steps[4] = {"label": steps[4]["label"], "status": "complete", "detail": "Saved to /logs/"}
    step_ph.markdown(render_steps(steps), unsafe_allow_html=True)

    if result.get("error_status"):
        st.warning(f"Notice: {result['error_status']}")

    st.session_state["last_result"] = result

# RESULTS — 3 TABS
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    ctx = result.get("context_json", {})
    top_scores = result.get("top_scores", [])
    kg_records = result.get("kg_raw_hits", [])
    reg_records = result.get("reg_records", [])
    reg_rules = result.get("reg_rules", [])
    battery = result.get("battery_result", {})
    inst_precedents = result.get("inst_regulatory_precedents", {})
    counts = result.get("record_counts", {})

    tab_names = ["COA Strategy", "Evidence & Scoring Detail"]
    if show_eval:
        tab_names.append("Evaluation Log")
    tabs = st.tabs(tab_names)

    # TAB 1 — COA Strategy (the complete recommendation)
    with tabs[0]:

        # --- What the agent understood (renamed from Strategy Context Audit) ---
        assumptions = ctx.get("assumptions_made", [])
        if assumptions:
            with st.expander("ℹ️ What the agent understood about your trial — click to review and correct", expanded=True):
                st.caption(
                    "The agent inferred the following parameters from your query. "
                    "If any are wrong, add the correct information to your description and re-run."
                )
                for a in assumptions:
                    st.markdown(f'<div class="assumption-box">⚠️ {a}</div>', unsafe_allow_html=True)

                st.markdown("**Parameters extracted:**")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Indication:** {ctx.get('indication','—')}")
                c2.markdown(f"**Phase:** {ctx.get('phase','—')}")
                c3.markdown(f"**Population:** {ctx.get('population_subtype','—')}")
                c1.markdown(f"**Administration:** {ctx.get('administration','—')}")
                c2.markdown(f"**Drug class:** {ctx.get('drug_class','—')}")
                c3.markdown(f"**Footprint:** {ctx.get('geographic_footprint','—')}")
                st.markdown(f"**Core domains (FDA lookup):** {', '.join(ctx.get('core_domains_required', []))}")
                st.markdown(f"**TPP claims:** {', '.join(ctx.get('tpp_claims', []))}")

        if result.get("error_status"):
            st.warning(f"Notice: {result['error_status']}")

        # --- Recommended battery summary (visual) ---
        if battery.get("battery"):
            st.subheader("Recommended COA Battery")
            st.caption(
                "Built by the domain coverage optimiser — selects highest-scoring non-redundant instrument "
                "per domain category, plus HTA-required instruments. "
                f"Covers {len(battery.get('covered_domains',[]))} of {len(ctx.get('core_domains_required',[]))} required domains."
            )
            # Risk level legend
            st.caption(
                "**Risk Level guide:** "
                "🟢 LOW = no regulatory concerns. "
                "🟡 MODERATE = addressable risks (e.g. missing MCID). "
                "🟠 HIGH = significant risk requiring active mitigation (e.g. not pre-specified). "
                "🔴 CRITICAL = fatal regulatory flaw if not addressed (e.g. recall bias with step-up dosing). "
                "Note: HTA-required instruments (EQ-5D) show LOW risk even if scientific score is moderate — "
                "they are mandatory inclusions, not optional."
            )
            for b in battery["battery"]:
                ecoa = any("+10" in f and "ecoa" in f.lower() for f in b.get("flags", []))
                ecoa_icon = "📱" if ecoa else "📄"
                op_parts = []
                op_net = b.get("operational_bonus", 0)
                for flag in b.get("flags", []):
                    if "ecoa ready (+10" in flag.lower():
                        op_parts.append("+10 eCOA")
                    if "open access (+5" in flag.lower():
                        op_parts.append("+5 open access")
                    if "translation gap" in flag.lower():
                        op_parts.append("-15 translation gap")
                op_str = " | ".join(op_parts) if op_parts else f"Net: {op_net:+d}"

                st.markdown(
                    f'<div class="battery-card">'
                    f'<div class="battery-role">{b.get("battery_role","")}</div>'
                    f'<b>{ecoa_icon} {b["instrument_name"]}</b> &nbsp; '
                    f'{risk_badge(b["risk_level"])} &nbsp; '
                    f'<span style="font-size:0.85rem">Scientific score: {b["scientific_score"]}/100 &nbsp;|&nbsp; '
                    f'Operational: {op_str}</span>'
                    + (f'<br><span style="font-size:0.82rem;color:#444">{b.get("battery_note","")}</span>' if b.get("battery_note") else "")
                    + f'</div>',
                    unsafe_allow_html=True
                )

            if battery.get("gaps"):
                st.markdown(
                    f'<div class="gap-warning">⚠️ <b>Domain coverage gap:</b> No instrument found for: '
                    f'{", ".join(battery["gaps"])}. '
                    f'The Reasoner has been instructed to search the web for instruments covering these domains.</div>',
                    unsafe_allow_html=True
                )
            st.divider()

        # --- Full recommendation text ---
        answer_text = result.get("answer", "No recommendation generated.")
        citation_index = result.get("citation_index", {})
        # Linkify KG citations inline
        linked_answer = linkify_citations(answer_text, citation_index)
        st.markdown(linked_answer)
        
        # Citation reference panel
        citation_index = result.get("citation_index", {})
        if citation_index:
            with st.expander(f"📎 Citation references ({len(citation_index)} KG sources) — click to view and trace", expanded=False):
                st.caption(
                    "These are the knowledge graph records cited in the recommendation above. "
                    "Labels like [TI-001] in the text refer to these entries. "
                    "Web citations appear as hyperlinks directly in the text above."
                )
                for label, data in sorted(citation_index.items()):
                    ctype = data.get("type", "")
                    if ctype == "trial_instrument":
                        summary = (
                            f"**[{label}]** Trial instrument: **{data.get('instrument','')}** "
                            f"in {data.get('trial','')} ({data.get('phase','')}) — "
                            f"Drug: {data.get('drug','')}"
                        )
                    elif ctype == "regulatory_review":
                        summary = (
                            f"**[{label}]** Regulatory review: **{data.get('agency','')}** "
                            f"decision on {data.get('drug','')} — {data.get('decision','')}"
                        )
                    elif ctype == "rejection":
                        summary = (
                            f"**[{label}]** Rejection record: **{data.get('agency','')}** "
                            f"on {data.get('drug','')} — {data.get('primary_reason','')[:100]}"
                        )
                    else:
                        summary = f"**[{label}]** {data}"

                    links = data.get("links", [])
                    link_html = " &nbsp;|&nbsp; ".join(
                        f'<a href="{l["url"]}" target="_blank">{l["label"]}</a>'
                        for l in links if l.get("url","").startswith("http")
                    )
                    st.markdown(summary)
                    if link_html:
                        st.markdown(f'<div class="source-card">{link_html}</div>', unsafe_allow_html=True)
                    st.divider()

        # Citation guide
        st.caption(
            "Citations: **[TI-XXX]** = trial instrument record from KG (see Evidence & Scoring Detail tab). "
            "**[RR-XXX]** = regulatory review record. "
            "**[IR-XXX]** = instrument reference. "
            "**[REJ-XXX]** = rejection reason from medical review. "
            "Web citations appear as [Source Name](url) — click to open. "
            "Switch to the **Evidence & Scoring Detail** tab to see all KG records with source links."
        )

        # --- Geographic and linguistic validation per instrument ---
        if battery.get("battery") and ctx.get("geographic_footprint"):
            st.divider()
            st.subheader("Language and translation status — per recommended instrument")
            footprint = ctx.get("geographic_footprint", "Global")
            geo = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(footprint, GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
            key_langs = geo.get("key_languages", [])
            regulatory_note = geo.get("regulatory_note", "")
            reference = geo.get("reference", "")

            for b in battery["battery"]:
                inst_name = b["instrument_name"]
                inst_lower = inst_name.lower()
                # Use known language counts first
                lang_count = next(
                    (v for k, v in KNOWN_LANGUAGE_COUNTS.items() if k in inst_lower),
                    0
                )
                if lang_count == 0:
                    # Try KG records
                    for rec in kg_records:
                        if rec.get("instrument_name") == inst_name:
                            raw = rec.get("languages", "")
                            if isinstance(raw, list):
                                lang_count = len([x for x in raw if x])
                            elif raw:
                                lang_count = len([x for x in str(raw).split("|") if x.strip()])
                            break

                if lang_count >= 50:
                    icon = "✅"
                    note = f"Strong coverage: approximately {lang_count} validated translations available."
                elif lang_count >= 15:
                    icon = "ℹ️"
                    note = f"Moderate coverage: approximately {lang_count} validated translations. Verify specific languages for your trial sites."
                elif lang_count > 0:
                    icon = "⚠️"
                    note = f"Limited coverage: approximately {lang_count} validated translations. Contact the instrument developer to commission translations for trial site languages."
                else:
                    icon = "⚠️"
                    note = "Language data unavailable in KG. Verify via PROQOLID (proqolid.org) or instrument developer."

                st.markdown(
                    f'<div class="geo-card">{icon} <b>{inst_name}</b><br>'
                    f'<span style="font-size:0.85rem">{note}<br>'
                    f'Key languages for {footprint} trial: {", ".join(key_langs[:8])}.<br>'
                    f'<small>{regulatory_note} ({reference})</small></span></div>',
                    unsafe_allow_html=True
                )

        # --- Regulatory precedent per instrument (inline) ---
        if inst_precedents:
            st.divider()
            st.subheader("Regulatory precedent — per recommended instrument")
            st.caption("These records show whether each recommended instrument has prior FDA/EMA acceptance in similar submissions.")
            for inst_name, reviews in inst_precedents.items():
                with st.expander(f"{inst_name} — {len(reviews)} precedent record(s)", expanded=False):
                    for j, rev in enumerate(reviews[:3], 1):
                        accepted = inst_name.lower() in str(rev.get("instruments_accepted", "")).lower()
                        icon = "✅" if accepted else "ℹ️"
                        st.markdown(
                            f'<div class="precedent-card">{icon} [{rev.get("agency","")}] '
                            f'{rev.get("drug_name","")} — Decision: {rev.get("decision","")}<br>'
                            f'<b>Instrument status:</b> {"Accepted onto label" if accepted else "Reviewed — acceptance unclear"}<br>'
                            f'<b>Claim type approved:</b> {rev.get("claim_type","Not specified")}<br>'
                            + (f'<b>Label language:</b> {str(rev.get("label_language",""))[:200]}<br>' if rev.get("label_language") else "")
                            + (f'<span style="color:#791F1F">⚠️ Rejection risk: {rev.get("rejection_reason_primary","")}</span>' if rev.get("rejection_reason_primary") else "")
                            + '</div>',
                            unsafe_allow_html=True
                        )
                        links = []
                        if rev.get("fda_label_url","").startswith("http"):
                            links.append(f'<a href="{rev["fda_label_url"]}" target="_blank">FDA label: {rev.get("drug_name","")}</a>')
                        if rev.get("ema_label_url","").startswith("http"):
                            links.append(f'<a href="{rev["ema_label_url"]}" target="_blank">EMA label: {rev.get("drug_name","")}</a>')
                        if links:
                            st.markdown('<div class="source-card">' + " &nbsp;|&nbsp; ".join(links) + "</div>", unsafe_allow_html=True)

        # --- Agent reasoning trace (optional) ---
        if show_reasoning:
            st.divider()
            with st.expander("Agent reasoning trace", expanded=False):
                st.markdown(f"**Model:** claude-haiku-4-5-20251001 (Analyzer) + claude-sonnet-4-20250514 (Reasoner)")
                st.markdown(f"**Web search:** enabled (fda.gov, ema.europa.eu, clinicaltrials.gov, pubmed, nih.gov)")
                st.markdown(f"**Output:** {len(result.get('answer',''))} characters | {counts.get('scored_instruments',0)} instruments scored")
                st.markdown(f"**KG records:** {counts.get('instrument_records',0)} instruments | {counts.get('regulatory_reviews',0)} reviews | {counts.get('regulatory_rules',0)} rules")

        # --- Download ---
        st.divider()
        st.download_button(
            "Download recommendation (.txt)",
            data=result.get("answer",""),
            file_name=f"COA_{ctx.get('indication','unknown')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

    # TAB 2 — Evidence & Scoring Detail
    with tabs[1]:
        with st.expander("ℹ️ How the scoring works — click to expand", expanded=False):
            st.markdown("""
**The 100-point scientific scale** measures how well each instrument fits THIS specific trial context.
It is NOT a general quality rating — a high score means the instrument is the right fit for your indication,
population, and TPP claims.

| Criterion | Max points | Regulatory basis |
|---|---|---|
| TPP/Core Domain Fit | +35 | FDA (2021) Core PRO Guidance |
| Regulatory Trust | +25 | FDA PRO Guidance (2009) Section V |
| Competitor/SoC Benchmark | +20 | FDA PRO Guidance (2009) Section III.B |
| MoA Sensitivity | +20 | FDA PFDD Guidance 1 (2017) |
| Validated MCID | +10 | FDA PRO Guidance (2009) Section V.C |

**Penalties** reduce the score when the instrument has a regulatory risk for THIS specific trial context:

| Penalty | Points | Risk level | Regulatory basis |
|---|---|---|---|
| Missing core domain | −50 | CRITICAL | FDA (2021) Core PRO Guidance |
| Recall period incompatible with dosing | −40 | CRITICAL | FDA PFDD Guidance 3 (2022) |
| Not pre-specified in SAP | −35 | HIGH | FDA PRO Guidance (2009) Section V |
| >30 items in Phase 3 (estimand burden) | −30 | HIGH | ICH E9(R1) Addendum (2019) |
| No validated MCID | −20 | MODERATE | FDA PRO Guidance (2009) Section V.C |
| Symptom-heavy instrument for asymptomatic patients | −20 | MODERATE | FDA PFDD Guidance 2 (2018) |

**The score is floored at 0.** The Risk Level carries the severity information independently.
An instrument scoring 0 with CRITICAL risk should not be used.
An instrument scoring 0 with LOW risk simply has no matching evidence in the KG.

**Operational bonuses** are independent of the 100-point cap and reflect practical considerations:
eCOA readiness (+10), open access licensing (+5), language coverage note (−5 to −10 if data unavailable).
            """)
        
        st.subheader("Instrument scoring detail")
        st.caption(
            "Each instrument is scored on a 0–100 scientific scale plus operational bonuses. "
            "KG records shown below are historical precedent records — the drug and trial named are the source of evidence, "
            "not a recommendation for that drug. Risk Level is set independently of the numeric score."
        )
        st.info(
            "**Citation guide:** Labels like [TI-001], [RR-003] appear in the recommendation text above. "
            "Each label links directly to a source when clicked. "
            "**[TI-XXX]** = a trial that used this instrument (ClinicalTrials.gov or publication). "
            "**[RR-XXX]** = an FDA or EMA review decision (drug label or EPAR). "
            "**[REJ-XXX]** = a rejection reason from a medical review document. "
            "**[IR-XXX]** = instrument reference data. "
            "KG records below show the full context for each label."
        )

        # --- Scoring detail per instrument ---
        if not top_scores:
            st.info("No instruments scored. The KG may be offline — recommendation used web search only.")
        else:
            for i, inst in enumerate(top_scores, 1):
                in_battery = inst["instrument_name"] in battery.get("battery_names", [])
                badge = " 🟢 **In recommended battery**" if in_battery else ""
                with st.expander(
                    f"{inst['instrument_name']} — Score: {inst['scientific_score']}/100 | Risk: {inst['risk_level']}{badge}",
                    expanded=(i <= 2 and in_battery)
                ):
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Scientific Score", f"{inst['scientific_score']}/100")
                    c2.metric("Positive Points", f"+{inst['raw_positive_score']}")
                    c3.metric("Penalty Points", f"-{inst['penalty_total']}")
                    c4.metric("Net Operational", f"{inst['operational_bonus']:+d}")
                    st.markdown(score_bar(inst["scientific_score"]), unsafe_allow_html=True)
                    st.markdown(risk_badge(inst["risk_level"]), unsafe_allow_html=True)

                    # Operational breakdown
                    op_detail = []
                    for flag in inst.get("flags",[]):
                        if "ecoa ready (+10" in flag.lower(): op_detail.append("eCOA Ready: +10")
                        if "open access (+5" in flag.lower(): op_detail.append("Open Access: +5")
                        if "translation gap (-15" in flag.lower(): op_detail.append("Translation Gap: -15")
                    if op_detail:
                        st.caption("Operational breakdown: " + " | ".join(op_detail) + f" = Net {inst['operational_bonus']:+d}")

                    st.markdown("**Score breakdown:**")
                    st.caption(
                        "🟩 Green = points awarded | 🟥 Red = penalty applied | "
                        "🟦 Blue = informational (no score change, confirms compatibility) | "
                        "⬜ Grey = neutral note"
                    )
                    for flag in inst.get("flags",[]):
                        st.markdown(f'<div class="{classify_flag(flag)}">{flag}</div>', unsafe_allow_html=True)

                    # --- KG Precedent Records (reframed) ---
                    st.markdown("**Historical precedent records from knowledge graph:**")
                    st.caption(
                        f"The following shows where {inst['instrument_name']} has been used in prior trials. "
                        "This is evidence for its regulatory familiarity — not a property of the instrument itself."
                    )
                    matching_kg = [r for r in kg_records if r.get("instrument_name") == inst["instrument_name"]]
                    if matching_kg:
                        for rec in matching_kg[:3]:
                            trial_label = f"{rec.get('trial_name','')} ({rec.get('nct_id','')})"
                            phase_label = rec.get("phase","")
                            drug_label = rec.get("drug_name","")
                            role_label = rec.get("endpoint_role","") or rec.get("pro_position","")
                            significance = rec.get("significance","")
                            key_finding = rec.get("key_finding","")
                            prespec = rec.get("prespecified","")

                            st.markdown(
                                f'<div class="precedent-card">'
                                f'<b>{trial_label}</b> ({phase_label}) — Drug: {drug_label}<br>'
                                f'Endpoint role: {role_label} | Significance: {significance} | Pre-specified: {prespec}'
                                + (f'<br>Key finding: {key_finding}' if key_finding else "")
                                + '</div>',
                                unsafe_allow_html=True
                            )
                            links = build_source_links(rec)
                            if links:
                                st.markdown(
                                    '<div class="source-card">' + " &nbsp;|&nbsp; ".join(links) + '</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.caption("No direct KG precedent records found — instrument referenced via web evidence only.")

                    # --- Regulatory precedent for this instrument ---
                    if inst["instrument_name"] in inst_precedents:
                        st.markdown(f"**Regulatory review precedents for {inst['instrument_name']}:**")
                        for rev in inst_precedents[inst["instrument_name"]][:2]:
                            accepted = inst["instrument_name"].lower() in str(rev.get("instruments_accepted","")).lower()
                            icon = "✅" if accepted else "ℹ️"
                            st.markdown(
                                f'<div class="precedent-card">{icon} '
                                f'{rev.get("agency","")} | {rev.get("drug_name","")} | '
                                f'{"ACCEPTED" if accepted else "Reviewed"} | '
                                f'Claim: {rev.get("claim_type","")}</div>',
                                unsafe_allow_html=True
                            )

        # --- Scoring methodology reference ---
        with st.expander("Scoring methodology and regulatory references", expanded=False):
            st.markdown("""
**Positive weights (max 100):**
- TPP/Core Fit +35 · FDA (2021) Core PRO Guidance
- Regulatory Trust +25 · FDA PRO Guidance (2009) Section V; EMA Reflection Paper (2005)
- Competitor/SoC Benchmark +20 · FDA PRO Guidance (2009) Section III.B
- MoA Sensitivity +20 · FDA PFDD Guidance 1 (2017)
- Validated MCID +10 · FDA PRO Guidance (2009) Section V.C

**Conditional penalties (score floored at 0; Risk Level independent):**
- Missing Core −50 CRITICAL · FDA (2021) Core PRO Guidance
- Recall Bias −40 CRITICAL · FDA PFDD Guidance 3 (2022)
- Pre-specification/Alpha −35 HIGH · FDA PRO Guidance (2009) Section V; ICH E9 (1998)
- Estimand Burden −30 HIGH · ICH E9(R1) Addendum (2019)
- No MCID −20 MODERATE · FDA PRO Guidance (2009) Section V.C
- Asymptomatic Burden −20 MODERATE · FDA PFDD Guidance 2 (2018)

**Operational bonuses (independent of 100-point cap):**
- eCOA Ready +10 · FDA eCOA Guidance (2023)
- Open Access +5
- Translation Gap −15 · FDA PRO Guidance (2009) Section IV.A; ISPOR ePRO Task Force (2009)
            """)

        # --- All KG records (grouped by instrument) ---
        st.divider()
        st.subheader(f"All knowledge graph records — {counts.get('instrument_records',0)} records retrieved")
        st.caption(f"Query: indication '{ctx.get('indication','')}' | synonyms: {', '.join(ctx.get('indication_synonyms',[])[:2])} | phase: {ctx.get('phase','')}")

        # Group by instrument name
        instrument_groups = {}
        for rec in kg_records:
            name = rec.get("instrument_name","Unknown")
            if name not in instrument_groups:
                instrument_groups[name] = []
            instrument_groups[name].append(rec)

        for inst_name, recs in instrument_groups.items():
            in_battery = inst_name in battery.get("battery_names",[])
            label = f"{'🟢 ' if in_battery else ''}{inst_name} — {len(recs)} precedent trial(s)"
            with st.expander(label, expanded=False):
                st.caption(f"Domain: {recs[0].get('instrument_domain','')} | Instrument type derived from KG records below.")
                for j, rec in enumerate(recs, 1):
                    st.markdown(f"**Trial {j}: {rec.get('trial_name','')} ({rec.get('nct_id','')})**")
                    c1,c2,c3 = st.columns(3)
                    c1.markdown(f"Drug: {rec.get('drug_name','')}")
                    c2.markdown(f"Phase: {rec.get('phase','')}")
                    c3.markdown(f"Role: {rec.get('endpoint_role','') or rec.get('pro_position','')}")
                    c1.markdown(f"Significance: {rec.get('significance','')}")
                    c2.markdown(f"MID met: {rec.get('mid_met','')}")
                    c3.markdown(f"Pre-specified: {rec.get('prespecified','')}")
                    if rec.get("key_finding"):
                        st.markdown(f"Key finding: {rec['key_finding']}")
                    links = build_source_links(rec)
                    if links:
                        st.markdown('<div class="source-card">' + " &nbsp;|&nbsp; ".join(links) + '</div>', unsafe_allow_html=True)

    # TAB 3 — Evaluation Log
    if show_eval:
        with tabs[2]:
            st.subheader("Evaluation log")
            st.caption("Every recommendation is logged automatically for human vs AI comparison.")

            log_dir = Path("logs")
            log_files = sorted(log_dir.glob("recommendation_*.json"), reverse=True)

            if not log_files:
                st.info("No recommendations logged yet.")
            else:
                st.markdown(f"**{len(log_files)} recommendations logged**")
                for lf in log_files[:10]:
                    try:
                        with open(lf) as f:
                            entry = json.load(f)
                        top = entry.get("top_5_instruments",[])
                        with st.expander(
                            f"{entry.get('timestamp','')} | {entry.get('indication','?')} | "
                            f"{entry.get('phase','')} | Top: {top[0]['name'] if top else 'none'}",
                            expanded=False
                        ):
                            st.markdown(f"**Query:** {entry.get('user_query','')[:250]}...")
                            st.markdown(f"**Assumptions:** {len(entry.get('assumptions_made',[]))}")
                            if entry.get("error_status"):
                                st.warning(f"Error: {entry['error_status']}")
                            for inst in top:
                                st.markdown(f"- {inst['name']}: {inst['score']}/100 | risk {inst['risk_level']}")
                            st.download_button(
                                f"Download log",
                                data=json.dumps(entry, indent=2, default=str),
                                file_name=lf.name,
                                mime="application/json",
                                key=f"dl_{lf.name}"
                            )
                    except Exception as e:
                        st.warning(f"Could not read {lf.name}: {e}")

                if st.button("Export all logs as CSV"):
                    rows = []
                    for lf in log_files:
                        try:
                            with open(lf) as f:
                                e = json.load(f)
                            top = e.get("top_5_instruments",[{}])
                            rows.append({
                                "timestamp": e.get("timestamp"),
                                "indication": e.get("indication"),
                                "phase": e.get("phase"),
                                "query_length": len(e.get("user_query","")),
                                "n_assumptions": len(e.get("assumptions_made",[])),
                                "top_instrument": top[0].get("name","") if top else "",
                                "top_score": top[0].get("score",0) if top else 0,
                                "top_risk": top[0].get("risk_level","") if top else "",
                                "kg_records": e.get("record_counts",{}).get("instrument_records",0),
                                "rejections_found": e.get("record_counts",{}).get("rejections_found",0),
                                "error": e.get("error_status",""),
                                "answer_chars": e.get("answer_length_chars",0)
                            })
                        except: pass
                    df = pd.DataFrame(rows)
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    st.download_button(
                        "Download CSV",
                        data=buf.getvalue(),
                        file_name=f"COA_eval_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
