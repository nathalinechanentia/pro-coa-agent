"""
Streamlit UI for PRO COA AI Agent
University of Cambridge × Evinova (AstraZeneca)
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import io

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="PRO COA AI Agent | Cambridge × Evinova",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORT AGENT
# ============================================================================
try:
    from agent import get_recommendation, HTA_PREFERENCES, GEOGRAPHIC_LANGUAGE_REQUIREMENTS
    AGENT_AVAILABLE = True
    AGENT_ERROR = None
except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
.step-complete{border-left:3px solid #1D9E75;padding:6px 12px;margin:3px 0;background:#f0faf6;border-radius:0 6px 6px 0;font-size:0.84rem;color:#085041}
.step-running{border-left:3px solid #EF9F27;padding:6px 12px;margin:3px 0;background:#fdf6e8;border-radius:0 6px 6px 0;font-size:0.84rem;color:#633806}
.step-pending{border-left:3px solid #D3D1C7;padding:6px 12px;margin:3px 0;background:#f8f8f6;border-radius:0 6px 6px 0;font-size:0.84rem;color:#888780}
.step-error{border-left:3px solid #E24B4A;padding:6px 12px;margin:3px 0;background:#fdf0f0;border-radius:0 6px 6px 0;font-size:0.84rem;color:#791F1F}
.risk-critical{background:#fdf0f0;color:#791F1F;border:1px solid #F09595;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.risk-high{background:#faeeda;color:#633806;border:1px solid #FAC775;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.risk-moderate{background:#fdf6e8;color:#854F0B;border:1px solid #EF9F27;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.risk-low{background:#e1f5ee;color:#085041;border:1px solid #5DCAA5;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:500}
.score-bar-outer{background:#e8e8e4;border-radius:4px;height:10px;width:100%;margin:4px 0}
.score-bar-inner{height:10px;border-radius:4px}
.flag-penalty{border-left:3px solid #E24B4A;padding:4px 10px;background:#fdf0f0;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.82rem}
.flag-bonus{border-left:3px solid #1D9E75;padding:4px 10px;background:#e1f5ee;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.82rem}
.flag-geo{border-left:3px solid #7F77DD;padding:4px 10px;background:#eeedfe;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.82rem}
.flag-neutral{border-left:3px solid #D3D1C7;padding:4px 10px;background:#f8f8f6;margin:2px 0;border-radius:0 4px 4px 0;font-size:0.82rem}
.source-card{border:1px solid #D3D1C7;border-radius:8px;padding:10px 14px;margin:4px 0;background:#fafaf8;font-size:0.85rem}
.source-card a{color:#185FA5;text-decoration:none}
.assumption-box{border:1px solid #FAC775;background:#faeeda;border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.88rem}
.rejection-card{border:1px solid #F09595;background:#fdf0f0;border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.85rem}
.rule-card{border:1px solid #AFA9EC;background:#eeedfe;border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.85rem}
.hta-card{border:1px solid #9FE1CB;background:#e1f5ee;border-radius:6px;padding:10px 14px;margin:6px 0}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
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
    if any(x in f for x in ["PENALTY","CRITICAL","MISSING","BIAS","BURDEN","NO MCID","TRANSLATION GAP","HTA NOTE"]):
        return "flag-penalty"
    if any(x in f for x in ["BONUS","+35","+25","+20","+10","+5","VALIDATED MCID (+","TPP/CORE","REGULATORY TRUST","COMPETITOR","MOA SENS","ECOA READY","OPEN ACCESS","HTA ALIGNMENT (+"]):
        return "flag-bonus"
    if any(x in f for x in ["GEO","TRANSLATION","LINGUISTIC"]):
        return "flag-geo"
    return "flag-neutral"

def build_source_links(record):
    """Build clickable source links using actual URLs from the KG where available."""
    links = []
    fda_url = record.get("fda_label_url", "")
    ema_url = record.get("ema_label_url", "")
    nct = record.get("nct_id", "")
    doi = record.get("publication_doi", "")
    year = record.get("publication_year", "")
    drug = record.get("drug_name", "")
    trial = record.get("trial_name", "")

    if fda_url and fda_url.startswith("http"):
        links.append(f'<a href="{fda_url}" target="_blank">FDA label: {drug}</a>')
    elif drug:
        links.append(f'<a href="https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug.replace(" ","+")}" target="_blank">FDA DailyMed: {drug}</a>')

    if ema_url and ema_url.startswith("http"):
        links.append(f'<a href="{ema_url}" target="_blank">EMA label: {drug}</a>')

    if nct and nct.startswith("NCT"):
        links.append(f'<a href="https://clinicaltrials.gov/study/{nct}" target="_blank">ClinicalTrials.gov: {nct}</a>')

    if doi:
        links.append(f'<a href="https://doi.org/{doi}" target="_blank">Publication DOI ({year})</a>')
    elif trial:
        links.append(f'<a href="https://pubmed.ncbi.nlm.nih.gov/?term={trial.replace(" ","+")}" target="_blank">PubMed: {trial}</a>')

    return links

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🏥 COA Strategy Agent")
st.sidebar.caption("University of Cambridge × Evinova (AstraZeneca)")
st.sidebar.divider()

st.sidebar.subheader("Trial Parameters")

indication = st.sidebar.text_input(
    "Indication *",
    placeholder="e.g. Multiple Myeloma, NSCLC, CRPC",
    help="Primary cancer indication. Drives core domain lookup and KG filter."
)
phase = st.sidebar.selectbox(
    "Trial Phase", ["Phase 3","Phase 2","Phase 1","Phase 4"],
    help="Phase 3 activates Estimand Burden penalty for instruments >30 items."
)
drug_class = st.sidebar.text_input(
    "Drug Class / Mechanism",
    placeholder="e.g. Bispecific, Proteasome Inhibitor, ICI",
    help="Used for MoA Sensitivity scoring. Bispecific/CAR-T infers step-up dosing."
)
administration = st.sidebar.selectbox(
    "Administration",
    ["Unknown / Infer","Step-up dosing","IV","Subcutaneous","Oral","Weekly IV"],
    help="Step-up dosing triggers Recall Bias penalty for instruments with >7-day recall."
)
population_subtype = st.sidebar.selectbox(
    "Patient Population",
    ["Unknown / Infer","Symptomatic","Asymptomatic/Smoldering","Mixed"],
    help="Symptomatic activates Missing Core penalty. Asymptomatic activates Asymptomatic Burden check."
)

st.sidebar.divider()
st.sidebar.subheader("Regulatory & Market Scope")

hta_markets = st.sidebar.multiselect(
    "HTA / Payer Markets",
    ["NICE","ICER","EUnetHTA","SMC"],
    default=["NICE","ICER"],
    help="NICE requires EQ-5D-5L. ICER requires a utility-based measure."
)
geographic_footprint = st.sidebar.selectbox(
    "Geographic Footprint",
    ["Global","EU","US-only","Unknown / Infer"],
    help="Activates Translation Gap penalty if validated translations are insufficient."
)

st.sidebar.divider()
st.sidebar.subheader("Knowledge Graph")
c1, c2, c3 = st.sidebar.columns(3)
c1.metric("Drugs", "36")
c2.metric("Trials", "131")
c3.metric("Instruments", "193")
st.sidebar.metric("Regulatory Reviews", "68")

st.sidebar.divider()
show_raw = st.sidebar.toggle("Show raw KG records", value=False)
show_eval = st.sidebar.toggle("Show Evaluation tab", value=True)
st.sidebar.caption("Project 2025gsk2 — Dept. of Chemical Engineering and Biotechnology, University of Cambridge")

# ============================================================================
# MAIN AREA — HEADER
# ============================================================================
st.title("PRO COA AI Agent")
st.markdown("**Evidence-based COA instrument selection for oncology clinical trials**")
st.caption("University of Cambridge, Department of Chemical Engineering and Biotechnology × Evinova (AstraZeneca)")

if not AGENT_AVAILABLE:
    st.error(f"Agent failed to load: {AGENT_ERROR}. Check your .env file and Neo4j connection.")
    st.stop()

st.divider()
st.subheader("Describe your trial")

user_query = st.text_area(
    "Trial description",
    height=130,
    placeholder=(
        "Describe your trial in plain language. Include: indication, patient population, "
        "drug mechanism, phase, what you want to measure, and any concerns.\n\n"
        "Example: We are running a Phase 3 trial of a BCMA bispecific antibody in "
        "relapsed/refractory multiple myeloma (≥3 prior lines). Step-up dosing Cycle 1. "
        "We want a label claim for treatment tolerability and physical function. "
        "What PRO instruments should we include?"
    )
)

col_btn, empty_space = st.columns([1, 1])
with col_btn:
    run_button = st.button("Generate COA Strategy", type="primary", use_container_width=True)

# ============================================================================
# AGENT EXECUTION
# ============================================================================
if run_button:
    if not user_query.strip():
        st.warning("Please enter a trial description.")
        st.stop()
    
    # Build sidebar context
    parts = []
    if indication: parts.append(f"Indication: {indication}")
    if drug_class: parts.append(f"Drug class: {drug_class}")
    if administration != "Unknown / Infer": parts.append(f"Administration: {administration}")
    if population_subtype != "Unknown / Infer": parts.append(f"Population: {population_subtype}")
    if geographic_footprint != "Unknown / Infer": parts.append(f"Geographic footprint: {geographic_footprint}")
    if hta_markets: parts.append(f"HTA markets: {', '.join(hta_markets)}")
    if phase: parts.append(f"Phase: {phase}")
    sidebar_context = ("\n\nAdditional context from filters:\n" + "\n".join(parts)) if parts else ""
    full_query = user_query + sidebar_context
    
    # Define steps
    steps = [
        {"label": "Step 1: Analyzer (Haiku) — extracting trial context", "status": "pending"},
        {"label": "Step 2: Knowledge Graph — querying Neo4j (trials, reviews, rules)", "status": "pending"},
        {"label": "Step 3: Scoring Engine — evaluating instruments (100-point scale)", "status": "pending"},
        {"label": "Step 4: Reasoner (Sonnet) — synthesising evidence + web search", "status": "pending"},
        {"label": "Step 5: Logging to evaluation dataset", "status": "pending"},
    ]
    step_ph = st.empty()
    step_ph.markdown(render_steps(steps), unsafe_allow_html=True)
    
    # Update Step 1 to running
    steps[0]["status"] = "running"
    steps[0]["detail"] = "Haiku parsing query..."
    step_ph.markdown(render_steps(steps), unsafe_allow_html=True)
    
    # Call agent
    result = get_recommendation(full_query)
    ctx = result.get("context_json", {})
    counts = result.get("record_counts", {})
    
    # Update all steps
    steps[0] = {"label": steps[0]["label"], "status": "complete",
                "detail": f"Indication: {ctx.get('indication','?')} | Phase: {ctx.get('phase','?')} | {len(ctx.get('assumptions_made',[]))} assumption(s)"}
    steps[1] = {"label": steps[1]["label"],
                "status": "error" if result.get("error_status") and "offline" in str(result.get("error_status","")) else "complete",
                "detail": f"{counts.get('instrument_records',0)} instruments | {counts.get('regulatory_reviews',0)} reviews | {counts.get('regulatory_rules',0)} rules"}
    steps[2] = {"label": steps[2]["label"], "status": "complete",
                "detail": f"{counts.get('scored_instruments',0)} scored | top: {result['top_scores'][0]['instrument_name'] if result.get('top_scores') else 'none'} | {counts.get('rejections_found',0)} rejection record(s)"}
    steps[3] = {"label": steps[3]["label"],
                "status": "complete" if result.get("answer") else "error",
                "detail": f"{len(result.get('answer',''))} chars generated"}
    steps[4] = {"label": steps[4]["label"], "status": "complete", "detail": "Saved to /logs/"}
    step_ph.markdown(render_steps(steps), unsafe_allow_html=True)
    
    if result.get("error_status"):
        st.warning(f"Notice: {result['error_status']}")
    
    st.session_state["last_result"] = result

# ============================================================================
# RESULTS — EIGHT TABS
# ============================================================================
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    ctx = result.get("context_json", {})
    top_scores = result.get("top_scores", [])
    kg_records = result.get("kg_raw_hits", [])
    reg_records = result.get("reg_records", [])
    reg_rules = result.get("reg_rules", [])
    hta_ctx = result.get("hta_context", {})
    counts = result.get("record_counts", {})

    tab_names = [
        "Strategy Recommendation",
        "Instrument Scoring",
        "Evidence & Sources",
        "Regulatory Reviews",
        "Rejection Risk Analysis",
        "HTA & Payer Alignment",
        "Agent Reasoning"
    ]
    if show_eval:
        tab_names.append("Evaluation Log")
    tabs = st.tabs(tab_names)

    # ========================================================================
    # TAB 1 — Strategy Recommendation
    # ========================================================================
    with tabs[0]:
        assumptions = ctx.get("assumptions_made", [])
        if assumptions:
            st.markdown(
                '<div class="assumption-box"><b>⚠️ Strategy Context Audit — Assumptions Made</b><br>'
                + "<br>".join(f"• {a}" for a in assumptions)
                + "<br><small>If any assumption is wrong, correct it in the sidebar and re-run.</small></div>",
                unsafe_allow_html=True
            )
        if result.get("error_status"):
            st.warning(f"Notice: {result['error_status']}")

        st.markdown(result.get("answer", "No recommendation generated."))

        st.download_button(
            "Download recommendation as .txt",
            data=result.get("answer", ""),
            file_name=f"COA_recommendation_{ctx.get('indication','unknown')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

    # ========================================================================
    # TAB 2 — Instrument Scoring
    # ========================================================================
    with tabs[1]:
        st.subheader(f"Instrument scoring — {counts.get('scored_instruments',0)} instruments evaluated")
        st.caption("Sorted: LOW risk first, then by score descending. Score is floored at 0; Risk Level carries penalty severity.")

        if not top_scores:
            st.info("No instruments scored. KG may be offline or no matching records found.")
        else:
            for i, inst in enumerate(top_scores, 1):
                with st.expander(
                    f"Rank {i}: {inst['instrument_name']} | Score: {inst['scientific_score']}/100 | "
                    f"Operational: {inst['operational_bonus']:+d} | Risk: {inst['risk_level']}",
                    expanded=(i <= 2)
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Scientific Score", f"{inst['scientific_score']}/100")
                    c2.metric("Positive Points", f"+{inst['raw_positive_score']}")
                    c3.metric("Penalty Points", f"-{inst['penalty_total']}")
                    c4.metric("Operational Bonus", f"{inst['operational_bonus']:+d}")
                    st.markdown(score_bar(inst["scientific_score"]), unsafe_allow_html=True)
                    st.markdown(risk_badge(inst["risk_level"]), unsafe_allow_html=True)

                    # Key clinical details
                    if inst.get("key_finding"):
                        st.markdown(f"**Key finding:** {inst['key_finding']}")
                    detail_cols = st.columns(3)
                    if inst.get("pro_position"):
                        detail_cols[0].markdown(f"**Endpoint position:** {inst['pro_position']}")
                    if inst.get("compliance_rate"):
                        detail_cols[1].markdown(f"**Compliance rate:** {inst['compliance_rate']}")
                    if inst.get("assessment_schedule"):
                        detail_cols[2].markdown(f"**Assessment schedule:** {inst['assessment_schedule']}")
                    if inst.get("key_toxicities"):
                        st.markdown(f"**Drug toxicities (MoA context):** {inst['key_toxicities']}")

                    # Score flags
                    st.markdown("**Score breakdown:**")
                    for flag in inst.get("flags", []):
                        st.markdown(f'<div class="{classify_flag(flag)}">{flag}</div>', unsafe_allow_html=True)

                    # Source links
                    links = build_source_links(inst)
                    if links:
                        st.markdown(
                            '<div class="source-card">' + "<br>".join(f"→ {l}" for l in links) + "</div>",
                            unsafe_allow_html=True
                        )

        with st.expander("Scoring methodology reference"):
            st.markdown("""
**Positive weights (max 100):**
- TPP/Core Fit +35 · FDA (2021) Core PRO Guidance
- Regulatory Trust +25 · FDA PRO Guidance (2009) Section V; EMA Reflection Paper (2005)
- Competitor/SoC Benchmark +20 · FDA PRO Guidance (2009) Section III.B
- MoA Sensitivity +20 · FDA PFDD Guidance 1 (2017)
- Validated MCID +10 · FDA PRO Guidance (2009) Section V.C

**Conditional penalties (score floored at 0; Risk Level set independently):**
- Missing Core -50 CRITICAL · FDA (2021) Core PRO Guidance
- Recall Bias -40 CRITICAL · FDA PFDD Guidance 3 (2022)
- Pre-specification/Alpha -35 HIGH · FDA PRO Guidance (2009) Section V; ICH E9 (1998)
- Estimand Burden -30 HIGH · ICH E9(R1) Addendum (2019)
- No MCID -20 MODERATE · FDA PRO Guidance (2009) Section V.C
- Asymptomatic Burden -20 MODERATE · FDA PFDD Guidance 2 (2018)

**Operational bonuses (independent of 100-point cap):**
- eCOA Ready +10 · FDA eCOA Guidance (2023)
- Open Access +5
- Translation Gap -15 · FDA PRO Guidance (2009) Section IV.A; ISPOR ePRO Task Force (2009)
            """)

    # ========================================================================
    # TAB 3 — Evidence & Sources
    # ========================================================================
    with tabs[2]:
        st.subheader(f"Knowledge graph evidence — {counts.get('instrument_records',0)} records")
        st.caption(
            f"Indication: {ctx.get('indication','')} | "
            f"Synonyms: {', '.join(ctx.get('indication_synonyms',[])[:2])} | "
            f"Phase: {ctx.get('phase','')}"
        )

        if not kg_records:
            st.info("No KG records retrieved. KG may be offline — recommendation used web search only.")
        else:
            for i, rec in enumerate(kg_records, 1):
                with st.expander(
                    f"[TI-{i:03d}] {rec.get('instrument_name','')} — "
                    f"{rec.get('drug_name','')} — {rec.get('trial_name','')} ({rec.get('nct_id','')})",
                    expanded=False
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Domain:** {rec.get('instrument_domain','')}")
                    c2.markdown(f"**Endpoint role:** {rec.get('endpoint_role','') or rec.get('instrument_endpoint_role','')}")
                    c3.markdown(f"**Phase:** {rec.get('phase','')}")
                    c1.markdown(f"**Significance:** {rec.get('significance','')}")
                    c2.markdown(f"**MID met:** {rec.get('mid_met','')}")
                    c3.markdown(f"**Pre-specified:** {rec.get('prespecified','') or rec.get('pro_prespecified','')}")
                    if rec.get("key_finding"):
                        st.markdown(f"**Key finding:** {rec['key_finding']}")
                    if rec.get("compliance_rate"):
                        st.markdown(f"**Compliance rate:** {rec['compliance_rate']}")
                    links = build_source_links(rec)
                    if links:
                        st.markdown(
                            '<div class="source-card">' + "<br>".join(f"→ {l}" for l in links) + "</div>",
                            unsafe_allow_html=True
                        )

        if show_raw and kg_records:
            st.subheader("Raw Neo4j records")
            st.json(kg_records)

    # ========================================================================
    # TAB 4 — Regulatory Reviews
    # ========================================================================
    with tabs[3]:
        st.subheader(f"Regulatory reviews — {counts.get('regulatory_reviews',0)} records")
        st.caption("FDA and EMA review decisions extracted from published medical review documents.")

        if not reg_records:
            st.info("No regulatory review records retrieved for this indication.")
        else:
            for i, rr in enumerate(reg_records, 1):
                decision = rr.get("decision","")
                icon = "✅" if any(x in decision.lower() for x in ["accept","approv","full"]) else "⚠️" if "partial" in decision.lower() else "❌"
                with st.expander(
                    f"[RR-{i:03d}] {rr.get('agency','')} | {rr.get('drug_name','')} | {icon} {decision}",
                    expanded=(i <= 2)
                ):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Agency:** {rr.get('agency','')}")
                    c2.markdown(f"**Indication:** {rr.get('indication_fda','') or rr.get('disease_area','')}")
                    c1.markdown(f"**Instruments accepted:** {rr.get('instruments_accepted','')}")
                    c2.markdown(f"**Claim type:** {rr.get('claim_type','')}")
                    if rr.get("approval_reason"):
                        st.markdown(f"**Why accepted:** {rr['approval_reason']}")
                    if rr.get("rejection_reason_primary"):
                        st.markdown(f"**Rejection reasons:** {rr['rejection_reason_primary']}")
                    if rr.get("label_language"):
                        st.info(f"**Final label language:** {rr['label_language']}")
                    # Label source links
                    drug = rr.get("drug_name","")
                    fda_url = rr.get("fda_label_url","")
                    ema_url = rr.get("ema_label_url","")
                    link_parts = []
                    if fda_url and fda_url.startswith("http"):
                        link_parts.append(f'<a href="{fda_url}" target="_blank">FDA label: {drug}</a>')
                    if ema_url and ema_url.startswith("http"):
                        link_parts.append(f'<a href="{ema_url}" target="_blank">EMA label: {drug}</a>')
                    if link_parts:
                        st.markdown('<div class="source-card">' + " &nbsp;|&nbsp; ".join(link_parts) + "</div>", unsafe_allow_html=True)

    # ========================================================================
    # TAB 5 — Rejection Risk Analysis
    # ========================================================================
    with tabs[4]:
        st.subheader(f"Rejection risk analysis — {counts.get('rejections_found',0)} rejection records")
        st.caption(
            "Rejection reasons extracted from actual FDA/EMA medical review documents. "
            "Use these to identify risks before submission. "
            "Systematic risks are patterns appearing across multiple drugs."
        )

        rejections = [r for r in reg_records if r.get("rejection_reason_primary") or r.get("rejection_reason_detailed")]

        if not rejections:
            st.info("No rejection records found for this indication in the knowledge graph.")
        else:
            for i, rr in enumerate(rejections, 1):
                with st.expander(
                    f"[REJ-{i:03d}] {rr.get('agency','')} — {rr.get('drug_name','')} — {rr.get('decision','')}",
                    expanded=True
                ):
                    st.markdown(
                        f'<div class="rejection-card">'
                        f'<b>Primary rejection reasons:</b><br>{rr.get("rejection_reason_primary","")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    if rr.get("rejection_reason_detailed"):
                        st.markdown("**Detailed analysis (from medical review):**")
                        st.markdown(
                            f'<div class="rejection-card">{rr["rejection_reason_detailed"]}</div>',
                            unsafe_allow_html=True
                        )
                    if rr.get("missing_data_issue"):
                        st.markdown(f"**Missing data issues:** {rr['missing_data_issue']}")
                    if rr.get("alpha_controlled"):
                        st.markdown(f"**Alpha controlled:** {rr['alpha_controlled']}")
                    if rr.get("label_language"):
                        st.info(f"**What was finally approved:** {rr['label_language']}")

        if reg_rules:
            st.divider()
            st.subheader(f"Published regulatory rules — {counts.get('regulatory_rules',0)} rules retrieved")
            for i, rule in enumerate(reg_rules, 1):
                with st.expander(
                    f"[RULE-{i:03d}] {rule.get('source_document','')} | Section {rule.get('section','')} | {rule.get('decision_type','')}",
                    expanded=False
                ):
                    st.markdown(
                        f'<div class="rule-card">'
                        f'<b>Rule:</b> {rule.get("rule_text","")}<br><br>'
                        f'<b>Context:</b> {rule.get("context","")}<br>'
                        f'<b>Stage:</b> {rule.get("lifecycle_stage","")} | '
                        f'<b>Stakeholder:</b> {rule.get("stakeholder","")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ========================================================================
    # TAB 6 — HTA & Payer Alignment
    # ========================================================================
    with tabs[5]:
        st.subheader("HTA and payer alignment")
        st.caption("HTA requirements differ from FDA/EMA. Missing EQ-5D for NICE means QALY calculation is impossible.")

        hta_markets_list = ctx.get("hta_markets", [])
        if not hta_markets_list:
            st.info("No HTA markets identified. Add them in the sidebar or your query.")
        else:
            inst_names_lower = [i["instrument_name"].lower() for i in top_scores]
            for body in hta_markets_list:
                if body in HTA_PREFERENCES:
                    h = HTA_PREFERENCES[body]
                    missing = [r for r in h["required_instruments"] if not any(r.lower() in n for n in inst_names_lower)]
                    ok = not missing
                    with st.expander(f"{'✅' if ok else '⚠️'} {body}", expanded=True):
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**Required:** {', '.join(h['required_instruments']) or 'None specified'}")
                        c2.markdown(f"**Preferred:** {', '.join(h['preferred_instruments'])}")
                        if missing:
                            st.markdown(
                                f'<div class="flag-penalty">⚠️ Missing for {body}: {", ".join(missing)}. '
                                f'{h["notes"]} [{h["reference"]}]</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="flag-bonus">✓ Required instruments present. {h["notes"]}</div>',
                                unsafe_allow_html=True
                            )
                        st.caption(f"Reference: {h['reference']}")

        st.divider()
        st.subheader("Geographic and linguistic validation")
        footprint = ctx.get("geographic_footprint", "Global")
        geo = GEOGRAPHIC_LANGUAGE_REQUIREMENTS.get(footprint, GEOGRAPHIC_LANGUAGE_REQUIREMENTS["Global"])
        st.markdown(
            f'<div class="hta-card"><b>Footprint: {footprint}</b><br>'
            f'Min validated translations: <b>{geo["min_languages"]}</b><br>'
            f'Key languages: {", ".join(geo["key_languages"])}<br>'
            f'{geo["regulatory_note"]}<br>'
            f'<small>Ref: {geo["reference"]}</small></div>',
            unsafe_allow_html=True
        )

    # ========================================================================
    # TAB 7 — Agent Reasoning
    # ========================================================================
    with tabs[6]:
        st.subheader("Agent reasoning trace")
        st.caption("Full transparency into every inference, parameter, and scoring decision. Override any assumption by adding it to your query.")

        st.markdown("#### Step 1: Analyzer output")
        assumptions = ctx.get("assumptions_made", [])
        if assumptions:
            for a in assumptions:
                st.markdown(f'<div class="assumption-box">⚠️ {a}</div>', unsafe_allow_html=True)
        else:
            st.success("No assumptions — all parameters were explicit in your query.")

        c1, c2 = st.columns(2)
        fields = [
            ("Indication", "indication"), ("Population", "population_subtype"),
            ("Phase", "phase"), ("Drug class", "drug_class"),
            ("Administration", "administration"), ("Dosing frequency", "dosing_frequency"),
            ("Geographic footprint", "geographic_footprint"), ("HTA markets", "hta_markets"),
        ]
        for j, (label, key) in enumerate(fields):
            (c1 if j % 2 == 0 else c2).markdown(f"**{label}:** {ctx.get(key,'—')}")

        st.markdown("**Core domains required (FDA indication lookup):**")
        st.info(", ".join(ctx.get("core_domains_required", [])) or "Not determined")
        st.markdown("**TPP claims:**")
        st.info(", ".join(ctx.get("tpp_claims", [])) or "Not specified")

        st.divider()
        st.markdown("#### Step 2: Knowledge graph queries")
        st.markdown(f"- Primary search: `{ctx.get('indication','')}`")
        st.markdown(f"- Synonym searches: `{', '.join(ctx.get('indication_synonyms',[])[:2])}`")
        st.markdown(f"- Returned: {counts.get('instrument_records',0)} instruments | {counts.get('regulatory_reviews',0)} reviews | {counts.get('regulatory_rules',0)} rules | {counts.get('rejections_found',0)} with rejection data")

        st.divider()
        st.markdown("#### Step 3: Scoring engine parameters")
        params = {
            "Population → Missing Core / Asymptomatic Burden": ctx.get("population_subtype"),
            "Administration → Recall Bias check": ctx.get("administration"),
            "Phase → Estimand Burden check": ctx.get("phase"),
            "Drug class → MoA Sensitivity": ctx.get("drug_class"),
            "Core domains checked": ctx.get("core_domains_required"),
            "Geographic footprint → Translation Gap": ctx.get("geographic_footprint"),
            "HTA markets → alignment flags": ctx.get("hta_markets"),
        }
        for k, v in params.items():
            st.markdown(f"- **{k}:** `{v}`")

        st.divider()
        st.markdown("#### Step 4: Reasoner settings")
        st.markdown("- Model: `claude-sonnet-4-20250514` | Max tokens: `8000`")
        st.markdown("- Web search: enabled (fda.gov, ema.europa.eu, clinicaltrials.gov, pubmed, nih.gov, ispor.org)")
        st.markdown("- Reasoning rules: 10 (including Rejection Pattern Analysis)")
        st.markdown(f"- Output: {len(result.get('answer',''))} characters across 10 sections")

    # ========================================================================
    # TAB 8 — Evaluation Log
    # ========================================================================
    if show_eval:
        with tabs[7]:
            st.subheader("Evaluation log")
            st.caption("Every recommendation is logged automatically. These logs are the evaluation dataset for human vs AI comparison (Project Objective iv).")

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
                        top = entry.get("top_5_instruments", [])
                        label = (
                            f"{entry.get('timestamp','')} | "
                            f"{entry.get('indication','unknown')} | "
                            f"{entry.get('phase','')} | "
                            f"Top: {top[0]['name'] if top else 'none'} "
                            f"({top[0]['score'] if top else '—'}/100)"
                        )
                        with st.expander(label, expanded=False):
                            st.markdown(f"**Query:** {entry.get('user_query','')[:250]}...")
                            st.markdown(f"**Assumptions:** {len(entry.get('assumptions_made',[]))}")
                            if entry.get("error_status"):
                                st.warning(f"Error: {entry['error_status']}")
                            for inst in top:
                                st.markdown(f"- {inst['name']}: {inst['score']}/100 | risk {inst['risk_level']} | op {inst.get('operational_bonus',0):+d}")
                            st.markdown(f"**Records:** {entry.get('record_counts',{})}")
                            st.download_button(
                                f"Download {lf.name}",
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
                                "kg_instrument_records": e.get("record_counts",{}).get("instrument_records",0),
                                "rejections_found": e.get("record_counts",{}).get("rejections_found",0),
                                "error": e.get("error_status",""),
                                "answer_chars": e.get("answer_length_chars",0)
                            })
                        except:
                            pass
                    df = pd.DataFrame(rows)
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    st.download_button(
                        "Download evaluation CSV",
                        data=buf.getvalue(),
                        file_name=f"COA_evaluation_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
