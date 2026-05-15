"""
Condition A (Baseline LLM) and Condition B (KG‑Only) for evaluation.

Usage:
    from evaluation_baselines import run_baseline_llm, run_kg_only, format_raw_kg_text

    # Condition A
    output_a = run_baseline_llm(scenario_text)

    # Condition B – first fetch KG records with agent.get_kg_data, then:
    raw_kg = format_raw_kg_text(raw_records)
    output_b = run_kg_only(scenario_text, raw_kg)
"""

import os
import json
from anthropic import Anthropic

# ── API key (same function used in agent.py) ────────────────────────────
def get_secret(key: str) -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, ""))
    except Exception:
        return ""

_client = Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
SONNET = "claude-sonnet-4-20250514"

# ── Fixed prompt for both conditions (from Section 3.1) ─────────────────
SYSTEM_PROMPT = (
    "You are a clinical outcome assessment (COA) expert specialising in oncology "
    "clinical trials. A pharmaceutical company is planning a clinical trial and "
    "needs a PRO measurement strategy."
)

USER_PROMPT_TEMPLATE = (
    "Based on your training knowledge (no web search), please recommend:\n"
    "1. Which PRO instruments to include and why.\n"
    "2. Which instruments were used in relevant historical comparator trials.\n"
    "3. How instruments should be positioned within the endpoint hierarchy "
    "(primary / secondary / exploratory).\n"
    "4. Any FDA and EMA regulatory considerations relevant to this trial.\n"
    "5. What the COA expert must decide before finalising the strategy.\n\n"
    "Organise your response clearly with section headings and provide your "
    "reasoning for each recommendation.\n\n"
    "Trial description:\n{scenario}"
)


def _call_llm(system: str, user: str, max_tokens: int = 4000) -> str:
    """Single call to Sonnet without web search (tools=[])."""
    resp = _client.messages.create(
        model=SONNET,
        max_tokens=max_tokens,
        system=system,
        tools=[],                 # <-- no web search, no KG augmentation
        messages=[{"role": "user", "content": user}],
    )
    return " ".join(b.text for b in resp.content if hasattr(b, "text") and b.text)


# ── Condition A ─────────────────────────────────────────────────────────
def run_baseline_llm(scenario_text: str) -> str:
    """
    Plain Sonnet output for the given scenario.
    No KG, no rules, no web search.
    """
    user = USER_PROMPT_TEMPLATE.format(scenario=scenario_text)
    return _call_llm(SYSTEM_PROMPT, user)


# ── Condition B helper: format KG records as plain text ─────────────────
def format_raw_kg_text(raw_records: list) -> str:
    """
    Convert a list of KG dictionaries (as returned by agent.get_kg_data)
    into a compact, human‑readable plain‑text block that preserves the
    unstructured nature of the raw retrieval.
    """
    if not raw_records:
        return "(No KG records retrieved.)"

    # Group by trial name for readability
    trials: dict = {}
    for r in raw_records:
        tname = (r.get("trial_name") or r.get("nct_id") or "Unknown trial")
        trials.setdefault(tname, []).append(r)

    lines = ["The following records were retrieved from a clinical trial knowledge graph.\n"]
    for tname, recs in trials.items():
        r0 = recs[0]
        lines.append(
            f"Trial: {tname} "
            f"({r0.get('phase','')}; {r0.get('drug_name','')}; {r0.get('drug_class_name','')}; "
            f"NCT: {r0.get('nct_id','')})"
        )
        for r in recs:
            inst = r.get("instrument_name", "?")
            role = r.get("endpoint_role") or r.get("pro_position", "")
            sig  = r.get("significance", "")
            kf   = r.get("key_finding", "")
            lines.append(
                f"  - {inst} | Role: {role} | Significance: {sig}"
            )
            if kf:
                lines.append(f"    Key finding: {kf[:200]}")
        lines.append("")
    return "\n".join(lines)


# ── Condition B ─────────────────────────────────────────────────────────
def run_kg_only(scenario_text: str, raw_kg_text: str) -> str:
    """
    Sonnet with raw KG text appended before the scenario.
    No scoring, no domain mapping, no regulatory rules, no web search.
    """
    kg_block = (
        "ADDITIONAL CONTEXT (Knowledge Graph Retrieval):\n"
        "The following records were retrieved from a clinical trial knowledge graph. "
        "Use them to inform your recommendations.\n\n"
        f"{raw_kg_text}\n"
    )
    user = USER_PROMPT_TEMPLATE.format(scenario=scenario_text)
    user = kg_block + user
    return _call_llm(SYSTEM_PROMPT, user)

if __name__ == "__main__":
    from agent import get_kg_data, analyze_trial_context
    
    scenario = "We are running a Phase 3 randomised controlled trial of a CD38-targeted monoclonal antibody..."

    # Condition A
    output_a = run_baseline_llm(scenario)

    # Condition B
    context = analyze_trial_context(scenario)
    raw_records, _, _ = get_kg_data(context)
    raw_kg = format_raw_kg_text(raw_records)
    output_b = run_kg_only(scenario, raw_kg)

    # Save or print the outputs
    with open("output_A.txt", "w") as f:
        f.write(output_a)
    with open("output_B.txt", "w") as f:
        f.write(output_b)