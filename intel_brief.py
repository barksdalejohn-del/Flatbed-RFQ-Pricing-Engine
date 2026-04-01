"""
Intelligence Brief Generator
Produces a 3-4 sentence operational brief for every spot quote,
telling the account manager whether they're in a strong or weak
negotiating position and why.

Uses Claude Sonnet with deep flatbed market domain knowledge extracted
from the AI Analyst framework.
"""

import anthropic
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# STATE → DEMAND LAYER MAPPING
# Maps each state to the demand layers most relevant to its freight economy.
# Used to filter demand indicators so the brief only cites what matters
# for the specific origin/destination pair.
# ──────────────────────────────────────────────────────────────────────────────

STATE_DEMAND_LAYERS = {
    # Energy states (drilling, pipe, OCTG)
    "TX": ["Energy", "Steel", "Construction", "HeavyEquip-Ag", "Renewables"],
    "OK": ["Energy", "HeavyEquip-Ag", "Renewables"],
    "NM": ["Energy"],
    "ND": ["Energy", "HeavyEquip-Ag"],
    "CO": ["Energy"],
    "LA": ["Energy"],
    "WY": ["Energy"],
    "PA": ["Energy", "Steel"],
    "WV": ["Energy"],

    # Steel Corridor
    "IN": ["Steel", "HeavyEquip-Ag"],
    "OH": ["Steel"],
    "MI": ["Steel"],
    "AL": ["Steel", "Lumber", "Construction"],

    # Lumber / PNW
    "OR": ["Lumber", "Lumber-PNW"],
    "WA": ["Lumber", "Lumber-PNW"],
    "ID": ["Lumber", "Lumber-PNW"],
    "MT": ["Lumber", "Lumber-PNW"],
    "GA": ["Lumber", "Construction"],
    "MS": ["Lumber"],
    "AR": ["Lumber", "HeavyEquip-Ag"],

    # Ag Belt (grain prices → equipment demand)
    "IA": ["HeavyEquip-Ag", "Renewables"],
    "IL": ["HeavyEquip-Ag", "Renewables", "ISM-Grid"],
    "NE": ["HeavyEquip-Ag"],
    "MN": ["HeavyEquip-Ag"],
    "KS": ["HeavyEquip-Ag", "Renewables"],
    "SD": ["HeavyEquip-Ag"],
    "MO": ["HeavyEquip-Ag"],
    "WI": ["HeavyEquip-Ag"],

    # Construction-heavy states
    "FL": ["Construction", "Permits-Starts"],
    "CA": ["Construction", "Permits-Starts", "Renewables"],
    "NC": ["Construction", "Permits-Starts"],
    "AZ": ["Construction", "Permits-Starts"],
    "TN": ["Construction"],
    "SC": ["Construction"],
    "VA": ["Construction"],
    "MD": ["Construction"],
    "NJ": ["Construction"],
    "NY": ["Construction"],
    "CT": ["Construction"],
    "MA": ["Construction"],
    "UT": ["Construction"],
    "NV": ["Construction", "Permits-Starts"],
}


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior intelligence analyst for a flatbed freight brokerage. You produce 3-4 sentence operational briefs for account managers quoting spot loads. You have deep expertise in flatbed freight markets.

## YOUR DOMAIN KNOWLEDGE

PRESSURE SCORES (0-100): Composite of 30D LTR baseline ratio (40%), 8D/30D divergence (35%), and absolute gap (25%). Classification: Soft (<15), Loosening (15-29), Balanced (30-44), Firming (45-64), Tightening (65-79), Acute Imbalance (80+). The midpoint of Balanced (37.5) is the neutral anchor — scores above mean carriers have leverage, below means brokers have leverage.

DAT BEST FIT RELIABILITY BY MARKET PHASE:
- Escalating markets (rising pressure scores): DAT is UNDERSTATING current carrier costs. Carriers will demand above Best Fit.
- Stable markets (flat scores): DAT is ALIGNED with reality.
- Softening markets (falling scores): DAT is OVERSTATING carrier costs. You can negotiate below Best Fit.
- Multi-shift states (2+ classification levels crossed in one week): DAT Best Fit is MOST unreliable.

THE 9 STATE-MARKET SCENARIOS:
- Hot State + Hot Market: MAXIMUM FIRMNESS. Everything confirms. Coverage is priority.
- Hot State + Warm Market: FIRM WITH NUANCE. State is tight, market not at extreme. Carriers can reposition to hotter markets nearby.
- Hot State + Cool Market: CAREFUL OPPORTUNITY. Brief window before carriers discover tighter markets. Act fast.
- Warm State + Hot Market: MARKET-LEVEL FIRMNESS. State average understates local reality. Quote to market signal.
- Warm State + Warm Market: STANDARD FIRMNESS. Straightforward moderate tightening.
- Warm State + Cool Market: COMPETITIVE OPPORTUNITY. Excess local capacity. You have leverage.
- Cool State + Hot Market: LOCALIZED PREMIUM. Local spike in broadly soft state. Size bet conservatively — spikes revert faster.
- Cool State + Warm Market: WATCHFUL. Could be noise or leading edge of turn. Monitor.
- Cool State + Cool Market: MAXIMUM LEVERAGE. Everything confirms overcapacity. Never quote below cost floor.

DEMAND LAYERS — WHY THEY MATTER FOR SPOT:
When demand indicators are rising for layers present at the origin, carriers have structural load options beyond your freight. This reduces your leverage even if the pressure score hasn't caught up yet (indicators lead, pressure scores lag). When demand indicators are falling at the origin, carriers are losing alternative load options — your freight becomes more valuable to them.

CONVICTION AND POSTURE:
Combined conviction score (0-6) reflects monthly structural positioning plus weekly momentum. High conviction + DEFENSIVE posture means the market is confirming tightening across multiple timeframes — protect cost risk. Low conviction + OPPORTUNISTIC/AGGRESSIVE posture means oversupply conditions — pursue volume.

RECENCY BIAS TRAP:
When a market has spiked rapidly and then STABILIZED at a high level, it is still technically tight but no longer escalating. DAT catches up. Do not add buffers for continued climbing if momentum is flat. Look for "high score + no momentum" as the signal.

## YOUR OUTPUT FORMAT

Your brief must:
1. Open with a clear position assessment in bold: **STRONG POSITION**, **NEUTRAL**, or **DISADVANTAGED**
2. Cite 2-3 specific signals driving your assessment — use actual numbers (pressure scores, LTR values, indicator values, directional adjustment percentage)
3. End with one clear action: quote aggressively and hold margin, or expect to pay up and move quickly

## RULES

- Never exceed 4 sentences
- Use the data provided — do not invent numbers
- Reference demand indicators only when they appear in the RELEVANT DEMAND INDICATORS section
- Do not repeat the quote range — the account manager already sees those numbers. Focus on WHY and WHAT TO DO
- If same-day mode is active, factor urgency into your recommendation
- Write in direct, confident language — this person is making decisions in seconds
- Do not use bullet points or headers — write flowing sentences
- Name the bet: if your assessment could be wrong, briefly note why in your final sentence
- When signals diverge (e.g., high pressure but falling demand indicators), hold the tension — surface the conflict, do not paper over it"""


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_relevant_indicators(orig_state, dest_state, demand_indicators):
    """Filter demand indicators to only those relevant to the origin/destination states.

    Returns list of dicts with key, label, latest, direction, unit.
    Always includes ISM (national relevance).
    """
    if not demand_indicators:
        return []

    # Collect relevant layers from both states
    orig_layers = set(STATE_DEMAND_LAYERS.get(orig_state, []))
    dest_layers = set(STATE_DEMAND_LAYERS.get(dest_state, []))
    relevant_layers = orig_layers | dest_layers
    # Always include ISM (national)
    relevant_layers.add("ISM-Grid")

    relevant = []
    for key, ind in demand_indicators.items():
        ind_layers = set(ind.get("layers", []))
        if ind_layers & relevant_layers:
            relevant.append({
                "key": key,
                "label": ind["label"],
                "latest": ind["latest"],
                "prior": ind.get("prior"),
                "direction": ind["direction"],
                "unit": ind["unit"],
                "layers": ind["layers"],
            })

    return relevant


def build_brief_context(
    orig_display, dest_display, miles,
    orig_sig, dest_sig,
    orig_pressure, dest_pressure,
    orig_ltr_8d, dest_ltr_8d,
    orig_ltr_30d, dest_ltr_30d,
    dir_adj, momentum_applied,
    macro, quoting_posture,
    best_fit, range_low, range_high,
    rate_strength, reports,
    vol_label, std_dev_used,
    same_day_on=False, same_day_mult=1.0,
    relevant_indicators=None,
    staleness_days=None,
    origin_live_ltr=None, dest_live_ltr=None,
):
    """Build the context string passed to Claude for brief generation."""

    lines = []
    lines.append(f"LANE: {orig_display} -> {dest_display} | {miles:,} miles")
    lines.append("")

    lines.append("ORIGIN MARKET:")
    lines.append(f"  Signal: {orig_sig or 'Unknown'} (Pressure Score: {orig_pressure or 'N/A'})")
    lines.append(f"  LTR 8-day: {orig_ltr_8d or 'N/A'} | 30-day: {orig_ltr_30d or 'N/A'}")
    if origin_live_ltr is not None:
        lines.append(f"  Live LTR (current day): {origin_live_ltr}")
    lines.append("")

    lines.append("DESTINATION MARKET:")
    lines.append(f"  Signal: {dest_sig or 'Unknown'} (Pressure Score: {dest_pressure or 'N/A'})")
    lines.append(f"  LTR 8-day: {dest_ltr_8d or 'N/A'} | 30-day: {dest_ltr_30d or 'N/A'}")
    if dest_live_ltr is not None:
        lines.append(f"  Live LTR (current day): {dest_live_ltr}")
    lines.append("")

    # Directional
    if dir_adj < -0.03:
        flow_label = "Carrier-friendly (broker advantage)"
    elif dir_adj > 0.03:
        flow_label = "Carrier-unfriendly (carrier advantage)"
    else:
        flow_label = "Balanced flow"

    lines.append(f"DIRECTIONAL ADJUSTMENT: {dir_adj:+.1%} — {flow_label}")
    lines.append(f"MOMENTUM CONFIRMED: {'Yes' if momentum_applied else 'No'}")
    lines.append("")

    # Macro
    if macro:
        lines.append("NATIONAL MACRO:")
        lines.append(f"  Regime: {macro.get('regime', 'N/A')} | Conviction: {macro.get('conviction_score', 'N/A')}/6")
        lines.append(f"  Label: {macro.get('output_label', 'N/A')}")
        lines.append("")

    if quoting_posture:
        lines.append(f"QUOTING POSTURE: {quoting_posture.get('posture', 'N/A')} (margin bias: {quoting_posture.get('margin_bias', 0):+.0%})")
        lines.append("")

    # DAT data
    lines.append("DAT RATE DATA:")
    lines.append(f"  Best Fit: ${best_fit:,.0f} | Range: ${range_low:,.0f} - ${range_high:,.0f}")
    lines.append(f"  Rate Strength: {rate_strength} | Reports: {reports}")
    lines.append(f"  Volatility: {vol_label} (StdDev: ${std_dev_used:,.0f})")
    lines.append("")

    # Same day
    if same_day_on:
        lines.append(f"SAME-DAY MODE: ACTIVE (multiplier: {same_day_mult:.2f}x)")
        lines.append("")

    # Staleness
    if staleness_days is not None:
        lines.append(f"SIGNAL AGE: {staleness_days} days old")
        lines.append("")

    # Demand indicators
    if relevant_indicators:
        lines.append("RELEVANT DEMAND INDICATORS:")
        for ind in relevant_indicators:
            arrow = {"RISING": "Rising", "FALLING": "Falling", "FLAT": "Flat"}[ind["direction"]]
            lines.append(f"  {ind['label']}: {ind['latest']} {ind['unit']} ({arrow})")
            if ind.get("prior"):
                pct_chg = (ind["latest"] - ind["prior"]) / ind["prior"] * 100
                lines.append(f"    vs prior month: {pct_chg:+.1f}%")
    else:
        lines.append("RELEVANT DEMAND INDICATORS: None specific to this lane")

    return "\n".join(lines)


def generate_intel_brief(context_string):
    """Call Claude API to generate the intelligence brief.

    Returns the brief text string, or None on failure.
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY", st.secrets.get("anthropic_api_key"))
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context_string}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"Intel brief generation failed: {e}")
        return None
