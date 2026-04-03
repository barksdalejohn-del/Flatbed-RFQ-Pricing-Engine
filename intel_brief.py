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

ORIGIN vs DESTINATION — THE CRITICAL DISTINCTION:
- ORIGIN pressure = carrier leverage over the broker. A tight origin means carriers have abundant load options and will be selective. The broker is competing for capacity. A soft or softening origin means carriers are losing leverage — the broker gains negotiating power.
- DESTINATION pressure = carrier reload opportunity = broker leverage. A tight destination means the carrier will have loads available after delivery — this makes YOUR load more attractive because they won't sit empty. A SOFT destination means the carrier faces a freight desert after delivery and must price in deadhead/repositioning cost to get back to freight.
- Therefore: Soft origin + tight destination = STRONG broker position (carrier needs loads, and your load delivers them to a good market). Tight origin + soft destination = DISADVANTAGED (carrier has options at origin and faces a freight desert at delivery).

STRUCTURAL DENSITY vs LIVE CONDITIONS — DO NOT CONFLATE:
- RELOAD OPPORTUNITY is determined by STRUCTURAL DENSITY (facility count from the FREIGHT DENSITY section), not by the live LTR on any given day. A freight-dense destination like DFW (Texas, 36 facilities) or Atlanta (Georgia, 13 facilities) provides structural reload opportunity even when today's live LTR is temporarily soft. Facilities don't disappear because the live LTR dipped — they generate freight across multiple demand layers on an ongoing basis.
- Live LTR softness at the destination tells you the market is temporarily loose RIGHT NOW — meaning the carrier can probably find a reload, but at lower rates. This is STILL a reload opportunity. A true "freight desert" is a destination with few or zero facilities (e.g., Nevada at 2, New Mexico at 1, Wyoming at 0).
- NEVER say "no reload advantage" for a destination with 5+ facilities. That destination has structural reload availability. You can say reload conditions are "soft" or "temporarily loose," but not absent.
- For intra-state or short-haul moves within freight-dense states (especially Texas, Alabama, Georgia, Indiana), the destination reload discussion matters less because the carrier never leaves the freight-rich ecosystem. Acknowledge this when applicable.

SUMMARY TABLE — WHO HAS LEVERAGE:
- Soft/softening origin = BROKER advantage (carriers losing leverage, need loads)
- Tight origin = CARRIER advantage (carriers have options, broker competes)
- Dense destination (high facility count) = BROKER advantage (carrier wants to go there for reloads)
- Sparse destination (low facility count) = CARRIER advantage (carrier demands premium for deadhead risk)
- Both origin softening AND destination structurally dense = signals COMPOUND in broker's favor → lean toward STRONG POSITION

- LANGUAGE RULE: When both origin and destination signals favor the same side (e.g., soft origin AND dense/tight destination both favor the broker), do NOT use contrastive language ("but," "however," "despite," "although") that implies one offsets the other. These signals COMPOUND — use additive language ("and," "reinforced by," "compounded by"). Contrastive language is only appropriate when origin and destination signals genuinely pull in opposite directions (e.g., tight origin working against the broker while dense destination works for them).

LIVE LTR vs WEEKLY DATA:
When live LTR (current day) is provided alongside weekly 8-day averages, the live data reflects TODAY's conditions. Weekly pressure scores can lag by days. When live and weekly signals diverge meaningfully, weight the live data and note the divergence. Example: if weekly says "Tightening" but live LTR has dropped to neutral, the market is softening faster than the weekly score reflects.

CALENDAR SEASONALITY:
- End-of-month (last 3-5 business days): Shippers push freight to meet quotas and bonuses. Predictable demand surge. Carrier leverage increases.
- End-of-quarter (March, June, September, December): End-of-month effect compounds. Q1 and Q4 close are typically the most intense. Factor this into every assessment during these windows.
- When the CALENDAR CONTEXT section says end-of-month or end-of-quarter is active, this is a significant factor that increases carrier leverage regardless of what pressure scores show.

GEOGRAPHIC FREIGHT DENSITY AND SEASONAL OPPORTUNITY COST:
When the FREIGHT DENSITY section shows high facility count at origin and low at destination, this means the carrier is leaving abundant freight opportunities to haul into a sparse market. Carriers price for this opportunity cost. Long-haul lanes (1,000+ miles) from dense origins to sparse destinations carry structural carrier premiums that DAT Best Fit often understates.

Critically, this opportunity cost is SEASONAL. The demand map facilities produce flatbed freight on seasonal cycles:
- Steel (mills, mini-mills): Peak Q1-Q3
- Lumber (SYP mills, plywood, OSB): Peak Q2-Q3
- Construction (cement, gypsum, roofing, precast): Peak Q2-Q4
- ISM-Grid (manufacturing hubs, Caterpillar, Ford, transformers): Peak Q1-Q3
- Energy (Permian, Gulf, Bakken — OCTG, pipe): Rig-count driven, year-round
- Renewables (wind/solar staging): Peak Q1-Q3
- HeavyEquip-Ag (Deere, CNH/Case): Peak Q3-Q4 harvest + Q1-Q3
- Permits-Starts (pre-build staging): Peak Q2-Q4

During peak flatbed season (roughly March-October), a carrier sitting at a dense origin like Chicago or Indianapolis has abundant high-frequency, short-haul loads from multiple active layers. Asking them to leave that for a long-haul run into a sparse destination is asking them to forgo their peak earning period. The premium they demand reflects that.

During off-season (November-February), the same origin's facilities are producing less. Construction slows, lumber demand drops, weather disrupts operations. The carrier's opportunity cost of leaving is lower — they may even prefer a long run to reposition for better freight.

When assessing freight density imbalance, always factor the current month against these seasonal peaks. A dense origin in July is far more expensive to pull a carrier from than the same origin in January.

PRESSURE SCORES (0-100): Composite of 30D LTR baseline ratio (40%), 8D/30D divergence (35%), and absolute gap (25%). Classification: Soft (<15), Loosening (15-29), Balanced (30-44), Firming (45-64), Tightening (65-79), Acute Imbalance (80+). The midpoint of Balanced (37.5) is the neutral anchor.

DAT BEST FIT RELIABILITY BY MARKET PHASE:
- Escalating markets (rising pressure scores): DAT is UNDERSTATING current carrier costs.
- Stable markets (flat scores): DAT is ALIGNED with reality.
- Softening markets (falling scores): DAT is OVERSTATING carrier costs.
- Multi-shift states (2+ classification levels crossed in one week): DAT Best Fit is MOST unreliable.
- End-of-month/quarter periods: DAT is typically UNDERSTATING because the surge is too rapid for the rolling average.

SAME-DAY RISK:
When RATERISK data is provided, it shows what carrier costs could escalate to if the load slips to same-day coverage. This is critical context even for next-day loads — if the carrier can't be booked today, tomorrow's cost exposure could be significant. Factor the RateRisk downside into your assessment of how urgently the analyst should secure coverage.

DEMAND LAYERS — WHY THEY MATTER FOR SPOT:
When demand indicators are rising for layers present at the origin, carriers have structural load options beyond your freight — this reduces broker leverage even if pressure scores haven't caught up (indicators lead, scores lag). When demand indicators are falling at the origin, carriers are losing alternative options — your freight becomes more valuable to them.

CONVICTION AND POSTURE:
Combined conviction score (0-6) reflects monthly structural positioning plus weekly momentum. High conviction + DEFENSIVE posture means tightening confirmed across timeframes. Low conviction + OPPORTUNISTIC/AGGRESSIVE posture means oversupply conditions.

RECENCY BIAS TRAP:
When a market spiked rapidly then STABILIZED at a high level, it is still tight but no longer escalating. DAT catches up. Do not add buffers for continued climbing if momentum is flat. Look for "high score + no momentum."

## YOUR OUTPUT FORMAT

Your brief must:
1. Open with a clear position assessment in bold: **STRONG POSITION**, **NEUTRAL**, or **DISADVANTAGED**
2. Cite 2-3 specific signals driving your assessment — use actual numbers (pressure scores, LTR values, indicator values, directional adjustment percentage)
3. End with one clear action: be aggressive and hold margin, or secure coverage promptly

## RULES

- Never exceed 4 sentences
- Use the data provided — do not invent numbers
- Reference demand indicators only when they appear in the RELEVANT DEMAND INDICATORS section
- Do not repeat the quote range — the account manager already sees those numbers. Focus on WHY and WHAT TO DO
- If same-day mode is active, factor urgency into your recommendation
- Write in direct, confident language — this person is making decisions in seconds
- Do not use bullet points or headers — write flowing sentences
- Name the direction and the signals, but do NOT prescribe magnitude. Never say "well above," "significantly below," or similar. Name the factors; let the analyst size the bet
- Name the bet: if your assessment could be wrong, briefly note why in your final sentence
- When signals diverge (e.g., high weekly pressure but falling live LTR), hold the tension — surface the conflict, do not paper over it"""


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


def get_calendar_context():
    """Determine end-of-month and end-of-quarter calendar context."""
    from datetime import datetime
    import calendar

    now = datetime.now()
    day = now.day
    month = now.month
    _, last_day = calendar.monthrange(now.year, month)
    days_remaining = last_day - day

    flags = []

    # End of month: last 5 calendar days
    if days_remaining <= 4:
        flags.append(f"END OF MONTH (day {day} of {last_day}, {days_remaining} days remaining)")

    # End of quarter: March, June, September, December
    if month in (3, 6, 9, 12) and days_remaining <= 4:
        quarter = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}[month]
        flags.append(f"END OF QUARTER ({quarter} close)")

    return flags


# Facility count by state from the demand map (synced with DEMAND_MAP in dashboard)
# Used to assess freight density imbalance between origin and destination
STATE_FACILITY_COUNT = {
    "TX": 36, "AL": 13, "GA": 13, "IN": 9, "NC": 9, "SC": 9,
    "CA": 8, "FL": 8, "MS": 8, "KY": 7, "LA": 7, "MO": 7, "OH": 7, "TN": 7,
    "PA": 6, "VA": 6, "IA": 5, "OK": 5, "WI": 5,
    "AZ": 4, "IL": 4, "NE": 4, "OR": 4,
    "ND": 3,
    "AR": 2, "CO": 2, "ID": 2, "KS": 2, "NV": 2, "NY": 2, "WA": 2,
    "MN": 1, "NM": 1, "UT": 1, "WV": 1,
    "AK": 0, "CT": 0, "DE": 0, "HI": 0, "MD": 0, "ME": 0, "MI": 0,
    "MT": 0, "NH": 0, "NJ": 0, "RI": 0, "SD": 0, "VT": 0, "WY": 0,
}


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
    orig_state=None, dest_state=None,
    rate_risk_triggered=False, rate_risk_costs=None,
    eom_eoq=None, density=None,
):
    """Build the context string passed to Claude for brief generation."""

    lines = []
    lines.append(f"LANE: {orig_display} -> {dest_display} | {miles:,} miles")
    lines.append("")

    # Calendar context
    cal_flags = get_calendar_context()
    if cal_flags:
        lines.append("CALENDAR CONTEXT:")
        for flag in cal_flags:
            lines.append(f"  *** {flag} ***")
        lines.append("")

    # EOM/EOQ multiplier applied
    if eom_eoq and eom_eoq.get("eom_active"):
        eom_label = "END OF QUARTER" if eom_eoq.get("eoq_active") else "END OF MONTH"
        eom_pct = (eom_eoq["multiplier"] - 1) * 100
        lines.append(f"CALENDAR CARRIER PREMIUM: +{eom_pct:.0f}% ({eom_label} — {eom_eoq['business_days_remaining']} business days remaining)")
        lines.append("")

    # Freight density factor applied
    if density and abs(density.get("factor", 0)) > 0.001:
        density_dir = "CARRIER PREMIUM" if density["factor"] > 0 else "BROKER ADVANTAGE"
        lines.append(f"FREIGHT DENSITY {density_dir}: {density['factor']:+.1%}")
        lines.append(f"  Origin ({orig_state}): {density['orig_facilities']} facilities | "
                     f"Destination ({dest_state}): {density['dest_facilities']} facilities")
        lines.append(f"  In-season demand layers at origin: {density['in_season_pct']:.0%}")
        lines.append("")

    lines.append("ORIGIN MARKET:")
    lines.append(f"  Signal: {orig_sig or 'Unknown'} (Pressure Score: {orig_pressure or 'N/A'})")
    lines.append(f"  LTR 8-day: {orig_ltr_8d or 'N/A'} | 30-day: {orig_ltr_30d or 'N/A'}")
    if origin_live_ltr is not None:
        lines.append(f"  Live LTR (current day): {origin_live_ltr}")
        if orig_ltr_8d and orig_ltr_8d > 0:
            live_div = (origin_live_ltr - orig_ltr_8d) / orig_ltr_8d * 100
            lines.append(f"  Live vs 8-day divergence: {live_div:+.0f}%")
    lines.append("")

    lines.append("DESTINATION MARKET:")
    lines.append(f"  Signal: {dest_sig or 'Unknown'} (Pressure Score: {dest_pressure or 'N/A'})")
    lines.append(f"  LTR 8-day: {dest_ltr_8d or 'N/A'} | 30-day: {dest_ltr_30d or 'N/A'}")
    if dest_live_ltr is not None:
        lines.append(f"  Live LTR (current day): {dest_live_ltr}")
        if dest_ltr_8d and dest_ltr_8d > 0:
            live_div = (dest_live_ltr - dest_ltr_8d) / dest_ltr_8d * 100
            lines.append(f"  Live vs 8-day divergence: {live_div:+.0f}%")
    lines.append("")

    # Freight density
    if orig_state or dest_state:
        orig_facilities = STATE_FACILITY_COUNT.get(orig_state, 0)
        dest_facilities = STATE_FACILITY_COUNT.get(dest_state, 0)
        lines.append("FREIGHT DENSITY (demand map facilities):")
        lines.append(f"  Origin ({orig_state}): {orig_facilities} facilities")
        lines.append(f"  Destination ({dest_state}): {dest_facilities} facilities")
        if orig_facilities >= 8 and dest_facilities <= 3:
            lines.append(f"  IMBALANCE: Carrier leaves freight-dense origin for sparse destination — structural carrier premium expected")
        elif dest_facilities >= 8 and orig_facilities <= 3:
            lines.append(f"  IMBALANCE: Carrier delivers into freight-dense destination — reload options favor broker")
        lines.append("")

    # Directional
    if dir_adj < -0.03:
        flow_label = "Carrier-friendly (broker advantage per weekly data)"
    elif dir_adj > 0.03:
        flow_label = "Carrier-unfriendly (carrier advantage per weekly data)"
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

    # RateRisk
    if rate_risk_triggered and rate_risk_costs:
        lines.append("RATERISK ALERT — SAME-DAY DOWNSIDE:")
        lines.append(f"  If this load slips to same-day coverage:")
        for hours, cost in sorted(rate_risk_costs.items(), reverse=True):
            lines.append(f"    {hours}h remaining: ${cost:,.0f}")
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
