import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ──────────────────────────────────────────────────────────────────────────────
# DIRECTIONAL PRICING INTELLIGENCE
# ──────────────────────────────────────────────────────────────────────────────

# Weights: origin is present reality (60%), destination is future opportunity (40%)
ORIGIN_WEIGHT = 0.60
DEST_WEIGHT = 0.40
MAX_ADJUSTMENT = 0.18  # ±18% cap
MOMENTUM_MODIFIER = 0.02  # ±2% when momentum confirms direction
BALANCED_MIDPOINT = 37.5  # Center of Balanced range (30-44)
NORMALIZATION_RANGE = 62.5  # Max distance from midpoint (100 - 37.5)

# Term dampening: how much directional signal applies by contract term
TERM_DAMPENING = {
    0: 1.0,    # Spot: full directional signal
    6: 0.50,   # 6-month: half signal (market will shift)
    12: 0.25,  # 12-month: quarter signal (mostly irrelevant)
}

# Staleness thresholds
STALENESS_WARN_DAYS = 7
STALENESS_CUTOFF_DAYS = 14


def load_dashboard_signals():
    """Load the dashboard signals JSON if available."""
    signals_path = os.path.join(DATA_DIR, "dashboard_signals.json")
    if not os.path.exists(signals_path):
        return None
    with open(signals_path) as f:
        signals = json.load(f)

    # Build city name lookup for direct matching (bypasses DAT market codes)
    city_lookup = {}
    for code, data in signals.get("markets", {}).items():
        if isinstance(data, dict) and "market_name" in data:
            name = data["market_name"].strip().upper()
            city_lookup[name] = data
            # Also add just city name without state for fuzzy matching
            city_only = name.split(",")[0].strip()
            if city_only not in city_lookup:
                city_lookup[city_only] = data
    signals["_city_lookup"] = city_lookup

    return signals


def get_signal_staleness(signals):
    """Check how stale the dashboard signals are. Returns (days_old, dampening_factor)."""
    if signals is None:
        return None, 0.0
    as_of = signals.get("as_of")
    if not as_of:
        return None, 0.0
    try:
        signal_date = datetime.strptime(as_of, "%Y-%m-%d")
        days_old = (datetime.now() - signal_date).days
        if days_old > STALENESS_CUTOFF_DAYS:
            return days_old, 0.0  # Too stale, disable
        elif days_old > STALENESS_WARN_DAYS:
            return days_old, 0.5  # Stale, dampen 50%
        else:
            return days_old, 1.0  # Fresh, full signal
    except (ValueError, TypeError):
        return None, 0.0


def normalize_pressure(pressure_score):
    """Convert pressure score (0-100) to normalized scale (-1.0 to +1.0)."""
    if pressure_score is None:
        return 0.0  # Balanced default
    return (pressure_score - BALANCED_MIDPOINT) / NORMALIZATION_RANGE


DAT_TO_DASHBOARD_MAP = {
    "CA_FRS": "CA_FRE", "CA_LAX": "CA_LOS", "CA_SDI": "CA_SDI",  # San Diego — no match, use state
    "CA_SFR": "CA_SAN", "CA_STK": "CA_STO", "FL_JAX": "FL_JAC",
    "IL_RFD": "IL_ROC", "IN_FTW": "IN_FT ", "IN_GRY": "IN_GAR",
    "IN_SBD": "IN_S B", "MI_RAP": "MI_GRA", "MN_STC": "MN_ST ",
    "MO_GIR": "MO_CAP", "MO_STL": "MO_ST ", "NE_NPL": "NE_N P",
    "NV_VEG": "NV_LAS", "NY_BRN": "NY_BRO", "OK_OKC": "OK_OKL",
    "SD_SXF": "SD_SIO", "TX_ANT": "TX_ALB", "TX_ELP": "TX_EL ",
    "TX_FTW": "TX_FT ", "UT_SLC": "UT_SAL", "VA_RCH": "VA_RIC",
}


def get_market_pressure(market_code, signals, city_name=None, state=None):
    """Look up pressure score. Tries city name first, then DAT code, then state fallback."""
    if signals is None:
        return None, None, None

    # Try city name lookup first (most reliable for spot quotes)
    city_lookup = signals.get("_city_lookup", {})
    if city_name:
        # Try "CITY, ST" format
        if state:
            full_name = f"{city_name.strip().upper()}, {state.strip().upper()}"
            if full_name in city_lookup:
                m = city_lookup[full_name]
                return m.get("pressure_score"), m.get("signal"), m.get("momentum")
        # Try city name only
        city_only = city_name.strip().upper()
        if city_only in city_lookup:
            m = city_lookup[city_only]
            return m.get("pressure_score"), m.get("signal"), m.get("momentum")

    markets = signals.get("markets", {})

    # Try market-level by code (exact match)
    if market_code and market_code in markets:
        m = markets[market_code]
        return m.get("pressure_score"), m.get("signal"), m.get("momentum")

    # Try mapped code (DAT codes that differ from dashboard codes)
    mapped = DAT_TO_DASHBOARD_MAP.get(market_code)
    if mapped and mapped in markets:
        m = markets[mapped]
        return m.get("pressure_score"), m.get("signal"), m.get("momentum")

    # Fall back to state-level
    st = state or (market_code[:2] if market_code else None)
    states = signals.get("states", {})
    if st and st.upper() in states:
        s = states[st.upper()]
        return s.get("pressure_score"), s.get("signal"), s.get("momentum")

    return None, None, None


def compute_directional_adjustment(orig_market, dest_market, signals, term=0,
                                    orig_city=None, orig_state=None,
                                    dest_city=None, dest_state=None):
    """
    Compute the directional pricing adjustment based on origin/destination market pressure.

    Returns dict with:
        adjustment_pct: the % adjustment to carrier cost (negative = discount, positive = premium)
        orig_pressure: origin pressure score
        dest_pressure: destination pressure score
        orig_signal: origin signal name
        dest_signal: destination signal name
        staleness_days: how old the signal data is
        dampened: whether the signal was dampened (staleness or term)
    """
    if signals is None:
        return {
            "adjustment_pct": 0.0,
            "orig_pressure": None, "dest_pressure": None,
            "orig_signal": None, "dest_signal": None,
            "orig_ltr_8d": None, "dest_ltr_8d": None,
            "orig_ltr_30d": None, "dest_ltr_30d": None,
            "staleness_days": None, "dampened": False,
            "momentum_applied": False,
        }

    # Get staleness
    staleness_days, staleness_factor = get_signal_staleness(signals)

    # Get pressure scores (city name lookup first, then code fallback)
    orig_pressure, orig_signal, orig_momentum = get_market_pressure(
        orig_market, signals, city_name=orig_city, state=orig_state)
    dest_pressure, dest_signal, dest_momentum = get_market_pressure(
        dest_market, signals, city_name=dest_city, state=dest_state)

    # Get LTR values for display
    orig_ltr_8d, orig_ltr_30d = None, None
    dest_ltr_8d, dest_ltr_30d = None, None

    # Try market level first, then state
    for source_key in ["markets", "states"]:
        source = signals.get(source_key, {})
        lookup_orig = orig_market if source_key == "markets" else (orig_market[:2] if orig_market else None)
        lookup_dest = dest_market if source_key == "markets" else (dest_market[:2] if dest_market else None)
        if lookup_orig and lookup_orig in source and orig_ltr_8d is None:
            orig_ltr_8d = source[lookup_orig].get("ltr_8d")
            orig_ltr_30d = source[lookup_orig].get("ltr_30d")
        if lookup_dest and lookup_dest in source and dest_ltr_8d is None:
            dest_ltr_8d = source[lookup_dest].get("ltr_8d")
            dest_ltr_30d = source[lookup_dest].get("ltr_30d")

    # Normalize pressures
    norm_orig = normalize_pressure(orig_pressure)
    norm_dest = normalize_pressure(dest_pressure)

    # Core formula: origin pushes cost UP, destination pulls cost DOWN
    raw_adjustment = (norm_orig * ORIGIN_WEIGHT) - (norm_dest * DEST_WEIGHT)

    # Momentum modifier: if momentum confirms direction, add ±2%
    momentum_applied = False
    momentum_bonus = 0.0
    if orig_momentum is not None and dest_momentum is not None:
        # Origin momentum negative (softening further) AND dest momentum positive (tightening further)
        # = strongest directional confirmation
        if orig_momentum < -0.02 and dest_momentum > 0.02:
            momentum_bonus = -MOMENTUM_MODIFIER  # Extra discount
            momentum_applied = True
        elif orig_momentum > 0.02 and dest_momentum < -0.02:
            momentum_bonus = MOMENTUM_MODIFIER  # Extra premium
            momentum_applied = True

    # Apply momentum
    adjusted = raw_adjustment + momentum_bonus

    # Apply cap
    capped = max(-1.0, min(1.0, adjusted))

    # Scale to max adjustment
    adjustment_pct = capped * MAX_ADJUSTMENT

    # Apply term dampening
    term_factor = TERM_DAMPENING.get(int(term), 0.5)
    adjustment_pct *= term_factor

    # Apply staleness dampening
    adjustment_pct *= staleness_factor

    dampened = (term_factor < 1.0) or (staleness_factor < 1.0)

    return {
        "adjustment_pct": round(adjustment_pct, 4),
        "orig_pressure": orig_pressure,
        "dest_pressure": dest_pressure,
        "orig_signal": orig_signal,
        "dest_signal": dest_signal,
        "orig_ltr_8d": orig_ltr_8d,
        "dest_ltr_8d": dest_ltr_8d,
        "orig_ltr_30d": orig_ltr_30d,
        "dest_ltr_30d": dest_ltr_30d,
        "staleness_days": staleness_days,
        "dampened": dampened,
        "momentum_applied": momentum_applied,
    }


def load_data():
    rate = pd.read_csv(os.path.join(DATA_DIR, "rate_matrix.csv"), index_col=0)
    stddev = pd.read_csv(os.path.join(DATA_DIR, "stddev_matrix.csv"), index_col=0)
    reports = pd.read_csv(os.path.join(DATA_DIR, "reports_matrix.csv"), index_col=0)
    miles = pd.read_csv(os.path.join(DATA_DIR, "miles_matrix.csv"), index_col=0)
    zip3 = pd.read_csv(os.path.join(DATA_DIR, "zip3_lookup.csv"), dtype={"Zip3": str})
    city = pd.read_csv(os.path.join(DATA_DIR, "city_lookup.csv"))
    city_zip3_path = os.path.join(DATA_DIR, "city_zip3_lookup.csv")
    if os.path.exists(city_zip3_path):
        city_zip3 = pd.read_csv(city_zip3_path, dtype={"Zip3": str})
    else:
        city_zip3 = pd.DataFrame(columns=["City", "State", "Zip3"])
    with open(os.path.join(DATA_DIR, "control_panel.json")) as f:
        params = json.load(f)
    return {
        "rate": rate, "stddev": stddev, "reports": reports, "miles": miles,
        "zip3": zip3, "city": city, "city_zip3": city_zip3, "params": params,
    }


def resolve_market(origin_input, state_input, data):
    if origin_input is None or str(origin_input).strip() == "":
        return None, None, None
    val = str(origin_input).strip()
    # Try as zip3 (3 digits) or zip5 (take first 3)
    digits = "".join(c for c in val if c.isdigit())
    if len(digits) >= 3:
        z3 = digits[:3]
        match = data["zip3"][data["zip3"]["Zip3"] == z3]
        if not match.empty:
            row = match.iloc[0]
            return row["DAT_Market"], row["Market_City"], row["State"]
    # Try as city name (DAT market cities first)
    upper_val = val.upper()
    city_df = data["city"]
    if state_input and str(state_input).strip():
        st = str(state_input).strip().upper()
        match = city_df[(city_df["City"] == upper_val) & (city_df["State"] == st)]
        if not match.empty:
            return match.iloc[0]["DAT_Market"], upper_val, st
    match = city_df[city_df["City"] == upper_val]
    if not match.empty:
        return match.iloc[0]["DAT_Market"], upper_val, match.iloc[0]["State"]
    # Fallback: city+state -> zip3 -> DAT market (covers ~25,000 US cities)
    city_zip3_df = data.get("city_zip3", pd.DataFrame())
    if not city_zip3_df.empty:
        if state_input and str(state_input).strip():
            st = str(state_input).strip().upper()
            match = city_zip3_df[(city_zip3_df["City"] == upper_val) & (city_zip3_df["State"] == st)]
        else:
            match = city_zip3_df[city_zip3_df["City"] == upper_val]
        if not match.empty:
            z3 = match.iloc[0]["Zip3"]
            st_found = match.iloc[0]["State"]
            z3_match = data["zip3"][data["zip3"]["Zip3"] == z3]
            if not z3_match.empty:
                row = z3_match.iloc[0]
                return row["DAT_Market"], upper_val, st_found
    # Try as direct market code
    if val.upper() in data["rate"].index:
        return val.upper(), val.upper(), val[:2].upper()
    return "NO MATCH", val, state_input


def lookup_lane(orig_market, dest_market, data):
    try:
        rpm = data["rate"].loc[orig_market, dest_market]
        sd = data["stddev"].loc[orig_market, dest_market]
        rpts = data["reports"].loc[orig_market, dest_market]
        mi = data["miles"].loc[orig_market, dest_market]
        if pd.isna(rpm) or rpm == 0:
            return None
        return {"rpm": float(rpm), "stddev": float(sd), "reports": int(rpts), "miles": int(mi)}
    except (KeyError, TypeError):
        return None


def classify_volatility(stddev, rpm):
    if rpm == 0:
        return "HIGH"
    cv = stddev / rpm
    if cv < 0.20:
        return "LOW"
    elif cv < 0.30:
        return "MED"
    return "HIGH"


def classify_liquidity(reports):
    if reports < 20:
        return "THIN"
    elif reports < 50:
        return "MODERATE"
    elif reports < 100:
        return "GOOD"
    return "DEEP"


def get_cycle_buffer(params):
    regime = params.get("regime", "EXPANSION")
    phase = params.get("phase", 4)
    ltr_dir = params.get("ltr_direction", "RISING")
    if regime == "CONTRACTION":
        cycle_adj = 0.05 if ltr_dir == "RISING" else 0.03
    else:
        cycle_adj = 0.02 if ltr_dir == "RISING" else 0.0
    if phase == 0:
        multiplier = 1.2
    elif phase == 1:
        multiplier = 0.85
    elif phase == 2:
        multiplier = 0.8
    elif phase == 3:
        multiplier = 1.0
    elif phase == 4:
        multiplier = 1.15
    elif phase == 5:
        multiplier = 1.2
    else:
        multiplier = 1.0
    return cycle_adj * multiplier


def get_vol_buffer(vol_tier, confidence, term, params):
    conf_num = str(confidence).replace("P", "").strip()
    term_key = str(int(term))
    table = params["vol_buffer_table"]
    if term_key not in table:
        term_key = "6" if int(term) > 0 else "0"
    tier_table = table[term_key].get(vol_tier, table[term_key]["MED"])
    if conf_num in tier_table:
        return tier_table[conf_num]
    return tier_table.get("75", 0.109)


def get_liq_adjustment(liq_tier, params):
    return params["liq_adj_table"].get(liq_tier, 0.0)


def price_lane(orig_market, dest_market, data, params_override=None, dashboard_signals=None,
               orig_city=None, orig_state=None, dest_city=None, dest_state=None):
    params = data["params"].copy()
    if params_override:
        params.update(params_override)

    lane = lookup_lane(orig_market, dest_market, data)
    if lane is None:
        return None

    rpm = lane["rpm"]
    sd = lane["stddev"]
    rpts = lane["reports"]
    mi = lane["miles"]

    if mi < 300:
        return {"status": "<300mi", "miles": mi, "orig_market": orig_market, "dest_market": dest_market}

    vol_tier = classify_volatility(sd, rpm)
    liq_tier = classify_liquidity(rpts)
    confidence = params.get("confidence", "P75")
    term = params.get("contract_term", 6)
    target_margin = params.get("target_margin", 0.12)
    fsc = params.get("fsc_per_mile", 0.53)

    # --- Compute directional adjustment ---
    directional = compute_directional_adjustment(orig_market, dest_market, dashboard_signals, term,
                                                  orig_city=orig_city, orig_state=orig_state,
                                                  dest_city=dest_city, dest_state=dest_state)
    dir_adj_pct = directional["adjustment_pct"]

    vol_buffer = get_vol_buffer(vol_tier, confidence, term, params)
    liq_adj = get_liq_adjustment(liq_tier, params)
    cycle_buffer = get_cycle_buffer(params)

    # --- For SPOT: directional replaces vol buffer AND cycle buffer ---
    #     The directional adjustment IS the cycle at the market level.
    #     No need to double-count with a national cycle overlay.
    # --- For CONTRACT: vol buffer + cycle buffer stay, directional is additive (dampened) ---
    if term == 0:
        # Spot: directional + liquidity only — no vol buffer, no cycle buffer
        total_buffer = dir_adj_pct + liq_adj
        effective_vol = 0.0
        effective_cycle = 0.0
    else:
        # Contract: full stack — vol buffer + liq + cycle + dampened directional
        total_buffer = vol_buffer + liq_adj + cycle_buffer + dir_adj_pct
        effective_vol = vol_buffer
        effective_cycle = cycle_buffer

    contract_rpm = rpm * (1 + total_buffer) * (1 + target_margin)
    flat_rate = contract_rpm * mi
    if term == 0:
        carrier_rpm = rpm * (1 + liq_adj + dir_adj_pct)
    else:
        carrier_rpm = rpm * (1 + liq_adj + cycle_buffer + dir_adj_pct)
    carrier_flat = carrier_rpm * mi
    carrier_fsc = carrier_flat + (fsc * mi)
    customer_fsc = flat_rate + (fsc * mi)
    margin_dollar = customer_fsc - carrier_fsc
    margin_pct = margin_dollar / customer_fsc if customer_fsc != 0 else 0

    # --- Quote range for spot ---
    quote_aggressive = None
    quote_target = None
    quote_defensive = None
    if term == 0:
        # Show a range: aggressive (10%), target (margin%), defensive (margin%+4%)
        aggressive_margin = max(target_margin - 0.04, 0.06)
        defensive_margin = target_margin + 0.04
        quote_aggressive = round(carrier_fsc * (1 + aggressive_margin), 2)
        quote_target = round(carrier_fsc * (1 + target_margin), 2)
        quote_defensive = round(carrier_fsc * (1 + defensive_margin), 2)

    return {
        "status": "MATCHED",
        "orig_market": orig_market,
        "dest_market": dest_market,
        "miles": mi,
        "dat_spot_rpm": round(rpm, 4),
        "stddev": round(sd, 4),
        "reports": rpts,
        "vol_tier": vol_tier,
        "liq_tier": liq_tier,
        "vol_buffer": round(effective_vol, 4),
        "liq_adj": round(liq_adj, 4),
        "cycle_buffer": round(effective_cycle if term == 0 else cycle_buffer, 4),
        "dir_adj_pct": round(dir_adj_pct, 4),
        "total_buffer": round(total_buffer, 4),
        "contract_rpm": round(contract_rpm, 4),
        "flat_rate": round(flat_rate, 2),
        "carrier_fsc": round(carrier_fsc, 2),
        "customer_fsc": round(customer_fsc, 2),
        "margin_dollar": round(margin_dollar, 2),
        "margin_pct": round(margin_pct, 4),
        # Directional intelligence
        "orig_signal": directional["orig_signal"],
        "dest_signal": directional["dest_signal"],
        "orig_pressure": directional["orig_pressure"],
        "dest_pressure": directional["dest_pressure"],
        "orig_ltr_8d": directional["orig_ltr_8d"],
        "dest_ltr_8d": directional["dest_ltr_8d"],
        "orig_ltr_30d": directional["orig_ltr_30d"],
        "dest_ltr_30d": directional["dest_ltr_30d"],
        "momentum_applied": directional["momentum_applied"],
        "staleness_days": directional["staleness_days"],
        "signal_dampened": directional["dampened"],
        # Quote range (spot only)
        "quote_aggressive": quote_aggressive,
        "quote_target": quote_target,
        "quote_defensive": quote_defensive,
    }


def load_previous_state_rates():
    """Load the most recent historical state-to-state CSV before the current one."""
    history_dir = os.path.join(DATA_DIR, "state_history")
    if not os.path.exists(history_dir):
        return None, None
    files = sorted([f for f in os.listdir(history_dir) if f.startswith("state_") and f.endswith(".csv")])
    if len(files) < 2:
        return None, None
    # Second-to-last file is the previous one
    prev_file = files[-2]
    prev_date = prev_file.replace("state_", "").replace(".csv", "")
    prev_path = os.path.join(history_dir, prev_file)
    return parse_state_to_state_csv(prev_path), prev_date


def compute_true_freshness(orig_market, dest_market, current_rates, previous_rates):
    """Compare current state RPM to previous state RPM for true market movement signal."""
    if current_rates is None or previous_rates is None:
        return None
    orig_state = orig_market[:2] if orig_market else None
    dest_state = dest_market[:2] if dest_market else None
    if orig_state is None or dest_state is None:
        return None
    try:
        current_rpm = current_rates.loc[orig_state, dest_state]
        prev_rpm = previous_rates.loc[orig_state, dest_state]
        if pd.isna(current_rpm) or pd.isna(prev_rpm) or prev_rpm == 0:
            return None
        return round(float(current_rpm) / float(prev_rpm), 4)
    except (KeyError, TypeError):
        return None


def compute_nowcast(orig_market, dest_market, state_rates, miles, fsc_per_mile=0.53):
    orig_state = orig_market[:2] if orig_market else None
    dest_state = dest_market[:2] if dest_market else None
    if orig_state is None or dest_state is None:
        return None, None
    if orig_state not in state_rates.index or dest_state not in state_rates.columns:
        return None, None
    rpm = state_rates.loc[orig_state, dest_state]
    if pd.isna(rpm) or rpm == 0:
        return None, None
    state_rpm = float(rpm)
    nowcast_with_fsc = round(state_rpm * miles + fsc_per_mile * miles, 2)
    return nowcast_with_fsc, state_rpm


def compute_gate(model_quote, market_signal, carrier_fsc, gate_tolerance, risk_tolerance=None,
                  green_threshold=0.05, red_threshold=0.15):
    if market_signal is None or pd.isna(market_signal) or market_signal == 0:
        return {"gate_quote": model_quote, "variance_pct": None, "variance_dollar": None,
                "zone": "N/A", "floor_margin_pct": None}

    variance_pct = (market_signal - carrier_fsc) / carrier_fsc if carrier_fsc != 0 else 0
    variance_dollar = market_signal - carrier_fsc
    abs_var = abs(variance_pct)

    if abs_var <= green_threshold:
        zone = "GREEN"
    elif abs_var <= red_threshold:
        zone = "YELLOW"
    else:
        zone = "RED"

    if abs_var <= gate_tolerance:
        gate_base = model_quote
    else:
        gate_base = market_signal * (1 + gate_tolerance)

    if risk_tolerance is not None and risk_tolerance != 0:
        gate_quote = gate_base * (1 + risk_tolerance)
    else:
        gate_quote = gate_base

    floor_margin_pct = (gate_quote - market_signal) / gate_quote if gate_quote != 0 else 0

    return {
        "gate_quote": round(gate_quote, 2),
        "variance_pct": round(variance_pct, 4),
        "variance_dollar": round(variance_dollar, 2),
        "zone": zone,
        "floor_margin_pct": round(floor_margin_pct, 4),
    }


def auto_risk_tolerance(variance_pct, qtr_kelly, green_threshold=0.05):
    abs_var = abs(variance_pct)
    if abs_var <= green_threshold:
        return None
    scale_factor = 0.20 + 0.40 * min(abs_var / 0.40, 1.0)
    s_raw = qtr_kelly * scale_factor
    s = max(s_raw, 0.01)
    s = min(s, 0.08)
    return round(s, 4)


def auto_decision(variance_pct, green_threshold=0.05):
    if variance_pct is None:
        return "MODEL"
    if variance_pct >= 0:
        if abs(variance_pct) <= green_threshold:
            return "MODEL"
        return "DAT"
    else:
        return "MODEL"


def parse_state_to_state_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    dest_row = rows[1]
    dest_states = []
    for i, h in enumerate(dest_row):
        if i == 0:
            continue
        s = h.strip().replace("Dest - ", "")
        dest_states.append(s)

    data = {}
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        origin = row[0].strip().replace("Origin - ", "")
        rates = {}
        for j, dest in enumerate(dest_states):
            val = row[j + 1].strip() if j + 1 < len(row) else ""
            if val:
                try:
                    rates[dest] = float(val)
                except ValueError:
                    pass
        data[origin] = rates

    df = pd.DataFrame(data).T
    df.index.name = "Origin"
    return df


import csv


def price_rfq(rfq_df, data, state_rates=None, params_override=None, dashboard_signals=None):
    params = data["params"].copy()
    if params_override:
        params.update(params_override)

    results = []
    for idx, row in rfq_df.iterrows():
        orig_input = row.get("orig_zip3") or row.get("orig_city")
        orig_state = row.get("orig_state")
        dest_input = row.get("dest_zip3") or row.get("dest_city")
        dest_state = row.get("dest_state")

        orig_market, orig_city, orig_st = resolve_market(orig_input, orig_state, data)
        dest_market, dest_city, dest_st = resolve_market(dest_input, dest_state, data)

        if orig_market is None or dest_market is None:
            results.append({"ta_id": idx + 1, "status": "MISSING INPUT"})
            continue
        if orig_market == "NO MATCH" or dest_market == "NO MATCH":
            results.append({
                "ta_id": idx + 1, "status": "NO MATCH",
                "orig_market": orig_market, "dest_market": dest_market,
                "orig_city": orig_city, "orig_state": orig_st,
                "dest_city": dest_city, "dest_state": dest_st,
            })
            continue

        pricing = price_lane(orig_market, dest_market, data, params_override, dashboard_signals,
                             orig_city=orig_city, orig_state=orig_st,
                             dest_city=dest_city, dest_state=dest_st)
        if pricing is None:
            results.append({"ta_id": idx + 1, "status": "NO DATA", "orig_market": orig_market, "dest_market": dest_market})
            continue

        pricing["ta_id"] = idx + 1
        pricing["orig_city"] = orig_city
        pricing["orig_state"] = orig_st
        pricing["dest_city"] = dest_city
        pricing["dest_state"] = dest_st

        if pricing["status"] == "MATCHED" and state_rates is not None:
            fsc = params.get("fsc_per_mile", 0.53)
            nowcast_fsc, state_rpm = compute_nowcast(
                orig_market, dest_market, state_rates, pricing["miles"], fsc
            )
            pricing["market_signal"] = nowcast_fsc
            pricing["state_rpm"] = state_rpm

            if nowcast_fsc is not None:
                pricing["freshness_ratio"] = round(state_rpm / pricing["dat_spot_rpm"], 4) if pricing["dat_spot_rpm"] != 0 else None

                state_gate_tol = params.get("state_gate_tolerance", 0.10)
                green_thresh = params.get("green_zone", 0.05)
                red_thresh = params.get("red_zone", 0.15)

                gate_result = compute_gate(
                    pricing["customer_fsc"], nowcast_fsc, pricing["carrier_fsc"],
                    state_gate_tol, green_threshold=green_thresh, red_threshold=red_thresh
                )
                pricing.update(gate_result)
                pricing["qtr_kelly"] = round(abs(gate_result["variance_pct"]) * 0.25, 4) if gate_result["variance_pct"] is not None else None

                if gate_result["variance_pct"] is not None and pricing["qtr_kelly"] is not None:
                    pricing["auto_risk_tol"] = auto_risk_tolerance(gate_result["variance_pct"], pricing["qtr_kelly"], green_thresh)
                    pricing["auto_decision"] = auto_decision(gate_result["variance_pct"], green_thresh)

                    if pricing["auto_risk_tol"] is not None:
                        gate_with_risk = compute_gate(
                            pricing["customer_fsc"], nowcast_fsc, pricing["carrier_fsc"],
                            state_gate_tol, pricing["auto_risk_tol"],
                            green_threshold=green_thresh, red_threshold=red_thresh
                        )
                        pricing["risk_adj_quote"] = gate_with_risk["gate_quote"]
                        pricing["risk_adj_floor_margin"] = gate_with_risk["floor_margin_pct"]
                    else:
                        pricing["risk_adj_quote"] = pricing["gate_quote"]
                        pricing["risk_adj_floor_margin"] = pricing["floor_margin_pct"]

                    if pricing["auto_decision"] == "MODEL":
                        pricing["final_quote"] = pricing["customer_fsc"]
                    else:
                        pricing["final_quote"] = pricing.get("risk_adj_quote", pricing["gate_quote"])
            else:
                pricing["market_signal"] = None
                pricing["state_rpm"] = None
                pricing["final_quote"] = pricing["customer_fsc"]
        else:
            pricing["final_quote"] = pricing.get("customer_fsc")

        results.append(pricing)

    return pd.DataFrame(results)
