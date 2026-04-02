import json
import os
import calendar
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from intel_brief import STATE_FACILITY_COUNT, STATE_DEMAND_LAYERS

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
    # Track which city-only names have duplicates across states
    city_only_entries = {}
    for code, data in signals.get("markets", {}).items():
        if isinstance(data, dict) and "market_name" in data:
            name = data["market_name"].strip().upper()
            city_lookup[name] = data  # "CITY, ST" format — always unique
            city_only = name.split(",")[0].strip()
            if city_only not in city_only_entries:
                city_only_entries[city_only] = []
            city_only_entries[city_only].append(data)
    # Only add city-only keys when there's no ambiguity (single state)
    for city_only, entries in city_only_entries.items():
        if len(entries) == 1:
            city_lookup[city_only] = entries[0]
        # Duplicates (e.g. Kansas City KS/MO) — skip city-only, require "City, ST" format
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


# ──────────────────────────────────────────────────────────────────────────────
# SAME-DAY PRICING MULTIPLIER
# ──────────────────────────────────────────────────────────────────────────────

SAME_DAY_BASE_URGENCY = 1.08  # 8% baseline premium for same-day
SAME_DAY_MAX_PREMIUM = 0.60  # Max time decay premium (60% at zero hours)
SAME_DAY_POWER = 1.5  # Curve steepness (>1 = ramps faster near end)
SAME_DAY_MAX_HOURS = 12  # Full business day window (6am - 6pm)

# Market tightness factor — uses signal labels, not raw LTR
SAME_DAY_MARKET_FACTOR = {
    "Soft": 1.00,
    "Loosening": 1.03,
    "Balanced": 1.06,
    "Firming": 1.10,
    "Tightening": 1.15,
    "Acute Imbalance": 1.20,
}

# Day of week factor
SAME_DAY_DOW_FACTOR = {
    0: 1.00,  # Monday
    1: 1.00,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.00,  # Thursday
    4: 1.05,  # Friday
    5: 1.15,  # Saturday
    6: 1.15,  # Sunday
}


def compute_same_day_multiplier(hours_remaining, origin_signal=None, day_of_week=None):
    """
    Compute same-day urgency multiplier.

    Args:
        hours_remaining: float, hours until end of business day (0-12)
        origin_signal: str, signal label from dashboard (e.g. "Firming")
        day_of_week: int, 0=Monday through 6=Sunday (auto-detected if None)

    Returns dict with:
        multiplier: float, total same-day multiplier (e.g. 1.57)
        base_urgency: float, base component
        time_decay: float, time decay component
        market_factor: float, market tightness component
        day_factor: float, day of week component
        hours_remaining: float, hours used in calculation
    """
    import math

    # Clamp hours to valid range
    hours = max(0.5, min(SAME_DAY_MAX_HOURS, float(hours_remaining)))

    # Time decay: power curve — ramps gradually early, steeper near end
    # At 10h: ~1.04, at 6h: ~1.21, at 4h: ~1.33, at 2h: ~1.46, at 1h: ~1.53
    urgency_ratio = (SAME_DAY_MAX_HOURS - hours) / SAME_DAY_MAX_HOURS
    time_decay = 1.0 + SAME_DAY_MAX_PREMIUM * (urgency_ratio ** SAME_DAY_POWER)

    # Market tightness from origin signal
    market_factor = SAME_DAY_MARKET_FACTOR.get(origin_signal, 1.06)  # Default to Balanced

    # Day of week
    if day_of_week is None:
        day_of_week = datetime.now().weekday()
    day_factor = SAME_DAY_DOW_FACTOR.get(day_of_week, 1.00)

    # Combined multiplier
    multiplier = SAME_DAY_BASE_URGENCY * time_decay * market_factor * day_factor

    return {
        "multiplier": round(multiplier, 3),
        "base_urgency": SAME_DAY_BASE_URGENCY,
        "time_decay": round(time_decay, 3),
        "market_factor": market_factor,
        "day_factor": day_factor,
        "hours_remaining": hours,
    }


# ──────────────────────────────────────────────────────────────────────────────
# EOM/EOQ CALENDAR MULTIPLIER (Spot Only)
# ──────────────────────────────────────────────────────────────────────────────

EOM_BASE_PREMIUM = 0.03        # +3% during last 5 business days of month
EOQ_ADDITIONAL_PREMIUM = 0.02  # +2% additional during quarter-end months (stacks)


def compute_eom_eoq_multiplier(ref_date=None):
    """Compute end-of-month / end-of-quarter carrier cost multiplier for spot quotes.

    EOM: last 5 business days of month → +3% on carrier cost
    EOQ: March/June/Sep/Dec → additional +2% (stacks with EOM, total +5%)

    Returns dict with:
        multiplier: float (e.g. 1.05 for EOQ window)
        eom_active: bool
        eoq_active: bool
        business_days_remaining: int
    """
    if ref_date is None:
        ref_date = datetime.now()

    month = ref_date.month
    year = ref_date.year
    day = ref_date.day
    _, last_day = calendar.monthrange(year, month)

    # Count business days remaining in month (excluding today)
    bdays_remaining = 0
    for d in range(day + 1, last_day + 1):
        dt = datetime(year, month, d)
        if dt.weekday() < 5:  # Mon-Fri
            bdays_remaining += 1

    eom_active = bdays_remaining <= 4  # 0-4 business days left = last 5 business days
    eoq_active = eom_active and month in (3, 6, 9, 12)

    mult = 1.0
    if eom_active:
        mult += EOM_BASE_PREMIUM
    if eoq_active:
        mult += EOQ_ADDITIONAL_PREMIUM

    return {
        "multiplier": round(mult, 3),
        "eom_active": eom_active,
        "eoq_active": eoq_active,
        "business_days_remaining": bdays_remaining,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FREIGHT DENSITY FACTOR (Spot Only)
# ──────────────────────────────────────────────────────────────────────────────

DENSITY_MAX_FACTOR = 0.06          # Cap at ±6%
DENSITY_IMBALANCE_THRESHOLD = 5    # Minimum facility difference to trigger
DENSITY_MAX_FACILITIES = 18        # TX = densest state (normalization denominator)

# Seasonal peak months by demand layer (from AI Analyst framework)
LAYER_PEAK_MONTHS = {
    "Steel":         [1, 2, 3, 4, 5, 6, 7, 8, 9],              # Q1-Q3
    "Lumber":        [4, 5, 6, 7, 8, 9],                         # Q2-Q3
    "Lumber-PNW":    [4, 5, 6, 7, 8, 9],                         # Q2-Q3
    "Construction":  [4, 5, 6, 7, 8, 9, 10, 11, 12],             # Q2-Q4
    "ISM-Grid":      [1, 2, 3, 4, 5, 6, 7, 8, 9],              # Q1-Q3
    "Energy":        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Year-round (rig-count driven)
    "Renewables":    [1, 2, 3, 4, 5, 6, 7, 8, 9],              # Q1-Q3
    "HeavyEquip-Ag": [1, 2, 3, 7, 8, 9, 10, 11, 12],           # Q3-Q4 harvest + Q1-Q3
    "Permits-Starts":[4, 5, 6, 7, 8, 9, 10, 11, 12],            # Q2-Q4
}


def compute_freight_density_factor(orig_state, dest_state, ref_date=None):
    """Compute freight density imbalance factor for spot quotes.

    Dense origin + sparse destination = positive factor (carrier premium).
    Sparse origin + dense destination = negative factor (broker advantage).
    Weighted by how many of the origin's demand layers are currently in-season.

    Args:
        orig_state: str, 2-letter state code
        dest_state: str, 2-letter state code
        ref_date: datetime, defaults to now

    Returns dict with:
        factor: float (e.g. 0.04 = +4% carrier premium)
        orig_facilities: int
        dest_facilities: int
        imbalance: int (orig - dest)
        in_season_pct: float (0-1, fraction of origin layers currently in peak)
        season_weight: float (0.3-1.0, dampened weight applied)
    """
    if ref_date is None:
        ref_date = datetime.now()

    month = ref_date.month
    orig_fac = STATE_FACILITY_COUNT.get(orig_state, 0) if orig_state else 0
    dest_fac = STATE_FACILITY_COUNT.get(dest_state, 0) if dest_state else 0
    imbalance = orig_fac - dest_fac

    # Compute in-season percentage for origin's demand layers
    orig_layers = STATE_DEMAND_LAYERS.get(orig_state, []) if orig_state else []
    if orig_layers:
        in_season_count = sum(1 for layer in orig_layers
                              if month in LAYER_PEAK_MONTHS.get(layer, []))
        in_season_pct = in_season_count / len(orig_layers)
    else:
        in_season_pct = 0.5  # Neutral default if no layer data

    # Season weight: in-season amplifies opportunity cost, off-season dampens it
    # Floor of 0.3 so off-season still has some effect
    season_weight = 0.3 + 0.7 * in_season_pct

    # Only apply if imbalance is meaningful
    if abs(imbalance) < DENSITY_IMBALANCE_THRESHOLD:
        factor = 0.0
    else:
        # Normalize imbalance against densest state
        raw_factor = imbalance / DENSITY_MAX_FACILITIES
        factor = raw_factor * season_weight * DENSITY_MAX_FACTOR

    # Cap
    factor = max(-DENSITY_MAX_FACTOR, min(DENSITY_MAX_FACTOR, factor))

    return {
        "factor": round(factor, 4),
        "orig_facilities": orig_fac,
        "dest_facilities": dest_fac,
        "imbalance": imbalance,
        "in_season_pct": round(in_season_pct, 2),
        "season_weight": round(season_weight, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# RATE CAST — REAL-TIME LTR DIVERGENCE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

RATE_CAST_LTR_DIVERGENCE_THRESHOLD = 0.38  # 38% live vs 8-day
RATE_CAST_ZSCORE_THRESHOLD = 2.0


def check_rate_cast_triggers(live_ltr, ltr_8d, state, signals):
    """
    Check if Rate Cast should appear.

    Triggers (either fires = show Rate Cast):
      1. State z-score > 2.0
      2. Live LTR vs 8-day LTR divergence > 38%

    Returns dict with:
      triggered: bool
      trigger_reason: str or None ('live_divergence', 'state_zscore', 'both')
      live_ltr: float
      ltr_8d: float
      divergence_pct: float or None
      z_score: float or None
    """
    result = {
        "triggered": False,
        "trigger_reason": None,
        "live_ltr": live_ltr,
        "ltr_8d": ltr_8d,
        "divergence_pct": None,
        "z_score": None,
    }

    if live_ltr is None or live_ltr <= 0:
        return result

    # Trigger 2: Live vs 8-day divergence
    if ltr_8d and ltr_8d > 0:
        div_pct = (live_ltr - ltr_8d) / ltr_8d
        result["divergence_pct"] = div_pct
        if div_pct > RATE_CAST_LTR_DIVERGENCE_THRESHOLD:
            result["triggered"] = True
            result["trigger_reason"] = "live_divergence"

    # Trigger 1: State z-score
    if signals and state:
        states = signals.get("states", {})
        st_upper = state.upper() if state else None
        if st_upper and st_upper in states:
            z = states[st_upper].get("z_score")
            result["z_score"] = z
            if z is not None and z > RATE_CAST_ZSCORE_THRESHOLD:
                result["triggered"] = True
                if result["trigger_reason"]:
                    result["trigger_reason"] = "both"
                else:
                    result["trigger_reason"] = "state_zscore"

    return result


# --- Live signal from LTR divergence ---
LIVE_SIGNAL_THRESHOLDS = [
    (0.60, "Acute Imbalance"),
    (0.50, "Tightening"),
    (0.12, "Firming"),
    (-0.12, "Balanced"),
    (-0.15, "Loosening"),
]


def compute_live_signal(live_ltr, ltr_30d):
    """Compute a signal label from live LTR vs 30-day LTR divergence.
    Uses the same divergence thresholds as the dashboard export."""
    if live_ltr is None or ltr_30d is None or ltr_30d <= 0:
        return None
    pct_diff = (live_ltr - ltr_30d) / ltr_30d
    for threshold, label in LIVE_SIGNAL_THRESHOLDS:
        if pct_diff >= threshold:
            return label
    return "Soft"


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


def get_cycle_buffer(params, dashboard_signals=None):
    """Compute cycle buffer.  Prefers live macro data from the enriched
    dashboard_signals bridge; falls back to control_panel.json values."""
    # --- Try live macro from bridge first ---
    if dashboard_signals and "macro" in dashboard_signals:
        macro = dashboard_signals["macro"]
        regime = macro.get("regime") or params.get("regime", "EXPANSION")
        phase = macro.get("phase") if macro.get("phase") is not None else params.get("phase", 4)
        ltr_dir = dashboard_signals.get("national", {}).get("ltr_direction") or params.get("ltr_direction", "RISING")
    else:
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
    cycle_buffer = get_cycle_buffer(params, dashboard_signals)

    # --- Quoting posture margin bias (from enriched bridge) ---
    posture_bias = 0.0
    posture_label = None
    if dashboard_signals and "quoting_posture" in dashboard_signals:
        qp = dashboard_signals["quoting_posture"]
        posture_bias = qp.get("margin_bias", 0.0)
        posture_label = qp.get("posture")

    # --- Spot-only factors: EOM/EOQ calendar multiplier + freight density ---
    eom_eoq = compute_eom_eoq_multiplier()
    density = compute_freight_density_factor(orig_state, dest_state)

    # --- For SPOT: directional replaces vol buffer AND cycle buffer ---
    #     The directional adjustment IS the cycle at the market level.
    #     No need to double-count with a national cycle overlay.
    #     EOM/EOQ multiplier and freight density factor apply only to spot.
    # --- For CONTRACT: vol buffer + cycle buffer stay, directional is additive (dampened) ---
    if term == 0:
        # Spot: directional + liquidity + density — no vol buffer, no cycle buffer
        density_factor = density["factor"]
        eom_mult = eom_eoq["multiplier"]
        total_buffer = dir_adj_pct + liq_adj + density_factor
        effective_vol = 0.0
        effective_cycle = 0.0
    else:
        # Contract: full stack — vol buffer + liq + cycle + dampened directional
        # No EOM/EOQ or density for contracts (these are short-term spot dynamics)
        density_factor = 0.0
        eom_mult = 1.0
        total_buffer = vol_buffer + liq_adj + cycle_buffer + dir_adj_pct
        effective_vol = vol_buffer
        effective_cycle = cycle_buffer

    contract_rpm = rpm * (1 + total_buffer) * (1 + target_margin) * eom_mult
    flat_rate = contract_rpm * mi
    if term == 0:
        carrier_rpm = rpm * (1 + liq_adj + dir_adj_pct + density_factor) * eom_mult
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
        # Apply quoting posture bias to margin targets
        effective_margin = target_margin + posture_bias
        aggressive_margin = max(effective_margin - 0.04, 0.06)
        defensive_margin = effective_margin + 0.04
        quote_aggressive = round(carrier_fsc * (1 + aggressive_margin), 2)
        quote_target = round(carrier_fsc * (1 + effective_margin), 2)
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
        # Quoting posture (from enriched bridge)
        "posture": posture_label,
        "posture_margin_bias": posture_bias,
        # EOM/EOQ calendar multiplier (spot only)
        "eom_eoq_mult": eom_mult,
        "eom_active": eom_eoq["eom_active"],
        "eoq_active": eom_eoq["eoq_active"],
        "bdays_remaining": eom_eoq["business_days_remaining"],
        # Freight density factor (spot only)
        "density_factor": round(density_factor, 4),
        "density_orig_facilities": density["orig_facilities"],
        "density_dest_facilities": density["dest_facilities"],
        "density_imbalance": density["imbalance"],
        "density_in_season_pct": density["in_season_pct"],
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
