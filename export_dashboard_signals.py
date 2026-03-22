"""
Export Dashboard Signals to JSON
Reads the Flatbed Market Signal Dashboard workbooks and exports
market intelligence data for the Pricing Engine to consume.

Run this after updating the dashboard weekly (or whenever LTR data refreshes).
Output: data/dashboard_signals.json
"""

import json
import os
import sys
from datetime import datetime

import openpyxl

# --- Paths ---
DASHBOARD_DIR = r"C:\Users\johnb\OneDrive - PS Logistics\Dashboard"
STATE_WB_PATH = os.path.join(DASHBOARD_DIR, "ltr_8day_vs_30day.xlsx")
MARKET_WB_PATH = os.path.join(DASHBOARD_DIR, "market_ltr_8day_vs_30day.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dashboard_signals.json")

# --- Signal thresholds (for market-level, based on divergence) ---
SIGNAL_THRESHOLDS = [
    (0.60, "Acute Imbalance"),
    (0.50, "Tightening"),
    (0.12, "Firming"),
    (-0.12, "Balanced"),
    (-0.15, "Loosening"),
]
# Below -0.15 = Soft

# --- Pressure score to signal mapping (state-level) ---
PRESSURE_THRESHOLDS = [
    (80, "Acute Imbalance"),
    (65, "Tightening"),
    (45, "Firming"),
    (30, "Balanced"),
    (15, "Loosening"),
]
# Below 15 = Soft


def classify_signal_from_divergence(pct_diff):
    """Classify market signal from 8D vs 30D divergence percentage."""
    if pct_diff is None:
        return "Balanced"
    for threshold, signal in SIGNAL_THRESHOLDS:
        if pct_diff >= threshold:
            return signal
    return "Soft"


def divergence_to_pressure_score(pct_diff, signal):
    """Estimate a pressure score from divergence for market-level data.
    State-level has actual computed scores; market-level only has divergence.
    We estimate using the midpoint of each signal's pressure range."""
    score_map = {
        "Acute Imbalance": 90,
        "Tightening": 72,
        "Firming": 55,
        "Balanced": 37,
        "Loosening": 22,
        "Soft": 7,
    }
    base = score_map.get(signal, 37)
    # Refine within the band using the divergence magnitude
    if signal == "Firming" and pct_diff is not None:
        # Firming range is 0.12-0.50, map to 45-64
        ratio = min(max((pct_diff - 0.12) / 0.38, 0), 1)
        base = 45 + ratio * 19
    elif signal == "Tightening" and pct_diff is not None:
        ratio = min(max((pct_diff - 0.50) / 0.10, 0), 1)
        base = 65 + ratio * 14
    elif signal == "Acute Imbalance" and pct_diff is not None:
        ratio = min(max((pct_diff - 0.60) / 0.40, 0), 1)
        base = 80 + ratio * 20
    elif signal == "Balanced" and pct_diff is not None:
        ratio = min(max((pct_diff + 0.12) / 0.24, 0), 1)
        base = 30 + ratio * 14
    elif signal == "Loosening" and pct_diff is not None:
        ratio = min(max((pct_diff + 0.15) / 0.03, 0), 1)
        base = 15 + ratio * 14
    elif signal == "Soft" and pct_diff is not None:
        ratio = min(max((pct_diff + 0.50) / 0.35, 0), 1)
        base = ratio * 14
    return round(base, 1)


def map_market_name_to_dat_code(market_name, state_abbrev):
    """Map dashboard market names like 'Birmingham, AL' to DAT codes like 'AL_BIR'.
    Uses first 3 letters of city + state abbreviation."""
    if not market_name or not state_abbrev:
        return None
    city = market_name.split(",")[0].strip()
    city_code = city[:3].upper()
    return f"{state_abbrev}_{city_code}"


def export_signals():
    """Main export function. Reads dashboard workbooks and writes JSON."""
    print(f"Reading state workbook: {STATE_WB_PATH}")
    state_wb = openpyxl.load_workbook(STATE_WB_PATH, data_only=True)

    # --- Extract state-level current data ---
    current_sheet = state_wb["Flatbed 8d vs 30d "] if "Flatbed 8d vs 30d " in state_wb.sheetnames else state_wb["Flatbed 8d vs 30d"]
    states = {}
    as_of_date = None
    for row in current_sheet.iter_rows(min_row=2, max_col=5, values_only=True):
        state, ltr_8d, ltr_30d, date_val, pct_diff = row
        if state is None or ltr_8d is None:
            continue
        state = str(state).strip()
        if len(state) != 2:
            continue
        if date_val and as_of_date is None:
            if isinstance(date_val, datetime):
                as_of_date = date_val.strftime("%Y-%m-%d")
            else:
                as_of_date = str(date_val)
        states[state] = {
            "ltr_8d": round(float(ltr_8d), 1) if ltr_8d else None,
            "ltr_30d": round(float(ltr_30d), 1) if ltr_30d else None,
            "divergence": round(float(pct_diff), 4) if pct_diff else None,
        }

    # --- Extract state-level pressure scores from history (most recent week) ---
    history_sheet = state_wb["LTR_State_History"]
    headers = [cell.value for cell in next(history_sheet.iter_rows(min_row=1, max_row=1))]

    # Find column indices
    col_map = {}
    for i, h in enumerate(headers):
        if h:
            col_map[str(h).strip()] = i

    # Read all rows, keep only the most recent date per state
    latest_data = {}
    latest_date = None
    national_data = {}
    for row in history_sheet.iter_rows(min_row=2, values_only=True):
        values = list(row)
        date_val = values[col_map.get("Report_Date", 0)]
        state = values[col_map.get("State", 1)]
        if date_val is None or state is None:
            continue

        if isinstance(date_val, datetime):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)

        if latest_date is None or date_str >= latest_date:
            if date_str > (latest_date or ""):
                latest_date = date_str
                latest_data = {}
            state = str(state).strip()

            pressure_score = values[col_map.get("Pressure_Score", 14)]
            pressure_state = values[col_map.get("Pressure_State", 15)]
            momentum = values[col_map.get("Momentum", 20)]
            pct_diff_zscore = values[col_map.get("PctDiff_ZScore", 18)]

            entry = {
                "pressure_score": round(float(pressure_score), 1) if pressure_score is not None else None,
                "signal": str(pressure_state) if pressure_state else None,
                "momentum": round(float(momentum), 4) if momentum is not None else None,
                "z_score": round(float(pct_diff_zscore), 2) if pct_diff_zscore is not None else None,
            }
            latest_data[state] = entry

            # Extract national data from any row
            nat_8d = values[col_map.get("National_LTR_8D", 6)]
            nat_30d = values[col_map.get("National_LTR_30D", 7)]
            nat_pct = values[col_map.get("National_Pct_Diff", 9)]
            if nat_8d is not None:
                national_data = {
                    "ltr_8d": round(float(nat_8d), 1),
                    "ltr_30d": round(float(nat_30d), 1) if nat_30d else None,
                    "divergence": round(float(nat_pct), 4) if nat_pct else None,
                }

    # Merge pressure data into state entries
    for state, pressure in latest_data.items():
        if state in states:
            states[state].update(pressure)
        else:
            states[state] = pressure

    state_wb.close()

    # --- Extract market-level data ---
    print(f"Reading market workbook: {MARKET_WB_PATH}")
    market_wb = openpyxl.load_workbook(MARKET_WB_PATH, data_only=True)
    market_sheet = market_wb["Market_Current_Week"]

    markets = {}
    for row in market_sheet.iter_rows(min_row=2, max_col=7, values_only=True):
        state_name, state_abbrev, market_name, ltr_8d, ltr_30d, pct_diff, div_signal = row
        if market_name is None or state_abbrev is None:
            continue

        state_abbrev = str(state_abbrev).strip()
        market_name = str(market_name).strip()

        # Map to DAT code
        dat_code = map_market_name_to_dat_code(market_name, state_abbrev)
        if dat_code is None:
            continue

        signal = str(div_signal) if div_signal else classify_signal_from_divergence(pct_diff)
        est_pressure = divergence_to_pressure_score(float(pct_diff) if pct_diff else None, signal)

        markets[dat_code] = {
            "market_name": market_name,
            "state": state_abbrev,
            "ltr_8d": round(float(ltr_8d), 1) if ltr_8d else None,
            "ltr_30d": round(float(ltr_30d), 1) if ltr_30d else None,
            "divergence": round(float(pct_diff), 4) if pct_diff else None,
            "signal": signal,
            "pressure_score": est_pressure,
        }

    market_wb.close()

    # --- Build the output JSON ---
    output = {
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of_date or latest_date,
        "national": national_data,
        "states": states,
        "markets": markets,
        "signal_thresholds": {
            "Acute Imbalance": {"min_pressure": 80, "min_divergence": 0.60},
            "Tightening": {"min_pressure": 65, "min_divergence": 0.50},
            "Firming": {"min_pressure": 45, "min_divergence": 0.12},
            "Balanced": {"min_pressure": 30, "min_divergence": -0.12},
            "Loosening": {"min_pressure": 15, "min_divergence": -0.15},
            "Soft": {"min_pressure": 0, "min_divergence": None},
        },
    }

    # Write JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to: {OUTPUT_PATH}")
    print(f"As of: {output['as_of']}")
    print(f"States: {len(output['states'])}")
    print(f"Markets: {len(output['markets'])}")

    # Summary
    signal_counts = {}
    for s_data in output["states"].values():
        sig = s_data.get("signal", "Unknown")
        signal_counts[sig] = signal_counts.get(sig, 0) + 1
    print(f"\nState signal distribution:")
    for sig, count in sorted(signal_counts.items()):
        print(f"  {sig}: {count}")

    return output


if __name__ == "__main__":
    export_signals()
