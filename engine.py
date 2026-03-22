import json
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data():
    rate = pd.read_csv(os.path.join(DATA_DIR, "rate_matrix.csv"), index_col=0)
    stddev = pd.read_csv(os.path.join(DATA_DIR, "stddev_matrix.csv"), index_col=0)
    reports = pd.read_csv(os.path.join(DATA_DIR, "reports_matrix.csv"), index_col=0)
    miles = pd.read_csv(os.path.join(DATA_DIR, "miles_matrix.csv"), index_col=0)
    zip3 = pd.read_csv(os.path.join(DATA_DIR, "zip3_lookup.csv"), dtype={"Zip3": str})
    city = pd.read_csv(os.path.join(DATA_DIR, "city_lookup.csv"))
    with open(os.path.join(DATA_DIR, "control_panel.json")) as f:
        params = json.load(f)
    return {
        "rate": rate, "stddev": stddev, "reports": reports, "miles": miles,
        "zip3": zip3, "city": city, "params": params,
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
    # Try as city name
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
        term_key = "6"
    tier_table = table[term_key].get(vol_tier, table[term_key]["MED"])
    if conf_num in tier_table:
        return tier_table[conf_num]
    return tier_table.get("75", 0.109)


def get_liq_adjustment(liq_tier, params):
    return params["liq_adj_table"].get(liq_tier, 0.0)


def price_lane(orig_market, dest_market, data, params_override=None):
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

    vol_buffer = get_vol_buffer(vol_tier, confidence, term, params)
    liq_adj = get_liq_adjustment(liq_tier, params)
    cycle_buffer = get_cycle_buffer(params)
    total_buffer = vol_buffer + liq_adj + cycle_buffer

    contract_rpm = rpm * (1 + total_buffer) * (1 + target_margin)
    flat_rate = contract_rpm * mi
    carrier_rpm = rpm * (1 + liq_adj + cycle_buffer)
    carrier_flat = carrier_rpm * mi
    carrier_fsc = carrier_flat + (fsc * mi)
    customer_fsc = flat_rate + (fsc * mi)
    margin_dollar = customer_fsc - carrier_fsc
    margin_pct = margin_dollar / customer_fsc if customer_fsc != 0 else 0

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
        "vol_buffer": round(vol_buffer, 4),
        "liq_adj": round(liq_adj, 4),
        "cycle_buffer": round(cycle_buffer, 4),
        "total_buffer": round(total_buffer, 4),
        "contract_rpm": round(contract_rpm, 4),
        "flat_rate": round(flat_rate, 2),
        "carrier_fsc": round(carrier_fsc, 2),
        "customer_fsc": round(customer_fsc, 2),
        "margin_dollar": round(margin_dollar, 2),
        "margin_pct": round(margin_pct, 4),
    }


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


def price_rfq(rfq_df, data, state_rates=None, params_override=None):
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

        pricing = price_lane(orig_market, dest_market, data, params_override)
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
