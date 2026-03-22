import streamlit as st
import pandas as pd
import numpy as np
import os
import io

st.set_page_config(page_title="Flatbed RFQ Pricing Engine", page_icon="🚛", layout="wide")


# --- Password Protection ---
def check_password():
    """Returns True if the user has entered the correct password."""
    def password_entered():
        if st.session_state.get("password") == st.secrets.get("app_password", "TA2026!pricing"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("## Flatbed RFQ Pricing Engine")
    st.markdown("**TA Services** — Authorized access only.")
    st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

from engine import (load_data, price_rfq, parse_state_to_state_csv, resolve_market,
                     load_previous_state_rates, compute_true_freshness)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { font-family: 'Segoe UI', Arial, sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #1e1e2e;
        border: 1px solid #3a3a4a;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #a0a0b0 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; }
    .zone-green { background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; }
    .zone-yellow { background-color: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 4px; }
    .zone-red { background-color: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def load_saved_state_rates():
    """Load the saved state-to-state CSV from the data folder (always available)."""
    saved_path = os.path.join(os.path.dirname(__file__), "data", "state_to_state.csv")
    if os.path.exists(saved_path):
        return parse_state_to_state_csv(saved_path)
    return None


@st.cache_data
def get_previous_state_rates():
    """Load the previous state-to-state CSV from history for true freshness comparison."""
    prev_rates, prev_date = load_previous_state_rates()
    return prev_rates, prev_date


@st.cache_data
def load_state_rates(file_content, filename):
    tmp = os.path.join(os.path.dirname(__file__), "data", "_tmp_state.csv")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(file_content)
    df = parse_state_to_state_csv(tmp)
    os.remove(tmp)
    return df


def parse_rfq_upload(uploaded_file):
    content = uploaded_file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_excel(io.BytesIO(content))

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "orig" in cl and "zip" in cl:
            col_map["orig_zip3"] = col
        elif "dest" in cl and "zip" in cl:
            col_map["dest_zip3"] = col
        elif "orig" in cl and "city" in cl:
            col_map["orig_city"] = col
        elif "dest" in cl and "city" in cl:
            col_map["dest_city"] = col
        elif "orig" in cl and "state" in cl:
            col_map["orig_state"] = col
        elif "dest" in cl and "state" in cl:
            col_map["dest_state"] = col
        elif "origin" in cl and "zip" not in cl and "state" not in cl and "city" not in cl:
            if "orig_zip3" not in col_map:
                col_map["orig_zip3"] = col
        elif "dest" in cl and "zip" not in cl and "state" not in cl and "city" not in cl:
            if "dest_zip3" not in col_map:
                col_map["dest_zip3"] = col

    result = pd.DataFrame()
    for target, source in col_map.items():
        result[target] = df[source].astype(str).str.strip()
        result[target] = result[target].replace({"nan": None, "": None, "None": None})

    for needed in ["orig_zip3", "orig_city", "orig_state", "dest_zip3", "dest_city", "dest_state"]:
        if needed not in result.columns:
            result[needed] = None

    return result


def format_currency(val):
    if pd.isna(val) or val is None:
        return ""
    return f"${val:,.2f}"


def format_pct(val):
    if pd.isna(val) or val is None:
        return ""
    return f"{val:.1%}"


def zone_badge(zone):
    if zone == "GREEN":
        return "🟢 Green"
    elif zone == "YELLOW":
        return "🟡 Yellow"
    elif zone == "RED":
        return "🔴 Red"
    return zone or ""


# --- Sidebar ---
st.sidebar.title("Control Panel")
data = get_data()
params = data["params"]

st.sidebar.markdown("### Market Regime")
col1, col2 = st.sidebar.columns(2)
regime = col1.selectbox("Regime", ["EXPANSION", "CONTRACTION"], index=0 if params["regime"] == "EXPANSION" else 1)
phase = col2.selectbox("Phase", [0, 1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4, 5].index(params["phase"]))
ltr_dir = col1.selectbox("LTR Direction", ["RISING", "FALLING"], index=0 if params["ltr_direction"] == "RISING" else 1)
nat_ltr = col2.number_input("National LTR", value=float(params["national_ltr"]), format="%.2f")

st.sidebar.markdown("### Pricing Parameters")
target_margin = st.sidebar.slider("Target Margin %", 0, 25, int(float(params["target_margin"]) * 100), 1, format="%d%%") / 100
term_options = {"Spot": 0, "6-Month": 6, "12-Month": 12}
term_default = {0: "Spot", 6: "6-Month", 12: "12-Month"}.get(params["contract_term"], "6-Month")
term_label = st.sidebar.selectbox("Contract Term", list(term_options.keys()),
    index=list(term_options.keys()).index(term_default))
term = term_options[term_label]
confidence = st.sidebar.selectbox("Confidence Level",
    ["P50", "P55", "P60", "P65", "P70", "P75", "P80", "P85", "P90"],
    index=["P50", "P55", "P60", "P65", "P70", "P75", "P80", "P85", "P90"].index(params["confidence"]))
fsc = st.sidebar.number_input("Fuel Surcharge ($/mi)", value=float(params["fsc_per_mile"]), format="%.2f", step=0.01)

st.sidebar.markdown("### Variance Thresholds")
c1, c2 = st.sidebar.columns(2)
green_zone = c1.number_input("Green Zone", value=float(params["green_zone"]), format="%.2f", step=0.01)
red_zone = c2.number_input("Red Zone", value=float(params["red_zone"]), format="%.2f", step=0.01)
gate_tol = st.sidebar.number_input("Lane Gate Tolerance", value=float(params["gate_tolerance"]), format="%.2f", step=0.01)
state_gate_tol = st.sidebar.number_input("State Data Gate Tolerance", value=0.10, format="%.2f", step=0.01,
    help="Wider tolerance for state-level data (coarser than lane-level). Default 10%.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Management")
st.sidebar.markdown(f"**DAT Market Data:** {params['dat_file_date']}")
st.sidebar.markdown(f"**Markets:** {data['rate'].shape[0]} x {data['rate'].shape[1]}")

# Auto-load saved state-to-state rates
saved_state_rates = load_saved_state_rates()
prev_state_rates, prev_state_date = get_previous_state_rates()
if saved_state_rates is not None:
    saved_state_path = os.path.join(os.path.dirname(__file__), "data", "state_to_state.csv")
    import datetime
    state_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(saved_state_path))
    st.sidebar.markdown(f"**State-to-State Data:** {state_mod_time.strftime('%b %d, %Y')}")
    st.sidebar.markdown(f"**State Pairs:** {saved_state_rates.shape[0]} x {saved_state_rates.shape[1]}")
    # Show history info
    history_dir = os.path.join(os.path.dirname(__file__), "data", "state_history")
    if os.path.exists(history_dir):
        history_files = sorted([f for f in os.listdir(history_dir) if f.startswith("state_") and f.endswith(".csv")])
        if len(history_files) >= 2:
            st.sidebar.markdown(f"**History:** {len(history_files)} snapshots")
            st.sidebar.markdown(f"**Previous:** {prev_state_date}")
            st.sidebar.caption("✅ True Freshness Ratio active")
        else:
            st.sidebar.caption(f"History: {len(history_files)} snapshot(s) — need 2+ for True Freshness")
else:
    st.sidebar.warning("No state-to-state data loaded. Upload below.")

dat_upload = st.sidebar.file_uploader("Update DAT Key Markets", type=["csv"], key="dat_refresh",
    help="Upload new DAT Key Markets CSV to refresh all rate matrices")
if dat_upload:
    from refresh_dat import refresh_from_dat_csv
    tmp_path = os.path.join(os.path.dirname(__file__), "data", "_tmp_dat_refresh.csv")
    with open(tmp_path, "wb") as f:
        f.write(dat_upload.getvalue())
    result = refresh_from_dat_csv(tmp_path)
    os.remove(tmp_path)
    st.sidebar.success(f"Refreshed {result['markets']}x{result['markets']} matrices — {result['dat_date']} data. Reload the page to use new data.")

state_upload = st.sidebar.file_uploader("Update State-to-State Rates", type=["csv"], key="state_refresh",
    help="Upload newer DAT state-to-state CSV to replace saved data. Previous version auto-archived.")
if state_upload:
    import datetime
    save_path = os.path.join(os.path.dirname(__file__), "data", "state_to_state.csv")
    history_dir = os.path.join(os.path.dirname(__file__), "data", "state_history")
    os.makedirs(history_dir, exist_ok=True)
    # Archive current file before replacing
    if os.path.exists(save_path):
        archive_date = datetime.datetime.fromtimestamp(os.path.getmtime(save_path)).strftime("%Y-%m-%d")
        archive_path = os.path.join(history_dir, f"state_{archive_date}.csv")
        if not os.path.exists(archive_path):
            import shutil
            shutil.copy2(save_path, archive_path)
    # Save new file
    with open(save_path, "wb") as f:
        f.write(state_upload.getvalue())
    # Also save new file to history with today's date
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    today_archive = os.path.join(history_dir, f"state_{today_str}.csv")
    with open(today_archive, "wb") as f:
        state_upload.seek(0)
        f.write(state_upload.read())
    st.sidebar.success(f"State-to-state data updated. Previous version archived as state_{archive_date}.csv. Reload to use new data.")
    st.cache_data.clear()

params_override = {
    "regime": regime, "phase": phase, "ltr_direction": ltr_dir,
    "national_ltr": nat_ltr, "target_margin": target_margin,
    "contract_term": term, "confidence": confidence,
    "fsc_per_mile": fsc, "green_zone": green_zone,
    "red_zone": red_zone, "gate_tolerance": gate_tol,
    "state_gate_tolerance": state_gate_tol,
}

# --- Main Area ---
st.title("Flatbed RFQ Pricing Engine")
st.caption("TA Services | Automated Cycle-Adjusted Spot Market Pricing")

tab_rfq, tab_cockpit, tab_quick = st.tabs(["Price RFQ", "Decision Cockpit", "Quick Quote"])

# === TAB: Price RFQ ===
with tab_rfq:
    st.markdown("### Upload Customer RFQ")
    st.markdown("Upload a CSV or Excel file with origin/destination columns. Supports Zip3, Zip5, city+state, or DAT market codes.")

    rfq_file = st.file_uploader("RFQ File", type=["csv", "xlsx", "xls"], key="rfq")

    # Auto-load saved state rates — no upload needed
    state_rates = saved_state_rates
    if state_rates is not None:
        st.caption(f"✅ State-to-state market signal active ({state_rates.shape[0]} x {state_rates.shape[1]} state pairs). Update via sidebar.")
    else:
        st.warning("No state-to-state data available. Upload via sidebar for market signal / variance analysis.")

    if rfq_file:
        rfq_df = parse_rfq_upload(rfq_file)
        st.markdown(f"**Parsed {len(rfq_df)} lanes from upload.**")

        with st.spinner("Pricing lanes..."):
            results = price_rfq(rfq_df, data, state_rates, params_override)

        matched = results[results["status"] == "MATCHED"]
        no_match = results[results["status"] == "NO MATCH"]
        short = results[results["status"] == "<300mi"]
        no_data = results[results["status"] == "NO DATA"]
        missing = results[results["status"] == "MISSING INPUT"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Lanes", len(results))
        m2.metric("Matched", len(matched))
        m3.metric("No Match", len(no_match))
        m4.metric("< 300 mi", len(short))
        m5.metric("No Data / Missing", len(no_data) + len(missing))

        if not matched.empty:
            st.markdown("---")
            st.markdown("### Pricing Results")

            display_cols = ["ta_id", "orig_market", "dest_market", "miles",
                           "dat_spot_rpm", "vol_tier", "liq_tier", "total_buffer",
                           "carrier_fsc", "customer_fsc", "margin_dollar", "margin_pct"]

            if state_rates is not None:
                display_cols += ["state_rpm", "freshness_ratio", "market_signal",
                                "variance_pct", "zone",
                                "auto_decision", "auto_risk_tol", "final_quote",
                                "risk_adj_floor_margin"]

            available = [c for c in display_cols if c in matched.columns]
            display_df = matched[available].copy()

            rename_map = {
                "ta_id": "TA", "orig_market": "Origin", "dest_market": "Dest",
                "miles": "Miles", "dat_spot_rpm": "Spot RPM", "vol_tier": "Vol",
                "liq_tier": "Liq", "total_buffer": "Buffer %",
                "carrier_fsc": "Carrier+FSC", "customer_fsc": "Model Quote",
                "margin_dollar": "Margin $", "margin_pct": "Margin %",
                "state_rpm": "State RPM", "freshness_ratio": "Fresh Ratio",
                "market_signal": "Mkt Signal", "variance_pct": "Variance %",
                "zone": "Zone", "auto_decision": "Decision",
                "auto_risk_tol": "Risk Tol", "final_quote": "Final Quote",
                "risk_adj_floor_margin": "Floor Margin %",
            }
            display_df = display_df.rename(columns=rename_map)

            st.dataframe(
                display_df.style
                    .format({
                        "Spot RPM": "${:.4f}",
                        "State RPM": "${:.4f}",
                        "Fresh Ratio": "{:.2f}x",
                        "Buffer %": "{:.1%}",
                        "Carrier+FSC": "${:,.2f}",
                        "Model Quote": "${:,.2f}",
                        "Margin $": "${:,.2f}",
                        "Margin %": "{:.1%}",
                        "Mkt Signal": "${:,.2f}",
                        "Variance %": "{:+.1%}",
                        "Risk Tol": "{:.2%}",
                        "Final Quote": "${:,.2f}",
                        "Floor Margin %": "{:.1%}",
                    }, na_rep="—")
                    .map(lambda v: "background-color: #d4edda" if v == "GREEN" else
                         ("background-color: #fff3cd" if v == "YELLOW" else
                          ("background-color: #f8d7da" if v == "RED" else "")),
                         subset=["Zone"] if "Zone" in display_df.columns else []),
                use_container_width=True, height=600, hide_index=True,
            )

            if state_rates is not None and "zone" in matched.columns:
                st.markdown("### Zone Distribution")
                zone_counts = matched["zone"].value_counts()
                zc1, zc2, zc3 = st.columns(3)
                zc1.metric("Green (0-5%)", zone_counts.get("GREEN", 0))
                zc2.metric("Yellow (5-15%)", zone_counts.get("YELLOW", 0))
                zc3.metric("Red (>15%)", zone_counts.get("RED", 0))

                dec_counts = matched.get("auto_decision", pd.Series()).value_counts()
                dc1, dc2 = st.columns(2)
                dc1.metric("MODEL decisions", dec_counts.get("MODEL", 0))
                dc2.metric("DAT decisions", dec_counts.get("DAT", 0))

            st.markdown("### Summary Statistics")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Avg Model Quote", format_currency(matched["customer_fsc"].mean()))
            sc2.metric("Avg Margin %", format_pct(matched["margin_pct"].mean()))
            sc3.metric("Total Revenue (Model)", format_currency(matched["customer_fsc"].sum()))
            sc4.metric("Total Margin $", format_currency(matched["margin_dollar"].sum()))

            st.markdown("### Export")
            export_csv = results.to_csv(index=False)
            st.download_button("Download Full Results (CSV)", export_csv, "rfq_pricing_results.csv", "text/csv")

        if not no_match.empty:
            with st.expander(f"No Match Lanes ({len(no_match)})"):
                st.dataframe(no_match[["ta_id", "orig_market", "dest_market"]].rename(
                    columns={"ta_id": "TA", "orig_market": "Origin Input", "dest_market": "Dest Input"}
                ), hide_index=True)

# === TAB: Decision Cockpit ===
with tab_cockpit:
    st.markdown("### Decision Cockpit")
    st.markdown("Review auto-calculated decisions, override Risk Tolerance and Decision per lane, then export final quotes.")

    if "results" not in dir() or rfq_file is None:
        st.info("Upload an RFQ in the 'Price RFQ' tab first, then return here to review lanes.")
    else:
        if not matched.empty and state_rates is not None:
            from engine import compute_gate

            cockpit_src = matched.copy().reset_index(drop=True)

            # Build the editable dataframe
            edit_data = pd.DataFrame({
                "TA": cockpit_src["ta_id"].astype(int),
                "Origin": cockpit_src["orig_market"],
                "Dest": cockpit_src["dest_market"],
                "Miles": cockpit_src["miles"].astype(int),
                "Carrier+FSC": cockpit_src["carrier_fsc"].round(2),
                "Model Quote": cockpit_src["customer_fsc"].round(2),
                "Mkt Signal": cockpit_src["market_signal"].round(2),
                "Variance %": cockpit_src["variance_pct"].round(4),
                "Zone": cockpit_src["zone"],
                "Qtr Kelly": cockpit_src.get("qtr_kelly", pd.Series([None]*len(cockpit_src))),
                "Risk Tol": cockpit_src.get("auto_risk_tol", pd.Series([None]*len(cockpit_src))),
                "Decision": cockpit_src.get("auto_decision", pd.Series(["MODEL"]*len(cockpit_src))),
            })

            # Replace NaN risk tol with 0.0 for editing
            edit_data["Risk Tol"] = edit_data["Risk Tol"].fillna(0.0).round(4)

            col_config = {
                "TA": st.column_config.NumberColumn("TA", disabled=True),
                "Origin": st.column_config.TextColumn("Origin", disabled=True),
                "Dest": st.column_config.TextColumn("Dest", disabled=True),
                "Miles": st.column_config.NumberColumn("Miles", disabled=True),
                "Carrier+FSC": st.column_config.NumberColumn("Carrier+FSC", format="$%,.2f", disabled=True),
                "Model Quote": st.column_config.NumberColumn("Model Quote", format="$%,.2f", disabled=True),
                "Mkt Signal": st.column_config.NumberColumn("Mkt Signal", format="$%,.2f", disabled=True),
                "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%", disabled=True),
                "Zone": st.column_config.TextColumn("Zone", disabled=True),
                "Qtr Kelly": st.column_config.NumberColumn("Qtr Kelly", format="%.4f", disabled=True),
                "Risk Tol": st.column_config.NumberColumn("Risk Tol", format="%.2f%%", min_value=0.0, max_value=0.15, step=0.005,
                    help="Editable — adjust risk tolerance per lane (0-15%)"),
                "Decision": st.column_config.SelectboxColumn("Decision", options=["MODEL", "DAT"],
                    help="Editable — choose MODEL (use model quote) or DAT (use gate-adjusted quote)"),
            }

            edited = st.data_editor(edit_data, column_config=col_config, use_container_width=True,
                                     height=500, hide_index=True, num_rows="fixed", key="cockpit_editor")

            # Recalculate quotes based on edited values
            st.markdown("---")
            st.markdown("### Recalculated Quotes")

            sgt = params_override.get("state_gate_tolerance", 0.10)
            gz = params_override.get("green_zone", 0.05)
            rz = params_override.get("red_zone", 0.15)

            final_rows = []
            for i, row in edited.iterrows():
                src = cockpit_src.iloc[i]
                risk_tol = row["Risk Tol"] if row["Risk Tol"] > 0 else None
                decision = row["Decision"]

                if decision == "MODEL":
                    final_quote = row["Model Quote"]
                    floor_margin = None
                    risk_adj_quote = None
                else:
                    gate = compute_gate(row["Model Quote"], row["Mkt Signal"], row["Carrier+FSC"],
                                       sgt, risk_tol, green_threshold=gz, red_threshold=rz)
                    risk_adj_quote = gate["gate_quote"]
                    floor_margin = gate["floor_margin_pct"]
                    final_quote = risk_adj_quote

                margin_on_final = final_quote - row["Carrier+FSC"]
                margin_pct_final = margin_on_final / final_quote if final_quote != 0 else 0

                final_rows.append({
                    "TA": int(row["TA"]),
                    "Origin": row["Origin"],
                    "Dest": row["Dest"],
                    "Miles": int(row["Miles"]),
                    "Zone": row["Zone"],
                    "Decision": decision,
                    "Risk Tol": risk_tol if risk_tol else 0,
                    "Carrier+FSC": row["Carrier+FSC"],
                    "Final Quote": round(final_quote, 2),
                    "Margin $": round(margin_on_final, 2),
                    "Margin %": round(margin_pct_final, 4),
                    "Floor Margin %": round(floor_margin, 4) if floor_margin is not None else None,
                })

            final_df = pd.DataFrame(final_rows)

            # Summary metrics
            total_rev = final_df["Final Quote"].sum()
            total_margin = final_df["Margin $"].sum()
            avg_margin = final_df["Margin %"].mean()
            model_count = (final_df["Decision"] == "MODEL").sum()
            dat_count = (final_df["Decision"] == "DAT").sum()

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Total Revenue", format_currency(total_rev))
            sm2.metric("Total Margin", format_currency(total_margin))
            sm3.metric("Avg Margin %", format_pct(avg_margin))
            sm4.metric("MODEL Lanes", model_count)
            sm5.metric("DAT Lanes", dat_count)

            st.dataframe(
                final_df.style.format({
                    "Carrier+FSC": "${:,.2f}",
                    "Final Quote": "${:,.2f}",
                    "Margin $": "${:,.2f}",
                    "Margin %": "{:.1%}",
                    "Risk Tol": "{:.2%}",
                    "Floor Margin %": "{:.1%}",
                }, na_rep="—").map(
                    lambda v: "font-weight: bold; color: #0d6efd" if v == "MODEL" else
                    ("font-weight: bold; color: #dc3545" if v == "DAT" else ""),
                    subset=["Decision"]
                ).map(
                    lambda v: "background-color: #d4edda" if v == "GREEN" else
                    ("background-color: #fff3cd" if v == "YELLOW" else
                     ("background-color: #f8d7da" if v == "RED" else "")),
                    subset=["Zone"]
                ),
                use_container_width=True, height=400, hide_index=True,
            )

            st.markdown("### Export")
            exp1, exp2 = st.columns(2)
            cockpit_csv = final_df.to_csv(index=False)
            exp1.download_button("Download Final Quotes (CSV)", cockpit_csv, "final_quotes.csv", "text/csv")

            # Customer-facing export (clean version)
            customer_df = final_df[["TA", "Origin", "Dest", "Miles", "Final Quote"]].copy()
            customer_df = customer_df.rename(columns={"Final Quote": "Quote Rate (All-In)"})
            customer_df["RPM"] = (customer_df["Quote Rate (All-In)"] / customer_df["Miles"]).round(4)
            customer_csv = customer_df.to_csv(index=False)
            exp2.download_button("Download Customer Quote (CSV)", customer_csv, "customer_quote.csv", "text/csv")

        elif state_rates is None:
            st.warning("Upload a state-to-state CSV in the 'Price RFQ' tab to enable variance analysis and the Decision Cockpit.")
        else:
            st.info("No matched lanes to display.")

# === TAB: Quick Quote ===
with tab_quick:
    st.markdown("### Quick Quote — Single Lane")
    st.markdown("Get an instant price for a single lane.")

    qq1, qq2 = st.columns(2)
    with qq1:
        st.markdown("**Origin**")
        orig_method = st.radio("Input method", ["Zip3", "City + State", "Market Code"], key="orig_method", horizontal=True)
        if orig_method == "Zip3":
            orig_val = st.text_input("Origin Zip3", placeholder="770", key="orig_zip")
            orig_state_val = None
        elif orig_method == "City + State":
            orig_val = st.text_input("Origin City", placeholder="HOUSTON", key="orig_city_q")
            orig_state_val = st.text_input("Origin State", placeholder="TX", key="orig_state_q")
        else:
            orig_val = st.text_input("Origin Market", placeholder="TX_HOU", key="orig_mkt")
            orig_state_val = None

    with qq2:
        st.markdown("**Destination**")
        dest_method = st.radio("Input method", ["Zip3", "City + State", "Market Code"], key="dest_method", horizontal=True)
        if dest_method == "Zip3":
            dest_val = st.text_input("Dest Zip3", placeholder="606", key="dest_zip")
            dest_state_val = None
        elif dest_method == "City + State":
            dest_val = st.text_input("Dest City", placeholder="CHICAGO", key="dest_city_q")
            dest_state_val = st.text_input("Dest State", placeholder="IL", key="dest_state_q")
        else:
            dest_val = st.text_input("Dest Market", placeholder="IL_CHI", key="dest_mkt")
            dest_state_val = None

    if st.button("Get Quote", type="primary"):
        orig_market, orig_city, orig_st = resolve_market(orig_val, orig_state_val, data)
        dest_market, dest_city, dest_st = resolve_market(dest_val, dest_state_val, data)

        if orig_market is None or orig_market == "NO MATCH":
            st.error(f"Could not resolve origin: {orig_val}")
        elif dest_market is None or dest_market == "NO MATCH":
            st.error(f"Could not resolve destination: {dest_val}")
        else:
            from engine import price_lane, compute_nowcast, compute_gate, auto_risk_tolerance, auto_decision
            result = price_lane(orig_market, dest_market, data, params_override)

            if result is None:
                st.error(f"No rate data for {orig_market} → {dest_market}")
            elif result.get("status") == "<300mi":
                st.warning(f"Lane is {result['miles']} miles — below 300mi minimum.")
            else:
                st.success(f"**{orig_market} → {dest_market}** | {result['miles']} miles")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("DAT Spot RPM", f"${result['dat_spot_rpm']:.4f}")
                r2.metric("Vol / Liq Tier", f"{result['vol_tier']} / {result['liq_tier']}")
                r3.metric("Total Buffer", f"{result['total_buffer']:.1%}")
                r4.metric("Carrier + FSC", format_currency(result['carrier_fsc']))

                r5, r6, r7, r8 = st.columns(4)
                r5.metric("Model Quote", format_currency(result['customer_fsc']))
                r6.metric("Margin $", format_currency(result['margin_dollar']))
                r7.metric("Margin %", format_pct(result['margin_pct']))
                r8.metric("Contract RPM", f"${result['contract_rpm']:.4f}")

                if state_rates is not None:
                    fsc_val = params_override.get("fsc_per_mile", 0.53)
                    nowcast_fsc, state_rpm_val = compute_nowcast(
                        orig_market, dest_market, state_rates, result["miles"], fsc_val)
                    if nowcast_fsc:
                        fresh_ratio = state_rpm_val / result["dat_spot_rpm"] if result["dat_spot_rpm"] else None
                        sgt = params_override.get("state_gate_tolerance", 0.10)
                        gz = params_override.get("green_zone", 0.05)
                        rz = params_override.get("red_zone", 0.15)
                        gate = compute_gate(result["customer_fsc"], nowcast_fsc, result["carrier_fsc"],
                                           sgt, green_threshold=gz, red_threshold=rz)
                        # Compute True Freshness if history available
                        true_fresh = None
                        if prev_state_rates is not None:
                            true_fresh = compute_true_freshness(orig_market, dest_market,
                                                                state_rates, prev_state_rates)

                        st.markdown("---")
                        st.markdown("**Market Signal (State-to-State)**")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("State RPM", f"${state_rpm_val:.4f}")
                        if true_fresh is not None:
                            pct_change = (true_fresh - 1) * 100
                            direction = "↑" if pct_change > 0.1 else ("↓" if pct_change < -0.1 else "→")
                            s2.metric("True Freshness", f"{true_fresh:.3f}x",
                                      delta=f"{direction} {abs(pct_change):.1f}% vs {prev_state_date}",
                                      delta_color="normal" if pct_change >= 0 else "inverse")
                        else:
                            s2.metric("Freshness Ratio", f"{fresh_ratio:.2f}x" if fresh_ratio else "—")
                        s3.metric("Market Signal (w/FSC)", format_currency(nowcast_fsc))
                        s4.metric("Carrier+FSC (Model)", format_currency(result["carrier_fsc"]))

                        s5, s6, s7, s8 = st.columns(4)
                        s5.metric("Variance", f"{gate['variance_pct']:+.1%}" if gate["variance_pct"] else "—")
                        s6.metric("Zone", zone_badge(gate["zone"]))
                        s7.metric("Gate Quote", format_currency(gate["gate_quote"]))
                        s8.metric("Variance $", format_currency(gate["variance_dollar"]))

                        if gate["variance_pct"] is not None:
                            qk = abs(gate["variance_pct"]) * 0.25
                            risk_tol = auto_risk_tolerance(gate["variance_pct"], qk, gz)
                            decision = auto_decision(gate["variance_pct"], gz)
                            if risk_tol:
                                gate_with_risk = compute_gate(result["customer_fsc"], nowcast_fsc,
                                    result["carrier_fsc"], sgt, risk_tol, green_threshold=gz, red_threshold=rz)
                                final = gate_with_risk["gate_quote"] if decision == "DAT" else result["customer_fsc"]
                                floor = gate_with_risk["floor_margin_pct"]
                            else:
                                final = result["customer_fsc"]
                                floor = gate.get("floor_margin_pct")

                            f1, f2, f3, f4 = st.columns(4)
                            f1.metric("Qtr Kelly", f"{qk:.4f}")
                            f2.metric("Auto Risk Tol", f"{risk_tol:.2%}" if risk_tol else "—")
                            f3.metric("Decision", decision)
                            f4.metric("Final Quote", format_currency(final))
