import streamlit as st
import pandas as pd
import numpy as np
import os
import io

st.set_page_config(page_title="Flatbed Spot Pricing Tool", page_icon="🚛", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# PASSWORD PROTECTION
# ──────────────────────────────────────────────────────────────────────────────

def check_password():
    """Returns True if the user has entered the correct password."""
    def password_entered():
        if st.session_state.get("password") == st.secrets.get("APP_PASSWORD", st.secrets.get("app_password", "TA2026!pricing")):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("## Flatbed Spot Pricing Tool")
    st.markdown("**TA Services** — Authorized access only.")
    st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

from engine import (load_data, resolve_market, load_dashboard_signals, get_signal_staleness,
                     compute_directional_adjustment, lookup_lane)
from vision_reader import extract_rateview_data


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────

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
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# EV CALCULATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def calculate_ev(quote_price, best_fit, std_dev, carrier_adj=None):
    """Calculate Expected Value for a given quote price.

    Args:
        quote_price: what we'd charge the customer (all-in)
        best_fit: DAT Best Fit (our best estimate of carrier cost)
        std_dev: standard deviation of carrier costs
        carrier_adj: directionally adjusted carrier cost (if different from best_fit)

    Returns dict with ev_per_load, p_profit, expected_100_loads, signal
    """
    from scipy.stats import norm

    # Use adjusted carrier cost as the mean if provided
    mean_cost = carrier_adj if carrier_adj else best_fit

    if std_dev <= 0:
        # No volatility data - simple margin calc
        margin = quote_price - mean_cost
        return {
            "ev_per_load": margin,
            "p_profit": 1.0 if margin > 0 else 0.0,
            "expected_100": margin * 100,
            "signal": "+" if margin > 0 else "-"
        }

    # P(carrier cost <= quote_price) = P(we make money)
    p_profit = norm.cdf(quote_price, loc=mean_cost, scale=std_dev)

    # Expected profit when we win (carrier cost < quote)
    # E[profit | win] = quote - E[carrier_cost | carrier_cost < quote]
    # E[X | X < a] = mu - sigma * phi((a-mu)/sigma) / Phi((a-mu)/sigma)
    z = (quote_price - mean_cost) / std_dev
    phi_z = norm.pdf(z)  # standard normal PDF at z
    Phi_z = norm.cdf(z)  # standard normal CDF at z

    if Phi_z > 0.001:
        expected_cost_when_win = mean_cost - std_dev * (phi_z / Phi_z)
        expected_profit_when_win = quote_price - expected_cost_when_win
    else:
        expected_profit_when_win = 0

    # Expected loss when we lose (carrier cost > quote)
    p_loss = 1 - p_profit
    if p_loss > 0.001:
        expected_cost_when_lose = mean_cost + std_dev * (phi_z / (1 - Phi_z))
        expected_loss_when_lose = expected_cost_when_lose - quote_price
    else:
        expected_loss_when_lose = 0

    # EV = P(win) * E[profit|win] - P(lose) * E[loss|lose]
    ev = p_profit * expected_profit_when_win - p_loss * expected_loss_when_lose

    return {
        "ev_per_load": round(ev, 2),
        "p_profit": round(p_profit, 4),
        "expected_100": round(ev * 100, 2),
        "signal": "+" if ev > 0 else "-"
    }


def find_breakeven(best_fit, std_dev, carrier_adj=None):
    """Find the minimum quote price where EV turns positive."""
    mean_cost = carrier_adj if carrier_adj else best_fit

    # Binary search for breakeven
    low = mean_cost * 0.8
    high = mean_cost * 1.5

    for _ in range(50):
        mid = (low + high) / 2
        ev = calculate_ev(mid, best_fit, std_dev, carrier_adj)
        if ev["ev_per_load"] > 0:
            high = mid
        else:
            low = mid

    return round((low + high) / 2, 2)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def format_currency(val):
    if pd.isna(val) or val is None:
        return ""
    return f"${val:,.2f}"


def format_pct(val):
    if pd.isna(val) or val is None:
        return ""
    return f"{val:.1%}"


SIGNAL_COLORS = {
    "Acute Imbalance": "#f85149",
    "Tightening": "#e07b39",
    "Firming": "#d29922",
    "Balanced": "#8b949e",
    "Loosening": "#3fb950",
    "Soft": "#58a6ff",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_dashboard_signals():
    return load_dashboard_signals()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROL PANEL
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Control Panel")
data = get_data()
params = data["params"]
dashboard_signals = get_dashboard_signals()

# Market Regime
st.sidebar.markdown("### Market Regime")
col1, col2 = st.sidebar.columns(2)
regime = col1.selectbox("Regime", ["EXPANSION", "CONTRACTION"],
                        index=0 if params["regime"] == "EXPANSION" else 1)
phase = col2.selectbox("Phase", [0, 1, 2, 3, 4, 5],
                       index=[0, 1, 2, 3, 4, 5].index(params["phase"]))
ltr_dir = col1.selectbox("LTR Direction", ["RISING", "FALLING"],
                         index=0 if params["ltr_direction"] == "RISING" else 1)
nat_ltr = col2.number_input("National LTR", value=float(params["national_ltr"]), format="%.2f")

# Target Margin
st.sidebar.markdown("### Pricing")
target_margin = st.sidebar.slider("Target Margin %", 0, 25,
                                  int(float(params["target_margin"]) * 100), 1,
                                  format="%d%%") / 100

st.sidebar.markdown("---")
# Dashboard signals status
st.sidebar.markdown("---")
if dashboard_signals is not None:
    sig_staleness, sig_factor = get_signal_staleness(dashboard_signals)
    sig_date = dashboard_signals.get("as_of", "Unknown")
    n_markets = len(dashboard_signals.get("markets", {}))
    st.sidebar.caption(f"Dashboard signals: {sig_date} · {n_markets} markets")
    if sig_staleness and sig_staleness > 7:
        st.sidebar.warning(f"Signals are {sig_staleness} days old")
else:
    st.sidebar.caption("No dashboard signals loaded")

params_override = {
    "regime": regime, "phase": phase, "ltr_direction": ltr_dir,
    "national_ltr": nat_ltr, "target_margin": target_margin,
    "contract_term": 0,  # Always spot
}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA — SPOT QUOTE INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

st.title("Flatbed Spot Pricing Tool")
st.caption("TA Services | Directional Market Intelligence")

# Initialize session state for extracted data
if "vision_data" not in st.session_state:
    st.session_state["vision_data"] = None

# --- Screenshot Upload Section (top) ---
st.markdown("### DAT RateView Screenshot")
st.caption("Win+Shift+S → select DAT area → click 📋 Paste button below → auto-extracts everything")

from streamlit_paste_button import paste_image_button as pib

# Paste button - captures clipboard image
paste_result = pib("📋 Paste Screenshot (Ctrl+V)", key="paste_btn")

# Check if image was pasted
screenshot_image_bytes = None

if paste_result is not None and paste_result.image_data is not None:
    # Convert pasted image to bytes
    import io
    buf = io.BytesIO()
    paste_result.image_data.save(buf, format="PNG")
    screenshot_image_bytes = buf.getvalue()
    st.image(screenshot_image_bytes, caption="Pasted RateView Screenshot", use_container_width=True)

# Also keep file uploader as fallback
screenshot_file = st.file_uploader("Or drag/browse a saved screenshot",
                                   type=["png", "jpg", "jpeg"],
                                   key="rateview_screenshot",
                                   label_visibility="collapsed")

if screenshot_file is not None and screenshot_image_bytes is None:
    screenshot_image_bytes = screenshot_file.getvalue()
    st.image(screenshot_image_bytes, caption="Uploaded RateView Screenshot", use_container_width=True)

# Process screenshot if we have one
if screenshot_image_bytes is not None:
    if st.button("Extract Data from Screenshot", type="primary"):
        with st.spinner("Analyzing screenshot with Claude Vision..."):
            extracted, error = extract_rateview_data(screenshot_image_bytes)

        if error:
            st.error(f"Vision extraction failed: {error}")
            st.info("Use the manual input fields below instead.")
        else:
            st.session_state["vision_data"] = extracted
            # Force-update origin/dest text input fields with extracted values
            if extracted.get("origin_city"):
                st.session_state["orig_city_q"] = extracted["origin_city"].upper()
            if extracted.get("origin_state"):
                st.session_state["orig_state_q"] = extracted["origin_state"].upper()
            if extracted.get("dest_city"):
                st.session_state["dest_city_q"] = extracted["dest_city"].upper()
            if extracted.get("dest_state"):
                st.session_state["dest_state_q"] = extracted["dest_state"].upper()
            st.success("Data extracted successfully!")
            st.rerun()

    # Show extracted data for confirmation
    if st.session_state["vision_data"] is not None:
        vd = st.session_state["vision_data"]

        # Show auto-detected origin/destination
        v_orig_city = vd.get("origin_city") or ""
        v_orig_state = vd.get("origin_state") or ""
        v_dest_city = vd.get("dest_city") or ""
        v_dest_state = vd.get("dest_state") or ""
        if v_orig_city and v_dest_city:
            st.info(f"Auto-detected: **{v_orig_city}, {v_orig_state}** → "
                    f"**{v_dest_city}, {v_dest_state}**")

        st.markdown("**Extracted Values** (edit if needed):")
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            vd_best_fit = st.number_input("Best Fit (all-in)", value=float(vd.get("best_fit") or 0),
                                          step=50.0, key="v_best_fit")
        with vc2:
            vd_range_low = st.number_input("Range Low", value=float(vd.get("range_low") or 0),
                                           step=50.0, key="v_range_low")
        with vc3:
            vd_range_high = st.number_input("Range High", value=float(vd.get("range_high") or 0),
                                            step=50.0, key="v_range_high")
        with vc4:
            vd_rate_strength = st.number_input("Rate Strength", value=int(vd.get("rate_strength") or 0),
                                               min_value=0, max_value=100, key="v_strength")

        vc5, vc6, vc7 = st.columns(3)
        with vc5:
            vd_reports = st.number_input("Reports", value=int(vd.get("reports") or 0),
                                         min_value=0, key="v_reports")
        with vc6:
            vd_companies = st.number_input("Companies", value=int(vd.get("companies") or 0),
                                           min_value=0, key="v_companies")
        with vc7:
            vd_miles = st.number_input("Miles (from screenshot)", value=int(vd.get("miles") or 0),
                                       min_value=0, key="v_miles")

        # Show lane trend if extracted
        if vd.get("lane_trend"):
            with st.expander(f"Lane Trend ({len(vd['lane_trend'])} months)", expanded=False):
                trend_df = pd.DataFrame(vd["lane_trend"])
                st.dataframe(trend_df, hide_index=True, use_container_width=True)

# --- Origin / Destination Input ---
st.markdown("---")
st.markdown("### Lane Details")

# Determine if vision data provides origin/dest defaults
vd_for_lane = st.session_state.get("vision_data")
has_vision_origin = (vd_for_lane and vd_for_lane.get("origin_city") and vd_for_lane.get("origin_state"))
has_vision_dest = (vd_for_lane and vd_for_lane.get("dest_city") and vd_for_lane.get("dest_state"))

qq1, qq2 = st.columns(2)
with qq1:
    st.markdown("**Origin**")
    # Default to City + State if vision extracted origin
    orig_method_options = ["Zip3", "City + State", "Market Code"]
    orig_method_default = 1
    orig_method = st.radio("Input method", orig_method_options,
                           index=orig_method_default,
                           key="orig_method", horizontal=True)
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
    dest_method_options = ["Zip3", "City + State", "Market Code"]
    dest_method_default = 1
    dest_method = st.radio("Input method", dest_method_options,
                           index=dest_method_default,
                           key="dest_method", horizontal=True)
    if dest_method == "Zip3":
        dest_val = st.text_input("Dest Zip3", placeholder="606", key="dest_zip")
        dest_state_val = None
    elif dest_method == "City + State":
        dest_val = st.text_input("Dest City", placeholder="CHICAGO", key="dest_city_q")
        dest_state_val = st.text_input("Dest State", placeholder="IL", key="dest_state_q")
    else:
        dest_val = st.text_input("Dest Market", placeholder="IL_CHI", key="dest_mkt")
        dest_state_val = None

# --- Manual Rate Entry (only if no screenshot data) ---
st.markdown("---")
st.markdown("### DAT RateView Data")

has_vision_rates = (st.session_state.get("vision_data") is not None
                    and float(st.session_state.get("vision_data", {}).get("best_fit") or 0) > 0)

if has_vision_rates:
    st.caption("Using screenshot-extracted rates above. Clear the screenshot to enter manually.")
    manual_best_fit = 0.0
    manual_range_low = 0.0
    manual_range_high = 0.0
    manual_rate_strength = 0
else:
    st.markdown("**Manual Input** (enter from DAT RateView if not using screenshot)")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        manual_best_fit = st.number_input("Best Fit (all-in w/ fuel)", min_value=0.0,
                                          step=50.0, value=0.0, key="manual_best_fit",
                                          help="Total rate from DAT RateView including fuel")
    with mc2:
        manual_range_low = st.number_input("Range Low", min_value=0.0, step=50.0,
                                           value=0.0, key="manual_range_low")
    with mc3:
        manual_range_high = st.number_input("Range High", min_value=0.0, step=50.0,
                                            value=0.0, key="manual_range_high")
    with mc4:
        manual_rate_strength = st.number_input("Rate Strength", min_value=0, max_value=100,
                                               step=1, value=0, key="manual_strength")


# --- Get Quote / Reset Buttons ---
st.markdown("---")
btn_col1, btn_col2 = st.columns([3, 1])
quote_clicked = btn_col1.button("Get Quote", type="primary", use_container_width=True)
reset_clicked = btn_col2.button("🔄 New Quote", use_container_width=True)

if reset_clicked:
    for key in list(st.session_state.keys()):
        if key not in ["password_correct"]:
            del st.session_state[key]
    st.rerun()

if quote_clicked:

    # Resolve markets
    orig_market, orig_city, orig_st = resolve_market(orig_val, orig_state_val, data)
    dest_market, dest_city, dest_st = resolve_market(dest_val, dest_state_val, data)

    if orig_market is None or orig_market == "NO MATCH":
        st.error(f"Could not resolve origin: {orig_val}")
    elif dest_market is None or dest_market == "NO MATCH":
        st.error(f"Could not resolve destination: {dest_val}")
    else:
        # Determine which data source to use (vision extracted > manual)
        vd = st.session_state.get("vision_data")
        if vd and vd.get("best_fit") and float(vd.get("best_fit", 0)) > 0:
            # Use vision-extracted data (potentially edited by user)
            best_fit = float(st.session_state.get("v_best_fit", vd.get("best_fit", 0)))
            range_low = float(st.session_state.get("v_range_low", vd.get("range_low", 0)))
            range_high = float(st.session_state.get("v_range_high", vd.get("range_high", 0)))
            rate_strength = int(st.session_state.get("v_strength", vd.get("rate_strength", 0)))
            reports = int(st.session_state.get("v_reports", vd.get("reports", 0)))
            companies = int(st.session_state.get("v_companies", vd.get("companies", 0)))
            screenshot_miles = int(st.session_state.get("v_miles", vd.get("miles", 0)))
            lane_trend = vd.get("lane_trend")
        else:
            # Use manual input
            best_fit = manual_best_fit
            range_low = manual_range_low
            range_high = manual_range_high
            rate_strength = manual_rate_strength
            reports = 0
            companies = 0
            screenshot_miles = 0
            lane_trend = None

        if best_fit <= 0:
            st.error("Enter a Best Fit rate (from screenshot or manual input) to generate a quote.")
        else:
            # Get directional adjustment
            directional = compute_directional_adjustment(orig_market, dest_market,
                                                         dashboard_signals, term=0)
            dir_adj = directional["adjustment_pct"]
            orig_sig = directional.get("orig_signal")
            dest_sig = directional.get("dest_signal")

            # Get miles from model data as fallback
            model_lane = lookup_lane(orig_market, dest_market, data)
            if screenshot_miles > 0:
                miles = screenshot_miles
            elif model_lane:
                miles = model_lane["miles"]
            else:
                miles = 0

            if miles <= 0:
                st.error("Could not determine mileage. Enter miles in the screenshot data or "
                         "ensure the lane exists in the rate matrix.")
            else:
                # ──────────────────────────────────────────────────────────
                # 1. LANE HEADER
                # ──────────────────────────────────────────────────────────
                # Display city/state if available, market code as fallback
                orig_display = f"{orig_city}, {orig_st}" if orig_city and orig_st else orig_market
                dest_display = f"{dest_city}, {dest_st}" if dest_city and dest_st else dest_market
                st.success(f"**{orig_display} → {dest_display}** | {miles:,} miles")

                # ──────────────────────────────────────────────────────────
                # 2. MARKET INTELLIGENCE
                # ──────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Market Intelligence")

                if orig_sig and dest_sig:
                    orig_color = SIGNAL_COLORS.get(orig_sig, "#8b949e")
                    dest_color = SIGNAL_COLORS.get(dest_sig, "#8b949e")

                    mi1, mi2 = st.columns(2)
                    with mi1:
                        ltr_8d_orig = directional.get("orig_ltr_8d", "---")
                        st.markdown(
                            f"**Origin:** {orig_display} — "
                            f"<span style='color:{orig_color};font-weight:bold'>{orig_sig}</span> "
                            f"(LTR 8D: {ltr_8d_orig})",
                            unsafe_allow_html=True)
                    with mi2:
                        ltr_8d_dest = directional.get("dest_ltr_8d", "---")
                        st.markdown(
                            f"**Dest:** {dest_display} — "
                            f"<span style='color:{dest_color};font-weight:bold'>{dest_sig}</span> "
                            f"(LTR 8D: {ltr_8d_dest})",
                            unsafe_allow_html=True)

                    # Flow description
                    if dir_adj < -0.03:
                        flow_icon = "<span style='color:#3fb950;font-size:1.2em'>&#9679;</span>"
                        flow_label = "Carrier-friendly — push hard on rate"
                    elif dir_adj > 0.03:
                        flow_icon = "<span style='color:#f85149;font-size:1.2em'>&#9679;</span>"
                        flow_label = "Carrier-unfriendly — protect margin"
                    else:
                        flow_icon = "<span style='color:#8b949e;font-size:1.2em'>&#9679;</span>"
                        flow_label = "Balanced flow — quote at market"

                    st.markdown(f"{flow_icon} **Flow:** {orig_sig} -> {dest_sig} — {flow_label}",
                                unsafe_allow_html=True)

                    if directional.get("momentum_applied"):
                        st.caption("Momentum confirms direction — additional +/-2% applied")
                else:
                    st.caption("No directional signals available — using market-neutral pricing")

                # ──────────────────────────────────────────────────────────
                # 3. CARRIER TARGET
                # ──────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Carrier Target")

                carrier_base = best_fit
                carrier_adjusted = carrier_base * (1 + dir_adj)
                carrier_low = min(carrier_base, carrier_adjusted)
                carrier_high = max(carrier_base, carrier_adjusted)

                ct1, ct2, ct3 = st.columns(3)
                ct1.metric("Target Low", format_currency(carrier_low),
                           delta=f"{((carrier_low / carrier_base - 1) * 100):+.1f}% vs Best Fit"
                           if abs(carrier_low - carrier_base) > 1 else "At market")
                ct2.metric("DAT Best Fit", format_currency(carrier_base))
                ct3.metric("Target High", format_currency(carrier_high),
                           delta=f"{((carrier_high / carrier_base - 1) * 100):+.1f}% vs Best Fit"
                           if abs(carrier_high - carrier_base) > 1 else "At market")

                if dir_adj < -0.03:
                    st.caption(f"Directional signal suggests carrier will discount "
                               f"{abs(dir_adj):.1%} below Best Fit. Push toward Target Low.")
                elif dir_adj > 0.03:
                    st.caption(f"Directional signal suggests carrier needs "
                               f"{abs(dir_adj):.1%} premium above Best Fit. Budget toward Target High.")
                else:
                    st.caption("Balanced market — carrier likely near DAT Best Fit.")

                # ──────────────────────────────────────────────────────────
                # 4. CUSTOMER QUOTE RANGE (Floor and Ceiling)
                # ──────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Customer Quote Range")

                aggressive_margin = max(target_margin - 0.04, 0.05)
                defensive_margin = target_margin + 0.04

                # Ceiling row — based on Target High (top)
                st.markdown("**Ceiling** (cost basis: Target High)")
                c1, c2, c3 = st.columns(3)

                ceil_agg = round(carrier_high * (1 + aggressive_margin), 2)
                ceil_tgt = round(carrier_high * (1 + target_margin), 2)
                ceil_def = round(carrier_high * (1 + defensive_margin), 2)

                ceil_agg_margin_d = ceil_agg - carrier_high
                ceil_agg_margin_p = ceil_agg_margin_d / ceil_agg * 100 if ceil_agg else 0
                ceil_tgt_margin_d = ceil_tgt - carrier_high
                ceil_tgt_margin_p = ceil_tgt_margin_d / ceil_tgt * 100 if ceil_tgt else 0
                ceil_def_margin_d = ceil_def - carrier_high
                ceil_def_margin_p = ceil_def_margin_d / ceil_def * 100 if ceil_def else 0

                c1.metric("Aggressive", format_currency(ceil_agg),
                          delta=f"${ceil_agg_margin_d:,.0f} | {ceil_agg_margin_p:.1f}%")
                c2.metric("Target", format_currency(ceil_tgt),
                          delta=f"${ceil_tgt_margin_d:,.0f} | {ceil_tgt_margin_p:.1f}%")
                c3.metric("Defensive", format_currency(ceil_def),
                          delta=f"${ceil_def_margin_d:,.0f} | {ceil_def_margin_p:.1f}%")

                # Floor row — based on Target Low (bottom)
                st.markdown("**Floor** (cost basis: Target Low)")
                f1, f2, f3 = st.columns(3)

                floor_agg = round(carrier_low * (1 + aggressive_margin), 2)
                floor_tgt = round(carrier_low * (1 + target_margin), 2)
                floor_def = round(carrier_low * (1 + defensive_margin), 2)

                floor_agg_margin_d = floor_agg - carrier_low
                floor_agg_margin_p = floor_agg_margin_d / floor_agg * 100 if floor_agg else 0
                floor_tgt_margin_d = floor_tgt - carrier_low
                floor_tgt_margin_p = floor_tgt_margin_d / floor_tgt * 100 if floor_tgt else 0
                floor_def_margin_d = floor_def - carrier_low
                floor_def_margin_p = floor_def_margin_d / floor_def * 100 if floor_def else 0

                f1.metric("Aggressive", format_currency(floor_agg),
                          delta=f"${floor_agg_margin_d:,.0f} | {floor_agg_margin_p:.1f}%")
                f2.metric("Target", format_currency(floor_tgt),
                          delta=f"${floor_tgt_margin_d:,.0f} | {floor_tgt_margin_p:.1f}%")
                f3.metric("Defensive", format_currency(floor_def),
                          delta=f"${floor_def_margin_d:,.0f} | {floor_def_margin_p:.1f}%")

                # ──────────────────────────────────────────────────────────
                # 5. EXPECTED VALUE ANALYSIS
                # ──────────────────────────────────────────────────────────
                has_range = range_low > 0 and range_high > 0 and range_high > range_low

                if has_range:
                    st.markdown("---")
                    st.markdown("### Expected Value Analysis")

                    # Calculate StdDev from DAT range
                    dat_std_dev = (range_high - range_low) / 4.0

                    # If 13-month history available, use historical StdDev
                    hist_std_dev = None
                    if lane_trend and len(lane_trend) >= 2:
                        monthly_stds = []
                        for month in lane_trend:
                            m_low = month.get("low")
                            m_high = month.get("high")
                            if m_low and m_high and m_high > m_low:
                                monthly_stds.append((m_high - m_low) / 4.0)
                        if monthly_stds:
                            hist_std_dev = sum(monthly_stds) / len(monthly_stds)

                    std_dev_used = hist_std_dev if hist_std_dev else dat_std_dev
                    std_source = (f"Historical ({len(lane_trend)} months)"
                                  if hist_std_dev else "DAT Range")

                    st.caption(f"StdDev: ${std_dev_used:,.0f} (source: {std_source}) | "
                               f"DAT Range: ${range_low:,.0f} - ${range_high:,.0f}")

                    # Find breakeven
                    carrier_mean = carrier_adjusted if abs(dir_adj) > 0.001 else best_fit
                    breakeven = find_breakeven(best_fit, std_dev_used, carrier_mean)

                    st.markdown(f"**Breakeven (EV=0): {format_currency(breakeven)}**")

                    # Confidence band label
                    cv = std_dev_used / best_fit if best_fit > 0 else 0

                    if cv < 0.05:
                        vol_label = "LOW VOLATILITY"
                        vol_color = "\U0001f7e2"
                        vol_advice = "Carrier cost is highly predictable — quote with confidence."
                    elif cv < 0.10:
                        vol_label = "MODERATE VOLATILITY"
                        vol_color = "\U0001f7e1"
                        vol_advice = "Some cost uncertainty — target quote recommended."
                    else:
                        vol_label = "HIGH VOLATILITY"
                        vol_color = "\U0001f534"
                        vol_advice = "Carrier cost is unpredictable — budget defensively."

                    # Data quality label based on rate_strength
                    if rate_strength >= 70:
                        quality_label = "GOOD"
                    elif rate_strength >= 40:
                        quality_label = "MODERATE"
                    else:
                        quality_label = "THIN"

                    st.markdown(f"""
**{vol_color} {vol_label}** (StdDev: ${std_dev_used:,.0f} | {cv*100:.1f}% of Best Fit)
**DAT data quality: {quality_label}** ({rate_strength} strength, {reports} reports, {companies} companies)
{vol_advice}
""")

                    # Generate wide-range price points
                    range_start = round((best_fit - 2 * std_dev_used) / 50) * 50
                    range_end = round((best_fit + 4 * std_dev_used) / 50) * 50

                    price_points = set()
                    # Add $50 increments across the range
                    p = range_start
                    while p <= range_end:
                        price_points.add(round(p, 2))
                        p += 50
                    # Always include breakeven and named price points
                    price_points.add(breakeven)
                    price_points.add(floor_agg)
                    price_points.add(floor_tgt)
                    price_points.add(floor_def)
                    price_points.add(ceil_agg)
                    price_points.add(ceil_tgt)
                    price_points.add(ceil_def)
                    price_points = sorted(price_points)

                    # Named price point labels for annotation
                    named_points = {
                        breakeven: "Breakeven",
                        floor_agg: "Floor Aggr",
                        floor_tgt: "Floor Tgt",
                        floor_def: "Floor Def",
                        ceil_agg: "Ceil Aggr",
                        ceil_tgt: "Ceil Tgt",
                        ceil_def: "Ceil Def",
                    }

                    # Find which price point is closest to breakeven
                    closest_be_price = min(price_points, key=lambda p: abs(p - breakeven))

                    # Build EV table
                    ev_rows = []
                    for price in price_points:
                        ev_result = calculate_ev(price, best_fit, std_dev_used, carrier_mean)
                        ev_val = ev_result["ev_per_load"]

                        # Signal: red for negative, white for near-breakeven, green for positive
                        if abs(ev_val) <= 5:
                            signal = "\u26aa"
                        elif ev_val < 0:
                            signal = "\U0001f534"
                        else:
                            signal = "\U0001f7e2"

                        label = named_points.get(price, "")
                        is_breakeven_row = (price == closest_be_price)

                        quote_str = format_currency(price)
                        if label:
                            quote_str = f"{quote_str} ({label})"
                        if is_breakeven_row:
                            quote_str = f"**{quote_str}**"

                        ev_rows.append({
                            "Quote": quote_str,
                            "EV/Load": format_currency(ev_val),
                            "100-Load": format_currency(ev_result["expected_100"]),
                            "P(Profit)": f"{ev_result['p_profit']:.0%}",
                            "Signal": signal,
                            "_ev": ev_val,
                            "_price": price,
                        })

                    ev_df = pd.DataFrame(ev_rows)
                    display_ev = ev_df[["Quote", "EV/Load", "100-Load", "P(Profit)", "Signal"]]
                    st.dataframe(display_ev, hide_index=True, use_container_width=True, height=600)

                    # Highlight key insight
                    best_ev_row = max(ev_rows, key=lambda r: r["_ev"])
                    if best_ev_row["_ev"] > 0:
                        st.caption(
                            f"Best EV: {best_ev_row['Quote']} at "
                            f"{best_ev_row['EV/Load']}/load "
                            f"({best_ev_row['100-Load']} over 100 loads)")
