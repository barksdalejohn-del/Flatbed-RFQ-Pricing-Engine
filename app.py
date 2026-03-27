import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

st.set_page_config(page_title="Flatbed Spot Pricing Tool", page_icon="🚛", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV RENAME — must run before password check so it applies on login page
# ──────────────────────────────────────────────────────────────────────────────
import streamlit.components.v1 as components

st.markdown("""
<style>
    /* Navigation header above page links in sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "Navigation";
        display: block;
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        padding: 0.5rem 1rem 0.25rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
function renameNav() {
    const doc = window.parent.document;
    const links = doc.querySelectorAll('section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span');
    links.forEach(function(span) {
        if (span.textContent.trim().toLowerCase() === 'app') {
            span.textContent = 'Spot Quote';
        }
    });
}
renameNav();
var obs = new MutationObserver(renameNav);
obs.observe(window.parent.document.body, {childList: true, subtree: true});
</script>
""", height=0)

# ──────────────────────────────────────────────────────────────────────────────
# PASSWORD PROTECTION
# ──────────────────────────────────────────────────────────────────────────────

def check_password():
    """Returns True if the user has entered the correct password.
    Uses both session_state and query_params for persistence across reconnections."""
    import hashlib
    correct_pw = st.secrets.get("APP_PASSWORD", st.secrets.get("app_password", "TA2026!pricing"))
    auth_token = hashlib.sha256(correct_pw.encode()).hexdigest()[:16]

    def password_entered():
        if st.session_state.get("password") == correct_pw:
            st.session_state["password_correct"] = True
            st.query_params["auth"] = auth_token
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Check session state first
    if st.session_state.get("password_correct", False):
        return True

    # Check query params as backup (survives session drops)
    if st.query_params.get("auth") == auth_token:
        st.session_state["password_correct"] = True
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
                     compute_directional_adjustment, compute_same_day_multiplier, lookup_lane)
from vision_reader import extract_rateview_data
from quote_log import ANALYSTS, save_quote, load_quote_log, update_outcome, delete_quote, detect_strategy


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
    """Calculate Expected Value using log-normal distribution.

    Trucking rates are right-skewed (hard floor, long right tail from spikes).
    Log-normal captures this better than normal distribution.

    Args:
        quote_price: what we'd charge the customer (all-in)
        best_fit: DAT Best Fit (our best estimate of carrier cost)
        std_dev: standard deviation of carrier costs (in dollar terms)
        carrier_adj: directionally adjusted carrier cost (if different from best_fit)

    Returns dict with ev_per_load, p_profit, expected_100_loads, signal
    """
    import numpy as np
    from scipy.stats import lognorm

    mean_cost = carrier_adj if carrier_adj else best_fit

    if std_dev <= 0 or mean_cost <= 0:
        margin = quote_price - mean_cost
        return {
            "ev_per_load": margin,
            "p_profit": 1.0 if margin > 0 else 0.0,
            "expected_100": margin * 100,
            "signal": "+" if margin > 0 else "-"
        }

    # Convert dollar-space mean/std to log-normal parameters
    # If X ~ LogNormal(mu, sigma), then E[X] = exp(mu + sigma^2/2)
    # and Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    cv = std_dev / mean_cost  # coefficient of variation
    sigma_sq = np.log(1 + cv**2)
    sigma = np.sqrt(sigma_sq)
    mu = np.log(mean_cost) - sigma_sq / 2

    # P(carrier cost <= quote_price) = P(we make money)
    if quote_price <= 0:
        p_profit = 0.0
    else:
        p_profit = lognorm.cdf(quote_price, s=sigma, scale=np.exp(mu))

    # E[X | X < a] for log-normal (truncated mean)
    # E[X | X < a] = E[X] * Phi((log(a) - mu - sigma^2) / sigma) / Phi((log(a) - mu) / sigma)
    from scipy.stats import norm
    if p_profit > 0.001 and quote_price > 0:
        log_a = np.log(quote_price)
        z1 = (log_a - mu - sigma_sq) / sigma
        z2 = (log_a - mu) / sigma
        expected_cost_when_win = mean_cost * norm.cdf(z1) / norm.cdf(z2)
        expected_profit_when_win = quote_price - expected_cost_when_win
    else:
        expected_profit_when_win = 0

    # E[X | X > a] for log-normal
    p_loss = 1 - p_profit
    if p_loss > 0.001 and quote_price > 0:
        log_a = np.log(quote_price)
        z1 = (log_a - mu - sigma_sq) / sigma
        z2 = (log_a - mu) / sigma
        expected_cost_when_lose = mean_cost * (1 - norm.cdf(z1)) / (1 - norm.cdf(z2))
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

# Target Margin
st.sidebar.markdown("### Pricing")
target_margin = st.sidebar.slider("Target Margin %", 0, 25,
                                  int(float(params["target_margin"]) * 100), 1,
                                  format="%d%%") / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Alerts & Modes")
deadhead_alert_on = st.sidebar.toggle("Soft Destination Alert (LTR < 20)", value=True,
                                       help="Warn when destination market LTR is below 20. Carriers may price in deadhead. Turn off in bear markets when low LTR is widespread.")

# Same-Day Mode
same_day_on = st.sidebar.toggle("⚡ Same-Day Quoting", value=False,
                                 help="Apply urgency multiplier for same-day pickup. Adjusts carrier target based on time remaining, market tightness, and day of week.")
same_day_hours = None
same_day_time_label = None
same_day_dow = None
if same_day_on:
    import pytz
    central = pytz.timezone("US/Central")
    now_central = datetime.now(central)
    today_dow = now_central.weekday()  # 0=Mon through 6=Sun
    current_hour_central = now_central.hour

    # Day of week selector
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_day = st.sidebar.selectbox("Day", day_names, index=today_dow,
                                         help="Defaults to today (Central time). Change to backtest past scenarios.")
    same_day_dow = day_names.index(selected_day)

    # Hours remaining selector — 1 to 8
    hours_options = [8, 7, 6, 5, 4, 3, 2, 1]
    # Default: estimate hours left based on Central time (6pm cutoff)
    est_hours_left = max(1, min(8, 18 - current_hour_central))
    default_index = hours_options.index(est_hours_left) if est_hours_left in hours_options else 4

    same_day_hours = st.sidebar.selectbox("Hours remaining", hours_options,
                                           index=default_index,
                                           help="Estimated hours left to find a truck. Defaults based on current time (Central).")
    same_day_time_label = f"{same_day_hours}h remaining"

    st.sidebar.caption(f"📅 {selected_day} · {same_day_hours} hours remaining")

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
    "regime": params.get("regime", "EXPANSION"),
    "phase": params.get("phase", 4),
    "ltr_direction": params.get("ltr_direction", "RISING"),
    "national_ltr": params.get("national_ltr", 57.11),
    "target_margin": target_margin,
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
    # If same-day is on, warn before resetting
    if same_day_on:
        if "same_day_confirm_reset" not in st.session_state:
            st.session_state["same_day_confirm_reset"] = True
            st.rerun()
        else:
            # Already showing confirmation — wait for button click below
            pass
    else:
        for key in list(st.session_state.keys()):
            if key not in ["password_correct"]:
                del st.session_state[key]
        st.rerun()

# Same-day reset confirmation dialog
if st.session_state.get("same_day_confirm_reset"):
    st.warning("⚡ **Same-Day Mode is still ON.** Are you quoting another same-day load?")
    confirm_col1, confirm_col2 = st.columns(2)
    if confirm_col1.button("Yes, keep Same-Day ON", use_container_width=True):
        del st.session_state["same_day_confirm_reset"]
        for key in list(st.session_state.keys()):
            if key not in ["password_correct", "same_day_confirm_reset"]:
                del st.session_state[key]
        st.rerun()
    if confirm_col2.button("No, turn it OFF", use_container_width=True):
        del st.session_state["same_day_confirm_reset"]
        for key in list(st.session_state.keys()):
            if key not in ["password_correct"]:
                del st.session_state[key]
        st.rerun()

if quote_clicked:
    # Store flag so results persist across reruns
    st.session_state["show_results"] = True
    st.session_state["result_inputs"] = {
        "orig_val": orig_val, "orig_state_val": orig_state_val,
        "dest_val": dest_val, "dest_state_val": dest_state_val,
    }

if st.session_state.get("show_results"):
    # Recover inputs from session state if this is a rerun (not a fresh click)
    if not quote_clicked:
        ri = st.session_state.get("result_inputs", {})
        orig_val = ri.get("orig_val", orig_val)
        orig_state_val = ri.get("orig_state_val", orig_state_val)
        dest_val = ri.get("dest_val", dest_val)
        dest_state_val = ri.get("dest_state_val", dest_state_val)

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
            # Use DAT market names from Lane Trend header for signal lookup (most accurate)
            # Falls back to user-entered city names if market names not available
            vd = st.session_state.get("vision_data") or {}
            signal_orig_city = vd.get("origin_market_name") or orig_city
            signal_orig_state = vd.get("origin_market_state") or orig_st
            signal_dest_city = vd.get("dest_market_name") or dest_city
            signal_dest_state = vd.get("dest_market_state") or dest_st

            # Get directional adjustment
            directional = compute_directional_adjustment(orig_market, dest_market,
                                                         dashboard_signals, term=0,
                                                         orig_city=signal_orig_city, orig_state=signal_orig_state,
                                                         dest_city=signal_dest_city, dest_state=signal_dest_state)
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

                    # Show market names (from Lane Trend header) for signal context
                    orig_mkt_display = f"{signal_orig_city.title()}, {signal_orig_state.upper()}" if signal_orig_city else orig_display
                    dest_mkt_display = f"{signal_dest_city.title()}, {signal_dest_state.upper()}" if signal_dest_city else dest_display

                    mi1, mi2 = st.columns(2)
                    with mi1:
                        ltr_8d_orig = directional.get("orig_ltr_8d", "---")
                        st.markdown(
                            f"**Origin:** {orig_mkt_display} — "
                            f"<span style='color:{orig_color};font-weight:bold'>{orig_sig}</span> "
                            f"(LTR 8D: {ltr_8d_orig})",
                            unsafe_allow_html=True)
                    with mi2:
                        ltr_8d_dest = directional.get("dest_ltr_8d", "---")
                        st.markdown(
                            f"**Dest:** {dest_mkt_display} — "
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

                    # Soft destination alert
                    if deadhead_alert_on:
                        dest_ltr = directional.get("dest_ltr_8d")
                        if dest_ltr is not None and isinstance(dest_ltr, (int, float)) and dest_ltr < 20:
                            st.warning(
                                f"**DESTINATION ALERT: {dest_mkt_display} — LTR {dest_ltr}**\n\n"
                                f"Extremely low demand at destination. Carrier may price in deadhead. "
                                f"Review market conditions before quoting."
                            )
                else:
                    st.caption("No directional signals available — using market-neutral pricing")

                # ──────────────────────────────────────────────────────────
                # 3. CARRIER TARGET
                # ──────────────────────────────────────────────────────────
                st.markdown("---")

                # Same-day multiplier
                same_day_result = None
                same_day_mult = 1.0
                if same_day_on and same_day_hours is not None:
                    orig_signal_label = directional.get("orig_signal") if directional else None
                    same_day_result = compute_same_day_multiplier(
                        same_day_hours,
                        origin_signal=orig_signal_label,
                        day_of_week=same_day_dow
                    )
                    same_day_mult = same_day_result["multiplier"]

                if same_day_on and same_day_result:
                    st.markdown("### ⚡ Carrier Target — Same-Day")
                    # Show multiplier breakdown
                    sd_cols = st.columns(4)
                    sd_cols[0].metric("Base Urgency", f"{same_day_result['base_urgency']:.0%}")
                    sd_cols[1].metric("Time Decay", f"{same_day_result['time_decay']:.2f}x",
                                      delta=f"{same_day_result['hours_remaining']:.1f}h left")
                    sd_cols[2].metric("Market Factor", f"{same_day_result['market_factor']:.2f}x",
                                      delta=directional.get("orig_signal", "Unknown") if directional else "Unknown")
                    day_names_display = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    sd_cols[3].metric("Day Factor", f"{same_day_result['day_factor']:.2f}x",
                                      delta=day_names_display[same_day_dow] if same_day_dow is not None else "Unknown")

                    st.caption(f"**Combined Same-Day Multiplier: {same_day_mult:.2f}x** "
                               f"({(same_day_mult - 1) * 100:+.0f}% above regular pricing)")
                    st.markdown("")
                else:
                    st.markdown("### Carrier Target")

                carrier_base = best_fit
                carrier_adjusted = carrier_base * (1 + dir_adj) * same_day_mult
                carrier_low = min(carrier_base * same_day_mult, carrier_adjusted)
                carrier_high = max(carrier_base * same_day_mult, carrier_adjusted)

                ct1, ct2, ct3 = st.columns(3)
                ct1.metric("Target Low", format_currency(carrier_low),
                           delta=f"{((carrier_low / best_fit - 1) * 100):+.1f}% vs Best Fit"
                           if abs(carrier_low - best_fit) > 1 else "At market")
                ct2.metric("DAT Best Fit", format_currency(best_fit))
                ct3.metric("Target High", format_currency(carrier_high),
                           delta=f"{((carrier_high / best_fit - 1) * 100):+.1f}% vs Best Fit"
                           if abs(carrier_high - best_fit) > 1 else "At market")

                if same_day_on:
                    st.caption(f"⚡ Same-day adjusted. Regular Target High would be "
                               f"{format_currency(best_fit * (1 + dir_adj))}.")
                elif dir_adj < -0.03:
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

                # Ceiling row (top)
                st.markdown("**Ceiling**")
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

                # Floor row (bottom)
                st.markdown("**Floor**")
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
                    st.markdown("### Expected Value Analysis *(Thinking In Bets)*")

                    # Calculate StdDev — blended approach
                    # 1. Current range StdDev (today's carrier spread)
                    dat_std_dev = (range_high - range_low) / 4.0

                    # 2. Historical StdDev from 13-month Mid values (market movement)
                    hist_mid_std = None
                    hist_range_std = None
                    if lane_trend and len(lane_trend) >= 3:
                        import numpy as np
                        # StdDev of Mid values — measures how much the market rate moves
                        mids = [m.get("mid") for m in lane_trend if m.get("mid")]
                        if len(mids) >= 3:
                            hist_mid_std = float(np.std(mids, ddof=1))

                        # Recent 3-month Mid StdDev (weighted toward current conditions)
                        recent_mids = mids[:3] if len(mids) >= 3 else mids
                        recent_mid_std = float(np.std(recent_mids, ddof=1)) if len(recent_mids) >= 2 else None

                    # Blended: 50% current range + 30% recent Mid + 20% full Mid
                    if hist_mid_std and recent_mid_std:
                        std_dev_used = (0.5 * dat_std_dev +
                                        0.3 * recent_mid_std +
                                        0.2 * hist_mid_std)
                        std_source = f"Blended ({len(lane_trend)} months)"
                    elif hist_mid_std:
                        std_dev_used = (0.6 * dat_std_dev + 0.4 * hist_mid_std)
                        std_source = f"Blended ({len(lane_trend)} months)"
                    else:
                        std_dev_used = dat_std_dev
                        std_source = "DAT Range"

                    stddev_html = (
                        '<p style="font-size:15px;">'
                        '<b>StdDev:</b> $' + f'{std_dev_used:,.0f}' + ' (source: ' + std_source + ')'
                        ' &nbsp;|&nbsp; '
                        '<b>DAT Range:</b> $' + f'{range_low:,.0f}' + ' - $' + f'{range_high:,.0f}'
                        '</p>'
                    )
                    st.markdown(stddev_html, unsafe_allow_html=True)

                    # Find breakeven — use same-day adjusted carrier if active
                    carrier_mean = carrier_high if (abs(dir_adj) > 0.001 or same_day_mult > 1.0) else best_fit
                    # Scale StdDev for same-day (wider uncertainty)
                    ev_std_dev = std_dev_used * same_day_mult if same_day_mult > 1.0 else std_dev_used
                    breakeven = find_breakeven(best_fit * same_day_mult, ev_std_dev, carrier_mean)

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

                    # Generate focused price points: 2 below breakeven, breakeven, then key quote points
                    price_points = set()

                    # 2 rows below breakeven (danger zone context)
                    step_below = round(std_dev_used * 0.75 / 50) * 50
                    if step_below < 50:
                        step_below = 50
                    price_points.add(round((breakeven - 2 * step_below) / 50) * 50)
                    price_points.add(round((breakeven - step_below) / 50) * 50)

                    # Breakeven
                    price_points.add(breakeven)

                    # Named quote points (floor/ceiling)
                    price_points.add(floor_agg)
                    price_points.add(floor_tgt)
                    price_points.add(floor_def)
                    price_points.add(ceil_agg)
                    price_points.add(ceil_tgt)
                    price_points.add(ceil_def)

                    # Remove any that are below the lowest danger row
                    min_show = breakeven - 2.5 * step_below
                    price_points = sorted(p for p in price_points if p >= min_show)

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
                        ev_result = calculate_ev(price, best_fit * same_day_mult, ev_std_dev, carrier_mean)
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

                    # Build color-coded HTML table
                    aggressive_prices = {floor_agg, ceil_agg}
                    target_prices = {floor_tgt, ceil_tgt}
                    defensive_prices = {floor_def, ceil_def}

                    html_rows = ""
                    for row in ev_rows:
                        price = row["_price"]
                        ev_val = row["_ev"]

                        # Color coding by strategy
                        if price in aggressive_prices:
                            row_color = "#f85149"  # red for aggressive
                        elif price in target_prices:
                            row_color = "#e3b341"  # yellow for target
                        elif price in defensive_prices:
                            row_color = "#3fb950"  # green for defensive
                        elif abs(price - breakeven) < 1:
                            row_color = "#ffffff"  # white bold for breakeven
                        elif ev_val < 0:
                            row_color = "#f8514980"  # faded red for danger zone
                        else:
                            row_color = "#8b949e"  # grey for unmarked

                        font_weight = "bold" if abs(price - breakeven) < 1 else "normal"

                        html_rows += f"""<tr style="color:{row_color};font-weight:{font_weight}">
                            <td style="padding:6px 8px;">{row['Quote']}</td>
                            <td style="padding:6px 8px;">{row['EV/Load']}</td>
                            <td style="padding:6px 8px;">{row['100-Load']}</td>
                            <td style="padding:6px 8px;">{row['P(Profit)']}</td>
                            <td style="padding:6px 8px;">{row['Signal']}</td>
                        </tr>"""

                    ev_html = f"""
                    <table style="width:100%;border-collapse:collapse;font-size:14px;">
                        <thead>
                            <tr style="color:#8b949e;border-bottom:1px solid #30363d;">
                                <th style="text-align:left;padding:8px;">Quote</th>
                                <th style="text-align:left;padding:8px;">EV/Load</th>
                                <th style="text-align:left;padding:8px;">100-Load</th>
                                <th style="text-align:left;padding:8px;">P(Profit)</th>
                                <th style="text-align:left;padding:8px;">Signal</th>
                            </tr>
                        </thead>
                        <tbody>
                            {html_rows}
                        </tbody>
                    </table>
                    <div style="margin-top:8px;font-size:12px;color:#8b949e;">
                        <span style="color:#f8514980;">■</span> Danger zone &nbsp;
                        <span style="color:#ffffff;">■</span> Breakeven &nbsp;
                        <span style="color:#f85149;">■</span> Aggressive &nbsp;
                        <span style="color:#e3b341;">■</span> Target &nbsp;
                        <span style="color:#3fb950;">■</span> Defensive
                    </div>
                    """
                    st.markdown(ev_html, unsafe_allow_html=True)

                    # Highlight key insight
                    best_ev_row = max(ev_rows, key=lambda r: r["_ev"])
                    if best_ev_row["_ev"] > 0:
                        st.caption(
                            f"Best EV: {best_ev_row['Quote']} at "
                            f"{best_ev_row['EV/Load']}/load "
                            f"({best_ev_row['100-Load']} over 100 loads)")

                # ──────────────────────────────────────────────────────────
                # SAVE QUOTE
                # ──────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Save Quote")

                save_col1, save_col2, save_col3 = st.columns([2, 2, 1])
                analyst_name = save_col1.selectbox("Analyst", ANALYSTS, key="analyst_select")
                if analyst_name == "Other":
                    analyst_name = save_col1.text_input("Enter name", key="analyst_other")

                quoted_amount = save_col2.number_input("Amount quoted to customer ($)",
                                                        min_value=0.0, step=50.0, key="quoted_amount",
                                                        help="Enter amount and press Enter, or click Save Quote")

                save_col3.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                save_button = save_col3.button("💾 Save Quote", use_container_width=True, type="primary")

                # Only Save Quote button triggers save — no Enter/Tab shortcuts
                should_save = False
                if save_button and quoted_amount > 0 and analyst_name:
                    # Check if we already saved this exact quote (prevent double-click)
                    save_key = f"{analyst_name}_{quoted_amount}_{best_fit}"
                    if st.session_state.get("last_saved_key") != save_key:
                        should_save = True
                        st.session_state["last_saved_key"] = save_key

                if should_save:
                    if quoted_amount > 0 and analyst_name:
                        # Build quote data dict from all available session data
                        quote_data = {
                            "analyst": analyst_name,
                            "origin": orig_display,
                            "destination": dest_display,
                            "miles": miles,
                            "dat_best_fit": best_fit,
                            "dat_range_low": range_low,
                            "dat_range_high": range_high,
                            "rate_strength": rate_strength,
                            "origin_signal": orig_sig if directional else "",
                            "dest_signal": dest_sig if directional else "",
                            "flow": f"{orig_sig} \u2192 {dest_sig}" if directional and orig_sig and dest_sig else "",
                            "dir_adj_pct": round(dir_adj * 100, 1) if directional else "",
                            "same_day": "Yes" if same_day_on else "No",
                            "same_day_multiplier": round(same_day_mult, 3) if same_day_on and same_day_mult > 1 else "",
                            "carrier_target_low": carrier_low,
                            "carrier_target_high": carrier_high,
                            "floor_aggressive": floor_agg,
                            "floor_target": floor_tgt,
                            "floor_defensive": floor_def,
                            "ceiling_aggressive": ceil_agg,
                            "ceiling_target": ceil_tgt,
                            "ceiling_defensive": ceil_def,
                            "breakeven": breakeven if 'breakeven' in dir() else "",
                            "stddev": std_dev_used if 'std_dev_used' in dir() else "",
                            "volatility_band": vol_label if 'vol_label' in dir() else "",
                            "quoted_amount": quoted_amount,
                            "strategy": "",
                            "p_profit_at_strategy": "",
                            "ev_at_quote": "",
                            "p_profit_at_quote": "",
                        }

                        # Detect strategy and compute P(Profit) for each strategy
                        strat_dict = {
                            "Floor Aggr": floor_agg, "Floor Tgt": floor_tgt,
                            "Floor Def": floor_def, "Ceil Aggr": ceil_agg,
                            "Ceil Tgt": ceil_tgt, "Ceil Def": ceil_def
                        }
                        p_profit_dict = None
                        if 'ev_std_dev' in dir() and ev_std_dev > 0 and 'carrier_mean' in dir():
                            p_profit_dict = {}
                            for sname, samt in strat_dict.items():
                                if samt and samt > 0:
                                    ev_result = calculate_ev(samt, best_fit * same_day_mult, ev_std_dev, carrier_mean)
                                    p_profit_dict[sname] = ev_result["p_profit"]

                        strat_str, p_profit_str = detect_strategy(quoted_amount, strat_dict, p_profit_dict)
                        quote_data["strategy"] = strat_str
                        quote_data["p_profit_at_strategy"] = p_profit_str

                        # Compute EV at quoted amount if we have the data
                        if 'ev_std_dev' in dir() and 'carrier_mean' in dir():
                            ev_at_q = calculate_ev(quoted_amount, best_fit * same_day_mult, ev_std_dev, carrier_mean)
                            quote_data["ev_at_quote"] = ev_at_q["ev_per_load"]
                            quote_data["p_profit_at_quote"] = f"{ev_at_q['p_profit']:.0%}"

                        success, message = save_quote(quote_data)
                        if success:
                            st.success("\u2705 Quote saved to log")
                        else:
                            st.error(f"Failed to save: {message}")
                    else:
                        st.warning("Enter the quoted amount and select an analyst")


# Quote History is on a separate page — use the sidebar navigation to access it.
