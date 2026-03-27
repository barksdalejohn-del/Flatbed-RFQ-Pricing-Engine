import streamlit as st
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV RENAME
# ──────────────────────────────────────────────────────────────────────────────
import streamlit.components.v1 as components

st.markdown("""
<style>
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
    var doc = window.parent.document;
    var links = doc.querySelectorAll('section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span');
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
# PASSWORD PROTECTION (shared with main app)
# ──────────────────────────────────────────────────────────────────────────────

def check_password():
    import hashlib
    correct_pw = st.secrets.get("APP_PASSWORD", st.secrets.get("app_password", "TA2026!pricing"))
    auth_token = hashlib.sha256(correct_pw.encode()).hexdigest()[:16]

    if st.session_state.get("password_correct", False):
        return True
    if st.query_params.get("auth") == auth_token:
        st.session_state["password_correct"] = True
        return True

    st.markdown("## Quote History")
    st.markdown("**TA Services** — Authorized access only.")
    st.text_input("Enter password:", type="password",
                  on_change=lambda: _pw_check(correct_pw, auth_token), key="password_hist")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return False

def _pw_check(correct_pw, auth_token):
    if st.session_state.get("password_hist") == correct_pw:
        st.session_state["password_correct"] = True
        st.query_params["auth"] = auth_token
    else:
        st.session_state["password_correct"] = False

if not check_password():
    st.stop()

from quote_log import load_quote_log, update_outcome, delete_quote

# ──────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("## 📋 Quote History")
st.caption("TA Services | Track quotes, update outcomes, measure performance")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

df_log, log_sha = load_quote_log()

if df_log.empty:
    st.info("No quotes logged yet. Go to the Spot Quote page to save your first quote.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# MONTHLY FILTER
# ──────────────────────────────────────────────────────────────────────────────

df_log["_month"] = pd.to_datetime(df_log["date"], errors="coerce").dt.to_period("M")
available_months = sorted(df_log["_month"].dropna().unique(), reverse=True)

if available_months:
    month_labels = [str(m) for m in available_months]
    selected_month = st.selectbox("Month", month_labels, index=0, key="hist_month_filter")
    df_filtered = df_log[df_log["_month"].astype(str) == selected_month].copy()
else:
    df_filtered = df_log.copy()

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS
# ──────────────────────────────────────────────────────────────────────────────

total_quotes = len(df_filtered)
won = len(df_filtered[df_filtered["outcome"] == "Won"])
lost = len(df_filtered[df_filtered["outcome"] == "Lost"])
pending = total_quotes - won - lost
decided = won + lost
win_rate = f"{(won / decided * 100):.1f}%" if decided > 0 else "—"

stat1, stat2, stat3, stat4, stat5 = st.columns(5)
stat1.metric("Total Quotes", total_quotes)
stat2.metric("Won", won)
stat3.metric("Lost", lost)
stat4.metric("Pending", pending)
stat5.metric("Win Rate", win_rate)

# ──────────────────────────────────────────────────────────────────────────────
# QUOTE LOG TABLE
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")

display_cols = ["date", "time_central", "analyst", "origin", "destination",
               "dat_best_fit", "quoted_amount", "strategy", "p_profit_at_strategy",
               "outcome", "actual_carrier_cost", "actual_margin_dollars", "actual_margin_pct"]
available_display_cols = [c for c in display_cols if c in df_filtered.columns]
display_df = df_filtered[available_display_cols].iloc[::-1].reset_index(drop=False)
display_df = display_df.rename(columns={"index": "row_id"})
show_df = display_df.drop(columns=["row_id"])
show_df.index = range(1, len(show_df) + 1)
st.dataframe(show_df, use_container_width=True, height=400)

# ──────────────────────────────────────────────────────────────────────────────
# UPDATE OUTCOME
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Update Outcome")

pending_df = df_filtered[df_filtered["outcome"].isin(["", "None"]) | df_filtered["outcome"].isna()].copy()
if pending_df.empty:
    st.info("All quotes have outcomes recorded.")
else:
    pending_df["label"] = pending_df.apply(
        lambda r: f"{r['date']} | {r['origin']} → {r['destination']} | ${r['quoted_amount']}", axis=1)

    update_col1, update_col2, update_col3, update_col4 = st.columns([2, 1, 1, 2])

    quote_labels = ["— Select —"] + pending_df["label"].tolist()
    selected_quote = update_col1.selectbox("Select quote", quote_labels,
                                            index=0,
                                            key="hist_update_quote_select")
    if selected_quote and selected_quote != "— Select —":
        actual_idx = pending_df[pending_df["label"] == selected_quote].index[0]

        outcome_options = ["— Select —", "Won", "Lost", "No Bid"]
        outcome_val = update_col2.selectbox("Outcome", outcome_options, index=0, key="hist_outcome_select")
        actual_cost_str = update_col3.text_input("Actual carrier cost ($)", value="", key="hist_actual_cost")
        update_notes = update_col4.text_input("Notes (optional)", key="hist_update_notes")

        if st.button("Update Outcome", type="primary"):
            if outcome_val == "— Select —":
                st.warning("Select an outcome (Won, Lost, or No Bid)")
            else:
                try:
                    actual_cost = float(actual_cost_str) if actual_cost_str.strip() else ""
                except ValueError:
                    actual_cost = ""
                success, msg = update_outcome(actual_idx, outcome_val,
                                               actual_cost if actual_cost and actual_cost > 0 else "",
                                               update_notes)
                if success:
                    st.success("✅ Outcome updated")
                    st.rerun()
                else:
                    st.error(f"Failed: {msg}")

# ──────────────────────────────────────────────────────────────────────────────
# DELETE QUOTE
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Delete Quote")

all_labels = df_filtered.apply(
    lambda r: f"{r['date']} | {r['analyst']} | {r['origin']} → {r['destination']} | ${r['quoted_amount']}", axis=1)
delete_options = list(zip(df_filtered.index.tolist(), all_labels.tolist()))

if delete_options:
    del_col1, del_col2 = st.columns([3, 1])
    delete_labels = ["— Select —"] + [label for _, label in delete_options]
    selected_delete = del_col1.selectbox("Select quote to delete",
                                          delete_labels,
                                          index=0,
                                          key="hist_delete_select")
    if selected_delete and selected_delete != "— Select —":
        delete_idx = [idx for idx, label in delete_options if label == selected_delete][0]
        del_col2.markdown("<br>", unsafe_allow_html=True)
        if del_col2.button("🗑️ Delete", type="secondary", use_container_width=True):
            success, msg = delete_quote(delete_idx)
            if success:
                st.success("Quote deleted")
                st.rerun()
            else:
                st.error(f"Failed: {msg}")

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
csv_download = df_log.to_csv(index=False)
st.download_button("📥 Download Full Log (CSV)", csv_download,
                  "quote_log.csv", "text/csv", use_container_width=True)
