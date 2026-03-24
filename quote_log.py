import streamlit as st
import pandas as pd
import requests
import base64
from datetime import datetime
import pytz
import io

REPO = "barksdalejohn-del/Flatbed-RFQ-Pricing-Engine"
FILE_PATH = "data/quote_log.csv"

ANALYSTS = ["John Barksdale", "Jeff Elliott", "Other"]

QUOTE_COLUMNS = [
    "date", "time_central", "analyst", "origin", "destination", "miles",
    "dat_best_fit", "dat_range_low", "dat_range_high", "rate_strength",
    "origin_signal", "dest_signal", "flow", "dir_adj_pct",
    "same_day", "same_day_multiplier",
    "carrier_target_low", "carrier_target_high",
    "floor_aggressive", "floor_target", "floor_defensive",
    "ceiling_aggressive", "ceiling_target", "ceiling_defensive",
    "breakeven", "stddev", "volatility_band",
    "quoted_amount", "ev_at_quote", "p_profit_at_quote",
    "outcome", "actual_carrier_cost", "actual_margin_dollars", "actual_margin_pct",
    "notes"
]

def _get_github_token():
    """Get GitHub token from Streamlit secrets."""
    return st.secrets.get("GITHUB_TOKEN", "")

def _get_file_from_github():
    """Fetch the quote_log.csv from GitHub. Returns (content_string, sha) or (None, None) if not found."""
    token = _get_github_token()
    if not token:
        return None, None

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        sha = data["sha"]
        return content, sha
    elif resp.status_code == 404:
        return None, None
    else:
        return None, None

def _push_file_to_github(content_string, sha=None, message="Update quote log"):
    """Push updated quote_log.csv to GitHub."""
    token = _get_github_token()
    if not token:
        return False, "No GitHub token configured"

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"

    encoded = base64.b64encode(content_string.encode("utf-8")).decode("utf-8")
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=headers, json=payload)
    if resp.status_code in (200, 201):
        return True, "Saved"
    else:
        return False, f"GitHub API error: {resp.status_code} - {resp.text[:200]}"

def load_quote_log():
    """Load quote log from GitHub as a DataFrame."""
    content, sha = _get_file_from_github()
    if content is None:
        return pd.DataFrame(columns=QUOTE_COLUMNS), None

    try:
        df = pd.read_csv(io.StringIO(content))
        # Ensure all columns exist
        for col in QUOTE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df, sha
    except Exception:
        return pd.DataFrame(columns=QUOTE_COLUMNS), sha

def save_quote(quote_data):
    """Append a new quote to the log and push to GitHub.
    quote_data is a dict with keys matching QUOTE_COLUMNS.
    Returns (success, message)."""

    # Add timestamp in Central time
    central = pytz.timezone("US/Central")
    now = datetime.now(central)
    quote_data["date"] = now.strftime("%Y-%m-%d")
    quote_data["time_central"] = now.strftime("%I:%M %p")

    # Set defaults for outcome fields
    quote_data.setdefault("outcome", "")
    quote_data.setdefault("actual_carrier_cost", "")
    quote_data.setdefault("actual_margin_dollars", "")
    quote_data.setdefault("actual_margin_pct", "")
    quote_data.setdefault("notes", "")

    # Load existing log
    df, sha = load_quote_log()

    # Create new row
    new_row = {col: quote_data.get(col, "") for col in QUOTE_COLUMNS}
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Push to GitHub
    csv_string = new_df.to_csv(index=False)
    success, message = _push_file_to_github(csv_string, sha,
                                             f"Log quote: {quote_data.get('origin', '?')} → {quote_data.get('destination', '?')}")
    return success, message

def update_outcome(row_index, outcome, actual_carrier_cost, notes=""):
    """Update the outcome for a previously logged quote.
    Returns (success, message)."""

    df, sha = load_quote_log()
    if df.empty or row_index >= len(df):
        return False, "Quote not found"

    df.at[row_index, "outcome"] = outcome
    df.at[row_index, "actual_carrier_cost"] = actual_carrier_cost

    if actual_carrier_cost and str(actual_carrier_cost).strip():
        try:
            carrier = float(actual_carrier_cost)
            quoted = float(df.at[row_index, "quoted_amount"]) if df.at[row_index, "quoted_amount"] else 0
            if quoted > 0:
                margin_d = quoted - carrier
                margin_p = (margin_d / quoted) * 100
                df.at[row_index, "actual_margin_dollars"] = round(margin_d, 2)
                df.at[row_index, "actual_margin_pct"] = round(margin_p, 1)
        except (ValueError, TypeError):
            pass

    df.at[row_index, "notes"] = notes

    csv_string = df.to_csv(index=False)
    success, message = _push_file_to_github(csv_string, sha, f"Update outcome: row {row_index}")
    return success, message
