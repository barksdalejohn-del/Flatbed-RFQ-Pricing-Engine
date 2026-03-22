"""
Refresh DAT market matrices from a new DAT Key Markets CSV download.

Usage:
    python refresh_dat.py "path/to/new_DAT_KEY_MARKETS.csv"

Or from the Streamlit app sidebar (upload button).

This replaces the old Excel macro. It reads the DAT CSV, maps city names
to market codes, and rebuilds all 4 matrices as CSVs in the data/ folder.
"""
import sys
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CITY_LOOKUP = os.path.join(DATA_DIR, "city_lookup.csv")


def build_city_to_market_map():
    cl = pd.read_csv(CITY_LOOKUP)
    mapping = {}
    for _, row in cl.iterrows():
        key = (row["City"].strip().upper(), row["State"].strip().upper())
        mapping[key] = row["DAT_Market"]
    return mapping


def refresh_from_dat_csv(dat_csv_path, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR

    city_map = build_city_to_market_map()
    df = pd.read_csv(dat_csv_path)

    # Map origin and dest cities to market codes
    def resolve(city, state):
        key = (str(city).strip().upper(), str(state).strip().upper())
        return city_map.get(key)

    df["orig_market"] = df.apply(lambda r: resolve(r["Orig City"], r["Orig State"]), axis=1)
    df["dest_market"] = df.apply(lambda r: resolve(r["Dest City"], r["Dest State"]), axis=1)

    unmatched = df[df["orig_market"].isna() | df["dest_market"].isna()]
    if len(unmatched) > 0:
        missing_origins = df[df["orig_market"].isna()][["Orig City", "Orig State"]].drop_duplicates()
        missing_dests = df[df["dest_market"].isna()][["Dest City", "Dest State"]].drop_duplicates()
        if len(missing_origins) > 0:
            print(f"WARNING: {len(missing_origins)} unmatched origin cities:")
            for _, r in missing_origins.iterrows():
                print(f"  {r['Orig City']}, {r['Orig State']}")
        if len(missing_dests) > 0:
            print(f"WARNING: {len(missing_dests)} unmatched destination cities:")
            for _, r in missing_dests.iterrows():
                print(f"  {r['Dest City']}, {r['Dest State']}")

    matched = df[df["orig_market"].notna() & df["dest_market"].notna()].copy()
    markets = sorted(matched["orig_market"].unique())

    def build_matrix(value_col, fill_value=0):
        pivot = matched.pivot_table(index="orig_market", columns="dest_market",
                                     values=value_col, aggfunc="first")
        pivot = pivot.reindex(index=markets, columns=markets, fill_value=fill_value)
        pivot.index.name = "Orig \\ Dest"
        return pivot

    rate_matrix = build_matrix("Spot Avg Linehaul Rate", 0)
    stddev_matrix = build_matrix("Spot Linehaul Rate StdDev", 0)
    reports_matrix = build_matrix("Spot # of Reports", 0)
    miles_matrix = build_matrix("PC-Miler Practical Mileage", 0)

    rate_matrix.to_csv(os.path.join(data_dir, "rate_matrix.csv"))
    stddev_matrix.to_csv(os.path.join(data_dir, "stddev_matrix.csv"))
    reports_matrix.to_csv(os.path.join(data_dir, "reports_matrix.csv"))
    miles_matrix.to_csv(os.path.join(data_dir, "miles_matrix.csv"))

    # Extract the date from filename or use today
    import re
    from datetime import date
    basename = os.path.basename(dat_csv_path)
    month_match = re.search(r'(\d{2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)', basename.upper())
    if month_match:
        months = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
                  "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
        year = str(date.today().year)
        dat_date = f"{year}-{months[month_match.group(2)]}"
    else:
        dat_date = date.today().strftime("%Y-%m")

    # Update control panel date
    import json
    cp_path = os.path.join(data_dir, "control_panel.json")
    if os.path.exists(cp_path):
        with open(cp_path) as f:
            cp = json.load(f)
        cp["dat_file_date"] = dat_date
        with open(cp_path, "w") as f:
            json.dump(cp, f, indent=2)

    print(f"\nRefresh complete:")
    print(f"  Source:          {basename}")
    print(f"  DAT data date:   {dat_date}")
    print(f"  Markets:         {len(markets)} x {len(markets)}")
    print(f"  Matched rows:    {len(matched)} / {len(df)}")
    print(f"  Rate Matrix:     {os.path.join(data_dir, 'rate_matrix.csv')}")
    print(f"  StdDev Matrix:   {os.path.join(data_dir, 'stddev_matrix.csv')}")
    print(f"  Reports Matrix:  {os.path.join(data_dir, 'reports_matrix.csv')}")
    print(f"  Miles Matrix:    {os.path.join(data_dir, 'miles_matrix.csv')}")

    return {
        "dat_date": dat_date,
        "markets": len(markets),
        "matched_rows": len(matched),
        "total_rows": len(df),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python refresh_dat.py <path_to_DAT_KEY_MARKETS.csv>")
        sys.exit(1)
    refresh_from_dat_csv(sys.argv[1])
