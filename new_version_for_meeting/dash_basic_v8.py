## new_version_for_meeting/dash_basic_v8_PRODUCTION.py 
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pulp as pl
from pulp import PULP_CBC_CMD
from datetime import timedelta, datetime
import numpy as np
import base64
import io
from time import perf_counter
import math

# -------------------------------
# CONFIGURATION
# -------------------------------
ORDERS_DATA_PATH = "new_version_for_meeting/example_data/orders_example_updated.csv"
FISH_GROUPS_DATA_PATH = "new_version_for_meeting/example_data/fish_groups_example_updated.csv"

# Define specific constraint parameters
ELITE_NUCLEUS_SITE = "Hemne"
HONSVIKGULEN_SITE = "Hønsvikgulen"
LEROY_CUSTOMER_PREFIX = "CUST001"

# Solver options
SOLVER_TIME_LIMIT_SECONDS = 60
W_DUMMY = 1000.0
W_PREF = 5.0
W_FIFO = 0.01

# Default temperature for degree-day calculations (Celsius)
DEFAULT_WATER_TEMP = 8.0

# -------------------------------
# DEGREE-DAYS CALCULATION
# -------------------------------

def calculate_degree_days_feasibility(
    stripping_start, stripping_stop,
    min_temp_prod, max_temp_prod,
    min_temp_customer, max_temp_customer,
    delivery_date,
    water_temp=DEFAULT_WATER_TEMP
):
    """
    Determines if a delivery date is feasible based on degree-day constraints.
    """
    if pd.isna(stripping_start) or pd.isna(stripping_stop) or pd.isna(delivery_date):
        return False
    
    if pd.isna(min_temp_prod) or pd.isna(max_temp_prod):
        return False
    
    if pd.isna(min_temp_customer) or pd.isna(max_temp_customer):
        return False
    
    if min_temp_customer < min_temp_prod:
        return False
    
    if max_temp_customer > max_temp_prod:
        return False
    
    days_from_earliest_strip = (delivery_date - stripping_start).days
    days_from_latest_strip = (delivery_date - stripping_stop).days
    
    if delivery_date < stripping_stop:
        return False
    
    earliest_eggs_dd = days_from_earliest_strip * water_temp
    latest_eggs_dd = days_from_latest_strip * water_temp
    
    if latest_eggs_dd < min_temp_customer:
        return False
    
    if earliest_eggs_dd > max_temp_customer:
        return False
    
    if latest_eggs_dd < min_temp_prod:
        return False
    
    if earliest_eggs_dd > max_temp_prod:
        return False
    
    return True


def calculate_delivery_window_dates(
    stripping_start, stripping_stop,
    min_temp_prod, max_temp_prod,
    water_temp=DEFAULT_WATER_TEMP
):
    """
    Calculate the earliest and latest possible delivery dates for a fish group.
    """
    if pd.isna(stripping_start) or pd.isna(stripping_stop):
        return (pd.NaT, pd.NaT)
    
    if pd.isna(min_temp_prod) or pd.isna(max_temp_prod):
        return (pd.NaT, pd.NaT)
    
    if min_temp_prod >= max_temp_prod:
        return (pd.NaT, pd.NaT)
    
    days_to_min = math.ceil(min_temp_prod / water_temp)
    earliest_delivery = stripping_stop + timedelta(days=days_to_min)
    
    days_to_max = int(max_temp_prod / water_temp)
    latest_delivery = stripping_start + timedelta(days=days_to_max)
    
    if earliest_delivery > latest_delivery:
        return (pd.NaT, pd.NaT)
    
    return (earliest_delivery, latest_delivery)


def add_delivery_windows(fish_groups, water_temp=DEFAULT_WATER_TEMP):
    """
    Add delivery window columns to fish groups DataFrame.
    """
    df = fish_groups.copy()
    
    windows = df.apply(
        lambda row: calculate_delivery_window_dates(
            row['StrippingStartDate'],
            row['StrippingStopDate'],
            row['MinTemp_prod'],
            row['MaxTemp_prod'],
            water_temp
        ) if pd.notna(row['StrippingStartDate']) and pd.notna(row['StrippingStopDate']) 
            and pd.notna(row['MinTemp_prod']) and pd.notna(row['MaxTemp_prod'])
        else (pd.NaT, pd.NaT),
        axis=1
    )
    
    df['EarliestDelivery'] = windows.apply(lambda x: x[0])
    df['LatestDelivery'] = windows.apply(lambda x: x[1])
    
    return df


# -------------------------------
# HELPER & VALIDATION FUNCTIONS
# -------------------------------

def process_organic(x):
    """Converts 'Organic' text or boolean to boolean."""
    if isinstance(x, bool):
        return x
    if pd.isnull(x) or str(x).strip() == "":
        return False
    s = str(x).strip().lower()
    if s in {"organic", "true", "yes", "1"}:
        return True
    if s in {"non-organic", "false", "no", "0"}:
        return False
    return False


def validate_orders(df, valid_groups):
    """
    Validate and preprocess the orders DataFrame.
    """
    warnings = []
    if df is None or df.empty:
        warnings.append("Orders DataFrame is empty.")
        return pd.DataFrame(columns=[
            "OrderNr","DeliveryDate","OrderStatus","CustomerID","CustomerName","Product",
            "Organic","Volume","LockedSite","PreferredSite","MinTemp_customer","MaxTemp_customer"
        ]), warnings

    required_cols = ["OrderNr","DeliveryDate","OrderStatus","CustomerID","CustomerName","Product",
                     "Organic","Volume","LockedSite","PreferredSite"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            warnings.append(f"Orders: Missing column '{col}' added with defaults.")

    valid_group_keys = {}
    for _, row in valid_groups.iterrows():
        site = row.get("Site", "")
        group_key = row.get("Site_Broodst_Season", "")
        if pd.isna(site) or pd.isna(group_key):
            continue
        if site not in valid_group_keys:
            valid_group_keys[site] = []
        valid_group_keys[site].append(group_key)

    df = df.copy()

    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Organic"] = df["Organic"].apply(process_organic)

    if "MinTemp_customer" in df.columns:
        df["MinTemp_customer"] = pd.to_numeric(df["MinTemp_customer"], errors="coerce").fillna(300)
    elif "MinTemp" in df.columns:
        df["MinTemp_customer"] = pd.to_numeric(df["MinTemp"], errors="coerce").fillna(7) * 50
        warnings.append("Converted temperature values to degree-days format")
    else:
        df["MinTemp_customer"] = 300
        warnings.append("Orders: MinTemp_customer not found. Using default value of 300 degree-days.")
    
    if "MaxTemp_customer" in df.columns:
        df["MaxTemp_customer"] = pd.to_numeric(df["MaxTemp_customer"], errors="coerce").fillna(500)
    elif "MaxTemp" in df.columns:
        df["MaxTemp_customer"] = pd.to_numeric(df["MaxTemp"], errors="coerce").fillna(9) * 50
        warnings.append("Converted temperature values to degree-days format")
    else:
        df["MaxTemp_customer"] = 500
        warnings.append("Orders: MaxTemp_customer not found. Using default value of 500 degree-days.")

    swap_mask = df["MinTemp_customer"] > df["MaxTemp_customer"]
    if swap_mask.any():
        df.loc[swap_mask, ["MinTemp_customer", "MaxTemp_customer"]] = df.loc[swap_mask, ["MaxTemp_customer", "MinTemp_customer"]].values
        warnings.append(f"Orders: Corrected {swap_mask.sum()} rows with reversed temperature values")

    def map_site_to_group(site_value):
        if pd.isnull(site_value) or str(site_value).strip() == "":
            return ""
        site_str = str(site_value).strip()
        if any(site_str == group_key for groups in valid_group_keys.values() for group_key in groups):
            return site_str
        if site_str in valid_group_keys and valid_group_keys[site_str]:
            return valid_group_keys[site_str][0]
        return ""

    df["LockedSite"] = df["LockedSite"].apply(map_site_to_group)
    df["PreferredSite"] = df["PreferredSite"].apply(map_site_to_group)

    initial_count = len(df)
    df = df.dropna(subset=["DeliveryDate"])
    df = df[df["Volume"] > 0]
    removed = initial_count - len(df)
    if removed > 0:
        warnings.append(f"Removed {removed} invalid orders (missing dates or zero volume)")

    df["OrderStatus"] = df["OrderStatus"].fillna("Aktiv")
    return df, warnings


def validate_fish_groups(df):
    """
    Validate and preprocess the fish_groups DataFrame.
    """
    warnings = []
    if df is None or df.empty:
        warnings.append("Fish Groups DataFrame is empty.")
        return pd.DataFrame(columns=[
            "Site","Site_Broodst_Season","StrippingStartDate","StrippingStopDate","SalesStartDate","SalesStopDate",
            "MinTemp_prod","MaxTemp_prod","Gain-eggs","Shield-eggs","Organic"
        ]), warnings

    df = df.copy()

    default_cols = {
        "Site": "",
        "Site_Broodst_Season": "",
        "StrippingStartDate": pd.NaT,
        "StrippingStopDate": pd.NaT,
        "SalesStartDate": pd.NaT,
        "SalesStopDate": pd.NaT,
        "Gain-eggs": 0,
        "Shield-eggs": 0,
        "Organic": False
    }
    for col, default in default_cols.items():
        if col not in df.columns:
            df[col] = default
            warnings.append(f"Fish groups: Missing column '{col}' added with defaults.")

    date_columns = ["StrippingStartDate", "StrippingStopDate", "SalesStartDate", "SalesStopDate"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if "MinTemp_prod" in df.columns:
        df["MinTemp_prod"] = pd.to_numeric(df["MinTemp_prod"], errors="coerce").fillna(300)
    else:
        df["MinTemp_prod"] = 300
        warnings.append("Fish groups: MinTemp_prod not found. Using default value of 300 degree-days.")
    
    if "MaxTemp_prod" in df.columns:
        df["MaxTemp_prod"] = pd.to_numeric(df["MaxTemp_prod"], errors="coerce").fillna(500)
    else:
        df["MaxTemp_prod"] = 500
        warnings.append("Fish groups: MaxTemp_prod not found. Using default value of 500 degree-days.")
    
    invalid = df[df["MinTemp_prod"] >= df["MaxTemp_prod"]]
    if not invalid.empty:
        warnings.append(f"{len(invalid)} fish groups have invalid temperature ranges")

    df["Gain-eggs"] = pd.to_numeric(df["Gain-eggs"], errors="coerce").fillna(0)
    df["Shield-eggs"] = pd.to_numeric(df["Shield-eggs"], errors="coerce").fillna(0)

    df["Organic"] = df["Organic"].apply(process_organic)

    if "GroupKey" not in df.columns:
        df["GroupKey"] = df["Site_Broodst_Season"]

    if "Site" not in df.columns or df["Site"].isna().all():
        df["Site"] = df["Site_Broodst_Season"].apply(lambda x: str(x).split("_")[0] if pd.notna(x) else "Unknown")
        warnings.append("Fish groups: Created 'Site' column from Site_Broodst_Season.")

    required_columns = ["GroupKey", "Site_Broodst_Season", "Site"]
    initial_count = len(df)
    for col in required_columns:
        df = df[df[col].notna() & (df[col] != "")]
    df = df.drop_duplicates(subset=["GroupKey" if "GroupKey" in df.columns else "Site_Broodst_Season"], keep='first')

    if len(df) < initial_count:
        warnings.append(f"Removed {initial_count - len(df)} rows due to invalid data or duplicates")

    return df, warnings


def parse_contents(contents, filename, content_type=None):
    """
    Decode uploaded file contents and return a Pandas DataFrame.
    """
    warnings = []
    if contents is None:
        return None, ["No file content"]

    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        name = (filename or "").lower()

        if name.endswith(".csv"):
            try:
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=None, engine="python")
                return df, warnings
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(decoded.decode("latin-1")), sep=None, engine="python")
                warnings.append(f"Parsed {filename} with alternative encoding")
                return df, warnings
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(decoded)), warnings
        else:
            return None, [f"Unsupported file format: {filename}"]
    except Exception as e:
        return None, [f"Error parsing file {filename}: {e}"]


def load_validated_data():
    """Loads and validates data from the specified CSV files."""
    orders_warnings, groups_warnings = [], []
    try:
        orders_df = pd.read_csv(ORDERS_DATA_PATH)
    except Exception as e:
        orders_df = pd.DataFrame({
            "OrderNr": [], "DeliveryDate": [], "OrderStatus": [], "CustomerID": [],
            "CustomerName": [], "Product": [], "Organic": [], "Volume": [],
            "LockedSite": [], "PreferredSite": [], "MinTemp_customer": [], "MaxTemp_customer": []
        })
        orders_warnings.append(f"Could not load example orders: {e}")

    try:
        fish_groups_df = pd.read_csv(FISH_GROUPS_DATA_PATH)
    except Exception as e:
        fish_groups_df = pd.DataFrame({
            "Site": [], "Site_Broodst_Season": [], "StrippingStartDate": [],
            "StrippingStopDate": [], "SalesStartDate": [], "SalesStopDate": [],
            "MinTemp_prod": [], "MaxTemp_prod": [],
            "Gain-eggs": [], "Shield-eggs": [], "Organic": []
        })
        groups_warnings.append(f"Could not load example fish groups: {e}")

    fish_groups_df, val_w_groups = validate_fish_groups(fish_groups_df)
    orders_df, val_w_orders = validate_orders(orders_df, fish_groups_df)

    warnings = orders_warnings + groups_warnings + val_w_groups + val_w_orders
    return orders_df, fish_groups_df, warnings


# -------------------------------
# SOLVER HELPERS
# -------------------------------

def build_feasible_pairs(active_orders, all_fish_groups):
    """Build feasible assignment pairs using degree-days constraints."""
    feasible = {i: set() for i in active_orders.index}
    dummy_idx_list = all_fish_groups[all_fish_groups["Site"] == "Dummy"].index
    if dummy_idx_list.empty:
        raise RuntimeError("Dummy group not found in all_fish_groups.")
    dummy_j = dummy_idx_list[0]

    constraints_applied = 0

    for i in active_orders.index:
        ddate = active_orders.loc[i, "DeliveryDate"]
        org_req = bool(active_orders.loc[i, "Organic"])
        product = str(active_orders.loc[i, "Product"])
        locked = str(active_orders.loc[i, "LockedSite"] or "").strip()
        min_temp_customer = float(active_orders.loc[i, "MinTemp_customer"])
        max_temp_customer = float(active_orders.loc[i, "MaxTemp_customer"])
        cust_id = str(active_orders.loc[i, "CustomerID"] or "")

        feasible[i].add(dummy_j)

        for j in all_fish_groups.index:
            if j == dummy_j:
                continue
            row = all_fish_groups.loc[j]

            if org_req and not bool(row["Organic"]):
                continue

            if locked and row["Site_Broodst_Season"] != locked:
                continue

            if product in ["Elite", "Nucleus"] and row["Site"] != ELITE_NUCLEUS_SITE:
                continue

            if row["Site"] == HONSVIKGULEN_SITE and not cust_id.startswith(LEROY_CUSTOMER_PREFIX):
                continue

            is_feasible = calculate_degree_days_feasibility(
                row["StrippingStartDate"], row["StrippingStopDate"],
                row["MinTemp_prod"], row["MaxTemp_prod"],
                min_temp_customer, max_temp_customer,
                ddate
            )
            
            if not is_feasible:
                constraints_applied += 1
                continue

            feasible[i].add(j)
    
    return feasible


def diagnose_dummy(order_row, fish_groups):
    """Diagnose why an order was assigned to Dummy."""
    reasons = []
    ddate = order_row.get("DeliveryDate")
    org_req = bool(order_row.get("Organic"))
    product = str(order_row.get("Product"))
    locked = str(order_row.get("LockedSite") or "").strip()
    min_temp_customer = float(order_row.get("MinTemp_customer", 300))
    max_temp_customer = float(order_row.get("MaxTemp_customer", 500))
    cust_id = str(order_row.get("CustomerID") or "")

    any_group = False
    org_ok = False
    locked_ok = False
    site_ok = False
    temp_ok = False
    customer_ok = False

    for _, row in fish_groups.iterrows():
        any_group = True
        if org_req and not bool(row["Organic"]):
            continue
        org_ok = True
        if locked and row["Site_Broodst_Season"] != locked:
            continue
        locked_ok = True
        if product in ["Elite", "Nucleus"] and row["Site"] != ELITE_NUCLEUS_SITE:
            continue
        site_ok = True
        if row["Site"] == HONSVIKGULEN_SITE and not cust_id.startswith(LEROY_CUSTOMER_PREFIX):
            continue
        customer_ok = True
        
        is_feasible = calculate_degree_days_feasibility(
            row["StrippingStartDate"], row["StrippingStopDate"],
            row["MinTemp_prod"], row["MaxTemp_prod"],
            min_temp_customer, max_temp_customer,
            ddate
        )
        if not is_feasible:
            continue
        temp_ok = True
        return "Insufficient capacity or constraint conflicts"

    if not any_group:
        reasons.append("No fish groups available")
    if org_req and not org_ok:
        reasons.append("No organic groups available")
    if locked and not locked_ok:
        reasons.append("Required group unavailable")
    if product in ["Elite", "Nucleus"] and not site_ok:
        reasons.append(f"{product} products only available from {ELITE_NUCLEUS_SITE}")
    if not customer_ok:
        reasons.append(f"Site restrictions apply")
    if not temp_ok:
        reasons.append("Delivery date incompatible with maturation requirements")
    return "; ".join(reasons) if reasons else "No compatible fish group"


# -------------------------------
# SOLVER FUNCTION
# -------------------------------
def solve_egg_allocation(orders, fish_groups):
    """
    Solves the egg allocation problem using optimization.
    """
    orders = orders.copy()
    fish_groups = fish_groups.copy()

    active_orders = orders[orders["OrderStatus"] != "Kansellert"].reset_index(drop=True)
    if active_orders.empty:
        status_str = "No active orders"
        results = orders.assign(AssignedGroup="No active orders", IsDummy=False, DummyReason="")
        capacity = fish_groups.assign(
            GainEggsUsed=0, ShieldEggsUsed=0, TotalEggsUsed=0,
            GainEggsRemaining=fish_groups["Gain-eggs"],
            ShieldEggsRemaining=fish_groups["Shield-eggs"],
            TotalEggsRemaining=fish_groups["Gain-eggs"] + fish_groups["Shield-eggs"]
        )
        capacity = add_delivery_windows(capacity)
        final_capacity = capacity[[
            "Site_Broodst_Season", "Site", "StrippingStartDate", "StrippingStopDate",
            "EarliestDelivery", "LatestDelivery", "MinTemp_prod", "MaxTemp_prod",
            "Gain-eggs", "Shield-eggs", "Organic", "GainEggsUsed", "ShieldEggsUsed",
            "TotalEggsUsed", "GainEggsRemaining", "ShieldEggsRemaining", "TotalEggsRemaining"
        ]]
        return {"status": status_str, "results": results, "remaining_capacity": final_capacity, "summary": {}}

    dummy_group = pd.DataFrame({
        "GroupKey": ["Dummy"], "Site": ["Dummy"], "Site_Broodst_Season": ["Dummy"],
        "StrippingStartDate": [pd.Timestamp("2000-01-01")], "StrippingStopDate": [pd.Timestamp("2100-12-31")],
        "SalesStartDate": [pd.Timestamp("2000-01-01")], "SalesStopDate": [pd.Timestamp("2100-12-31")],
        "MinTemp_prod": [0], "MaxTemp_prod": [10000],
        "Gain-eggs": [float("inf")], "Shield-eggs": [float("inf")], "Organic": [True]
    })
    all_fish_groups = pd.concat([fish_groups, dummy_group], ignore_index=True)

    feasible = build_feasible_pairs(active_orders, all_fish_groups)
    prob = pl.LpProblem("FishEggAllocation", pl.LpMinimize)
    x = {(i, j): pl.LpVariable(f"x_{i}_{j}", cat="Binary") for i in active_orders.index for j in feasible[i]}

    real_groups = all_fish_groups[all_fish_groups["Site"] != "Dummy"]
    earliest_start = real_groups["StrippingStartDate"].min() if not real_groups.empty else pd.NaT
    group_fifo_pen = {}
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] != "Dummy" and pd.notna(earliest_start):
            delta = (all_fish_groups.loc[j, "StrippingStopDate"] - earliest_start).days
            group_fifo_pen[j] = max(0.0, float(delta))
        else:
            group_fifo_pen[j] = 0.0
    preferred_map = {i: str(active_orders.loc[i, "PreferredSite"] or "") for i in active_orders.index}
    dummy_j = all_fish_groups[all_fish_groups["Site"] == "Dummy"].index[0]

    obj_terms = []
    for (i, j), var in x.items():
        is_dummy = 1 if j == dummy_j else 0
        pref_mismatch = 1 if (preferred_map[i] and all_fish_groups.loc[j, "Site_Broodst_Season"] != preferred_map[i]) else 0
        fifo_pen = group_fifo_pen.get(j, 0.0)
        obj_terms.append(W_DUMMY * var * is_dummy + W_PREF * var * pref_mismatch + W_FIFO * var * fifo_pen)
    prob += pl.lpSum(obj_terms)

    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in feasible[i]) == 1

    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] == "Dummy":
            continue
        Gcap = all_fish_groups.loc[j, "Gain-eggs"]
        Scap = all_fish_groups.loc[j, "Shield-eggs"]

        gain_i = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Gain" and j in feasible[i]]
        shield_i = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Shield" and j in feasible[i]]
        elite_i = [i for i in active_orders.index if active_orders.loc[i, "Product"] in ["Elite", "Nucleus"] and j in feasible[i]]

        all_gain_orders = gain_i + elite_i
        if all_gain_orders:
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in all_gain_orders) <= Gcap

        if shield_i:
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in shield_i) <= Scap

    try:
        start = perf_counter()
        PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_SECONDS).solve(prob)
        solve_time = perf_counter() - start
        solver_status = pl.LpStatus[prob.status]
    except Exception as e:
        solver_status = f"Error: {e}"
        solve_time = 0.0

    results = active_orders.copy()
    results["AssignedGroup"] = None
    results["IsDummy"] = False

    gain_used = {j: 0.0 for j in all_fish_groups.index}
    shield_used = {j: 0.0 for j in all_fish_groups.index}

    for i in results.index:
        assigned_j = None
        for j in feasible[i]:
            val = pl.value(x[i, j])
            if val is not None and round(val) == 1:
                assigned_j = j
                break
        if assigned_j is None:
            assigned_j = dummy_j
        results.loc[i, "AssignedGroup"] = all_fish_groups.loc[assigned_j, "Site_Broodst_Season"]
        results.loc[i, "IsDummy"] = (all_fish_groups.loc[assigned_j, "Site"] == "Dummy")
        if all_fish_groups.loc[assigned_j, "Site"] != "Dummy":
            volume = float(results.loc[i, "Volume"])
            product = str(results.loc[i, "Product"])
            if product in ["Gain", "Elite", "Nucleus"]:
                gain_used[assigned_j] += volume
            elif product == "Shield":
                shield_used[assigned_j] += volume

    all_res = orders.copy()
    all_res["AssignedGroup"] = None
    all_res["IsDummy"] = False
    order_results_map = {row["OrderNr"]: (row["AssignedGroup"], row["IsDummy"]) for _, row in results.iterrows()}

    for i, row in all_res.iterrows():
        order_nr = row["OrderNr"]
        if order_nr in order_results_map:
            all_res.loc[i, "AssignedGroup"] = order_results_map[order_nr][0]
            all_res.loc[i, "IsDummy"] = order_results_map[order_nr][1]
        elif row["OrderStatus"] == "Kansellert":
            all_res.loc[i, "AssignedGroup"] = "Cancelled"
            all_res.loc[i, "IsDummy"] = False

    res_active = all_res[all_res["OrderStatus"] != "Kansellert"].copy()
    res_active["DummyReason"] = ""
    for idx, r in res_active.iterrows():
        if bool(r["IsDummy"]):
            res_active.at[idx, "DummyReason"] = diagnose_dummy(r, fish_groups)
    all_res = all_res.merge(res_active[["OrderNr", "DummyReason"]], on="OrderNr", how="left")
    all_res["DummyReason"] = all_res["DummyReason"].fillna("")

    capacity = fish_groups.copy()
    capacity["GainEggsUsed"] = 0.0
    capacity["ShieldEggsUsed"] = 0.0
    capacity["TotalEggsUsed"] = 0.0
    capacity["GainEggsRemaining"] = capacity["Gain-eggs"].astype(float)
    capacity["ShieldEggsRemaining"] = capacity["Shield-eggs"].astype(float)
    capacity["TotalEggsRemaining"] = capacity["Gain-eggs"].astype(float) + capacity["Shield-eggs"].astype(float)

    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] != "Dummy":
            site_broodst_season = all_fish_groups.loc[j, "Site_Broodst_Season"]
            if site_broodst_season in capacity["Site_Broodst_Season"].values:
                idx = capacity[capacity["Site_Broodst_Season"] == site_broodst_season].index
                if len(idx) > 0:
                    capacity.loc[idx[0], "GainEggsUsed"] = gain_used[j]
                    capacity.loc[idx[0], "ShieldEggsUsed"] = shield_used[j]
                    capacity.loc[idx[0], "TotalEggsUsed"] = gain_used[j] + shield_used[j]
                    capacity.loc[idx[0], "GainEggsRemaining"] = max(0.0, capacity.loc[idx[0], "Gain-eggs"] - gain_used[j])
                    capacity.loc[idx[0], "ShieldEggsRemaining"] = max(0.0, capacity.loc[idx[0], "Shield-eggs"] - shield_used[j])
                    capacity.loc[idx[0], "TotalEggsRemaining"] = max(0.0, 
                        (capacity.loc[idx[0], "Gain-eggs"] + capacity.loc[idx[0], "Shield-eggs"]) - 
                        gain_used[j] - shield_used[j]
                    )

    capacity = add_delivery_windows(capacity)

    final_capacity = capacity[[
        "Site_Broodst_Season", "Site", "StrippingStartDate", "StrippingStopDate",
        "EarliestDelivery", "LatestDelivery", "MinTemp_prod", "MaxTemp_prod",
        "Gain-eggs", "Shield-eggs", "Organic", "GainEggsUsed", "ShieldEggsUsed",
        "TotalEggsUsed", "GainEggsRemaining", "ShieldEggsRemaining", "TotalEggsRemaining"
    ]].copy()

    total_orders = int((orders["OrderStatus"] != "Kansellert").sum())
    assigned_dummy = int(all_res[(all_res["OrderStatus"] != "Kansellert") & (all_res["IsDummy"])].shape[0])
    summary = {
        "total_orders": total_orders,
        "dummy_orders": assigned_dummy,
        "organic_orders": int(orders[(orders["OrderStatus"] != "Kansellert") & (orders["Organic"])].shape[0]),
        "solve_time_sec": solve_time,
        "status": solver_status,
        "binaries": len(x)
    }
    status_str = f"{solver_status}, {solve_time:.2f}s"
    return {
        "status": status_str,
        "results": all_res,
        "remaining_capacity": final_capacity,
        "summary": summary
    }


# -------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------
def create_buffer_graph(remaining, results, selected_groups=None):
    """
    Creates a visualization of weekly inventory buffer over time.
    """
    if remaining is None or results is None or len(remaining) == 0 or "Site_Broodst_Season" not in remaining.columns:
        return px.line(title="No capacity data available")

    df = remaining.copy()
    
    if "StrippingStartDate" in df.columns:
        df["SalesStartDate"] = pd.to_datetime(df["StrippingStartDate"])
    else:
        df["SalesStartDate"] = pd.to_datetime(df.get("SalesStartDate", pd.NaT))
    
    if "StrippingStopDate" in df.columns:
        df["SalesStopDate"] = pd.to_datetime(df["StrippingStopDate"])
    else:
        df["SalesStopDate"] = pd.to_datetime(df.get("SalesStopDate", pd.NaT))

    if selected_groups:
        df = df[df["Site_Broodst_Season"].isin(selected_groups)]
        if df.empty:
            return px.line(title="No groups selected")

    group_keys = df["Site_Broodst_Season"].unique()
    start_date = df["SalesStartDate"].min()
    end_date = df["SalesStopDate"].max() + pd.Timedelta(weeks=4)
    if pd.isna(start_date) or pd.isna(end_date):
        return px.line(title="Invalid date range")

    weekly_dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    initial_capacity = {group: (row["GainEggsRemaining"] + row["ShieldEggsRemaining"] + row["TotalEggsUsed"])
                        for group, row in df.set_index("Site_Broodst_Season").iterrows()}
    results_copy = results.copy()
    results_copy["DeliveryDate"] = pd.to_datetime(results_copy["DeliveryDate"])

    buffer_data = []
    for week in weekly_dates:
        for group in group_keys:
            assigned_orders = results_copy[
                (results_copy["AssignedGroup"] == group) &
                (results_copy["DeliveryDate"] <= week) &
                (results_copy["IsDummy"] == False)
            ]
            used_volume = assigned_orders["Volume"].sum()
            initial_vol = initial_capacity.get(group, 0)
            remaining_vol = max(0, initial_vol - used_volume)
            buffer_data.append({
                "Week": week, "Group": group,
                "RemainingCapacity": remaining_vol / 1e6
            })

    if not buffer_data:
        return px.line(title="No buffer data generated")

    buffer_df = pd.DataFrame(buffer_data)
    fig = px.line(
        buffer_df, x="Week", y="RemainingCapacity", color="Group",
        title="Weekly Inventory Buffer by Fish Group", markers=True
    )
    fig.update_layout(
        xaxis_title="Week", yaxis_title="Remaining Capacity (Millions)",
        template="plotly_white", font=dict(family="Arial, sans-serif", size=14),
        legend_title_text="Fish Group", margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_yaxes(tickformat=".2f")
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    return fig


def get_input_table_style():
    """Styling for INPUT tables."""
    return {
        "style_table": {
            "overflowX": "auto", "maxHeight": "400px",
            "overflowY": "auto", "borderRadius": "10px"
        },
        "style_cell": {
            "minWidth": "90px", "width": "120px", "maxWidth": "200px",
            "whiteSpace": "normal", "overflow": "hidden", "textOverflow": "ellipsis",
            "textAlign": "left", "padding": "8px", "fontFamily": "Arial, sans-serif",
            "fontSize": "14px", "border": "1px solid #e0e0e0",
        },
        "style_header": {
            "backgroundColor": "#007bff", "color": "white", "fontWeight": "bold",
            "textAlign": "center", "padding": "10px", "borderBottom": "2px solid #0056b3",
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            {"if": {"state": "selected"}, "backgroundColor": "#cce5ff", "border": "1px solid #007bff"},
        ]
    }


def get_results_table_style():
    """Enhanced styling for RESULTS tables."""
    return {
        "style_table": {
            "overflowX": "auto", "maxHeight": "400px",
            "overflowY": "auto", "borderRadius": "10px"
        },
        "style_cell": {
            "minWidth": "90px", "width": "120px", "maxWidth": "200px",
            "whiteSpace": "normal", "overflow": "hidden", "textOverflow": "ellipsis",
            "textAlign": "left", "padding": "8px", "fontFamily": "Arial, sans-serif",
            "fontSize": "14px", "border": "1px solid #e0e0e0",
        },
        "style_header": {
            "backgroundColor": "#007bff", "color": "white", "fontWeight": "bold",
            "textAlign": "center", "padding": "10px", "borderBottom": "2px solid #0056b3",
        },
        "style_filter": {
            "backgroundColor": "#e7f3ff",
            "border": "2px solid #007bff",
            "fontWeight": "bold",
            "padding": "5px"
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            {"if": {"state": "selected"}, "backgroundColor": "#cce5ff", "border": "1px solid #007bff"},
            {
                "if": {
                    "filter_query": '{AssignedGroup} != "Cancelled" && {AssignedGroup} != "Dummy" && {AssignedGroup} != ""',
                    "column_id": "OrderNr"
                },
                "backgroundColor": "#d4edda", "color": "black"
            }
        ]
    }


def build_order_columns_schema(group_keys):
    """Columns schema for order table."""
    group_keys = sorted([g for g in group_keys if isinstance(g, str)]) if group_keys is not None else []
    order_columns = [
        {"name": "OrderNr", "id": "OrderNr"},
        {"name": "OrderStatus", "id": "OrderStatus", "presentation": "dropdown"},
        {"name": "CustomerID", "id": "CustomerID"},
        {"name": "CustomerName", "id": "CustomerName"},
        {"name": "Product", "id": "Product", "presentation": "dropdown"},
        {"name": "Organic", "id": "Organic", "presentation": "dropdown"},
        {"name": "Volume", "id": "Volume", "type": "numeric"},
        {"name": "DeliveryDate", "id": "DeliveryDate", "type": "datetime"},
        {"name": "LockedSite", "id": "LockedSite", "presentation": "dropdown"},
        {"name": "PreferredSite", "id": "PreferredSite", "presentation": "dropdown"},
        {"name": "MinTemp_customer", "id": "MinTemp_customer", "type": "numeric"},
        {"name": "MaxTemp_customer", "id": "MaxTemp_customer", "type": "numeric"},
    ]
    return order_columns


def build_order_dropdowns(group_keys):
    product_options = ["Gain", "Shield", "Elite", "Nucleus"]
    group_options = [{"label": x, "value": x} for x in sorted([g for g in group_keys if isinstance(g, str)])]
    dropdowns = {
        "OrderStatus": {"options": [{"label": x, "value": x} for x in ["Aktiv", "Bekreftet", "Planlagt", "Kansellert"]]},
        "Product": {"options": [{"label": x, "value": x} for x in product_options]},
        "Organic": {"options": [{"label": "Yes", "value": True}, {"label": "No", "value": False}]},
        "LockedSite": {"options": group_options},
        "PreferredSite": {"options": [{"label": "", "value": ""}] + group_options},
    }
    return dropdowns


# -------------------------------
# START WITH EMPTY DATA
# -------------------------------
orders_data = pd.DataFrame(columns=[
    "OrderNr","DeliveryDate","OrderStatus","CustomerID","CustomerName","Product",
    "Organic","Volume","LockedSite","PreferredSite","MinTemp_customer","MaxTemp_customer"
])

fish_groups_data = pd.DataFrame(columns=[
    "Site","Site_Broodst_Season","StrippingStartDate","StrippingStopDate",
    "MinTemp_prod","MaxTemp_prod","Gain-eggs","Shield-eggs","Organic",
    "SalesStartDate","SalesStopDate"
])

initial_warnings = []

# -------------------------------
# DASH APP SETUP
# -------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
app.title = "AquaGen Egg Planning System"

app.layout = dbc.Container([
    html.H1("🐟 AquaGen Egg Allocation Planner", 
            className="text-center my-4 py-3 bg-primary text-white rounded"),
    
    html.P("Optimize egg allocation to customer orders using degree-days maturation tracking",
           className="text-center lead mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H4("Data Source", className="mb-3"),
            dcc.RadioItems(
                id="data-source-selector",
                options=[
                    {"label": " Upload Your Data", "value": "upload"},
                    {"label": " Use Example Data", "value": "example"}
                ],
                value="upload",
                labelStyle={"display": "block", "marginBottom": "10px"},
                className="mb-3"
            ),
            html.Div(
                id="upload-container",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📤 Upload Data Files", className="card-title"),
                            html.P("Upload your orders and fish groups data to begin planning", className="card-text"),
                            
                            html.Div([
                                html.Label("Orders File (.csv, .xlsx)", className="fw-bold"),
                                dcc.Upload(
                                    id="upload-orders",
                                    children=html.Div([
                                        "Drag and Drop or ",
                                        html.A("Browse Files", className="text-primary fw-bold")
                                    ]),
                                    style={
                                        'width': '100%', 'height': '70px', 'lineHeight': '70px',
                                        'borderWidth': '2px', 'borderStyle': 'dashed', 
                                        'borderRadius': '8px', 'borderColor': '#007bff',
                                        'textAlign': 'center', 'marginTop': '5px',
                                        'backgroundColor': '#f8f9fa', 'cursor': 'pointer'
                                    },
                                    multiple=False
                                ),
                                html.Div(id="upload-orders-status", className="mt-2"),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Fish Groups File (.csv, .xlsx)", className="fw-bold"),
                                dcc.Upload(
                                    id="upload-fish-groups",
                                    children=html.Div([
                                        "Drag and Drop or ",
                                        html.A("Browse Files", className="text-primary fw-bold")
                                    ]),
                                    style={
                                        'width': '100%', 'height': '70px', 'lineHeight': '70px',
                                        'borderWidth': '2px', 'borderStyle': 'dashed',
                                        'borderRadius': '8px', 'borderColor': '#007bff',
                                        'textAlign': 'center', 'marginTop': '5px',
                                        'backgroundColor': '#f8f9fa', 'cursor': 'pointer'
                                    },
                                    multiple=False
                                ),
                                html.Div(id="upload-fish-groups-status", className="mt-2"),
                            ]),
                        ])
                    ]),
                ],
                style={"display": "block"}
            )
        ], width=12)
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Orders", className="mt-4"),
            html.P("Customer orders with delivery requirements and degree-days specifications", className="text-muted"),
            dash_table.DataTable(
                id="order-table",
                data=orders_data.to_dict("records"),
                columns=build_order_columns_schema([]),
                editable=True, 
                row_deletable=True, 
                page_action='native', 
                page_current=0,
                page_size=10, 
                sort_action='native',
                dropdown=build_order_dropdowns([]),
                fixed_rows={"headers": True}, 
                **get_input_table_style()
            ),
            dbc.Button("Add Order Row", id="add-order-row-button", n_clicks=0, 
                      color="primary", size="sm", className="mt-2"),
            
            html.H3("Fish Groups", className="mt-4"),
            html.P("Production groups with capacity and maturation parameters", className="text-muted"),
            dash_table.DataTable(
                id="fish-group-table",
                data=fish_groups_data.to_dict("records"),
                columns=[{"name": col, "id": col, "editable": True} for col in fish_groups_data.columns],
                editable=True, 
                row_deletable=True, 
                page_action='native', 
                page_current=0,
                page_size=10, 
                sort_action='native',
                fixed_rows={"headers": True}, 
                **get_input_table_style()
            ),
            dbc.Button("Add Fish Group Row", id="add-fish-group-row-button", n_clicks=0, 
                      color="primary", size="sm", className="mt-2"),
        ], width=7),
        
        dbc.Col([
            html.H3("System Overview", className="mt-4"),
            dcc.Markdown("""
                ### Allocation Constraints
                
                The system automatically allocates orders to fish groups while respecting:
                
                **Biological Requirements**
                - Degree-days maturation tracking
                - Minimum and maximum maturation thresholds
                - Quality window compliance
                - Delivery timing validation
                
                **Capacity Management**
                - Separate Gain and Shield egg inventories
                - Volume availability per group
                - Organic certification requirements
                
                **Business Rules**
                - Elite and Nucleus products: Hemne site only
                - Hønsvikgulen site: Lerøy customers only
                - Locked and preferred site assignments
                - Order status handling (Active, Confirmed, Planned, Cancelled)
                
                **Optimization Goals**
                1. Maximize successful order assignments
                2. Respect preferred site assignments
                3. Optimize inventory utilization
                
                ### Data Format
                
                **Orders require:**
                - Order number, delivery date, customer details
                - Product type, volume, organic status
                - MinTemp_customer and MaxTemp_customer (degree-days)
                
                **Fish Groups require:**
                - Site and group identifier
                - Stripping dates, capacity (Gain/Shield eggs)
                - MinTemp_prod and MaxTemp_prod (degree-days)
            """, className="p-3 bg-light rounded"),
            
            dbc.Alert(id="validation-alert", color="info", is_open=True,
                      children="Upload data files or load example data to begin", 
                      className="mt-3"),
            
            dbc.Button("Run Optimization", id="solve-button", n_clicks=0,
                       color="success", size="lg", className="mt-4 w-100", disabled=True),
            html.Div(id="solver-summary", className="mt-3"),
        ], width=5, className="p-4"),
    ], className="my-4"),
    
    dcc.Loading(
        id="loading", type="circle",
        children=html.Div(
            id="results-section", className="mt-4", style={"display": "none"},
            children=[
                html.Hr(className="my-4"),
                html.H2("📊 Allocation Results", className="text-center mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Filter Results", className="card-title"),
                                dbc.Checklist(
                                    id="filter-dummy-toggle",
                                    options=[{"label": " Show only unassigned orders", "value": "dummy"}],
                                    value=[],
                                    switch=True,
                                    className="mb-2"
                                ),
                                html.P("Use column filters below for advanced filtering", 
                                       className="text-muted small mb-0")
                            ])
                        ], color="light", className="mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H3("Order Assignments", className="mt-4"),
                        html.P("Detailed allocation of each order to fish groups", className="text-muted"),
                        dash_table.DataTable(
                            id="results-table", 
                            filter_action='native',
                            page_action='native',
                            page_size=10,
                            sort_action='native',
                            **get_results_table_style()
                        ),
                        
                        html.H3("Capacity Summary", className="mt-5"),
                        html.P("Remaining capacity and delivery windows per fish group", className="text-muted"),
                        dbc.Alert([
                            html.Strong("Delivery Windows: "),
                            "EarliestDelivery = earliest viable delivery date based on maturation. ",
                            "LatestDelivery = latest delivery before quality degradation."
                        ], color="info", className="small"),
                        dash_table.DataTable(
                            id="capacity-table", 
                            filter_action='native',
                            page_action='native',
                            page_size=10,
                            sort_action='native',
                            **get_results_table_style()
                        ),
                    ], width=12),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H3("Inventory Projection", className="mt-5"),
                        html.P("Weekly remaining capacity forecast", className="text-muted"),
                        dcc.Dropdown(
                            id="buffer-group-filter", 
                            multi=True, 
                            placeholder="Filter by fish group..."
                        ),
                        dcc.Graph(id="buffer-visualization", className="mt-3"),
                    ], width=12),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Export Results", className="mt-4"),
                        dbc.ButtonGroup([
                            dbc.Button("📥 Download CSV", id="download-csv-button", color="info", className="me-2"),
                            dbc.Button("📥 Download Excel", id="download-xlsx-button", color="secondary"),
                        ]),
                        dcc.Download(id="download-csv"),
                        dcc.Download(id="download-xlsx"),
                    ], width=12)
                ], className="mt-4 mb-5"),
            ]
        )
    )
], fluid=True)

# -------------------------------
# CALLBACKS
# -------------------------------

@app.callback(
    Output("upload-orders-status", "children"),
    Input("upload-orders", "contents"),
    State("upload-orders", "filename")
)
def show_orders_upload_status(contents, filename):
    if contents is None:
        return None
    return dbc.Alert(f"✓ Loaded: {filename}", color="success", className="small py-1 px-2 mb-0")


@app.callback(
    Output("upload-fish-groups-status", "children"),
    Input("upload-fish-groups", "contents"),
    State("upload-fish-groups", "filename")
)
def show_fish_groups_upload_status(contents, filename):
    if contents is None:
        return None
    return dbc.Alert(f"✓ Loaded: {filename}", color="success", className="small py-1 px-2 mb-0")


@app.callback(
    [Output("solve-button", "disabled"),
     Output("validation-alert", "children"),
     Output("validation-alert", "is_open"),
     Output("validation-alert", "color")],
    [Input("order-table", "data"),
     Input("fish-group-table", "data")]
)
def update_solve_button_state(order_data, fish_group_data):
    """Enable solve button only when both tables have data."""
    orders_empty = not order_data or len(order_data) == 0
    groups_empty = not fish_group_data or len(fish_group_data) == 0
    
    if orders_empty and groups_empty:
        return True, "Upload data files or load example data to begin", True, "info"
    elif orders_empty:
        return True, "Orders data missing. Please upload orders file.", True, "warning"
    elif groups_empty:
        return True, "Fish groups data missing. Please upload fish groups file.", True, "warning"
    else:
        orders_count = len(order_data)
        groups_count = len(fish_group_data)
        return False, f"Ready: {orders_count} orders and {groups_count} fish groups loaded", True, "success"


@app.callback(
    Output("upload-container", "style"),
    Input("data-source-selector", "value")
)
def toggle_upload_container(data_source):
    if data_source == "upload":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("order-table", "dropdown"),
    Input("fish-group-table", "data")
)
def update_order_dropdowns(fish_group_data):
    df = pd.DataFrame(fish_group_data) if fish_group_data else pd.DataFrame(columns=["Site_Broodst_Season"])
    group_keys = df["Site_Broodst_Season"].dropna().unique().tolist() if "Site_Broodst_Season" in df.columns else []
    return build_order_dropdowns(group_keys)


@app.callback(
    [Output("order-table", "data"),
     Output("order-table", "columns")],
    [Input("upload-orders", "contents"),
     Input("data-source-selector", "value"),
     Input("add-order-row-button", "n_clicks")],
    [State("order-table", "data"),
     State("upload-orders", "filename"),
     State("fish-group-table", "data")]
)
def update_order_table(uploaded_contents, data_source, n_clicks, current_data, filename, fish_group_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    fg_df = pd.DataFrame(fish_group_data) if fish_group_data else pd.DataFrame(columns=["Site_Broodst_Season"])
    group_keys = fg_df["Site_Broodst_Season"].dropna().unique().tolist() if "Site_Broodst_Season" in fg_df.columns else []
    columns_schema = build_order_columns_schema(group_keys)
    table_data = current_data if current_data else []

    if triggered_id == "data-source-selector":
        if data_source == "example":
            odf, _, _ = load_validated_data()
            return odf.to_dict("records"), columns_schema
        else:
            return table_data, columns_schema
    
    elif triggered_id == "upload-orders":
        if uploaded_contents is not None:
            df, _ = parse_contents(uploaded_contents, filename)
            if df is not None:
                fg_valid, _ = validate_fish_groups(fg_df)
                df_valid, _ = validate_orders(df, fg_valid)
                return df_valid.to_dict("records"), columns_schema
    
    elif triggered_id == "add-order-row-button" and n_clicks > 0:
        new_row = {
            "OrderNr": "", "OrderStatus": "Aktiv", "CustomerID": "", "CustomerName": "",
            "Product": "Gain", "Organic": False, "Volume": 0, "DeliveryDate": "",
            "LockedSite": "", "PreferredSite": "", "MinTemp_customer": 300, "MaxTemp_customer": 500,
        }
        table_data = table_data + [new_row]

    return table_data, columns_schema


@app.callback(
    Output("fish-group-table", "data"),
    [Input("upload-fish-groups", "contents"),
     Input("data-source-selector", "value"),
     Input("add-fish-group-row-button", "n_clicks")],
    [State("fish-group-table", "data"),
     State("upload-fish-groups", "filename")]
)
def update_fish_group_table(uploaded_contents, data_source, n_clicks, current_data, filename):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    table_data = current_data if current_data else []

    if triggered_id == "data-source-selector":
        if data_source == "example":
            _, fish_groups_df, _ = load_validated_data()
            return fish_groups_df.to_dict("records")
        else:
            return table_data
    
    elif triggered_id == "upload-fish-groups":
        if uploaded_contents is not None:
            df, _ = parse_contents(uploaded_contents, filename)
            if df is not None:
                df_valid, _ = validate_fish_groups(df)
                return df_valid.to_dict("records")
    
    elif triggered_id == "add-fish-group-row-button" and n_clicks > 0:
        if not table_data:
            new_row = {
                "Site": "", "Site_Broodst_Season": "", 
                "StrippingStartDate": "", "StrippingStopDate": "",
                "MinTemp_prod": 300, "MaxTemp_prod": 1200,
                "Gain-eggs": 0, "Shield-eggs": 0, "Organic": False,
                "SalesStartDate": "", "SalesStopDate": ""
            }
        else:
            new_row = {col: "" for col in table_data[0].keys()}
        table_data.append(new_row)

    return table_data


@app.callback(
    [Output("results-section", "style"),
     Output("results-table", "data"), Output("results-table", "columns"),
     Output("capacity-table", "data"), Output("capacity-table", "columns"),
     Output("buffer-visualization", "figure"), Output("solve-button", "children"),
     Output("buffer-group-filter", "options"), Output("buffer-group-filter", "value"),
     Output("solver-summary", "children")],
    [Input("solve-button", "n_clicks"),
     Input("filter-dummy-toggle", "value")],
    [State("order-table", "data"), State("fish-group-table", "data"),
     State("buffer-group-filter", "value"),
     State("results-table", "data")],
    prevent_initial_call=True
)
def update_results(n_clicks, dummy_filter, order_data, fish_group_data, selected_groups, current_results):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == "filter-dummy-toggle" and current_results:
        results_df = pd.DataFrame(current_results)
        display_results = results_df.copy()
        if "dummy" in dummy_filter:
            display_results = display_results[display_results["IsDummy"] == True]
        results_cols = [{"name": c, "id": c} for c in display_results.columns]
        return dash.no_update, display_results.to_dict("records"), results_cols, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if triggered_id == "solve-button" and n_clicks > 0:
        orders_df = pd.DataFrame(order_data)
        fish_groups_df = pd.DataFrame(fish_group_data)
        fish_groups_df, w1 = validate_fish_groups(fish_groups_df)
        orders_df, w2 = validate_orders(orders_df, fish_groups_df)

        solution = solve_egg_allocation(orders_df, fish_groups_df)
        results_df = solution["results"]
        capacity_df = solution["remaining_capacity"]
        status = solution["status"]

        display_results = results_df.copy()
        if "dummy" in dummy_filter:
            display_results = display_results[display_results["IsDummy"] == True]

        results_cols = [{"name": c, "id": c} for c in display_results.columns]
        capacity_cols = [{"name": c, "id": c} for c in capacity_df.columns]

        group_options = sorted(capacity_df["Site_Broodst_Season"].dropna().unique().tolist()) if not capacity_df.empty else []
        filter_options = [{"label": g, "value": g} for g in group_options]
        filter_value = selected_groups if selected_groups else []

        fig = create_buffer_graph(capacity_df, results_df, selected_groups=filter_value)
        button_text = "Run Optimization"

        sumd = solution.get("summary", {})
        summary_display = dbc.Alert([
            html.H5("Optimization Summary", className="alert-heading"),
            html.Hr(),
            html.P([
                html.Strong("Total Orders: "), f"{sumd.get('total_orders', 0)}", html.Br(),
                html.Strong("Successfully Assigned: "), f"{sumd.get('total_orders', 0) - sumd.get('dummy_orders', 0)}", html.Br(),
                html.Strong("Unassigned: "), f"{sumd.get('dummy_orders', 0)}", html.Br(),
                html.Strong("Organic Orders: "), f"{sumd.get('organic_orders', 0)}", html.Br(),
                html.Strong("Processing Time: "), f"{sumd.get('solve_time_sec', 0):.2f} seconds", html.Br(),
                html.Strong("Status: "), f"{sumd.get('status', 'N/A')}"
            ])
        ], color="success" if sumd.get('dummy_orders', 0) == 0 else "warning")

        return (
            {"margin": "20px", "display": "block"}, display_results.to_dict("records"),
            results_cols, capacity_df.to_dict("records"), capacity_cols, fig,
            button_text, filter_options, filter_value, summary_display
        )
    
    return dash.no_update


@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-button", "n_clicks"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def export_results_csv(n_clicks, results_data):
    if n_clicks:
        df = pd.DataFrame(results_data)
        return dcc.send_data_frame(df.to_csv, "allocation_results.csv", index=False)
    return None


@app.callback(
    Output("download-xlsx", "data"),
    Input("download-xlsx-button", "n_clicks"),
    State("results-table", "data"),
    State("capacity-table", "data"),
    prevent_initial_call=True
)
def export_results_excel(n_clicks, results_data, capacity_data):
    if n_clicks:
        results_df = pd.DataFrame(results_data)
        capacity_df = pd.DataFrame(capacity_data)

        def to_excel(bytes_io):
            with pd.ExcelWriter(bytes_io, engine="xlsxwriter") as writer:
                results_df.to_excel(writer, sheet_name="Order Assignments", index=False)
                capacity_df.to_excel(writer, sheet_name="Capacity Summary", index=False)

        return dcc.send_bytes(to_excel, "allocation_results.xlsx")
    return None


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8050))
    host = '0.0.0.0'
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    print(f"🐟 Starting AquaGen Roe Planning System on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
