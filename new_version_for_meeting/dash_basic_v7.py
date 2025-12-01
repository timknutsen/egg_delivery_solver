## new_version_for_meeting/dash_basic_v6_FIXED_UI.py 
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
# DEGREE-DAYS CALCULATION (FIXED)
# -------------------------------

def calculate_degree_days_feasibility(
    stripping_start, stripping_stop,
    min_temp_prod, max_temp_prod,
    min_temp_customer, max_temp_customer,
    delivery_date,
    water_temp=DEFAULT_WATER_TEMP
):
    """
    🔴 FIXED: Determines if a delivery date is feasible based on degree-day constraints.
    
    CORRECTED Logic:
    1. Customer requirements must fit within production limits:
       - min_temp_customer >= min_temp_prod (roe must be ready)
       - max_temp_customer <= max_temp_prod (roe must not be overripe)
    
    2. Delivery date must allow sufficient degree-day accumulation:
       - LATEST stripped eggs must reach min_temp_customer
       - EARLIEST stripped eggs must not exceed max_temp_customer
    
    Key insight: The valid delivery window is when ALL eggs in the batch
    meet both production and customer requirements simultaneously.
    """
    # Validate inputs
    if pd.isna(stripping_start) or pd.isna(stripping_stop) or pd.isna(delivery_date):
        return False
    
    if pd.isna(min_temp_prod) or pd.isna(max_temp_prod):
        return False
    
    if pd.isna(min_temp_customer) or pd.isna(max_temp_customer):
        return False
    
    # Constraint 1: Customer requirements must fit within production limits
    if min_temp_customer < min_temp_prod:
        return False
    
    if max_temp_customer > max_temp_prod:
        return False
    
    # 🔴 FIXED: Correct degree-day accumulation logic
    days_from_earliest_strip = (delivery_date - stripping_start).days
    days_from_latest_strip = (delivery_date - stripping_stop).days
    
    # Handle delivery before stripping period ends
    if delivery_date < stripping_stop:
        return False
    
    earliest_eggs_dd = days_from_earliest_strip * water_temp
    latest_eggs_dd = days_from_latest_strip * water_temp
    
    # 🔴 FIXED: Constraint 2 - ALL eggs must meet customer requirements
    if latest_eggs_dd < min_temp_customer:
        return False
    
    if earliest_eggs_dd > max_temp_customer:
        return False
    
    # 🔴 FIXED: Constraint 3 - ALL eggs must still be within production quality window
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
    🔴 FIXED: Calculate the earliest and latest possible delivery dates for a fish group.
    
    CORRECTED Logic:
    - Earliest delivery: when LATEST (youngest) stripped eggs reach min_temp_prod
    - Latest delivery: when EARLIEST (oldest) stripped eggs reach max_temp_prod
    
    This ensures ALL eggs in the batch are within the sellable quality window.
    """
    if pd.isna(stripping_start) or pd.isna(stripping_stop):
        return (pd.NaT, pd.NaT)
    
    if pd.isna(min_temp_prod) or pd.isna(max_temp_prod):
        return (pd.NaT, pd.NaT)
    
    if min_temp_prod >= max_temp_prod:
        return (pd.NaT, pd.NaT)
    
    # 🔴 FIXED: Earliest delivery = when YOUNGEST eggs (stripped last) are ready
    days_to_min = math.ceil(min_temp_prod / water_temp)
    earliest_delivery = stripping_stop + timedelta(days=days_to_min)
    
    # 🔴 FIXED: Latest delivery = when OLDEST eggs (stripped first) become overripe
    days_to_max = int(max_temp_prod / water_temp)
    latest_delivery = stripping_start + timedelta(days=days_to_max)
    
    # 🔴 FIXED: Validate that window is possible
    if earliest_delivery > latest_delivery:
        return (pd.NaT, pd.NaT)
    
    return (earliest_delivery, latest_delivery)


def add_delivery_windows(fish_groups, water_temp=DEFAULT_WATER_TEMP):
    """
    🔴 FIXED: Add delivery window columns to fish groups DataFrame.
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
    Now expects MinTemp_customer and MaxTemp_customer columns (degree-days).
    Returns (df, warnings)
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

    # Handle temperature columns
    if "MinTemp_customer" in df.columns:
        df["MinTemp_customer"] = pd.to_numeric(df["MinTemp_customer"], errors="coerce").fillna(300)
    elif "MinTemp" in df.columns:
        df["MinTemp_customer"] = pd.to_numeric(df["MinTemp"], errors="coerce").fillna(7) * 50
        warnings.append("Note: Converted MinTemp (°C) to MinTemp_customer (degree-days) using approximation (temp × 50)")
    else:
        df["MinTemp_customer"] = 300
        warnings.append("Orders: MinTemp_customer not found. Using default value of 300 degree-days.")
    
    if "MaxTemp_customer" in df.columns:
        df["MaxTemp_customer"] = pd.to_numeric(df["MaxTemp_customer"], errors="coerce").fillna(500)
    elif "MaxTemp" in df.columns:
        df["MaxTemp_customer"] = pd.to_numeric(df["MaxTemp"], errors="coerce").fillna(9) * 50
        warnings.append("Note: Converted MaxTemp (°C) to MaxTemp_customer (degree-days) using approximation (temp × 50)")
    else:
        df["MaxTemp_customer"] = 500
        warnings.append("Orders: MaxTemp_customer not found. Using default value of 500 degree-days.")

    swap_mask = df["MinTemp_customer"] > df["MaxTemp_customer"]
    if swap_mask.any():
        df.loc[swap_mask, ["MinTemp_customer", "MaxTemp_customer"]] = df.loc[swap_mask, ["MaxTemp_customer", "MinTemp_customer"]].values
        warnings.append(f"Orders: Swapped {swap_mask.sum()} rows where MinTemp_customer > MaxTemp_customer")

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
        warnings.append(f"Orders: Removed {removed} orders with invalid dates or zero/non-numeric volume.")

    df["OrderStatus"] = df["OrderStatus"].fillna("Aktiv")
    return df, warnings


def validate_fish_groups(df):
    """
    Validate and preprocess the fish_groups DataFrame.
    Now expects MinTemp_prod and MaxTemp_prod columns (degree-days).
    Returns (df, warnings)
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
        warnings.append(f"Fish groups: {len(invalid)} groups have MinTemp_prod >= MaxTemp_prod. These may cause issues.")

    df["Gain-eggs"] = pd.to_numeric(df["Gain-eggs"], errors="coerce").fillna(0)
    df["Shield-eggs"] = pd.to_numeric(df["Shield-eggs"], errors="coerce").fillna(0)

    df["Organic"] = df["Organic"].apply(process_organic)

    if "GroupKey" not in df.columns:
        df["GroupKey"] = df["Site_Broodst_Season"]

    if "Site" not in df.columns or df["Site"].isna().all():
        df["Site"] = df["Site_Broodst_Season"].apply(lambda x: str(x).split("_")[0] if pd.notna(x) else "Unknown")
        warnings.append("Fish groups: 'Site' column not found or empty. Created from Site_Broodst_Season.")

    required_columns = ["GroupKey", "Site_Broodst_Season", "Site"]
    initial_count = len(df)
    for col in required_columns:
        df = df[df[col].notna() & (df[col] != "")]
    df = df.drop_duplicates(subset=["GroupKey" if "GroupKey" in df.columns else "Site_Broodst_Season"], keep='first')

    if len(df) < initial_count:
        warnings.append(f"Fish groups: Removed {initial_count - len(df)} rows due to invalid data or duplicates.")

    return df, warnings


# -------------------------------
# PARSE UPLOADED FILE CONTENT
# -------------------------------
def parse_contents(contents, filename, content_type=None):
    """
    Decode uploaded file contents and return a Pandas DataFrame.
    Supports CSV and Excel files.
    Returns (df, warnings)
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
                warnings.append(f"Parsed {filename} with latin-1 encoding fallback.")
                return df, warnings
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(decoded)), warnings
        else:
            return None, [f"Unsupported file format: {filename}"]
    except Exception as e:
        return None, [f"Error parsing file {filename}: {e}"]


# -------------------------------
# DATA LOADING FUNCTION (Example Data)
# -------------------------------
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
        orders_warnings.append(f"Error loading {ORDERS_DATA_PATH}: {e}")

    try:
        fish_groups_df = pd.read_csv(FISH_GROUPS_DATA_PATH)
    except Exception as e:
        fish_groups_df = pd.DataFrame({
            "Site": [], "Site_Broodst_Season": [], "StrippingStartDate": [],
            "StrippingStopDate": [], "SalesStartDate": [], "SalesStopDate": [],
            "MinTemp_prod": [], "MaxTemp_prod": [],
            "Gain-eggs": [], "Shield-eggs": [], "Organic": []
        })
        groups_warnings.append(f"Error loading {FISH_GROUPS_DATA_PATH}: {e}")

    fish_groups_df, val_w_groups = validate_fish_groups(fish_groups_df)
    orders_df, val_w_orders = validate_orders(orders_df, fish_groups_df)

    warnings = orders_warnings + groups_warnings + val_w_groups + val_w_orders
    return orders_df, fish_groups_df, warnings


# -------------------------------
# SOLVER HELPERS
# -------------------------------

def build_feasible_pairs(active_orders, all_fish_groups):
    """🔴 FIXED: Build feasible assignment pairs using corrected degree-days constraints."""
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
    
    print(f"✅ Applied {constraints_applied} degree-days exclusion constraints (CORRECTED logic)")
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
        return "Capacity limitation or objective penalties"

    if not any_group:
        reasons.append("No fish groups available")
    if org_req and not org_ok:
        reasons.append("Organic mismatch (no organic groups)")
    if locked and not locked_ok:
        reasons.append("Locked group not available")
    if product in ["Elite", "Nucleus"] and not site_ok:
        reasons.append(f"{product} restricted to site {ELITE_NUCLEUS_SITE}")
    if not customer_ok:
        reasons.append(f"Customer restriction for site {HONSVIKGULEN_SITE}")
    if not temp_ok:
        reasons.append("No groups within degree-days delivery window (corrected calculation)")
    return "; ".join(reasons) if reasons else "No feasible group"


# -------------------------------
# SOLVER FUNCTION
# -------------------------------
def solve_egg_allocation(orders, fish_groups):
    """
    Solves the egg allocation problem using PuLP with corrected degree-days constraints.
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

    print("🔧 Building feasible pairs with CORRECTED degree-days constraints...")
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
        print(f"✅ Solver completed: {solver_status} in {solve_time:.2f}s")
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
            all_res.loc[i, "AssignedGroup"] = "Skipped-Cancelled"
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
    status_str = f"{solver_status}, {solve_time:.2f}s, {len(x)} binaries"
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
        title="Weekly Inventory Buffer by Fish Group (Degree-Days Model - CORRECTED)", markers=True
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
    """Styling for INPUT tables (NO filtering)."""
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
    """Enhanced styling for RESULTS tables (WITH filtering)."""
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
                    "filter_query": '{AssignedGroup} != "Skipped-Cancelled" && {AssignedGroup} != "Dummy" && {AssignedGroup} != ""',
                    "column_id": "OrderNr"
                },
                "backgroundColor": "#d4edda", "color": "black"
            }
        ]
    }


def build_order_columns_schema(group_keys):
    """Columns schema for order table with dropdowns and types."""
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
# 🔴 CHANGED: START WITH EMPTY DATA
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

initial_warnings = ["Please upload data files or click 'Load Example Data' to begin."]

# -------------------------------
# DASH APP SETUP
# -------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Fish Egg Allocation Solver - Degree-Days Model (FIXED)"

app.layout = dbc.Container([
    html.H1("🔧 Fish Egg Allocation Solver (Degree-Days - DELIVERY WINDOW FIXED)", 
            className="text-center my-4 py-3 bg-primary text-white rounded"),
    
    dbc.Alert([
        html.H4("✅ Critical Fix Applied", className="alert-heading"),
        html.P([
            html.Strong("Fixed: "), "Delivery window calculation now correctly uses:",
            html.Br(),
            "• Earliest delivery = when LAST (youngest) stripped eggs reach MinTemp_prod",
            html.Br(),
            "• Latest delivery = when FIRST (oldest) stripped eggs reach MaxTemp_prod",
            html.Br(),
            "• Safety: Using math.ceil() for minimum degree-days to ensure quality",
        ])
    ], color="success", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H4("Data Source Selection", className="mb-3"),
            dcc.RadioItems(
                id="data-source-selector",
                options=[
                    {"label": " Upload Your Data (Start Here)", "value": "upload"},
                    {"label": " Load Example Data", "value": "example"}
                ],
                value="upload",
                labelStyle={"display": "block", "marginBottom": "10px"},
                className="mb-3"
            ),
            html.Div(
                id="upload-container",
                children=[
                    dbc.Alert([
                        html.Strong("📤 Upload Your Files"),
                        html.Br(),
                        "Please upload both Orders and Fish Groups files to continue.",
                        html.Br(),
                        "Supported formats: CSV, Excel (.xlsx, .xls)"
                    ], color="info", className="mb-3"),
                    
                    html.H5("Upload Orders File"),
                    dcc.Upload(
                        id="upload-orders",
                        children=html.Div([
                            "Drag and Drop or ",
                            html.A("Click to Select Orders File", style={"fontWeight": "bold"})
                        ]),
                        style={
                            'width': '100%', 'height': '80px', 'lineHeight': '80px',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 
                            'borderRadius': '10px', 'borderColor': '#007bff',
                            'textAlign': 'center', 'marginBottom': '15px',
                            'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),
                    html.Div(id="upload-orders-status", className="mb-3"),
                    
                    html.H5("Upload Fish Groups File"),
                    dcc.Upload(
                        id="upload-fish-groups",
                        children=html.Div([
                            "Drag and Drop or ",
                            html.A("Click to Select Fish Groups File", style={"fontWeight": "bold"})
                        ]),
                        style={
                            'width': '100%', 'height': '80px', 'lineHeight': '80px',
                            'borderWidth': '2px', 'borderStyle': 'dashed',
                            'borderRadius': '10px', 'borderColor': '#007bff',
                            'textAlign': 'center',
                            'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),
                    html.Div(id="upload-fish-groups-status", className="mb-3"),
                ],
                style={"display": "block"}
            )
        ], width=12)
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Orders (Input)", className="mt-4"),
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
                      color="primary", className="mt-2"),
            
            html.H3("Fish Groups (Input)", className="mt-4"),
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
                      color="primary", className="mt-2"),
        ], width=7),
        
        dbc.Col([
            html.H3("Problem Description & Constraints", className="mt-4"),
            dcc.Markdown("""
                ### Overview (Degree-Days Model - CORRECTED)
                This application allocates customer orders to fish egg groups using **degree-days** for temperature tracking.

                **🔧 FIXED - Delivery Window Calculation:**
                - **EarliestDelivery**: When LAST (youngest) stripped eggs reach MinTemp_prod
                - **LatestDelivery**: When FIRST (oldest) stripped eggs reach MaxTemp_prod
                - **Result**: ALL eggs in batch meet quality requirements simultaneously

                **Key Features:**
                - **MinTemp_customer / MaxTemp_customer**: Degree-days required/accepted by customer
                - **MinTemp_prod / MaxTemp_prod**: Degree-days when roe is sellable/overripe
                - **Delivery windows shown in capacity table**
                - Delivery feasibility calculated using corrected degree-day accumulation

                **Constraints:**
                - Each active order assigned once
                - **Gain uses ONLY Gain capacity; Shield uses ONLY Shield capacity**
                - **Delivery date must allow degree-day requirements to be met (CORRECTED)**
                - Organic orders only to organic groups
                - Locked group must be used
                - Elite/Nucleus only from Hemne
                - Hønsvikgulen delivers only to Lerøy customers
                
                **Technical Details:**
                - Uses `math.ceil()` for safety on minimum degree-days
                - Validates that delivery windows are physically possible
                - Default water temperature: 8°C
            """, className="p-3 bg-light rounded"),
            
            dbc.Alert(id="validation-alert", color="info", is_open=bool(initial_warnings),
                      children="; ".join(initial_warnings) if initial_warnings else "", 
                      duration=None, className="mt-2"),
            
            dbc.Button("Solve Allocation Problem", id="solve-button", n_clicks=0,
                       color="success", size="lg", className="mt-4", disabled=True),
            html.Div(id="solver-summary", className="mt-3"),
        ], width=5, className="p-4"),
    ], className="my-4"),
    
    dcc.Loading(
        id="loading", type="circle",
        children=html.Div(
            id="results-section", className="mt-4", style={"display": "none"},
            children=[
                html.H2("Solver Results (CORRECTED LOGIC)", className="text-center mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Filter Options", className="card-title"),
                                dbc.Checklist(
                                    id="filter-dummy-toggle",
                                    options=[{"label": " Show ONLY Dummy Assignments", "value": "dummy"}],
                                    value=[],
                                    switch=True,
                                    className="mb-2"
                                ),
                                html.P([
                                    html.Strong("💡 Filter Tips: "),
                                    "Use the switch above to show only dummy assignments, or type directly in the column filter boxes below."
                                ], className="text-muted small mb-0")
                            ])
                        ], className="mb-3", color="light")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H3("Order Assignments (Results)", className="mt-4"),
                        dash_table.DataTable(
                            id="results-table", 
                            filter_action='native',
                            page_action='native',
                            page_size=10,
                            sort_action='native',
                            **get_results_table_style()
                        ),
                        
                        html.H3("Remaining Capacity with Delivery Windows (Results)", className="mt-4"),
                        html.P([
                            html.Strong("📅 Delivery Windows (CORRECTED): "),
                            "EarliestDelivery = when youngest eggs are ready. ",
                            "LatestDelivery = when oldest eggs become overripe. ",
                            "This ensures ALL eggs meet quality standards."
                        ], className="text-info small"),
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
                        html.H3("Weekly Inventory Buffer", className="mt-4"),
                        dcc.Dropdown(id="buffer-group-filter", multi=True, placeholder="Filter groups..."),
                        dcc.Graph(id="buffer-visualization"),
                    ], width=12),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Download Results (CSV)", id="download-csv-button", 
                                  color="info", className="mt-2 me-2"),
                        dcc.Download(id="download-csv"),
                        dbc.Button("Download Excel (Results+Capacity)", id="download-xlsx-button", 
                                  color="secondary", className="mt-2"),
                        dcc.Download(id="download-xlsx"),
                    ], width=12)
                ], className="mt-3"),
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
    return dbc.Alert(f"✅ Loaded: {filename}", color="success", className="mb-2")


@app.callback(
    Output("upload-fish-groups-status", "children"),
    Input("upload-fish-groups", "contents"),
    State("upload-fish-groups", "filename")
)
def show_fish_groups_upload_status(contents, filename):
    if contents is None:
        return None
    return dbc.Alert(f"✅ Loaded: {filename}", color="success", className="mb-2")


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
        return True, "Please upload data files or load example data to begin.", True, "info"
    elif orders_empty:
        return True, "Orders table is empty. Please upload orders data.", True, "warning"
    elif groups_empty:
        return True, "Fish groups table is empty. Please upload fish groups data.", True, "warning"
    else:
        return False, "✅ Data loaded successfully. Ready to solve!", True, "success"


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
     Output("buffer-group-filter", "options"), Output("buffer-group-filter", "value")],
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
        return dash.no_update, display_results.to_dict("records"), results_cols, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
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
        button_text = f"Solve Allocation Problem (Status: {status})"

        return (
            {"margin": "20px", "display": "block"}, display_results.to_dict("records"),
            results_cols, capacity_df.to_dict("records"), capacity_cols, fig,
            button_text, filter_options, filter_value
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
        return dcc.send_data_frame(df.to_csv, "solver_results_degree_days_FIXED.csv", index=False)
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
                results_df.to_excel(writer, sheet_name="Results", index=False)
                capacity_df.to_excel(writer, sheet_name="Capacity", index=False)

        return dcc.send_bytes(to_excel, "solver_output_degree_days_FIXED.xlsx")
    return None


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8050))
    host = '0.0.0.0'
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    print(f"🔧 Starting Degree-Days Model server (DELIVERY WINDOW FIXED + UI FIXED) on {host}:{port}")
    app.run(host=host, port=port, debug=debug)