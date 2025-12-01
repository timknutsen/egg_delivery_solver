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
    
    Logic:
    1. Customer requirements must fit within production limits:
       - min_temp_customer >= min_temp_prod (roe must be ready)
       - max_temp_customer <= max_temp_prod (roe must not be overripe)
    
    2. Delivery date must allow sufficient degree-day accumulation:
       - Earliest delivery: when min_temp_customer is reached
       - Latest delivery: before max_temp_customer is exceeded
    
    Args:
        stripping_start: Date when stripping period begins
        stripping_stop: Date when stripping period ends
        min_temp_prod: Minimum degree-days for roe to be sellable
        max_temp_prod: Maximum degree-days before roe becomes overripe
        min_temp_customer: Minimum degree-days required by customer
        max_temp_customer: Maximum degree-days accepted by customer
        delivery_date: Requested delivery date
        water_temp: Average water temperature (default 8°C)
    
    Returns:
        bool: True if delivery is feasible, False otherwise
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
        return False  # Customer wants roe before it's ready
    
    if max_temp_customer > max_temp_prod:
        return False  # Customer accepts roe that's already overripe
    
    # Calculate degree-days from stripping to delivery
    # For eggs stripped at stripping_start, delivered at delivery_date
    days_from_earliest_strip = (delivery_date - stripping_start).days
    earliest_degree_days = days_from_earliest_strip * water_temp
    
    # For eggs stripped at stripping_stop, delivered at delivery_date
    days_from_latest_strip = (delivery_date - stripping_stop).days
    latest_degree_days = days_from_latest_strip * water_temp
    
    # If delivery is before stripping ends, use only the earliest strip date
    if delivery_date < stripping_stop:
        latest_degree_days = earliest_degree_days
    
    # Constraint 2: Delivery date must allow meeting customer requirements
    # The earliest stripped eggs must have at least min_temp_customer
    if earliest_degree_days < min_temp_customer:
        return False  # Even earliest eggs won't be ready by delivery
    
    # The latest stripped eggs must not exceed max_temp_customer
    if latest_degree_days > max_temp_customer:
        return False  # Even latest eggs will be too old by delivery
    
    # Additional check: ensure we're within production window
    if earliest_degree_days > max_temp_prod:
        return False  # All eggs will be overripe by delivery
    
    return True


def calculate_delivery_window_dates(
    stripping_start, stripping_stop,
    min_temp_prod, max_temp_prod,
    water_temp=DEFAULT_WATER_TEMP
):
    """
    Calculate the earliest and latest possible delivery dates for a fish group.
    
    Returns:
        (earliest_delivery, latest_delivery): Date range for deliveries
    """
    if pd.isna(stripping_start) or pd.isna(stripping_stop):
        return (pd.NaT, pd.NaT)
    
    if pd.isna(min_temp_prod) or pd.isna(max_temp_prod):
        return (pd.NaT, pd.NaT)
    
    # Earliest delivery: min_temp_prod days after stripping starts
    days_to_min = int(min_temp_prod / water_temp)
    earliest_delivery = stripping_start + timedelta(days=days_to_min)
    
    # Latest delivery: max_temp_prod days after stripping stops
    days_to_max = int(max_temp_prod / water_temp)
    latest_delivery = stripping_stop + timedelta(days=days_to_max)
    
    return (earliest_delivery, latest_delivery)


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

    # Ensure required columns exist
    required_cols = ["OrderNr","DeliveryDate","OrderStatus","CustomerID","CustomerName","Product",
                     "Organic","Volume","LockedSite","PreferredSite"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            warnings.append(f"Orders: Missing column '{col}' added with defaults.")

    # Convert to valid group keys dictionary for easy lookup
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

    # Convert data types
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Organic"] = df["Organic"].apply(process_organic)

    # Handle temperature columns - support both old and new naming
    if "MinTemp_customer" in df.columns:
        df["MinTemp_customer"] = pd.to_numeric(df["MinTemp_customer"], errors="coerce").fillna(300)
    elif "MinTemp" in df.columns:
        # Convert old MinTemp (Celsius) to degree-days (rough approximation)
        df["MinTemp_customer"] = pd.to_numeric(df["MinTemp"], errors="coerce").fillna(7) * 50
        warnings.append("Note: Converted MinTemp (°C) to MinTemp_customer (degree-days) using approximation (temp × 50)")
    else:
        df["MinTemp_customer"] = 300
        warnings.append("Orders: MinTemp_customer not found. Using default value of 300 degree-days.")
    
    if "MaxTemp_customer" in df.columns:
        df["MaxTemp_customer"] = pd.to_numeric(df["MaxTemp_customer"], errors="coerce").fillna(500)
    elif "MaxTemp" in df.columns:
        # Convert old MaxTemp (Celsius) to degree-days (rough approximation)
        df["MaxTemp_customer"] = pd.to_numeric(df["MaxTemp"], errors="coerce").fillna(9) * 50
        warnings.append("Note: Converted MaxTemp (°C) to MaxTemp_customer (degree-days) using approximation (temp × 50)")
    else:
        df["MaxTemp_customer"] = 500
        warnings.append("Orders: MaxTemp_customer not found. Using default value of 500 degree-days.")

    # Ensure MinTemp_customer <= MaxTemp_customer
    swap_mask = df["MinTemp_customer"] > df["MaxTemp_customer"]
    if swap_mask.any():
        df.loc[swap_mask, ["MinTemp_customer", "MaxTemp_customer"]] = df.loc[swap_mask, ["MaxTemp_customer", "MinTemp_customer"]].values
        warnings.append(f"Orders: Swapped {swap_mask.sum()} rows where MinTemp_customer > MaxTemp_customer")

    # Update site references to use group keys
    def map_site_to_group(site_value):
        if pd.isnull(site_value) or str(site_value).strip() == "":
            return ""
        site_str = str(site_value).strip()
        # If it's already a full group key, verify it exists
        if any(site_str == group_key for groups in valid_group_keys.values() for group_key in groups):
            return site_str
        # If it's a site name, try to find the first matching group key
        if site_str in valid_group_keys and valid_group_keys[site_str]:
            return valid_group_keys[site_str][0]
        # Otherwise, it's invalid
        return ""

    df["LockedSite"] = df["LockedSite"].apply(map_site_to_group)
    df["PreferredSite"] = df["PreferredSite"].apply(map_site_to_group)

    # Remove rows with invalid critical data
    initial_count = len(df)
    df = df.dropna(subset=["DeliveryDate"])
    df = df[df["Volume"] > 0]
    removed = initial_count - len(df)
    if removed > 0:
        warnings.append(f"Orders: Removed {removed} orders with invalid dates or zero/non-numeric volume.")

    # Fill missing statuses
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

    # Ensure essential columns exist
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

    # Convert data types - ensure column exists before attempting conversion
    date_columns = ["StrippingStartDate", "StrippingStopDate", "SalesStartDate", "SalesStopDate"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Handle degree-day columns
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
    
    # Validate degree-day constraints
    invalid = df[df["MinTemp_prod"] >= df["MaxTemp_prod"]]
    if not invalid.empty:
        warnings.append(f"Fish groups: {len(invalid)} groups have MinTemp_prod >= MaxTemp_prod. These may cause issues.")

    # Ensure numeric columns exist and are numeric
    df["Gain-eggs"] = pd.to_numeric(df["Gain-eggs"], errors="coerce").fillna(0)
    df["Shield-eggs"] = pd.to_numeric(df["Shield-eggs"], errors="coerce").fillna(0)

    # Process organic status
    df["Organic"] = df["Organic"].apply(process_organic)

    # Ensure GroupKey exists (use Site_Broodst_Season)
    if "GroupKey" not in df.columns:
        df["GroupKey"] = df["Site_Broodst_Season"]

    # Ensure Site column exists (if missing or blank)
    if "Site" not in df.columns or df["Site"].isna().all():
        df["Site"] = df["Site_Broodst_Season"].apply(lambda x: str(x).split("_")[0] if pd.notna(x) else "Unknown")
        warnings.append("Fish groups: 'Site' column not found or empty. Created from Site_Broodst_Season.")

    # Remove rows with invalid critical data
    required_columns = ["GroupKey", "Site_Broodst_Season", "Site"]
    initial_count = len(df)
    for col in required_columns:
        df = df[df[col].notna() & (df[col] != "")]
    # Drop duplicate groups
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
                # auto-detect delimiter
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

    # Validate data
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

        # Always allow Dummy so every order is assignable
        feasible[i].add(dummy_j)

        for j in all_fish_groups.index:
            if j == dummy_j:
                continue
            row = all_fish_groups.loc[j]

            # Organic feasibility
            if org_req and not bool(row["Organic"]):
                continue

            # Locked group
            if locked and row["Site_Broodst_Season"] != locked:
                continue

            # Elite/Nucleus site restriction
            if product in ["Elite", "Nucleus"] and row["Site"] != ELITE_NUCLEUS_SITE:
                continue

            # Hønsvikgulen -> Lerøy only
            if row["Site"] == HONSVIKGULEN_SITE and not cust_id.startswith(LEROY_CUSTOMER_PREFIX):
                continue

            # DEGREE-DAYS CONSTRAINT (NEW!)
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
    
    print(f"Applied {constraints_applied} degree-days exclusion constraints during feasibility building")
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
        
        # Check degree-days feasibility
        is_feasible = calculate_degree_days_feasibility(
            row["StrippingStartDate"], row["StrippingStopDate"],
            row["MinTemp_prod"], row["MaxTemp_prod"],
            min_temp_customer, max_temp_customer,
            ddate
        )
        if not is_feasible:
            continue
        temp_ok = True
        # If reached, there is at least one feasible group -> capacity must be limiting
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
        reasons.append("No groups within degree-days delivery window")
    return "; ".join(reasons) if reasons else "No feasible group"


# -------------------------------
# SOLVER FUNCTION
# -------------------------------
def solve_egg_allocation(orders, fish_groups):
    """
    Solves the egg allocation problem using PuLP with degree-days constraints.
    """
    # Create working copies
    orders = orders.copy()
    fish_groups = fish_groups.copy()

    # Filter active orders
    active_orders = orders[orders["OrderStatus"] != "Kansellert"].reset_index(drop=True)
    if active_orders.empty:
        status_str = "No active orders"
        # Prepare outputs
        results = orders.assign(AssignedGroup="No active orders", IsDummy=False, DummyReason="")
        capacity = fish_groups.assign(
            GainEggsUsed=0, ShieldEggsUsed=0, TotalEggsUsed=0,
            GainEggsRemaining=fish_groups["Gain-eggs"],
            ShieldEggsRemaining=fish_groups["Shield-eggs"],
            TotalEggsRemaining=fish_groups["Gain-eggs"] + fish_groups["Shield-eggs"]
        )
        final_capacity = capacity[[
            "Site_Broodst_Season", "Site", "SalesStartDate", "SalesStopDate",
            "MinTemp_prod", "MaxTemp_prod",
            "Gain-eggs", "Shield-eggs", "Organic", "GainEggsUsed", "ShieldEggsUsed",
            "TotalEggsUsed", "GainEggsRemaining", "ShieldEggsRemaining", "TotalEggsRemaining"
        ]]
        return {"status": status_str, "results": results, "remaining_capacity": final_capacity, "summary": {}}

    # Add dummy group for unassignable orders
    dummy_group = pd.DataFrame({
        "GroupKey": ["Dummy"], "Site": ["Dummy"], "Site_Broodst_Season": ["Dummy"],
        "StrippingStartDate": [pd.Timestamp("2000-01-01")], "StrippingStopDate": [pd.Timestamp("2100-12-31")],
        "SalesStartDate": [pd.Timestamp("2000-01-01")], "SalesStopDate": [pd.Timestamp("2100-12-31")],
        "MinTemp_prod": [0], "MaxTemp_prod": [10000],
        "Gain-eggs": [float("inf")], "Shield-eggs": [float("inf")], "Organic": [True]
    })
    all_fish_groups = pd.concat([fish_groups, dummy_group], ignore_index=True)

    # Build feasible pairs with degree-days constraints
    print("Building feasible pairs with degree-days constraints...")
    feasible = build_feasible_pairs(active_orders, all_fish_groups)
    prob = pl.LpProblem("FishEggAllocation", pl.LpMinimize)
    x = {(i, j): pl.LpVariable(f"x_{i}_{j}", cat="Binary") for i in active_orders.index for j in feasible[i]}

    # Precompute FIFO penalties and preferred mapping
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

    # Objective
    obj_terms = []
    for (i, j), var in x.items():
        is_dummy = 1 if j == dummy_j else 0
        pref_mismatch = 1 if (preferred_map[i] and all_fish_groups.loc[j, "Site_Broodst_Season"] != preferred_map[i]) else 0
        fifo_pen = group_fifo_pen.get(j, 0.0)
        obj_terms.append(W_DUMMY * var * is_dummy + W_PREF * var * pref_mismatch + W_FIFO * var * fifo_pen)
    prob += pl.lpSum(obj_terms)

    # Assignment: each order exactly once over feasible options
    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in feasible[i]) == 1

    # Capacity constraints
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] == "Dummy":
            continue
        Gcap = all_fish_groups.loc[j, "Gain-eggs"]
        Scap = all_fish_groups.loc[j, "Shield-eggs"]

        gain_i = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Gain" and j in feasible[i]]
        shield_i = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Shield" and j in feasible[i]]
        elite_i = [i for i in active_orders.index if active_orders.loc[i, "Product"] in ["Elite", "Nucleus"] and j in feasible[i]]

        prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in gain_i) <= Gcap
        prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in (gain_i + shield_i)) <= (Gcap + Scap)
        if elite_i:
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in elite_i) <= Gcap

    # Solve
    try:
        start = perf_counter()
        PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_SECONDS).solve(prob)
        solve_time = perf_counter() - start
        solver_status = pl.LpStatus[prob.status]
        print(f"Solver completed: {solver_status} in {solve_time:.2f}s")
    except Exception as e:
        solver_status = f"Error: {e}"
        solve_time = 0.0

    # Process results for active orders
    results = active_orders.copy()
    results["AssignedGroup"] = None
    results["IsDummy"] = False

    # Track usage
    gain_used = {j: 0.0 for j in all_fish_groups.index}
    shield_used = {j: 0.0 for j in all_fish_groups.index}
    elite_used = {j: 0.0 for j in all_fish_groups.index}

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
            if product == "Gain":
                gain_used[assigned_j] += volume
            elif product == "Shield":
                shield_used[assigned_j] += volume
            elif product in ["Elite", "Nucleus"]:
                elite_used[assigned_j] += volume

    # Merge results back into all orders
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

    # Add DummyReason for active orders
    res_active = all_res[all_res["OrderStatus"] != "Kansellert"].copy()
    res_active["DummyReason"] = ""
    for idx, r in res_active.iterrows():
        if bool(r["IsDummy"]):
            res_active.at[idx, "DummyReason"] = diagnose_dummy(r, fish_groups)
    all_res = all_res.merge(res_active[["OrderNr", "DummyReason"]], on="OrderNr", how="left")
    all_res["DummyReason"] = all_res["DummyReason"].fillna("")

    # Calculate remaining capacity
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
                    total_gain_used = gain_used[j] + elite_used[j]
                    capacity.loc[idx[0], "GainEggsUsed"] = total_gain_used
                    capacity.loc[idx[0], "ShieldEggsUsed"] = shield_used[j]
                    capacity.loc[idx[0], "TotalEggsUsed"] = total_gain_used + shield_used[j]
                    capacity.loc[idx[0], "GainEggsRemaining"] = max(0.0, capacity.loc[idx[0], "Gain-eggs"] - total_gain_used)
                    capacity.loc[idx[0], "ShieldEggsRemaining"] = max(0.0, capacity.loc[idx[0], "Shield-eggs"] - shield_used[j])
                    capacity.loc[idx[0], "TotalEggsRemaining"] = max(0.0, (capacity.loc[idx[0], "Gain-eggs"] + capacity.loc[idx[0], "Shield-eggs"]) - total_gain_used - shield_used[j])

    final_capacity = capacity[[
        "Site_Broodst_Season", "Site", "SalesStartDate", "SalesStopDate",
        "MinTemp_prod", "MaxTemp_prod",
        "Gain-eggs", "Shield-eggs", "Organic", "GainEggsUsed", "ShieldEggsUsed",
        "TotalEggsUsed", "GainEggsRemaining", "ShieldEggsRemaining", "TotalEggsRemaining"
    ]].copy()

    # Summary
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
    Creates a visualization of weekly inventory buffer over time,
    grouped by Site_Broodst_Season.
    """
    if remaining is None or results is None or len(remaining) == 0 or "Site_Broodst_Season" not in remaining.columns:
        return px.line(title="No capacity data available")

    df = remaining.copy()
    df["SalesStartDate"] = pd.to_datetime(df["SalesStartDate"])
    df["SalesStopDate"] = pd.to_datetime(df["SalesStopDate"])

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
        title="Weekly Inventory Buffer by Fish Group (Degree-Days Model)", markers=True
    )
    fig.update_layout(
        xaxis_title="Week", yaxis_title="Remaining Capacity (Millions)",
        template="plotly_white", font=dict(family="Arial, sans-serif", size=14),
        legend_title_text="Fish Group", margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_yaxes(tickformat=".2f")
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    return fig


def get_table_style():
    """Enhanced styling for tables."""
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
# INITIAL DATA LOAD (Example Data)
# -------------------------------
orders_data, fish_groups_data, initial_warnings = load_validated_data()

# -------------------------------
# DASH APP SETUP
# -------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Fish Egg Allocation Solver - Degree-Days Model"

app.layout = dbc.Container([
    html.H1("Fish Egg Allocation Solver (Degree-Days Model)", className="text-center my-4 py-3 bg-primary text-white rounded"),
    dbc.Row([
        dbc.Col([
            dcc.RadioItems(
                id="data-source-selector",
                options=[
                    {"label": "Load Example Data", "value": "example"},
                    {"label": "Upload Your Data", "value": "upload"}
                ],
                value="example",
                labelStyle={"display": "inline-block", "marginRight": "20px"}
            ),
            html.Div(
                id="upload-container",
                children=[
                    html.H4("Upload Orders File"),
                    dcc.Upload(
                        id="upload-orders",
                        children=html.Div(["Drag and Drop or ", html.A("Select Orders File")]),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                               'textAlign': 'center', 'marginBottom': '10px'},
                        multiple=False
                    ),
                    html.H4("Upload Fish Groups File"),
                    dcc.Upload(
                        id="upload-fish-groups",
                        children=html.Div(["Drag and Drop or ", html.A("Select Fish Groups File")]),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                               'textAlign': 'center'},
                        multiple=False
                    )
                ],
                style={"display": "none"}
            )
        ], width=12)
    ], className="my-3"),
    dbc.Row([
        dbc.Col([
            html.H3("Orders", className="mt-4"),
            dash_table.DataTable(
                id="order-table",
                data=orders_data.to_dict("records"),
                columns=build_order_columns_schema(fish_groups_data["Site_Broodst_Season"].unique()),
                editable=True, row_deletable=True, page_action='native', page_current=0,
                page_size=10, sort_action='native', filter_action='native',
                dropdown=build_order_dropdowns(fish_groups_data["Site_Broodst_Season"].unique()),
                fixed_rows={"headers": True}, **get_table_style()
            ),
            dbc.Button("Add Order Row", id="add-order-row-button", n_clicks=0, color="primary", className="mt-2"),
            html.H3("Fish Groups", className="mt-4"),
            dash_table.DataTable(
                id="fish-group-table",
                data=fish_groups_data.to_dict("records"),
                columns=[{"name": col, "id": col, "editable": True} for col in fish_groups_data.columns],
                editable=True, row_deletable=True, page_action='native', page_current=0,
                page_size=10, sort_action='native', filter_action='native',
                fixed_rows={"headers": True}, **get_table_style()
            ),
            dbc.Button("Add Fish Group Row", id="add-fish-group-row-button", n_clicks=0, color="primary", className="mt-2"),
        ], width=7),
        dbc.Col([
            html.H3("Problem Description & Constraints", className="mt-4"),
            dcc.Markdown("""
                ### Overview (Degree-Days Model)
                This application allocates customer orders to fish egg groups using **degree-days** for temperature tracking.

                **Key Changes:**
                - **MinTemp_customer / MaxTemp_customer**: Degree-days required/accepted by customer
                - **MinTemp_prod / MaxTemp_prod**: Degree-days when roe is sellable/overripe
                - Delivery feasibility calculated using degree-day accumulation

                **Constraints:**
                - Each active order assigned once
                - Gain uses Gain capacity; Shield uses Shield plus leftover Gain
                - **Delivery date must allow degree-day requirements to be met**
                - Organic orders only to organic groups
                - Locked group must be used
                - Elite/Nucleus only from Hemne
                - Hønsvikgulen delivers only to Lerøy customers
                - FIFO preference (earlier stripping stop favored)
                
                **Degree-Days Logic:**
                - Customer requirements must fit within production limits
                - Delivery date must allow sufficient time for degree-day accumulation
                - Default water temperature: 8°C
            """, className="p-3 bg-light rounded"),
            dbc.Alert(id="validation-alert", color="warning", is_open=bool(initial_warnings),
                      children="; ".join(initial_warnings) if initial_warnings else "", duration=9000, className="mt-2"),
            dbc.Button("Solve Allocation Problem", id="solve-button", n_clicks=0,
                       color="success", size="lg", className="mt-4"),
            html.Div(id="solver-summary", className="mt-3"),
        ], width=5, className="p-4"),
    ], className="my-4"),
    dcc.Loading(
        id="loading", type="circle",
        children=html.Div(
            id="results-section", className="mt-4", style={"display": "none"},
            children=[
                html.H2("Solver Results", className="text-center mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.H3("Order Assignments", className="mt-4"),
                        dash_table.DataTable(id="results-table", **get_table_style()),
                        html.H3("Remaining Capacity", className="mt-4"),
                        dash_table.DataTable(id="capacity-table", **get_table_style()),
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
                        dbc.Button("Download Results (CSV)", id="download-csv-button", color="info", className="mt-2 me-2"),
                        dcc.Download(id="download-csv"),
                        dbc.Button("Download Excel (Results+Capacity)", id="download-xlsx-button", color="secondary", className="mt-2"),
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

    if triggered_id in ["upload-orders", "data-source-selector"]:
        if data_source == "upload" and uploaded_contents is not None:
            df, _ = parse_contents(uploaded_contents, filename)
            if df is not None:
                fg_valid, _ = validate_fish_groups(fg_df)
                df_valid, _ = validate_orders(df, fg_valid)
                return df_valid.to_dict("records"), columns_schema
        else:
            odf, _, _ = load_validated_data()
            return odf.to_dict("records"), columns_schema
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

    if triggered_id in ["upload-fish-groups", "data-source-selector"]:
        if data_source == "upload" and uploaded_contents is not None:
            df, _ = parse_contents(uploaded_contents, filename)
            if df is not None:
                df_valid, _ = validate_fish_groups(df)
                return df_valid.to_dict("records")
        else:
            _, fish_groups_df, _ = load_validated_data()
            return fish_groups_df.to_dict("records")
    elif triggered_id == "add-fish-group-row-button" and n_clicks > 0:
        if not table_data:
            _, fish_groups_df, _ = load_validated_data()
            columns = fish_groups_df.columns
            new_row = {col: "" for col in columns}
        else:
            new_row = {col: "" for col in table_data[0].keys()}
        table_data.append(new_row)

    return table_data

@app.callback(
    [Output("results-section", "style"),
     Output("results-table", "data"), Output("results-table", "columns"),
     Output("capacity-table", "data"), Output("capacity-table", "columns"),
     Output("buffer-visualization", "figure"), Output("solve-button", "children"),
     Output("validation-alert", "children"), Output("validation-alert", "is_open"),
     Output("validation-alert", "color"), Output("solver-summary", "children"),
     Output("buffer-group-filter", "options"), Output("buffer-group-filter", "value")],
    [Input("solve-button", "n_clicks")],
    [State("order-table", "data"), State("fish-group-table", "data"),
     State("buffer-group-filter", "value")],
    prevent_initial_call=True
)
def update_results(n_clicks, order_data, fish_group_data, selected_groups):
    if n_clicks > 0:
        orders_df = pd.DataFrame(order_data)
        fish_groups_df = pd.DataFrame(fish_group_data)
        fish_groups_df, w1 = validate_fish_groups(fish_groups_df)
        orders_df, w2 = validate_orders(orders_df, fish_groups_df)
        warnings = w1 + w2

        solution = solve_egg_allocation(orders_df, fish_groups_df)
        results_df = solution["results"]
        capacity_df = solution["remaining_capacity"]
        status = solution["status"]

        results_cols = [{"name": c, "id": c} for c in results_df.columns]
        capacity_cols = [{"name": c, "id": c} for c in capacity_df.columns]

        group_options = sorted(capacity_df["Site_Broodst_Season"].dropna().unique().tolist()) if not capacity_df.empty else []
        filter_options = [{"label": g, "value": g} for g in group_options]
        filter_value = selected_groups if selected_groups else []

        fig = create_buffer_graph(capacity_df, results_df, selected_groups=filter_value)
        button_text = f"Solve Allocation Problem (Status: {status})"
        alert_text = "; ".join(warnings) if warnings else "Optimization completed successfully!"
        alert_open = True
        alert_color = "warning" if warnings else "success"

        sumd = solution.get("summary", {})
        summary_children = html.Ul([
            html.Li(f"Active orders: {sumd.get('total_orders', 0)}"),
            html.Li(f"Dummy assignments: {sumd.get('dummy_orders', 0)}"),
            html.Li(f"Organic orders: {sumd.get('organic_orders', 0)}"),
            html.Li(f"Solve time: {sumd.get('solve_time_sec', 0):.2f}s"),
            html.Li(f"Status: {sumd.get('status', 'N/A')}"),
            html.Li(f"Binaries: {sumd.get('binaries', 0)}"),
        ])

        return (
            {"margin": "20px", "display": "block"}, results_df.to_dict("records"),
            results_cols, capacity_df.to_dict("records"), capacity_cols, fig,
            button_text, alert_text, alert_open, alert_color, summary_children,
            filter_options, filter_value
        )
    return ({"display": "none"}, [], [], [], [], {}, "Solve Allocation Problem", "", False, "warning", "", [], [])

@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-button", "n_clicks"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def export_results_csv(n_clicks, results_data):
    if n_clicks:
        df = pd.DataFrame(results_data)
        return dcc.send_data_frame(df.to_csv, "solver_results_degree_days.csv", index=False)
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

        return dcc.send_bytes(to_excel, "solver_output_degree_days.xlsx")
    return None

# -------------------------------
# RUN LOCAL OR VIA RENDER
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8050))
    host = '0.0.0.0'
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    print(f"Starting Degree-Days Model server on {host}:{port} with debug={debug}")
    app.run(host=host, port=port, debug=debug)
