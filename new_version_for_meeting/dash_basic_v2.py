import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pulp as pl
from datetime import timedelta
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
# Define file paths for the updated example data
ORDERS_DATA_PATH = "example_data/orders_example_updated.csv"
FISH_GROUPS_DATA_PATH = "example_data/fish_groups_example_updated.csv"

# Define specific constraint parameters (ADJUST THESE AS NEEDED)
ELITE_NUCLEUS_SITE = "Hemne"  # Site(s) allowed for Elite/Nucleus
HONSVIKGULEN_SITE = "Hønsvikgulen"  # Site with customer restriction
LERØY_CUSTOMER_PREFIX = "CUST001"  # CustomerID prefix for Lerøy companies

# -------------------------------
# HELPER & VALIDATION FUNCTIONS
# -------------------------------

def process_organic(x):
    """Converts 'Organic' text or boolean to boolean."""
    if isinstance(x, bool):
        return x
    if pd.isnull(x) or str(x).strip() == "":
        return False
    elif str(x).strip().lower() == "organic":
        return True
    else:
        print(f"Warning: Invalid Organic value '{x}' found. Setting to False.")
        return False

def validate_orders(df, valid_groups):
    """
    Validate and preprocess the orders DataFrame.
    - Ensures required columns exist
    - Parses dates, numbers, organic status
    - Validates site references against valid groups
    """
    if df.empty:
        print("Warning: Orders DataFrame is empty.")
        return df
    
    # Convert to valid group keys dictionary for easy lookup
    valid_group_keys = {}
    for _, row in valid_groups.iterrows():
        site = row["Site"]
        group_key = row["Site_Broodst_Season"]
        if site not in valid_group_keys:
            valid_group_keys[site] = []
        valid_group_keys[site].append(group_key)
    
    df = df.copy()
    
    # Convert data types
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Organic"] = df["Organic"].apply(process_organic)
    df["MinTemp"] = pd.to_numeric(df["MinTemp"], errors="coerce").fillna(7)
    df["MaxTemp"] = pd.to_numeric(df["MaxTemp"], errors="coerce").fillna(9)
    
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
            print(f"Note: Mapping site '{site_str}' to first matching group key '{valid_group_keys[site_str][0]}'")
            return valid_group_keys[site_str][0]
        # Otherwise, it's invalid
        print(f"Warning: Site value '{site_str}' not found in valid sites or group keys.")
        return ""
    
    df["LockedSite"] = df["LockedSite"].apply(map_site_to_group)
    df["PreferredSite"] = df["PreferredSite"].apply(map_site_to_group)
    
    # Remove rows with invalid critical data
    initial_count = len(df)
    df = df.dropna(subset=["DeliveryDate"])
    df = df[df["Volume"] > 0]
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} orders with invalid dates or zero volume.")
    
    return df

def validate_fish_groups(df):
    """
    Validate and preprocess the fish_groups DataFrame.
    - Parses dates, numbers, organic status
    - Ensures required columns exist
    """
    if df.empty:
        print("Warning: Fish Groups DataFrame is empty.")
        return df
    
    df = df.copy()
    
    # Convert data types
    date_columns = ["StrippingStartDate", "StrippingStopDate", "SalesStartDate", "SalesStopDate"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    
    df["Gain-eggs"] = pd.to_numeric(df["Gain-eggs"], errors="coerce").fillna(0)
    df["Shield-eggs"] = pd.to_numeric(df["Shield-eggs"], errors="coerce").fillna(0)
    df["Organic"] = df["Organic"].apply(process_organic)
    
    # Ensure GroupKey exists (use Site_Broodst_Season)
    if "GroupKey" not in df.columns:
        df["GroupKey"] = df["Site_Broodst_Season"]
    
    # Remove rows with invalid critical data
    initial_count = len(df)
    df = df.dropna(subset=date_columns + ["GroupKey"])
    df = df.drop_duplicates(subset=["GroupKey"], keep='first')  # Ensure unique groups
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} fish groups with invalid dates or duplicates.")
    
    return df

# -------------------------------
# DELIVERY WINDOW CALCULATION (PLACEHOLDER)
# -------------------------------

def calculate_delivery_window(stripping_start, stripping_stop, min_temp, max_temp):
    """
    PLACEHOLDER: Calculates the valid delivery window based on stripping dates and temperature.
    TODO: Replace this with the actual logic from the Klekkekalulator Excel files.
    """
    if pd.isna(stripping_start) or pd.isna(stripping_stop):
        return (pd.NaT, pd.NaT)
    
    # Example placeholder logic - very simplified approximation
    avg_temp = (min_temp + max_temp) / 2
    
    # Rough approximation of degree-days to hatch
    degree_days = 350  # Example value
    min_days = int(degree_days / max_temp) if max_temp > 0 else 50  # Fast development
    max_days = int(degree_days / min_temp) if min_temp > 0 else 90  # Slow development
    
    delivery_start = stripping_start + timedelta(days=min_days)
    delivery_stop = stripping_stop + timedelta(days=max_days)
    
    return (delivery_start, delivery_stop)

# -------------------------------
# DATA LOADING FUNCTION
# -------------------------------

def load_validated_data():
    """Loads and validates data from the specified CSV files."""
    try:
        orders_df = pd.read_csv(ORDERS_DATA_PATH)
        print(f"Loaded {len(orders_df)} orders from {ORDERS_DATA_PATH}")
    except Exception as e:
        print(f"Error loading {ORDERS_DATA_PATH}: {e}")
        orders_df = pd.DataFrame({
            "OrderNr": [],
            "DeliveryDate": [],
            "OrderStatus": [],
            "CustomerID": [],
            "CustomerName": [],
            "Product": [],
            "Organic": [],
            "Volume": [],
            "LockedSite": [],
            "PreferredSite": [],
            "MinTemp": [],
            "MaxTemp": []
        })
    
    try:
        fish_groups_df = pd.read_csv(FISH_GROUPS_DATA_PATH)
        print(f"Loaded {len(fish_groups_df)} fish groups from {FISH_GROUPS_DATA_PATH}")
    except Exception as e:
        print(f"Error loading {FISH_GROUPS_DATA_PATH}: {e}")
        fish_groups_df = pd.DataFrame({
            "Site": [],
            "Site_Broodst_Season": [],
            "StrippingStartDate": [],
            "StrippingStopDate": [],
            "SalesStartDate": [],
            "SalesStopDate": [],
            "Gain-eggs": [],
            "Shield-eggs": [],
            "Organic": []
        })
    
    # Validate data
    fish_groups_df = validate_fish_groups(fish_groups_df)
    orders_df = validate_orders(orders_df, fish_groups_df)
    
    return orders_df, fish_groups_df

# -------------------------------
# SOLVER FUNCTION
# -------------------------------

def solve_egg_allocation(orders, fish_groups):
    """
    Solves the egg allocation problem using PuLP.
    - Uses Site_Broodst_Season (GroupKey) as the primary identifier
    - Implements temperature-based delivery window calculation
    - Adds special constraints for Elite/Nucleus products and Hønsvikgulen
    """
    # Create working copies
    orders = orders.copy()
    fish_groups = fish_groups.copy()
    
    # Filter active orders
    active_orders = orders[orders["OrderStatus"] != "Kansellert"].reset_index(drop=True)
    if active_orders.empty:
        print("No active orders to process.")
        return {
            "status": "No active orders",
            "results": orders.assign(AssignedGroup="No active orders", IsDummy=False),
            "remaining_capacity": fish_groups.assign(
                TotalEggs=fish_groups["Gain-eggs"] + fish_groups["Shield-eggs"],
                TotalEggsUsed=0,
                TotalEggsRemaining=fish_groups["Gain-eggs"] + fish_groups["Shield-eggs"],
                GainEggsUsed=0,
                ShieldEggsUsed=0,
                GainEggsRemaining=fish_groups["Gain-eggs"],
                ShieldEggsRemaining=fish_groups["Shield-eggs"]
            )
        }
    
    # Add dummy group for unassignable orders
    dummy_group = pd.DataFrame({
        "GroupKey": ["Dummy"],
        "Site": ["Dummy"],
        "Site_Broodst_Season": ["Dummy"],
        "StrippingStartDate": [pd.Timestamp("2000-01-01")],
        "StrippingStopDate": [pd.Timestamp("2100-12-31")],
        "SalesStartDate": [pd.Timestamp("2000-01-01")],
        "SalesStopDate": [pd.Timestamp("2100-12-31")],
        "Gain-eggs": [float("inf")],
        "Shield-eggs": [float("inf")],
        "Organic": [True]
    })
    
    all_fish_groups = pd.concat([fish_groups, dummy_group], ignore_index=True)
    
    # Problem setup
    prob = pl.LpProblem("FishEggAllocation", pl.LpMinimize)
    
    # Decision variables: x[i, j] = 1 if order i is assigned to group j
    x = {}
    for i in active_orders.index:
        for j in all_fish_groups.index:
            x[i, j] = pl.LpVariable(f"x_{i}_{j}", cat="Binary")
    
    # Penalty variables
    dummy_penalty = {i: pl.LpVariable(f"dummy_penalty_{i}", lowBound=0) for i in active_orders.index}
    pref_penalty = {}
    for i, row in active_orders.iterrows():
        if pd.notna(row["PreferredSite"]) and row["PreferredSite"] != "":
            pref_penalty[i] = pl.LpVariable(f"pref_penalty_{i}", lowBound=0)
    
    # FIFO penalty (based on StrippingStopDate)
    real_groups = all_fish_groups[all_fish_groups["Site"] != "Dummy"]
    if not real_groups.empty:
        earliest_stop = real_groups["StrippingStopDate"].min()
        group_penalty = {}
        for j in all_fish_groups.index:
            if all_fish_groups.loc[j, "Site"] != "Dummy":
                delta = (all_fish_groups.loc[j, "StrippingStopDate"] - earliest_stop).days
                group_penalty[j] = float(max(0, delta))
            else:
                group_penalty[j] = 0.0
    else:
        group_penalty = {j: 0.0 for j in all_fish_groups.index}
    
    fifo_weight = 0.01
    
    # Objective function
    prob += (
        1000 * pl.lpSum(dummy_penalty.values()) +  # High penalty for dummy
        10 * pl.lpSum(pref_penalty.values()) +    # Medium penalty for preferred site
        pl.lpSum(x[i, j] * group_penalty[j] * fifo_weight  # Small FIFO penalty
                for i in active_orders.index for j in all_fish_groups.index)
    )
    
    # Constraint 1: Each order assigned exactly once
    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in all_fish_groups.index) == 1
    
    # Constraint 2: Capacity constraints
    gain_orders = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Gain"]
    shield_orders = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Shield"]
    elite_nucleus_orders = [i for i in active_orders.index if active_orders.loc[i, "Product"] in ["Elite", "Nucleus"]]
    
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] != "Dummy":
            Gcap = all_fish_groups.loc[j, "Gain-eggs"]
            Scap = all_fish_groups.loc[j, "Shield-eggs"]
            
            # Gain orders limited by Gain capacity
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in gain_orders) <= Gcap
            
            # Total regular orders (Gain+Shield) limited by total capacity
            regular_orders = gain_orders + shield_orders
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in regular_orders) <= (Gcap + Scap)
            
            # Elite/Nucleus orders use their own capacity or can go to Dummy
            # For simplicity, we assume they're allocated from the same pool as Gain
            if elite_nucleus_orders:
                prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in elite_nucleus_orders) <= Gcap
    
    # Constraint 3: Organic requirement
    for i in active_orders.index:
        if active_orders.loc[i, "Organic"]:
            for j in all_fish_groups.index:
                if (not all_fish_groups.loc[j, "Organic"]) and all_fish_groups.loc[j, "Site"] != "Dummy":
                    prob += x[i, j] == 0
    
    # Constraint 4: Locked site requirement
    for i in active_orders.index:
        locked_site = active_orders.loc[i, "LockedSite"]
        if pd.notna(locked_site) and locked_site != "":
            for j in all_fish_groups.index:
                group_key = all_fish_groups.loc[j, "Site_Broodst_Season"]
                if group_key != locked_site and all_fish_groups.loc[j, "Site"] != "Dummy":
                    prob += x[i, j] == 0
    
    # Constraint 5: Dummy penalty
    dummy_idx = all_fish_groups[all_fish_groups["Site"] == "Dummy"].index
    if not dummy_idx.empty:
        for i in active_orders.index:
            prob += dummy_penalty[i] >= x[i, dummy_idx[0]]
    
    # Constraint 6: Preferred site penalty
    for i in pref_penalty:
        pref_site = active_orders.loc[i, "PreferredSite"]
        for j in all_fish_groups.index:
            group_key = all_fish_groups.loc[j, "Site_Broodst_Season"]
            if group_key != pref_site and all_fish_groups.loc[j, "Site"] != "Dummy":
                prob += pref_penalty[i] >= x[i, j]
    
    # Constraint 7: Temperature-based delivery window
    for i in active_orders.index:
        ddate = active_orders.loc[i, "DeliveryDate"]
        min_temp = active_orders.loc[i, "MinTemp"]
        max_temp = active_orders.loc[i, "MaxTemp"]
        
        for j in all_fish_groups.index:
            if all_fish_groups.loc[j, "Site"] != "Dummy":
                strip_start = all_fish_groups.loc[j, "StrippingStartDate"]
                strip_stop = all_fish_groups.loc[j, "StrippingStopDate"]
                delivery_start, delivery_stop = calculate_delivery_window(strip_start, strip_stop, min_temp, max_temp)
                
                if pd.isna(delivery_start) or pd.isna(delivery_stop) or pd.isna(ddate) or ddate < delivery_start or ddate > delivery_stop:
                    prob += x[i, j] == 0
    
    # Constraint 8: Elite/Nucleus only from specific sites
    for i in active_orders.index:
        if active_orders.loc[i, "Product"] in ["Elite", "Nucleus"]:
            for j in all_fish_groups.index:
                if all_fish_groups.loc[j, "Site"] != ELITE_NUCLEUS_SITE and all_fish_groups.loc[j, "Site"] != "Dummy":
                    prob += x[i, j] == 0
    
    # Constraint 9: Hønsvikgulen only for Lerøy customers
    # Check by CustomerID prefix instead of exact match
    honsvikgulen_idx = all_fish_groups[all_fish_groups["Site"] == HONSVIKGULEN_SITE].index
    for i in active_orders.index:
        customer_id = active_orders.loc[i, "CustomerID"]
        is_leroy = isinstance(customer_id, str) and customer_id.startswith(LERØY_CUSTOMER_PREFIX)
        if not is_leroy:
            for j in honsvikgulen_idx:
                prob += x[i, j] == 0
    
    # Solve the problem
    try:
        prob.solve()
        solver_status = pl.LpStatus[prob.status]
    except Exception as e:
        print(f"Solver error: {e}")
        solver_status = "Error"
    
    # Process results for active orders
    results = active_orders.copy()
    results["AssignedGroup"] = None
    results["IsDummy"] = False
    
    # Track usage
    gain_used = {j: 0.0 for j in all_fish_groups.index}
    shield_used = {j: 0.0 for j in all_fish_groups.index}
    elite_used = {j: 0.0 for j in all_fish_groups.index}  # Track Elite/Nucleus separately
    
    for i in results.index:
        for j in all_fish_groups.index:
            if pl.value(x[i, j]) == 1:
                results.loc[i, "AssignedGroup"] = all_fish_groups.loc[j, "Site_Broodst_Season"]
                results.loc[i, "IsDummy"] = (all_fish_groups.loc[j, "Site"] == "Dummy")
                
                # Track usage
                if all_fish_groups.loc[j, "Site"] != "Dummy":
                    volume = results.loc[i, "Volume"]
                    product = results.loc[i, "Product"]
                    if product == "Gain":
                        gain_used[j] += volume
                    elif product == "Shield":
                        shield_used[j] += volume
                    elif product in ["Elite", "Nucleus"]:
                        elite_used[j] += volume  # Track separately for reporting
                break
    
    # Merge results back into all orders
    all_res = orders.copy()
    all_res["AssignedGroup"] = None
    all_res["IsDummy"] = False
    
    # Map results back using OrderNr as the key
    order_results_map = {row["OrderNr"]: (row["AssignedGroup"], row["IsDummy"]) for _, row in results.iterrows()}
    
    for i, row in all_res.iterrows():
        order_nr = row["OrderNr"]
        if order_nr in order_results_map:
            all_res.loc[i, "AssignedGroup"] = order_results_map[order_nr][0]
            all_res.loc[i, "IsDummy"] = order_results_map[order_nr][1]
        elif row["OrderStatus"] == "Kansellert":
            all_res.loc[i, "AssignedGroup"] = "Skipped-Cancelled"
            all_res.loc[i, "IsDummy"] = False
    
    # Calculate remaining capacity
    capacity = fish_groups.copy()
    capacity["GainEggsUsed"] = 0
    capacity["ShieldEggsUsed"] = 0
    capacity["TotalEggsUsed"] = 0
    capacity["GainEggsRemaining"] = capacity["Gain-eggs"]
    capacity["ShieldEggsRemaining"] = capacity["Shield-eggs"]
    capacity["TotalEggsRemaining"] = capacity["Gain-eggs"] + capacity["Shield-eggs"]
    
    # Update used/remaining values based on assignments
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] != "Dummy":
            site_broodst_season = all_fish_groups.loc[j, "Site_Broodst_Season"]
            if site_broodst_season in capacity["Site_Broodst_Season"].values:
                idx = capacity[capacity["Site_Broodst_Season"] == site_broodst_season].index
                if len(idx) > 0:
                    # Include Elite/Nucleus usage in Gain usage for accounting
                    total_gain_used = gain_used[j] + elite_used[j]
                    capacity.loc[idx[0], "GainEggsUsed"] = total_gain_used
                    capacity.loc[idx[0], "ShieldEggsUsed"] = shield_used[j]
                    capacity.loc[idx[0], "TotalEggsUsed"] = total_gain_used + shield_used[j]
                    capacity.loc[idx[0], "GainEggsRemaining"] = max(0, capacity.loc[idx[0], "Gain-eggs"] - total_gain_used)
                    capacity.loc[idx[0], "ShieldEggsRemaining"] = max(0, capacity.loc[idx[0], "Shield-eggs"] - shield_used[j])
                    capacity.loc[idx[0], "TotalEggsRemaining"] = max(0, capacity.loc[idx[0], "TotalEggsRemaining"] - total_gain_used - shield_used[j])
    
    # Filter columns for the final remaining capacity
    final_capacity = capacity[[
        "Site_Broodst_Season", "Site", 
        "SalesStartDate", "SalesStopDate", 
        "Gain-eggs", "Shield-eggs", "Organic",
        "GainEggsUsed", "ShieldEggsUsed", "TotalEggsUsed",
        "GainEggsRemaining", "ShieldEggsRemaining", "TotalEggsRemaining"
    ]]
    
    return {
        "status": solver_status,
        "results": all_res,
        "remaining_capacity": final_capacity
    }

# -------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------

def create_buffer_graph(remaining, results):
    """
    Creates a visualization of weekly inventory buffer over time,
    grouped by Site_Broodst_Season.
    """
    if remaining.empty or "Site_Broodst_Season" not in remaining.columns:
        return px.line(title="No capacity data available")
    
    # Create a copy of the remaining capacity data
    df = remaining.copy()
    df["SalesStartDate"] = pd.to_datetime(df["SalesStartDate"])
    df["SalesStopDate"] = pd.to_datetime(df["SalesStopDate"])
    
    # Prepare data for weekly tracking
    group_keys = df["Site_Broodst_Season"].unique()
    start_date = df["SalesStartDate"].min()
    end_date = df["SalesStopDate"].max() + pd.Timedelta(weeks=4)
    if pd.isna(start_date) or pd.isna(end_date):
        return px.line(title="Invalid date range")
    
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    
    # Initial capacity map
    initial_capacity = {group: row["GainEggsRemaining"] + row["ShieldEggsRemaining"] + row["TotalEggsUsed"]
                        for group, row in df.set_index("Site_Broodst_Season").iterrows()}
    
    # Prepare data for buffer calculation
    results_copy = results.copy()
    results_copy["DeliveryDate"] = pd.to_datetime(results_copy["DeliveryDate"])
    
    buffer_data = []
    
    for week in weekly_dates:
        for group in group_keys:
            # Get orders assigned to this group with delivery dates up to this week
            assigned_orders = results_copy[
                (results_copy["AssignedGroup"] == group) & 
                (results_copy["DeliveryDate"] <= week) & 
                (results_copy["IsDummy"] == False)
            ]
            
            # Calculate volume used up to this week
            used_volume = assigned_orders["Volume"].sum()
            
            # Calculate remaining capacity
            initial_vol = initial_capacity.get(group, 0)
            remaining_vol = max(0, initial_vol - used_volume)
            
            buffer_data.append({
                "Week": week,
                "Group": group,
                "RemainingCapacity": remaining_vol / 1e6  # Convert to millions
            })
    
    if not buffer_data:
        return px.line(title="No buffer data generated")
    
    buffer_df = pd.DataFrame(buffer_data)
    
    # Create the visualization
    fig = px.line(
        buffer_df,
        x="Week",
        y="RemainingCapacity",
        color="Group",
        title="Weekly Inventory Buffer by Fish Group",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Remaining Capacity (Millions)",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=14),
        legend_title_text="Fish Group",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    
    return fig

def get_table_style():
    """Enhanced styling for tables."""
    return {
        "style_table": {
            "overflowX": "auto",
            "maxHeight": "400px",
            "overflowY": "auto",
            "borderRadius": "10px"
        },
        "style_cell": {
            "minWidth": "90px",
            "width": "120px",
            "maxWidth": "180px",
            "whiteSpace": "normal",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "textAlign": "left",
            "padding": "10px",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "14px",
            "border": "1px solid #e0e0e0",
        },
        "style_header": {
            "backgroundColor": "#007bff",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
            "padding": "10px",
            "borderBottom": "2px solid #0056b3",
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            {"if": {"state": "selected"}, "backgroundColor": "#cce5ff", "border": "1px solid #007bff"},
            {
                "if": {
                    "filter_query": '{AssignedGroup} != "Skipped-Cancelled" and {AssignedGroup} != "Dummy" and {AssignedGroup} != ""',
                    "column_id": "OrderNr"
                },
                "backgroundColor": "#d4edda",
                "color": "black"
            }
        ]
    }

# -------------------------------
# INITIAL DATA LOAD
# -------------------------------

print("Initial data load...")
orders_data, fish_groups_data = load_validated_data()

# -------------------------------
# DASH APP SETUP
# -------------------------------

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    html.H1("Fish Egg Allocation Solver", className="text-center my-4 py-3 bg-primary text-white rounded"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Orders", className="mt-4"),
            dash_table.DataTable(
                id="order-table",
                data=orders_data.to_dict("records"),
                columns=[{"name": col, "id": col, "editable": True} for col in orders_data.columns],
                editable=True,
                row_deletable=True,
                page_action='native',
                page_current=0,
                page_size=10,
                sort_action='native',
                filter_action='native',
                **get_table_style()
            ),
            dbc.Button("Add Order Row", id="add-order-row-button", n_clicks=0, color="primary", className="mt-2"),
            
            html.H3("Fish Groups", className="mt-4"),
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
                filter_action='native',
                **get_table_style()
            ),
            dbc.Button("Add Fish Group Row", id="add-fish-group-row-button", n_clicks=0, color="primary", className="mt-2"),
        ], width=6),
        
        dbc.Col([
            html.H3("Problem Description & Constraints", className="mt-4"),
            dcc.Markdown("""
                ### Overview
                This application optimally allocates customer orders to fish egg groups based on multiple constraints and business rules. It helps fish hatchery managers efficiently assign orders to specific broodstock groups while respecting capacity limits and special requirements.
                
                ### Key Functions
                * Import and validate fish group data with capacities and dates
                * Import and validate customer orders with requirements
                * Solve the allocation problem using mathematical optimization
                * Visualize results and remaining capacity over time
                * Export allocation results to CSV/Excel for reporting
                
                ### Key Constraints
                
                1. **Orders**: Each active order must be assigned exactly once.
                
                2. **Capacity**: 
                   * Gain orders use only Gain capacity
                   * Shield orders can use leftover Gain capacity as well as Shield capacity
                   * Total capacity (Gain+Shield) must not be exceeded
                
                3. **Temperature Window**: 
                   * Order delivery date must fall within the calculated temperature-based delivery window
                   * Window is determined by stripping dates and customer's temperature requirements
                
                4. **Organic**: 
                   * Organic orders can only be assigned to organic-certified fish groups
                   * Non-organic orders can use either organic or non-organic groups
                
                5. **Locked/Preferred Groups**: 
                   * Orders with a locked group must be assigned to that specific group
                   * Orders with preferred groups are prioritized (but not required) to use those groups
                
                6. **Special Products**: 
                   * Elite/Nucleus products can only come from the Hemne site
                   * Regular products (Gain/Shield) have no site restrictions
                
                7. **Customer Restriction**: 
                   * Hønsvikgulen site can only deliver to Lerøy customers
                   * Other sites can deliver to any customer
                
                8. **FIFO Preference**: 
                   * Preference for using groups with earlier stripping end dates
                   * Helps maintain inventory freshness
                
                **Note**: The temperature window calculation currently uses a placeholder algorithm. Replace with actual logic from the Klekkekalulator files for production use.
            """, className="p-3 bg-light rounded"),
            
            dbc.Button("Solve Allocation Problem", id="solve-button", n_clicks=0, color="success", size="lg", className="mt-4"),
        ], width=6, className="p-4"),
    ], className="my-4"),
    
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(
            id="results-section",
            className="mt-4",
            style={"display": "none"},
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
                        dcc.Graph(id="buffer-visualization"),
                    ], width=12),
                ]),

                # Download buttons for results
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Download Results (CSV)", id="download-csv-button", color="info", className="mt-2 me-2"),
                        dcc.Download(id="download-csv"),
                        dbc.Button("Download Results (Excel)", id="download-xlsx-button", color="secondary", className="mt-2"),
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
    Output("order-table", "data"),
    Input("add-order-row-button", "n_clicks"),
    State("order-table", "data"),
    prevent_initial_call=True
)
def add_order_row(n_clicks, rows):
    if n_clicks > 0:
        new_row = {col: "" for col in orders_data.columns}
        rows.append(new_row)
    return rows

@app.callback(
    Output("fish-group-table", "data"),
    Input("add-fish-group-row-button", "n_clicks"),
    State("fish-group-table", "data"),
    prevent_initial_call=True
)
def add_fish_group_row(n_clicks, rows):
    if n_clicks > 0:
        new_row = {col: "" for col in fish_groups_data.columns}
        rows.append(new_row)
    return rows

@app.callback(
    [
        Output("results-section", "style"),
        Output("results-table", "data"),
        Output("results-table", "columns"),
        Output("capacity-table", "data"),
        Output("capacity-table", "columns"),
        Output("buffer-visualization", "figure"),
        Output("solve-button", "children")
    ],
    Input("solve-button", "n_clicks"),
    [State("order-table", "data"), State("fish-group-table", "data")],
    prevent_initial_call=True
)
def update_results(n_clicks, order_data, fish_group_data):
    if n_clicks > 0:
        # Convert table data back to DataFrames
        orders_df = pd.DataFrame(order_data)
        fish_groups_df = pd.DataFrame(fish_group_data)
        
        # Validate data
        fish_groups_df = validate_fish_groups(fish_groups_df)
        orders_df = validate_orders(orders_df, fish_groups_df)
        
        # Solve the allocation problem
        solution = solve_egg_allocation(orders_df, fish_groups_df)
        results_df = solution["results"]
        capacity_df = solution["remaining_capacity"]
        status = solution["status"]
        
        # Prepare data for display
        results_cols = [{"name": c, "id": c} for c in results_df.columns]
        capacity_cols = [{"name": c, "id": c} for c in capacity_df.columns]
        
        # Create visualization
        fig = create_buffer_graph(capacity_df, results_df)
        
        # Update button text with solver status
        button_text = f"Solve Allocation Problem (Status: {status})"
        
        return (
            {"margin": "20px", "display": "block"},
            results_df.to_dict("records"),
            results_cols,
            capacity_df.to_dict("records"),
            capacity_cols,
            fig,
            button_text
        )
    
    return ({"display": "none"}, [], [], [], [], {}, "Solve Allocation Problem")

@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-button", "n_clicks"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def export_results_csv(n_clicks, results_data):
    if n_clicks:
        df = pd.DataFrame(results_data)
        return dcc.send_data_frame(df.to_csv, "solver_results.csv", index=False)
    return None

@app.callback(
    Output("download-xlsx", "data"),
    Input("download-xlsx-button", "n_clicks"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def export_results_excel(n_clicks, results_data):
    if n_clicks:
        df = pd.DataFrame(results_data)
        return dcc.send_data_frame(df.to_excel, "solver_results.xlsx", sheet_name="SolverResults", index=False)
    return None

# -------------------------------
# RUN LOCAL OR VIA RENDER
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8050))  # default 8050 for local
    host = '0.0.0.0'
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    print(f"Starting server on {host}:{port} with debug={debug}")
    # Use app.run(...) instead of app.run_server(...) to avoid ObsoleteAttributeException
    app.run(host=host, port=port, debug=debug)
