import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pulp as pl

# -------------------------------
# VALIDATION FUNCTIONS
# -------------------------------

def process_organic(x):
    # If already a boolean, return it.
    if isinstance(x, bool):
        return x
    if pd.isnull(x) or str(x).strip() == "":
        return False
    elif str(x).strip().lower() == "organic":
        return True
    else:
        print(f"Warning: Invalid Organic value '{x}' found. Setting to False.")
        return False

def validate_orders(df, valid_sites):
    """
    For orders:
      1. Keep only rows where Product is "Gain" or "Shield".
      2. Process Organic using process_organic.
      3. For LockedSite: if non-empty, warn if not in valid_sites.
      4. For PreferredSite: only allow values found in valid_sites.
         (Also print out the unique PreferredSite values.)
    """
    valid_products = ["Gain", "Shield"]
    initial_count = len(df)
    df = df[df["Product"].isin(valid_products)].copy()
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} orders due to invalid Product values (only Gain and Shield are allowed).")
    
    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, "Organic"] = df["Organic"].apply(process_organic)
    
    def validate_locked(site):
        if pd.isnull(site) or str(site).strip() == "":
            return ""
        elif str(site).strip() not in valid_sites:
            print(f"Warning: LockedSite value '{site}' is not in the list of valid sites: {valid_sites}.")
        return str(site).strip()
    
    df.loc[:, "LockedSite"] = df["LockedSite"].apply(validate_locked)
    
    def validate_preferred(site):
        if pd.isnull(site) or str(site).strip() == "":
            return ""
        elif str(site).strip() not in valid_sites:
            print(f"Warning: PreferredSite value '{site}' is not in the list of valid sites: {valid_sites}.")
        return str(site).strip()
    
    df.loc[:, "PreferredSite"] = df["PreferredSite"].apply(validate_preferred)
    
    unique_preferred = df["PreferredSite"].unique().tolist()
    print("Unique PreferredSite values in orders data:", unique_preferred)
    
    return df

def validate_fish_groups(df):
    """
    For fish groups:
      - Process Organic using process_organic.
      - Convert numeric columns to numbers (invalid entries set to 0).
      - Convert date columns to datetime (dropping rows with invalid dates).
    """
    df = df.copy()
    df.loc[:, "Organic"] = df["Organic"].apply(process_organic)
    
    df.loc[:, "Gain-eggs"] = pd.to_numeric(df["Gain-eggs"], errors="coerce").fillna(0)
    df.loc[:, "Shield-eggs"] = pd.to_numeric(df["Shield-eggs"], errors="coerce").fillna(0)
    
    df.loc[:, "StrippingStartDate"] = pd.to_datetime(df["StrippingStartDate"], errors="coerce")
    df.loc[:, "StrippingStopDate"] = pd.to_datetime(df["StrippingStopDate"], errors="coerce")
    invalid_dates = df[df["StrippingStartDate"].isnull() | df["StrippingStopDate"].isnull()]
    if not invalid_dates.empty:
        print("Warning: Some rows in fish groups have invalid dates and will be dropped:")
        print(invalid_dates)
    df = df.dropna(subset=["StrippingStartDate", "StrippingStopDate"])
    
    return df

# -------------------------------
# DATA LOADING FUNCTION
# -------------------------------

def load_validated_data():
    orders_path = "example_data/orders_example.csv"
    fish_groups_path = "example_data/fish_groups_example.csv"
    try:
        orders_df = pd.read_csv(orders_path)
    except Exception as e:
        print(f"Error loading {orders_path}: {e}")
        orders_df = pd.DataFrame({
            "OrderNr": [],
            "DeliveryDate": [],
            "OrderStatus": [],
            "CustomerName": [],
            "Product": [],
            "Organic": [],
            "Volume": [],
            "LockedSite": [],
            "PreferredSite": []
        })
    try:
        fish_groups_df = pd.read_csv(fish_groups_path)
    except Exception as e:
        print(f"Error loading {fish_groups_path}: {e}")
        fish_groups_df = pd.DataFrame({
            "Site": [],
            "StrippingStartDate": [],
            "StrippingStopDate": [],
            "Gain-eggs": [],
            "Shield-eggs": [],
            "Organic": []
        })
    fish_groups_df = validate_fish_groups(fish_groups_df)
    valid_sites = fish_groups_df["Site"].unique().tolist()
    orders_df = validate_orders(orders_df, valid_sites)
    
    if not orders_df.empty:
        orders_df.loc[:, "DeliveryDate"] = pd.to_datetime(orders_df["DeliveryDate"], errors="coerce")
        orders_df.loc[:, "Volume"] = pd.to_numeric(orders_df["Volume"], errors="coerce").fillna(0)
    
    return orders_df, fish_groups_df

# -------------------------------
# SOLVER FUNCTION
# -------------------------------

def solve_egg_allocation(orders, fish_groups):
    orders = orders.copy()
    fish_groups = fish_groups.copy()
    orders.loc[:, "DeliveryDate"] = pd.to_datetime(orders["DeliveryDate"])
    fish_groups.loc[:, "StrippingStartDate"] = pd.to_datetime(fish_groups["StrippingStartDate"])
    fish_groups.loc[:, "StrippingStopDate"] = pd.to_datetime(fish_groups["StrippingStopDate"])
    
    orders.loc[:, "Volume"] = pd.to_numeric(orders["Volume"], errors="coerce").fillna(0)
    fish_groups.loc[:, "Gain-eggs"] = pd.to_numeric(fish_groups["Gain-eggs"], errors="coerce").fillna(0)
    fish_groups.loc[:, "Shield-eggs"] = pd.to_numeric(fish_groups["Shield-eggs"], errors="coerce").fillna(0)
    
    active_orders = orders[orders["OrderStatus"] != "Kansellert"].reset_index(drop=True)
    
    # Add dummy site
    dummy_df = pd.DataFrame({
        "Site": ["Dummy"],
        "StrippingStartDate": [pd.to_datetime("2024-01-01")],
        "StrippingStopDate": [pd.to_datetime("2024-12-31")],
        "Gain-eggs": [float("inf")],
        "Shield-eggs": [float("inf")],
        "Organic": [True]
    })
    fish_groups_reset = fish_groups.reset_index(drop=True)
    all_fish_groups = pd.concat([fish_groups_reset, dummy_df], ignore_index=True)
    dummy_idx = all_fish_groups.index[all_fish_groups["Site"] == "Dummy"].tolist()[0]
    
    prob = pl.LpProblem("FishEggAllocation", pl.LpMinimize)
    
    x = {}
    for i in active_orders.index:
        for j in all_fish_groups.index:
            x[i, j] = pl.LpVariable(f"x_{i}_{j}", cat="Binary")
    
    dummy_penalty = {i: pl.LpVariable(f"dummy_penalty_{i}", lowBound=0) for i in active_orders.index}
    pref_penalty = {}
    for i, row in active_orders.iterrows():
        if pd.notna(row["PreferredSite"]) and row["PreferredSite"] != "":
            pref_penalty[i] = pl.LpVariable(f"pref_penalty_{i}", lowBound=0)
    
    real_groups = all_fish_groups[all_fish_groups["Site"] != "Dummy"]
    earliest_stop = real_groups["StrippingStopDate"].min()
    group_penalty = {}
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] != "Dummy":
            delta = (all_fish_groups.loc[j, "StrippingStopDate"] - earliest_stop).days
            group_penalty[j] = float(delta)
        else:
            group_penalty[j] = 0.0
    fifo_weight = 0.01
    
    prob += (
        1000 * pl.lpSum(dummy_penalty.values())
        + 10 * pl.lpSum(pref_penalty.values())
        + pl.lpSum(x[i, j] * group_penalty[j] * fifo_weight
                   for i in active_orders.index for j in all_fish_groups.index)
    )
    
    # Each order assigned exactly once
    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in all_fish_groups.index) == 1
    
    # Gains vs. Shield capacity
    gain_orders = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Gain"]
    shield_orders = [i for i in active_orders.index if active_orders.loc[i, "Product"] == "Shield"]
    
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, "Site"] != "Dummy":
            Gcap = all_fish_groups.loc[j, "Gain-eggs"]
            Scap = all_fish_groups.loc[j, "Shield-eggs"]
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in gain_orders) <= Gcap
            prob += pl.lpSum(x[i, j] * active_orders.loc[i, "Volume"] for i in (gain_orders + shield_orders)) <= (Gcap + Scap)
    
    # Organic requirement
    for i in active_orders.index:
        if active_orders.loc[i, "Organic"]:
            for j in all_fish_groups.index:
                if (not all_fish_groups.loc[j, "Organic"]) and (all_fish_groups.loc[j, "Site"] != "Dummy"):
                    prob += x[i, j] == 0
    
    # Locked site requirement
    for i in active_orders.index:
        locked_site = active_orders.loc[i, "LockedSite"]
        if pd.notna(locked_site) and locked_site != "":
            for j in all_fish_groups.index:
                if (all_fish_groups.loc[j, "Site"] != locked_site) and (all_fish_groups.loc[j, "Site"] != "Dummy"):
                    prob += x[i, j] == 0
    
    # Dummy penalty
    for i in active_orders.index:
        prob += dummy_penalty[i] >= x[i, dummy_idx]
    
    # Preferred site penalty
    for i in active_orders.index:
        if i in pref_penalty:
            pref_site = active_orders.loc[i, "PreferredSite"]
            not_pref = [
                jj for jj in all_fish_groups.index 
                if (all_fish_groups.loc[jj, "Site"] != pref_site) and (all_fish_groups.loc[jj, "Site"] != "Dummy")
            ]
            prob += pref_penalty[i] >= pl.lpSum(x[i, jj] for jj in not_pref)
    
    # Date constraints
    for i in active_orders.index:
        ddate = active_orders.loc[i, "DeliveryDate"]
        for j in all_fish_groups.index:
            if all_fish_groups.loc[j, "Site"] != "Dummy":
                start = all_fish_groups.loc[j, "StrippingStartDate"]
                stop = all_fish_groups.loc[j, "StrippingStopDate"]
                if ddate < start or ddate > stop:
                    prob += x[i, j] == 0
    
    prob.writeLP("fish_egg_allocation.lp")
    prob.solve()
    solver_status = pl.LpStatus[prob.status]
    
    # Extract solution
    results = active_orders.copy()
    results["AssignedGroup"] = None
    results["IsDummy"] = False
    for i in results.index:
        for j in all_fish_groups.index:
            if pl.value(x[i, j]) == 1:
                group_site = all_fish_groups.loc[j, "Site"]
                results.loc[i, "AssignedGroup"] = group_site
                results.loc[i, "IsDummy"] = (group_site == "Dummy")
                break
    
    # Merge results with original (including cancelled orders)
    all_res = orders.copy()
    all_res["AssignedGroup"] = None
    all_res["IsDummy"] = False
    for i, row in results.iterrows():
        idx = all_res.index[all_res["OrderNr"] == row["OrderNr"]]
        if len(idx) > 0:
            all_res.loc[idx[0], "AssignedGroup"] = row["AssignedGroup"]
            all_res.loc[idx[0], "IsDummy"] = row["IsDummy"]
    
    # Mark cancelled as "Skipped-Cancelled"
    cancelled_idx = all_res[all_res["OrderStatus"] == "Kansellert"].index
    all_res.loc[cancelled_idx, "AssignedGroup"] = "Skipped-Cancelled"
    all_res.loc[cancelled_idx, "IsDummy"] = False
    
    # Calculate capacity usage
    capacity = fish_groups.copy()
    capacity["TotalEggs"] = capacity["Gain-eggs"] + capacity["Shield-eggs"]
    total_used = []
    for j, grp in capacity.iterrows():
        site_name = grp["Site"]
        used_here = all_res[all_res["AssignedGroup"] == site_name]["Volume"].sum()
        total_used.append(used_here)
    capacity["TotalEggsUsed"] = total_used
    capacity["TotalEggsRemaining"] = capacity["TotalEggs"] - capacity["TotalEggsUsed"]
    
    final_capacity = capacity[[
        "Site", 
        "StrippingStartDate", 
        "StrippingStopDate", 
        "TotalEggs", 
        "Organic", 
        "TotalEggsUsed", 
        "TotalEggsRemaining"
    ]]
    
    return {
        "status": solver_status,
        "results": all_res,
        "remaining_capacity": final_capacity
    }

# -------------------------------
# VISUALIZATION & STYLING
# -------------------------------

def create_buffer_graph(remaining, results):
    df = remaining.copy()
    df["StrippingStartDate"] = pd.to_datetime(df["StrippingStartDate"])
    df["StrippingStopDate"] = pd.to_datetime(df["StrippingStopDate"])
    buffer_data = []
    facilities = df["Site"].unique()
    start_date = df["StrippingStartDate"].min()
    end_date = df["StrippingStopDate"].max() + pd.Timedelta(weeks=4)
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    
    current_rem = {
        fac: float(df.loc[df["Site"] == fac, "TotalEggs"].iloc[0]) 
        for fac in facilities
    }
    
    for week in weekly_dates:
        for facility in facilities:
            assigned_orders = results[
                (results["AssignedGroup"] == facility) &
                (pd.to_datetime(results["DeliveryDate"]) <= week) &
                (results["IsDummy"] == False)
            ]
            used_this_week = assigned_orders["Volume"].sum()
            current_rem[facility] -= used_this_week
            if current_rem[facility] < 0:
                current_rem[facility] = 0
            buffer_data.append({
                "Week": week,
                "Facility": facility,
                "TotalRemaining": current_rem[facility] / 1e6
            })
    
    buffer_df = pd.DataFrame(buffer_data)
    fig = px.line(
        buffer_df, 
        x="Week", 
        y="TotalRemaining", 
        color="Facility",
        title="Weekly Inventory Buffer (Merged Gains+Shield)",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Available Roe (Millions)",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=14),
        legend_title_text="Facility",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    return fig

def get_table_style():
    """
    Enhanced style for large tables:
      - Constrain column widths
      - Scroll if table is too wide or too tall
    """
    return {
        "style_table": {
            "overflowX": "auto",
            "maxHeight": "400px",  # limit vertical expansion, scroll if exceeded
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
            {
                "if": {"state": "selected"}, 
                "backgroundColor": "#cce5ff", 
                "border": "1px solid #007bff"
            },
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
                # Pagination
                page_action='native',
                page_current=0,
                page_size=10,  # show 10 rows per page
                # Sorting & Filtering
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
                # Pagination
                page_action='native',
                page_current=0,
                page_size=10,  # show 10 rows per page
                # Sorting & Filtering
                sort_action='native',
                filter_action='native',
                **get_table_style()
            ),
            dbc.Button("Add Fish Group Row", id="add-fish-group-row-button", n_clicks=0, color="primary", className="mt-2"),
        ], width=6),
        
        dbc.Col([
            html.H3("Problem Description & Constraints", className="mt-4"),
            dcc.Markdown("""
                **Intent**  
                This solver automates the allocation of customer orders (each with a certain volume of eggs) 
                to different fish groups (sites).  
                
                **Key Constraints**  
                1. **Assign each active (non-cancelled) order exactly once**.  
                2. **Shield→Gain**: If a product is Shield, it can use leftover Gains capacity.  
                3. **Gains capacity** must not be exceeded by Gains orders alone.  
                4. **Total capacity** (Gains+Shield) must not be exceeded by Gains+Shield orders combined.  
                5. **Organic**: Organic orders only go to Organic sites (or Dummy).  
                6. **Locked site**: Orders locked to a site can only be assigned there or to Dummy.  
                7. **Preferred site**: Soft constraint (a small penalty if not assigned there).  
                8. **Dummy**: Large penalty to avoid usage if possible.  
                9. **Date window**: An order’s delivery date must lie within a site's stripping dates.  
                10. **FIFO**: A small penalty for using groups with later stop-dates.
                
                **Merged Gains+Shield Reporting**  
                - Gains and Shield capacities are merged into "TotalEggs" to avoid negative leftovers.
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
                        html.H3("Remaining Capacity (Merged)", className="mt-4"),
                        dash_table.DataTable(id="capacity-table", **get_table_style()),
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Weekly Inventory Buffer", className="mt-4"),
                        dcc.Graph(id="buffer-visualization"),
                    ], width=12),
                ]),
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
        orders_df = pd.DataFrame(order_data)
        fish_groups_df = pd.DataFrame(fish_group_data)
        
        # Re-validate with the updated data
        fish_groups_df = validate_fish_groups(fish_groups_df)
        valid_sites = fish_groups_df["Site"].unique().tolist()
        orders_df = validate_orders(orders_df, valid_sites)
        
        orders_df.loc[:, "DeliveryDate"] = pd.to_datetime(orders_df["DeliveryDate"], errors="coerce")
        orders_df.loc[:, "Volume"] = pd.to_numeric(orders_df["Volume"], errors="coerce").fillna(0)
        
        solution = solve_egg_allocation(orders_df, fish_groups_df)
        results_df = solution["results"]
        capacity_df = solution["remaining_capacity"]
        status = solution["status"]
        
        results_cols = [{"name": c, "id": c} for c in results_df.columns]
        capacity_cols = [{"name": c, "id": c} for c in capacity_df.columns]
        
        fig = create_buffer_graph(capacity_df, results_df)
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

if __name__ == "__main__":
    app.run_server(debug=True)

