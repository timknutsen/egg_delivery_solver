# %%
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pulp as pl

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

# --------------------------------------------------------------------------------
# EXAMPLE DATA: FIFO + Shield→Gain
# --------------------------------------------------------------------------------

default_orders = pd.DataFrame({
    'OrderNr': ['O001', 'O002', 'O003', 'O004', 'O005', 'O006', 'O007', 'O008', 'O009'],
    'DeliveryDate': [
        '2024-08-15',  # Gains
        '2024-09-01',  # Gains
        '2024-10-01',  # Shield
        '2024-09-25',  # Shield (Kansellert)
        '2024-08-20',  # Gains (Locked to Hønsvikgulen)
        '2024-09-05',  # Shield
        '2024-11-01',  # Gains
        '2024-08-25',  # Large Shield
        '2024-09-05'   # Even larger Shield
    ],
    'OrderStatus': [
        'Bekreftet', 'Bekreftet', 'Bekreftet', 'Kansellert',
        'Bekreftet', 'Bekreftet', 'Bekreftet', 'Bekreftet', 'Bekreftet'
    ],
    'CustomerName': [
        'AquaGen AS', 'NTNU', 'SalMar Farming AS', 'Ewos Innovation AS',
        'Lerøy Midt AS', 'Marine Harvest', 'Cermaq', 'New Shield Customer', 'Mixed Farming AS'
    ],
    'Product': [
        'Gain', 'Gain', 'Shield', 'Shield',
        'Gain', 'Shield', 'Gain', 'Shield', 'Shield'
    ],
    'Organic': [
        False, True, False, False,
        False, False, True, False, False
    ],
    'Volume': [
        500000, 300000, 400000, 200000,
        600000, 250000, 350000, 1000000, 1500000
    ],
    'LockedSite': [
        None, None, None, None,
        'Hønsvikgulen', None, None, None, None
    ],
    'PreferredSite': [
        'Vestseøra', 'Bogen', 'Kilavågen Land', None,
        None, 'Vestseøra', 'Kilavågen Land', None, None
    ]
})

default_fish_groups = pd.DataFrame({
    'Site': ['Vestseøra', 'Kilavågen Land', 'Bogen', 'Hønsvikgulen'],
    'StrippingStartDate': ['2024-08-05', '2024-09-26', '2024-08-05', '2024-07-16'],
    'StrippingStopDate':  ['2024-09-02', '2024-11-21', '2024-09-09', '2024-08-27'],
    # Enough Gains/Shield to force interesting allocations
    'Gain-eggs':   [7996198, 16451359, 1200000, 1500000],
    'Shield-eggs': [7996198, 16451359, 200000, 0],
    'Organic':     [True, False, True, False]
})

# --------------------------------------------------------------------------------
# SOLVER FUNCTION
# --------------------------------------------------------------------------------

def solve_egg_allocation(orders, fish_groups):
    """
    Solver implementing:
      1) FIFO: small penalty for using fish groups with a later StrippingStopDate
      2) Shield→Gain: a Shield order can also use leftover Gains capacity.
    
    CONSTRAINTS:
      - Each non-cancelled order assigned exactly once (to real site or Dummy).
      - Gains orders can only use Gains capacity at a site.
      - Shield orders can use Gains + Shield capacity combined.
      - If an order is Organic, it can only be assigned to an Organic site or Dummy.
      - If an order is locked to a site, it must go there or Dummy.
      - Preferred site mismatch => small penalty (soft constraint).
      - Dummy usage => large penalty (to avoid dummy if feasible).
      - If the order date is outside the site’s [Start, Stop], the order cannot go there.
      - FIFO => small penalty if using “later” stop-date groups first.
    """
    # Convert dates to datetime
    orders['DeliveryDate'] = pd.to_datetime(orders['DeliveryDate'])
    fish_groups['StrippingStartDate'] = pd.to_datetime(fish_groups['StrippingStartDate'])
    fish_groups['StrippingStopDate'] = pd.to_datetime(fish_groups['StrippingStopDate'])
    
    # Convert numeric columns
    orders['Volume'] = pd.to_numeric(orders['Volume'], errors='coerce').fillna(0)
    fish_groups['Gain-eggs'] = pd.to_numeric(fish_groups['Gain-eggs'], errors='coerce').fillna(0)
    fish_groups['Shield-eggs'] = pd.to_numeric(fish_groups['Shield-eggs'], errors='coerce').fillna(0)
    
    # Filter out cancelled
    active_orders = orders[orders['OrderStatus'] != 'Kansellert'].copy().reset_index(drop=True)
    
    # Add dummy site
    dummy_df = pd.DataFrame({
        'Site': ['Dummy'],
        'StrippingStartDate': [pd.to_datetime('2024-01-01')],
        'StrippingStopDate': [pd.to_datetime('2024-12-31')],
        'Gain-eggs': [float('inf')],
        'Shield-eggs': [float('inf')],
        'Organic': [True]
    })
    fish_groups_reset = fish_groups.reset_index(drop=True)
    all_fish_groups = pd.concat([fish_groups_reset, dummy_df], ignore_index=True)
    dummy_idx = all_fish_groups.index[all_fish_groups['Site'] == 'Dummy'].tolist()[0]
    
    # Create problem
    prob = pl.LpProblem("FishEggAllocation", pl.LpMinimize)
    
    # Decision variables
    x = {}
    for i in active_orders.index:
        for j in all_fish_groups.index:
            x[i, j] = pl.LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # Penalties
    dummy_penalty = {i: pl.LpVariable(f"dummy_penalty_{i}", lowBound=0) for i in active_orders.index}
    pref_penalty = {}
    for i, row in active_orders.iterrows():
        if pd.notna(row['PreferredSite']):
            pref_penalty[i] = pl.LpVariable(f"pref_penalty_{i}", lowBound=0)
    
    # FIFO penalty setup
    real_groups = all_fish_groups[all_fish_groups['Site'] != 'Dummy']
    earliest_stop = real_groups['StrippingStopDate'].min()
    group_penalty = {}
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, 'Site'] != 'Dummy':
            delta = (all_fish_groups.loc[j, 'StrippingStopDate'] - earliest_stop).days
            group_penalty[j] = float(delta)
        else:
            group_penalty[j] = 0.0
    fifo_weight = 0.01
    
    # Objective
    prob += (
        1000 * pl.lpSum(dummy_penalty.values())
        + 10 * pl.lpSum(pref_penalty.values())
        + pl.lpSum(x[i, j] * group_penalty[j] * fifo_weight
                   for i in active_orders.index
                   for j in all_fish_groups.index)
    )
    
    # 1) Each order assigned exactly once
    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in all_fish_groups.index) == 1
    
    # Gains vs Shield capacity
    gain_orders = [i for i in active_orders.index if active_orders.loc[i, 'Product'] == 'Gain']
    shield_orders = [i for i in active_orders.index if active_orders.loc[i, 'Product'] == 'Shield']
    
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, 'Site'] != 'Dummy':
            Gcap = all_fish_groups.loc[j, 'Gain-eggs']
            Scap = all_fish_groups.loc[j, 'Shield-eggs']
            # Gains usage <= Gains capacity
            prob += pl.lpSum(
                x[i, j] * active_orders.loc[i, 'Volume']
                for i in gain_orders
            ) <= Gcap
            # Gains + Shield usage <= Gains + Shield capacity
            prob += pl.lpSum(
                x[i, j] * active_orders.loc[i, 'Volume']
                for i in (gain_orders + shield_orders)
            ) <= (Gcap + Scap)
    
    # 3) Organic requirement
    for i in active_orders.index:
        if active_orders.loc[i, 'Organic']:
            for j in all_fish_groups.index:
                if (not all_fish_groups.loc[j, 'Organic']) and (all_fish_groups.loc[j, 'Site'] != 'Dummy'):
                    prob += x[i, j] == 0
    
    # 4) Locked site requirement
    for i in active_orders.index:
        locked_site = active_orders.loc[i, 'LockedSite']
        if pd.notna(locked_site):
            for j in all_fish_groups.index:
                if (all_fish_groups.loc[j, 'Site'] != locked_site) and (all_fish_groups.loc[j, 'Site'] != 'Dummy'):
                    prob += x[i, j] == 0
    
    # 5) Dummy penalty
    for i in active_orders.index:
        prob += dummy_penalty[i] >= x[i, dummy_idx]
    
    # 6) Preferred site penalty
    for i in active_orders.index:
        if i in pref_penalty:
            pref_site = active_orders.loc[i, 'PreferredSite']
            not_pref = [jj for jj in all_fish_groups.index
                        if (all_fish_groups.loc[jj, 'Site'] != pref_site) and (all_fish_groups.loc[jj, 'Site'] != 'Dummy')]
            prob += pref_penalty[i] >= pl.lpSum(x[i, jj] for jj in not_pref)
    
    # 7) Date constraint
    for i in active_orders.index:
        ddate = active_orders.loc[i, 'DeliveryDate']
        for j in all_fish_groups.index:
            if all_fish_groups.loc[j, 'Site'] != 'Dummy':
                start = all_fish_groups.loc[j, 'StrippingStartDate']
                stop = all_fish_groups.loc[j, 'StrippingStopDate']
                if ddate < start or ddate > stop:
                    prob += x[i, j] == 0
    
    # Write LP for debugging
    prob.writeLP("fish_egg_allocation.lp")
    prob.solve()
    solver_status = pl.LpStatus[prob.status]
    
    # Extract solution
    results = active_orders.copy()
    results['AssignedGroup'] = None
    results['IsDummy'] = False
    for i in results.index:
        for j in all_fish_groups.index:
            if pl.value(x[i, j]) == 1:
                group_site = all_fish_groups.loc[j, 'Site']
                results.loc[i, 'AssignedGroup'] = group_site
                results.loc[i, 'IsDummy'] = (group_site == 'Dummy')
                break
    
    # Combine with original (to include cancelled)
    all_res = orders.copy()
    all_res['AssignedGroup'] = None
    all_res['IsDummy'] = False
    for i, row in results.iterrows():
        idx = all_res.index[all_res['OrderNr'] == row['OrderNr']]
        if len(idx) > 0:
            all_res.loc[idx[0], 'AssignedGroup'] = row['AssignedGroup']
            all_res.loc[idx[0], 'IsDummy'] = row['IsDummy']
    
    # Mark cancelled as "Skipped-Cancelled"
    cancelled_idx = all_res[all_res['OrderStatus'] == 'Kansellert'].index
    all_res.loc[cancelled_idx, 'AssignedGroup'] = 'Skipped-Cancelled'
    all_res.loc[cancelled_idx, 'IsDummy'] = False
    
    # ------------------------------------------------------------------------------
    # APPROACH A: Merge Gains + Shield in final reporting (Total eggs).
    # ------------------------------------------------------------------------------
    capacity = fish_groups.copy()
    capacity['TotalEggs'] = capacity['Gain-eggs'] + capacity['Shield-eggs']
    
    # Calculate total usage at each site (both Gains + Shield)
    total_used = []
    for j, grp in capacity.iterrows():
        site_name = grp['Site']
        used_here = all_res[all_res['AssignedGroup'] == site_name]['Volume'].sum()
        total_used.append(used_here)
    capacity['TotalEggsUsed'] = total_used
    capacity['TotalEggsRemaining'] = capacity['TotalEggs'] - capacity['TotalEggsUsed']
    
    # We keep only the merged columns (no Gains-eggs-remaining or Shield-eggs-remaining)
    # for clarity and to avoid negative leftover in separate columns.
    final_capacity = capacity[[
        'Site', 'StrippingStartDate', 'StrippingStopDate',
        'TotalEggs', 'Organic', 'TotalEggsUsed', 'TotalEggsRemaining'
    ]]
    
    return {
        'status': solver_status,
        'results': all_res,
        'remaining_capacity': final_capacity
    }

# --------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------

def create_buffer_graph(remaining, results):
    """
    We can still do a naive 'weekly' leftover plot, but we only track 'TotalEggsRemaining'.
    For each site, we consider the entire Gains+Shield capacity as one pool.
    """
    # We'll rename columns for clarity
    df = remaining.copy()
    df['StrippingStartDate'] = pd.to_datetime(df['StrippingStartDate'])
    df['StrippingStopDate'] = pd.to_datetime(df['StrippingStopDate'])
    
    buffer_data = []
    facilities = df['Site'].unique()
    start_date = df['StrippingStartDate'].min()
    end_date = df['StrippingStopDate'].max() + pd.Timedelta(weeks=4)
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    
    # We'll track total capacity usage as a single pool
    current_rem = {fac: float(df.loc[df['Site'] == fac, 'TotalEggs'].iloc[0]) for fac in facilities}
    
    for week in weekly_dates:
        for facility in facilities:
            # Subtract volumes for orders delivered on/before this week
            assigned_orders = results[
                (results['AssignedGroup'] == facility)
                & (pd.to_datetime(results['DeliveryDate']) <= week)
                & (results['IsDummy'] == False)
            ]
            used_this_week = assigned_orders['Volume'].sum()
            
            # Reduce the local capacity
            current_rem[facility] -= used_this_week
            if current_rem[facility] < 0:
                current_rem[facility] = 0  # No negative leftover for the plot
            
            buffer_data.append({
                'Week': week,
                'Facility': facility,
                'TotalRemaining': current_rem[facility] / 1e6
            })
    
    buffer_df = pd.DataFrame(buffer_data)
    fig = px.line(
        buffer_df, x='Week', y='TotalRemaining', color='Facility',
        title='Weekly Inventory Buffer (Merged Gains+Shield)', markers=True
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
    return {
        'style_table': {'overflowX': 'auto', 'borderRadius': '10px', 'overflow': 'hidden'},
        'style_cell': {
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px',
            'border': '1px solid #e0e0e0',
        },
        'style_header': {
            'backgroundColor': '#007bff',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'padding': '10px',
            'borderBottom': '2px solid #0056b3',
        },
        'style_data_conditional': [
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
            {'if': {'state': 'selected'}, 'backgroundColor': '#cce5ff', 'border': '1px solid #007bff'},
            {
                'if': {
                    'filter_query': '{AssignedGroup} != "Skipped-Cancelled" and {AssignedGroup} != "Dummy" and {AssignedGroup} != ""',
                    'column_id': 'OrderNr'
                },
                'backgroundColor': '#d4edda',
                'color': 'black'
            }
        ]
    }

# --------------------------------------------------------------------------------
# DASH LAYOUT
# --------------------------------------------------------------------------------

app.layout = dbc.Container([
    html.H1("Fish Egg Allocation Solver", className="text-center my-4 py-3 bg-primary text-white rounded"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Orders", className="mt-4"),
            dash_table.DataTable(
                id='order-table',
                data=default_orders.to_dict('records'),
                columns=[{'name': col, 'id': col, 'editable': True} for col in default_orders.columns],
                editable=True,
                row_deletable=True,
                **get_table_style()
            ),
            dbc.Button('Add Order Row', id='add-order-row-button', n_clicks=0, color="primary", className="mt-2"),
            
            html.H3("Fish Groups", className="mt-4"),
            dash_table.DataTable(
                id='fish-group-table',
                data=default_fish_groups.to_dict('records'),
                columns=[{'name': col, 'id': col, 'editable': True} for col in default_fish_groups.columns],
                editable=True,
                row_deletable=True,
                **get_table_style()
            ),
            dbc.Button('Add Fish Group Row', id='add-fish-group-row-button', n_clicks=0, color="primary", className="mt-2"),
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
                5. **Organic**: If an order is Organic, it must go to an Organic site (or Dummy).  
                6. **Locked site**: If an order is locked to a site, it can only go there or Dummy.  
                7. **Preferred site**: Soft constraint (small penalty if we don't assign it there).  
                8. **Dummy**: Large penalty to avoid usage if possible.  
                9. **Date window**: An order's delivery date must lie within a site's [Start, Stop].  
                10. **FIFO**: We add a small penalty for using groups with a later stop-date first.  

                **Merged Gains+Shield Reporting**  
                - In the final capacity table, Gains and Shield are combined into a single "TotalEggs" 
                  column. This avoids negative leftover if a Shield order partially uses Gains capacity.
            """, className="p-3 bg-light rounded"),
            
            dbc.Button('Solve Allocation Problem', id='solve-button', n_clicks=0, color="success", size="lg", className="mt-4"),
        ], width=6, className="p-4"),
    ], className="my-4"),
    
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(
            id='results-section',
            className="mt-4",
            style={'display': 'none'},
            children=[
                html.H2("Solver Results", className="text-center mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.H3("Order Assignments", className="mt-4"),
                        dash_table.DataTable(id='results-table', **get_table_style()),
                        html.H3("Remaining Capacity (Merged)", className="mt-4"),
                        dash_table.DataTable(id='capacity-table', **get_table_style()),
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Weekly Inventory Buffer", className="mt-4"),
                        dcc.Graph(id='buffer-visualization'),
                    ], width=12),
                ]),
            ]
        )
    )
], fluid=True)

# --------------------------------------------------------------------------------
# CALLBACKS
# --------------------------------------------------------------------------------

@app.callback(
    Output('order-table', 'data'),
    Input('add-order-row-button', 'n_clicks'),
    State('order-table', 'data'),
    prevent_initial_call=True
)
def add_order_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({col: '' for col in default_orders.columns})
    return rows

@app.callback(
    Output('fish-group-table', 'data'),
    Input('add-fish-group-row-button', 'n_clicks'),
    State('fish-group-table', 'data'),
    prevent_initial_call=True
)
def add_fish_group_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({col: '' for col in default_fish_groups.columns})
    return rows

@app.callback(
    [
        Output('results-section', 'style'),
        Output('results-table', 'data'),
        Output('results-table', 'columns'),
        Output('capacity-table', 'data'),
        Output('capacity-table', 'columns'),
        Output('buffer-visualization', 'figure'),
        Output('solve-button', 'children')
    ],
    Input('solve-button', 'n_clicks'),
    [State('order-table', 'data'), State('fish-group-table', 'data')],
    prevent_initial_call=True
)
def update_results(n_clicks, order_data, fish_group_data):
    if n_clicks > 0:
        # Convert to DataFrame
        orders_df = pd.DataFrame(order_data)
        fish_groups_df = pd.DataFrame(fish_group_data)
        
        # Solve
        solution = solve_egg_allocation(orders_df, fish_groups_df)
        
        # Extract
        results_df = solution['results']
        capacity_df = solution['remaining_capacity']
        status = solution['status']
        
        # Prepare columns
        results_cols = [{'name': c, 'id': c} for c in results_df.columns]
        capacity_cols = [{'name': c, 'id': c} for c in capacity_df.columns]
        
        # Create plot
        fig = create_buffer_graph(capacity_df, results_df)
        
        # Status
        button_text = f"Solve Allocation Problem (Status: {status})"
        
        return (
            {'margin': '20px', 'display': 'block'},
            results_df.to_dict('records'),
            results_cols,
            capacity_df.to_dict('records'),
            capacity_cols,
            fig,
            button_text
        )
    
    return ({'display': 'none'}, [], [], [], [], {}, "Solve Allocation Problem")

if __name__ == '__main__':
    app.run_server(debug=True)

