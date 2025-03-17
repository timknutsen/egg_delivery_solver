# %%
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pulp as pl

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

# Default data for orders
default_orders = pd.DataFrame({
    'OrderNr': ['O001', 'O002', 'O003', 'O004', 'O005'],
    'DeliveryDate': ['2024-09-15', '2024-10-10', '2024-09-20', '2024-09-25', '2024-09-30'],
    'OrderStatus': ['Bekreftet', 'Bekreftet', 'Bekreftet', 'Kansellert', 'Bekreftet'],
    'CustomerName': ['AquaGen AS', 'NTNU', 'SalMar Farming AS', 'Ewos Innovation AS', 'Lerøy Midt AS'],
    'Product': ['Gain', 'Shield', 'Gain', 'Shield', 'Gain'],
    'Organic': [False, False, True, False, False],
    'Volume': [500000, 300000, 400000, 200000, 1000000],
    'LockedSite': [None, None, None, None, 'Hønsvikgulen'],
    'PreferredSite': ['Vestseøra', 'Kilavågen Land', 'Bogen', None, None]
})

# Default data for fish groups
default_fish_groups = pd.DataFrame({
    'Site': ['Vestseøra', 'Kilavågen Land', 'Bogen', 'Hønsvikgulen'],
    'StrippingStartDate': ['2024-08-05', '2024-09-26', '2024-08-05', '2024-07-16'],
    'StrippingStopDate': ['2024-09-02', '2024-11-21', '2024-09-09', '2024-08-27'],
    'Gain-eggs': [7996198, 16451359, 8728493, 16600150],
    'Shield-eggs': [7996198, 16451359, 8728493, 0],
    'Organic': [True, False, True, False]
})

# Solver function with corrected temporal constraint
def solve_egg_allocation(orders, fish_groups):
    # Convert dates to datetime
    orders['DeliveryDate'] = pd.to_datetime(orders['DeliveryDate'])
    fish_groups['StrippingStartDate'] = pd.to_datetime(fish_groups['StrippingStartDate'])
    fish_groups['StrippingStopDate'] = pd.to_datetime(fish_groups['StrippingStopDate'])
    
    # Filter out cancelled orders
    active_orders = orders[orders['OrderStatus'] != 'Kansellert'].copy().reset_index(drop=True)
    
    # Add dummy fish group
    dummy_group = pd.DataFrame({
        'Site': ['Dummy'],
        'StrippingStartDate': [pd.to_datetime('2024-01-01')],
        'StrippingStopDate': [pd.to_datetime('2024-12-31')],
        'Gain-eggs': [float('inf')],
        'Shield-eggs': [float('inf')],
        'Organic': [True]
    })
    
    fish_groups_reset = fish_groups.reset_index(drop=True)
    all_fish_groups = pd.concat([fish_groups_reset, dummy_group], ignore_index=True)
    dummy_site_idx = all_fish_groups.index[all_fish_groups['Site'] == 'Dummy'].tolist()[0]
    
    # Create the PuLP problem
    prob = pl.LpProblem("FishEggAllocation", pl.LpMinimize)
    
    # Decision variables
    x = {}
    for i in active_orders.index:
        for j in all_fish_groups.index:
            x[i, j] = pl.LpVariable(f"assign_{i}_{j}", cat='Binary')
    
    # Penalty variables
    dummy_penalties = {i: pl.LpVariable(f"dummy_penalty_{i}", lowBound=0) for i in active_orders.index}
    pref_site_penalties = {}
    for i, order in active_orders.iterrows():
        if pd.notna(order['PreferredSite']):
            pref_site_penalties[i] = pl.LpVariable(f"pref_site_penalty_{i}", lowBound=0)
    
    # Objective function
    prob += (1000 * pl.lpSum(dummy_penalties.values()) +
             10 * pl.lpSum(pref_site_penalties.values()))
    
    # Each order must be assigned to exactly one group
    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in all_fish_groups.index) == 1
    
    # Capacity constraints
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, 'Site'] != 'Dummy':
            prob += pl.lpSum(
                x[i, j] * active_orders.loc[i, 'Volume'] 
                for i in active_orders.index if active_orders.loc[i, 'Product'] == 'Gain'
            ) <= all_fish_groups.loc[j, 'Gain-eggs']
            prob += pl.lpSum(
                x[i, j] * active_orders.loc[i, 'Volume'] 
                for i in active_orders.index if active_orders.loc[i, 'Product'] == 'Shield'
            ) <= all_fish_groups.loc[j, 'Shield-eggs']
    
    # Organic requirement
    for i in active_orders.index:
        if active_orders.loc[i, 'Organic']:
            for j in all_fish_groups.index:
                if not all_fish_groups.loc[j, 'Organic'] and all_fish_groups.loc[j, 'Site'] != 'Dummy':
                    prob += x[i, j] == 0
    
    # Locked site requirement
    for i in active_orders.index:
        if pd.notna(active_orders.loc[i, 'LockedSite']):
            locked_site = active_orders.loc[i, 'LockedSite']
            for j in all_fish_groups.index:
                if all_fish_groups.loc[j, 'Site'] != locked_site and all_fish_groups.loc[j, 'Site'] != 'Dummy':
                    prob += x[i, j] == 0
    
    # Dummy penalties
    for i in active_orders.index:
        prob += dummy_penalties[i] >= x[i, dummy_site_idx]
    
    # Preferred site penalties
    for i in active_orders.index:
        if pd.notna(active_orders.loc[i, 'PreferredSite']):
            pref_site = active_orders.loc[i, 'PreferredSite']
            non_pref_sites = [j for j in all_fish_groups.index 
                              if all_fish_groups.loc[j, 'Site'] != pref_site and all_fish_groups.loc[j, 'Site'] != 'Dummy']
            if non_pref_sites:
                prob += pref_site_penalties[i] >= pl.lpSum(x[i, j] for j in non_pref_sites)
    
    # Corrected temporal constraint: Delivery date must be within stripping window
    for i in active_orders.index:
        delivery_date = active_orders.loc[i, 'DeliveryDate']
        for j in all_fish_groups.index:
            if all_fish_groups.loc[j, 'Site'] != 'Dummy':
                stripping_start_date = all_fish_groups.loc[j, 'StrippingStartDate']
                stripping_stop_date = all_fish_groups.loc[j, 'StrippingStopDate']
                if delivery_date < stripping_start_date or delivery_date > stripping_stop_date:
                    prob += x[i, j] == 0
    
    # Solve the problem
    prob.solve()
    
    # Extract results
    results = active_orders.copy()
    results['AssignedGroup'] = None
    results['IsDummy'] = False
    for i in results.index:
        for j in all_fish_groups.index:
            if pl.value(x[i, j]) == 1:
                results.loc[i, 'AssignedGroup'] = all_fish_groups.loc[j, 'Site']
                results.loc[i, 'IsDummy'] = (all_fish_groups.loc[j, 'Site'] == 'Dummy')
                break
    
    # Combine with original orders to include cancelled ones
    all_results = orders.copy()
    all_results['AssignedGroup'] = None
    all_results['IsDummy'] = False
    for i, row in results.iterrows():
        orig_idx = all_results.index[all_results['OrderNr'] == row['OrderNr']]
        if len(orig_idx) > 0:
            all_results.loc[orig_idx[0], 'AssignedGroup'] = row['AssignedGroup']
            all_results.loc[orig_idx[0], 'IsDummy'] = row['IsDummy']
    
    # Mark cancelled orders
    cancelled_idx = all_results[all_results['OrderStatus'] == 'Kansellert'].index
    all_results.loc[cancelled_idx, 'AssignedGroup'] = 'Skipped-Cancelled'
    all_results.loc[cancelled_idx, 'IsDummy'] = False
    
    # Calculate remaining capacity
    remaining = fish_groups.copy()
    remaining['Gain-eggs-used'] = 0
    remaining['Shield-eggs-used'] = 0
    for j, group in fish_groups.iterrows():
        site = group['Site']
        gain_used = all_results[(all_results['AssignedGroup'] == site) & (all_results['Product'] == 'Gain')]['Volume'].sum()
        shield_used = all_results[(all_results['AssignedGroup'] == site) & (all_results['Product'] == 'Shield')]['Volume'].sum()
        remaining.loc[j, 'Gain-eggs-used'] = gain_used
        remaining.loc[j, 'Shield-eggs-used'] = shield_used
        remaining.loc[j, 'Gain-eggs-remaining'] = group['Gain-eggs'] - gain_used
        remaining.loc[j, 'Shield-eggs-remaining'] = group['Shield-eggs'] - shield_used
    
    return {
        'status': pl.LpStatus[prob.status],
        'results': all_results,
        'remaining_capacity': remaining
    }

# Updated buffer graph function to reflect assigned orders
def create_buffer_graph(remaining, results):
    remaining['StrippingStartDate'] = pd.to_datetime(remaining['StrippingStartDate'])
    remaining['StrippingStopDate'] = pd.to_datetime(remaining['StrippingStopDate'])
    
    # Initialize remaining capacity with original values
    buffer_data = []
    facilities = remaining['Site'].unique()
    start_date = remaining['StrippingStartDate'].min()
    end_date = remaining['StrippingStopDate'].max() + pd.Timedelta(weeks=4)
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    
    # Create a copy of remaining capacity to track changes over time
    current_remaining = remaining.copy()
    
    for week in weekly_dates:
        for facility in facilities:
            facility_groups = current_remaining[current_remaining['Site'] == facility].iloc[0]
            total_gain = facility_groups['Gain-eggs-remaining']
            total_shield = facility_groups['Shield-eggs-remaining']
            total_remaining = total_gain + total_shield
            
            # Adjust remaining capacity based on assigned orders up to this week
            assigned_orders = results[
                (results['AssignedGroup'] == facility) & 
                (pd.to_datetime(results['DeliveryDate']) <= week) &
                (results['IsDummy'] == False)
            ]
            for _, order in assigned_orders.iterrows():
                if order['Product'] == 'Gain':
                    total_gain -= order['Volume']
                elif order['Product'] == 'Shield':
                    total_shield -= order['Volume']
            
            # Update current remaining for next iteration
            current_remaining.loc[current_remaining['Site'] == facility, 'Gain-eggs-remaining'] = total_gain
            current_remaining.loc[current_remaining['Site'] == facility, 'Shield-eggs-remaining'] = total_shield
            total_remaining = max(total_gain + total_shield, 0)  # Ensure non-negative
            
            buffer_data.append({
                'Week': week,
                'Facility': facility,
                'TotalRemaining': total_remaining / 1e6  # Convert to millions for readability
            })
    
    buffer_df = pd.DataFrame(buffer_data)
    
    fig = px.line(buffer_df, x='Week', y='TotalRemaining', color='Facility',
                  title='Weekly Inventory Buffer per Facility', markers=True)
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Available Roe (Millions)",
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=14),
        legend_title_text="Facility",
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    return fig

# Table styling function for consistency
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
        ]
    }

# App layout with Bootstrap and enhanced styling
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
            html.H3("Problem Description", className="mt-4"),
            dcc.Markdown("""
                ### Fish Egg Allocation Problem
                
                **Key Objectives:**
                - Reserve customer orders against fish groups/cylinders based on delivery dates and temperature conditions.
                - Automate the allocation to reduce manual work.
                - Manage waste by optimizing the splitting of fish groups into batches.
                - Visualize inventory buffers with a weekly graph per production facility.
            """, className="p-3 bg-light rounded"),
            dbc.Button('Solve Allocation Problem', id='solve-button', n_clicks=0, color="success", size="lg", className="mt-4"),
        ], width=6, className="p-4"),
    ], className="my-4"),
    
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(id='results-section', className="mt-4", style={'display': 'none'}, children=[
            html.H2("Solver Results", className="text-center mb-4"),
            dbc.Row([
                dbc.Col([
                    html.H3("Order Assignments", className="mt-4"),
                    dash_table.DataTable(id='results-table', **get_table_style()),
                    html.H3("Remaining Capacity", className="mt-4"),
                    dash_table.DataTable(id='capacity-table', **get_table_style()),
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Weekly Inventory Buffer", className="mt-4"),
                    dcc.Graph(id='buffer-visualization'),
                ], width=12),
            ]),
        ])
    )
], fluid=True)

# Callbacks for adding rows to tables
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

# Callback for solving the problem and displaying results
@app.callback(
    [Output('results-section', 'style'),
     Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('capacity-table', 'data'),
     Output('capacity-table', 'columns'),
     Output('buffer-visualization', 'figure'),
     Output('solve-button', 'children')],
    Input('solve-button', 'n_clicks'),
    [State('order-table', 'data'),
     State('fish-group-table', 'data')],
    prevent_initial_call=True
)
def update_results(n_clicks, order_data, fish_group_data):
    if n_clicks > 0:
        # Convert data to DataFrame
        orders_df = pd.DataFrame(order_data)
        fish_groups_df = pd.DataFrame(fish_group_data)
        
        # Convert numeric and boolean columns
        if not orders_df.empty:
            for col in ['Volume']:
                if col in orders_df.columns:
                    orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')
            orders_df['Organic'] = orders_df['Organic'].apply(
                lambda x: True if str(x).lower() in ['true', '1', 't', 'y', 'yes', 'organic'] else False
            )
        if not fish_groups_df.empty:
            for col in ['Gain-eggs', 'Shield-eggs']:
                if col in fish_groups_df.columns:
                    fish_groups_df[col] = pd.to_numeric(fish_groups_df[col], errors='coerce')
            fish_groups_df['Organic'] = fish_groups_df['Organic'].apply(
                lambda x: True if str(x).lower() in ['true', '1', 't', 'y', 'yes', 'organic'] else False
            )
        
        # Solve the allocation problem
        solution = solve_egg_allocation(orders_df, fish_groups_df)
        
        # Prepare results
        results_df = solution['results']
        capacity_df = solution['remaining_capacity']
        
        # Prepare columns for DataTable
        results_columns = [{'name': col, 'id': col} for col in results_df.columns]
        capacity_columns = [{'name': col, 'id': col} for col in capacity_df.columns]
        
        # Create weekly inventory buffer visualization with adjusted capacity
        buffer_fig = create_buffer_graph(capacity_df, results_df)
        
        # Update button text with solver status
        status = solution['status']
        button_text = f"Solve Allocation Problem (Status: {status})"
        
        return (
            {'margin': '20px', 'display': 'block'},
            results_df.to_dict('records'),
            results_columns,
            capacity_df.to_dict('records'),
            capacity_columns,
            buffer_fig,
            button_text
        )
    
    return ({'display': 'none'}, [], [], [], [], {}, "Solve Allocation Problem")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
