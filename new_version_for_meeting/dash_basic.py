# %%
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pulp as pl
import plotly.express as px
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

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

# Solver function
def solve_egg_allocation(orders, fish_groups):
    # Convert dates to datetime
    orders['DeliveryDate'] = pd.to_datetime(orders['DeliveryDate'])
    fish_groups['StrippingStartDate'] = pd.to_datetime(fish_groups['StrippingStartDate'])
    fish_groups['StrippingStopDate'] = pd.to_datetime(fish_groups['StrippingStopDate'])
    
    # Filter out cancelled orders
    active_orders = orders[orders['OrderStatus'] != 'Kansellert'].copy()
    active_orders = active_orders.reset_index(drop=True)
    
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
    dummy_penalties = {}
    for i in active_orders.index:
        dummy_penalties[i] = pl.LpVariable(f"dummy_penalty_{i}", lowBound=0)
    
    pref_site_penalties = {}
    for i, order in active_orders.iterrows():
        if pd.notna(order['PreferredSite']):
            pref_site_penalties[i] = pl.LpVariable(f"pref_site_penalty_{i}", lowBound=0)
    
    # Objective function
    prob += (
        1000 * pl.lpSum(dummy_penalties.values()) +
        10 * pl.lpSum(pref_site_penalties.values())
    )
    
    # Each order must be assigned to exactly one group
    for i in active_orders.index:
        prob += pl.lpSum(x[i, j] for j in all_fish_groups.index) == 1
    
    # Capacity constraints
    for j in all_fish_groups.index:
        if all_fish_groups.loc[j, 'Site'] != 'Dummy':
            # Gain eggs capacity
            prob += pl.lpSum(
                x[i, j] * active_orders.loc[i, 'Volume'] 
                for i in active_orders.index 
                if active_orders.loc[i, 'Product'] == 'Gain'
            ) <= all_fish_groups.loc[j, 'Gain-eggs']
            
            # Shield eggs capacity
            prob += pl.lpSum(
                x[i, j] * active_orders.loc[i, 'Volume'] 
                for i in active_orders.index 
                if active_orders.loc[i, 'Product'] == 'Shield'
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
    
    # Define dummy penalties
    for i in active_orders.index:
        prob += dummy_penalties[i] >= x[i, dummy_site_idx]
    
    # Define preferred site penalties
    for i in active_orders.index:
        if pd.notna(active_orders.loc[i, 'PreferredSite']):
            pref_site = active_orders.loc[i, 'PreferredSite']
            non_pref_sites = [j for j in all_fish_groups.index 
                             if all_fish_groups.loc[j, 'Site'] != pref_site 
                             and all_fish_groups.loc[j, 'Site'] != 'Dummy']
            if non_pref_sites:
                prob += pref_site_penalties[i] >= pl.lpSum(x[i, j] for j in non_pref_sites)
    
    # Solve the problem
    solver_status = prob.solve()
    
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
    
    # Set cancelled orders
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

# App layout
app.layout = html.Div([
    html.H1("Fish Egg Allocation Solver", style={'textAlign': 'center', 'margin': '20px'}),
    
    html.Div([
        html.Div([
            html.H3("Orders"),
            dash_table.DataTable(
                id='order-table',
                data=default_orders.to_dict('records'),
                columns=[{'name': col, 'id': col, 'editable': True} for col in default_orders.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': 'lightblue'},
                editable=True,
                row_deletable=True,
            ),
            html.Button('Add Order Row', id='add-order-row-button', n_clicks=0, style={'margin': '10px'}),
            
            html.H3("Fish Groups"),
            dash_table.DataTable(
                id='fish-group-table',
                data=default_fish_groups.to_dict('records'),
                columns=[{'name': col, 'id': col, 'editable': True} for col in default_fish_groups.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': 'lightblue'},
                editable=True,
                row_deletable=True,
            ),
            html.Button('Add Fish Group Row', id='add-fish-group-row-button', n_clicks=0, style={'margin': '10px'}),
        ], style={'margin': '20px'}),
        
        html.Div([
            html.H3("Problem Description"),
            dcc.Markdown("""
                ### Fish Egg Allocation Problem
                
                **Priority 1 Constraints:**
                * All customer orders (except cancelled) must be assigned to a fish group
                * If no suitable fish group is available, assign to a dummy group
                * Orders should get priority from preferred/agreed facility when possible
                
                **Solver Approach:**
                1. Create binary variables for order-to-fish-group assignments
                2. Set penalties for using dummy group and not using preferred sites
                3. Add constraints for capacity, organic requirements, and locked sites
                4. Optimize to minimize penalties, ensuring best possible allocation
            """),
            
            html.Button(
                'Solve Allocation Problem', 
                id='solve-button', 
                n_clicks=0, 
                style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px', 'fontSize': '16px', 'margin': '20px 0'}
            ),
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    html.Div(id='results-section', style={'margin': '20px', 'display': 'none'}, children=[
        html.H2("Allocation Results"),
        html.Div([
            html.H3("Order Assignments"),
            dash_table.DataTable(
                id='results-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': 'lightblue'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{IsDummy} eq true',
                        },
                        'backgroundColor': '#ffcccb',
                    },
                    {
                        'if': {
                            'filter_query': '{AssignedGroup} eq "Skipped-Cancelled"',
                        },
                        'backgroundColor': '#d3d3d3',
                    }
                ]
            ),
            
            html.H3("Remaining Capacity"),
            dash_table.DataTable(
                id='capacity-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': 'lightblue'},
            ),
        ]),
        
        html.Div([
            html.H3("Visualization"),
            dcc.Graph(id='allocation-visualization'),
        ], style={'marginTop': '30px'})
    ])
])

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
     Output('allocation-visualization', 'figure')],
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
        
        # Convert numeric columns
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
        
        # Create visualization - Sankey diagram
        sankey_fig = create_sankey_diagram(results_df)
        
        return (
            {'margin': '20px', 'display': 'block'},
            results_df.to_dict('records'),
            results_columns,
            capacity_df.to_dict('records'),
            capacity_columns,
            sankey_fig
        )
    
    # Default return
    return (
        {'display': 'none'},
        [], [],
        [], [], 
        {}
    )

def create_sankey_diagram(results_df):
    # Filter out cancelled orders
    active_results = results_df[results_df['AssignedGroup'] != 'Skipped-Cancelled'].copy()
    
    # Prepare source, target and value lists for Sankey diagram
    source = []
    target = []
    value = []
    
    # All unique orders and fish groups
    orders = active_results['OrderNr'].tolist()
    fish_groups = active_results['AssignedGroup'].unique().tolist()
    
    # Create node labels
    node_labels = orders + fish_groups
    
    # Fill source, target and value lists
    for idx, row in active_results.iterrows():
        source_idx = node_labels.index(row['OrderNr'])
        target_idx = node_labels.index(row['AssignedGroup'])
        source.append(source_idx)
        target.append(target_idx)
        value.append(row['Volume'])
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=["blue" if i < len(orders) else "green" for i in range(len(node_labels))]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(
        title_text="Order to Fish Group Allocation",
        font_size=12,
        height=600
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


