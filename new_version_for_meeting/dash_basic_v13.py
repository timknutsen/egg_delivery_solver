import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pulp as pl
from pulp import PULP_CBC_CMD
import traceback

# ==========================================
# PARAMS for Advanced Model
# ==========================================
ADVANCED_MODEL_PARAMS = {
    "TemperatureGridPoints": 5,
    "DaysToReadyDD": 300,
    "ShelfLifeDD": 340,
    "WaterTemp": 8.0
}

# ==========================================
# ADVANCED LOGIC
# ==========================================

def build_advanced_feasibility_set(orders_df, fish_groups_df, params):
    """Creates a DataFrame of all feasible (order, group, temp) combinations."""
    feasible_combinations = []
    days_to_ready = params["DaysToReadyDD"] / params["WaterTemp"]
    shelf_days = params["ShelfLifeDD"] / params["WaterTemp"]
    for o_idx, order in orders_df.iterrows():
        for g_idx, group in fish_groups_df.iterrows():
            t_low = max(group['MinTemp_prod'], order['MinTemp_customer'])
            t_high = min(group['MaxTemp_prod'], order['MaxTemp_customer'])
            if t_low >= t_high: continue
            temp_candidates = np.linspace(t_low, t_high, params["TemperatureGridPoints"])
            for temp_candidate_dd in temp_candidates:
                strip_stop_date = pd.to_datetime(group['StrippingStopDate'])
                ready_date = strip_stop_date + timedelta(days=days_to_ready)
                expiry_date = ready_date + timedelta(days=shelf_days)
                required_delivery_date = pd.to_datetime(order['DeliveryDate'])
                if ready_date <= required_delivery_date <= expiry_date:
                    feasible_combinations.append({
                        'OrderNr': order['OrderNr'],
                        'Group': group['Site_Broodst_Season'],
                        'TemperatureDD': round(temp_candidate_dd, 2),
                        'DeliveryDate': required_delivery_date.date(),
                        'Volume': order['Volume']
                    })
    return pd.DataFrame(feasible_combinations) if feasible_combinations else pd.DataFrame()

def solve_advanced_allocation(orders_df, fish_groups_df, feasible_set_df):
    """Solves allocation using the advanced, temperature-discretized model."""
    if feasible_set_df.empty:
        print("Feasible set is empty. No solution possible.")
        # Return a structure indicating which orders failed
        unallocated = orders_df[['OrderNr']].copy()
        unallocated['Reason'] = 'No feasible production window found'
        return pd.DataFrame(), unallocated

    prob = pl.LpProblem("AdvancedEggAllocation", pl.LpMinimize)
    feasible_set_df['id'] = feasible_set_df.index
    y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in feasible_set_df['id']}
    
    objective_terms = []
    for idx, row in feasible_set_df.iterrows():
        order_info = orders_df[orders_df['OrderNr'] == row['OrderNr']].iloc[0]
        t_mid = (order_info['MinTemp_customer'] + order_info['MaxTemp_customer']) / 2
        cost = abs(row['TemperatureDD'] - t_mid)
        objective_terms.append(cost * y[row['id']])
    prob += pl.lpSum(objective_terms)

    for order_nr in orders_df['OrderNr']:
        choices_for_order = feasible_set_df[feasible_set_df['OrderNr'] == order_nr]['id']
        if len(choices_for_order) > 0:
             prob += pl.lpSum(y[i] for i in choices_for_order) == 1

    for g_idx, group in fish_groups_df.iterrows():
        group_name = group['Site_Broodst_Season']
        total_capacity = group['Gain-eggs'] + group['Shield-eggs']
        choices_in_group = feasible_set_df[feasible_set_df['Group'] == group_name]
        if not choices_in_group.empty:
            prob += pl.lpSum(y[i] * choices_in_group.loc[i, 'Volume'] for i in choices_in_group['id']) <= total_capacity

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    
    allocated_orders = []
    if pl.LpStatus[prob.status] == 'Optimal':
        for i, var in y.items():
            if pl.value(var) and round(pl.value(var)) == 1:
                allocated_orders.append(feasible_set_df.loc[i])
    
    allocated_df = pd.DataFrame(allocated_orders)
    
    # Identify unallocated orders
    allocated_order_nrs = allocated_df['OrderNr'].unique() if not allocated_df.empty else []
    unallocated_df = orders_df[~orders_df['OrderNr'].isin(allocated_order_nrs)].copy()
    unallocated_df['Reason'] = 'Could not be assigned due to capacity or other constraints'

    return allocated_df, unallocated_df

def create_advanced_visualization(fish_groups_df, allocated_df):
    """Creates a Gantt chart for the advanced model results, focusing on groups."""
    try:
        fig = go.Figure()
        
        # Use a copy to avoid SettingWithCopyWarning
        groups_df = fish_groups_df.copy()
        groups_df['StrippingStartDate'] = pd.to_datetime(groups_df['StrippingStartDate'])
        groups_df['StrippingStopDate'] = pd.to_datetime(groups_df['StrippingStopDate'])
        
        y_positions = {group: i for i, group in enumerate(groups_df['Site_Broodst_Season'])}
        
        # Calculate overall production window for each group
        days_to_ready = ADVANCED_MODEL_PARAMS["DaysToReadyDD"] / ADVANCED_MODEL_PARAMS["WaterTemp"]
        shelf_days = ADVANCED_MODEL_PARAMS["ShelfLifeDD"] / ADVANCED_MODEL_PARAMS["WaterTemp"]

        for _, group in groups_df.iterrows():
            group_name = group['Site_Broodst_Season']
            y_pos = y_positions[group_name]
            
            # The earliest possible delivery date
            earliest_delivery = group['StrippingStopDate'] + timedelta(days=days_to_ready)
            # The latest possible delivery date
            latest_delivery = earliest_delivery + timedelta(days=shelf_days)
            
            # Plot the main production window (like the old 'Red' bar)
            fig.add_trace(go.Scatter(
                x=[earliest_delivery, latest_delivery], y=[y_pos, y_pos],
                mode='lines', line=dict(color='#ff7f0e', width=20),
                name=f'Delivery Window', legendgroup=group_name,
                hovertemplate=f"<b>{group_name}</b><br>Delivery Window<br>%{{x}}<extra></extra>"
            ))

        # Plot allocated orders as markers on the chart
        if not allocated_df.empty:
            for _, allocation in allocated_df.iterrows():
                y_pos = y_positions.get(allocation['Group'])
                if y_pos is not None:
                    fig.add_trace(go.Scatter(
                        x=[allocation['DeliveryDate']], y=[y_pos],
                        mode='markers', marker=dict(color='green', size=15, symbol='star'),
                        name=f"Order {allocation['OrderNr']}", legendgroup='allocations',
                        hovertemplate=f"<b>Order {allocation['OrderNr']}</b><br>Group: {allocation['Group']}<br>Temp: {allocation['TemperatureDD']} DD<br>Date: {allocation['DeliveryDate']}<extra></extra>"
                    ))

        fig.update_layout(
            title="Group Delivery Windows & Order Allocations",
            xaxis_title="Date", yaxis_title="Fish Group",
            yaxis=dict(tickmode='array', tickvals=list(y_positions.values()), ticktext=list(y_positions.keys())),
            height=600, hovermode='closest', showlegend=True
        )
        return fig
    except Exception as e:
        print(f"Error creating visualization: {e}"); traceback.print_exc()
        return go.Figure().update_layout(title_text="Error creating visualization.")

# ==========================================
# EXAMPLE DATA & DASH APP LAYOUT
# ==========================================
fish_groups = pd.DataFrame([
    {'Site': 'Hemne', 'Site_Broodst_Season': 'Hemne_Normal_24/25', 'StrippingStartDate': '2024-09-01', 'StrippingStopDate': '2024-09-22', 'MinTemp_prod': 300, 'MaxTemp_prod': 640, 'Gain-eggs': 5000000.0, 'Shield-eggs': 0.0, 'Organic': False},
    {'Site': 'Vestseøra', 'Site_Broodst_Season': 'Vestseøra_Organic_24/25', 'StrippingStartDate': '2024-08-25', 'StrippingStopDate': '2024-09-15', 'MinTemp_prod': 300, 'MaxTemp_prod': 640, 'Gain-eggs': 3000000.0, 'Shield-eggs': 2000000.0, 'Organic': True}
])
orders = pd.DataFrame([
    {'OrderNr': 1001, 'DeliveryDate': '2024-11-15', 'Product': 'Elite', 'Volume': 800000.0, 'MinTemp_customer': 400.0, 'MaxTemp_customer': 560.0},
    {'OrderNr': 1002, 'DeliveryDate': '2024-11-28', 'Product': 'Gain', 'Volume': 1200000.0, 'MinTemp_customer': 400.0, 'MaxTemp_customer': 560.0},
    {'OrderNr': 1003, 'DeliveryDate': '2024-12-10', 'Product': 'Shield', 'Volume': 600000.0, 'MinTemp_customer': 400.0, 'MaxTemp_customer': 560.0}
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Advanced Egg Allocation Planner", className="text-center my-4 p-3 text-white bg-primary rounded"), width=12)),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader(html.H4("Input Data")), dbc.CardBody([html.H5("Fish Groups"), dash_table.DataTable(id='fish-groups-table', data=fish_groups.to_dict('records'), style_table={'overflowX': 'auto'}), html.Hr(), html.H5("Orders"), dash_table.DataTable(id='orders-table', data=orders.to_dict('records'), style_table={'overflowX': 'auto'})])])], width=12)], className="mb-4"),
    dbc.Row(dbc.Col(dbc.Button("Run Advanced Allocation", id="run-button", color="success", size="lg", className="w-100"), width=12), className="mb-4"),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="results-output"))
], fluid=True)

# ==========================================
# FINAL, INTEGRATED DASH APP CALLBACK
# ==========================================
@app.callback(
    Output("results-output", "children"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True
)
def run_advanced_allocation_report(n_clicks):
    # Step 1: Build the feasibility set using the advanced logic
    feasible_set_df = build_advanced_feasibility_set(orders, fish_groups, ADVANCED_MODEL_PARAMS)
    
    # Step 2: Run the advanced solver
    allocated_df, unallocated_df = solve_advanced_allocation(orders, fish_groups, feasible_set_df)

    # Step 3: Create summary statistics
    total_orders = len(orders)
    assigned_count = len(allocated_df)
    unassigned_count = total_orders - assigned_count
    success_rate = (assigned_count / total_orders * 100) if total_orders > 0 else 0
    
    summary_stats = html.Div([
        dbc.Row([
            dbc.Col(html.P(f"Total Orders: {total_orders}"), width=3),
            dbc.Col(html.P(f"Successfully Assigned: {assigned_count}"), width=3),
            dbc.Col(html.P(f"Unassigned: {unassigned_count}"), width=3),
            dbc.Col(html.P(f"Success Rate: {success_rate:.1f}%"), width=3),
        ])
    ])

    # Step 4: Create the visualization
    fig = create_advanced_visualization(fish_groups, allocated_df)

    # Step 5: Format results for display
    # Combine allocated and unallocated for a single results table
    results_for_display = allocated_df[['OrderNr', 'Group', 'TemperatureDD', 'DeliveryDate', 'Volume']].copy()
    results_for_display.rename(columns={'Group': 'AssignedGroup'}, inplace=True)
    
    if not unallocated_df.empty:
        unallocated_display = unallocated_df[['OrderNr', 'Reason']].copy()
        unallocated_display['AssignedGroup'] = 'UNASSIGNED'
        results_for_display = pd.concat([results_for_display, unallocated_display], ignore_index=True)

    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.H4("Advanced Allocation Results")),
            dbc.CardBody([
                html.H5("Summary", className="mt-3"),
                summary_stats,
                html.Hr(),
                html.H5("Order Assignments", className="mt-3"),
                dash_table.DataTable(
                    data=results_for_display.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{AssignedGroup} = "UNASSIGNED"'}, 'backgroundColor': '#ffdddd'},
                        {'if': {'filter_query': '{AssignedGroup} != "UNASSIGNED"'},'backgroundColor': '#ddffdd'}
                    ]
                ),
                html.Hr(),
                html.H5("Delivery Window Analysis", className="mt-3"),
                dcc.Graph(figure=fig)
            ])
        ])
    ])

# ==========================================
# RUN THE APP
# ==========================================
if __name__ == '__main__':
    app.run_server(debug=True)
