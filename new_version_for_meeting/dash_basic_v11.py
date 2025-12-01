import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pulp as pl
from pulp import PULP_CBC_CMD
import sys
import io

# ==========================================
# PARAMS for Advanced Model (NEW)
# ==========================================
ADVANCED_MODEL_PARAMS = {
    "TemperatureGridPoints": 5,
    # These values are derived from the example data's degree-day requirements
    "DaysToReadyDD": 300,  # Corresponds to MinTemp_prod
    "ShelfLifeDD": 340,    # Corresponds to (MaxTemp_prod - MinTemp_prod) -> (640 - 300)
    "WaterTemp": 8.0       # Default water temperature for calculations
}

# ==========================================
# CORE ALLOCATION & VISUALIZATION LOGIC
# ==========================================

def generate_batches(fish_group, water_temp=8.0):
    """Generate weekly batches with normal distribution"""
    try:
        strip_start = pd.to_datetime(fish_group['StrippingStartDate'])
        strip_stop = pd.to_datetime(fish_group['StrippingStopDate'])
        weeks = pd.date_range(strip_start, strip_stop, freq='W-MON')
        if len(weeks) == 0:
            weeks = pd.DatetimeIndex([strip_start])
        if len(weeks) > 5:
            weeks = weeks[:5]
        n = len(weeks)
        indices = np.arange(n)
        weights = np.exp(-0.5 * ((indices - (n - 1) / 2) / max(n / 4, 1)) ** 2)
        weights = weights / weights.sum()
        batches = []
        for i, strip_date in enumerate(weeks):
            maturation_days = fish_group['MinTemp_prod'] / water_temp
            production_days = fish_group['MaxTemp_prod'] / water_temp
            maturation_end = strip_date + timedelta(days=maturation_days)
            production_end = strip_date + timedelta(days=production_days)
            batches.append({
                'BatchID': f"{fish_group['Site_Broodst_Season']}_W{i+1}",
                'Group': fish_group['Site_Broodst_Season'],
                'Site': fish_group['Site'],
                'StripDate': strip_date,
                'MaturationEnd': maturation_end,
                'ProductionEnd': production_end,
                'GainCapacity': float(fish_group['Gain-eggs']) * weights[i],
                'ShieldCapacity': float(fish_group['Shield-eggs']) * weights[i],
                'MinTemp_prod': fish_group['MinTemp_prod'],
                'MaxTemp_prod': fish_group['MaxTemp_prod'],
                'Organic': fish_group['Organic']
            })
        return pd.DataFrame(batches)
    except Exception as e:
        print(f"Error generating batches: {e}")
        raise

def calculate_green_window(batch, order, water_temp=8.0):
    """Calculate customer-compatible delivery window (GREEN zone)"""
    try:
        cust_min_days = order['MinTemp_customer'] / water_temp
        cust_max_days = order['MaxTemp_customer'] / water_temp
        cust_min_date = batch['StripDate'] + timedelta(days=cust_min_days)
        cust_max_date = batch['StripDate'] + timedelta(days=cust_max_days)
        green_start = max(batch['MaturationEnd'], cust_min_date)
        green_end = min(batch['ProductionEnd'], cust_max_date)
        if green_start >= green_end:
            return None, None
        return green_start, green_end
    except Exception as e:
        print(f"Error calculating green window: {e}")
        return None, None

def check_feasibility(batch, order, water_temp=8.0):
    """Check if order delivery date falls in GREEN window"""
    try:
        green_start, green_end = calculate_green_window(batch, order, water_temp)
        if green_start is None:
            return False
        delivery_date = pd.to_datetime(order['DeliveryDate'])
        return green_start <= delivery_date <= green_end
    except Exception as e:
        print(f"Error checking feasibility: {e}")
        return False

def solve_allocation(orders, batches):
    """Solve batch-level allocation using optimization"""
    try:
        feasible = {i: [] for i in orders.index}
        for order_idx, order in orders.iterrows():
            for batch_idx, batch in batches.iterrows():
                if check_feasibility(batch, order):
                    feasible[order_idx].append(batch_idx)
        dummy_idx = len(batches)
        for i in orders.index:
            feasible[i].append(dummy_idx)
        prob = pl.LpProblem("EggAllocation", pl.LpMinimize)
        x = {(i, j): pl.LpVariable(f"x_{i}_{j}", cat="Binary") for i in orders.index for j in feasible[i]}
        prob += pl.lpSum(1000000 * x[i, j] if j == dummy_idx else 0 for (i, j) in x.keys())
        for i in orders.index:
            prob += pl.lpSum(x[i, j] for j in feasible[i]) == 1
        for j in batches.index:
            gain_orders = [i for i in orders.index if orders.loc[i, 'Product'] in ['Gain', 'Elite', 'Nucleus'] and j in feasible[i]]
            if gain_orders:
                prob += pl.lpSum(x[i, j] * orders.loc[i, 'Volume'] for i in gain_orders) <= batches.loc[j, 'GainCapacity']
            shield_orders = [i for i in orders.index if orders.loc[i, 'Product'] == 'Shield' and j in feasible[i]]
            if shield_orders:
                prob += pl.lpSum(x[i, j] * orders.loc[i, 'Volume'] for i in shield_orders) <= batches.loc[j, 'ShieldCapacity']
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
        results = orders.copy()
        results['AssignedBatch'] = 'UNASSIGNED'
        results['Success'] = False
        for i in orders.index:
            for j in feasible[i]:
                var_val = pl.value(x.get((i, j)))
                if var_val and round(var_val) == 1:
                    if j != dummy_idx:
                        results.loc[i, 'AssignedBatch'] = batches.loc[j, 'BatchID']
                        results.loc[i, 'Success'] = True
                    break
        return results
    except Exception as e:
        print(f"Error in solve_allocation: {e}")
        raise

def create_visualization(batches_df, results):
    """Create Gantt-style visualization of batch windows"""
    try:
        fig = go.Figure()
        groups = batches_df['Group'].unique()
        y_positions = {group: i for i, group in enumerate(groups)}
        y_range = [0, len(groups) - 1 + 0.5] if groups.size > 0 else [-1, 1]
        for _, batch in batches_df.iterrows():
            y_base = y_positions[batch['Group']]
            batch_offset = list(batches_df[batches_df['Group'] == batch['Group']].index).index(batch.name) * 0.15
            y_pos = y_base + batch_offset
            fig.add_trace(go.Scatter(x=[batch['StripDate'], batch['MaturationEnd']], y=[y_pos, y_pos], mode='lines', line=dict(color='#1f77b4', width=20), name='Maturation', legendgroup='maturation', showlegend=(batch.name == 0), hovertemplate=f"<b>{batch['BatchID']}</b><br>Maturation<br>%{{x}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[batch['MaturationEnd'], batch['ProductionEnd']], y=[y_pos, y_pos], mode='lines', line=dict(color='#ff7f0e', width=20), name='Production Window', legendgroup='production', showlegend=(batch.name == 0), hovertemplate=f"<b>{batch['BatchID']}</b><br>Production<br>%{{x}}<extra></extra>"))
            for order_idx, order in results.iterrows():
                green_start, green_end = calculate_green_window(batch, order)
                if green_start is not None:
                    fig.add_trace(go.Scatter(x=[green_start, green_end], y=[y_pos, y_pos], mode='lines', line=dict(color='#2ca02c', width=15), name='Customer Window', legendgroup='green', showlegend=(batch.name == 0 and order_idx == 0), hovertemplate=f"<b>{batch['BatchID']}</b><br>Green Zone (Order {order['OrderNr']})<br>%{{x}}<extra></extra>"))
        for _, order in results.iterrows():
            delivery_date = pd.to_datetime(order['DeliveryDate']).to_pydatetime()
            color = 'green' if order['Success'] else 'red'
            fig.add_vline(x=delivery_date, line_dash="dash", line_color=color, line_width=2)
            fig.add_annotation(x=delivery_date, y=1.05, yref="paper", text=f"Order {order['OrderNr']}", showarrow=False, font=dict(color=color))
        fig.update_layout(title="Batch-Level Delivery Window Analysis", xaxis_title="Date", yaxis_title="Fish Group", yaxis=dict(tickmode='array', tickvals=list(y_positions.values()), ticktext=list(y_positions.keys()), range=y_range), height=600, hovermode='closest', showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().update_layout(title_text="Error creating visualization. Check console for details.")

# ==========================================
# NEW ADVANCED PREPROCESSING FUNCTION
# ==========================================
def build_advanced_feasibility_set(orders_df, fish_groups_df, params):
    """
    Implements the advanced preprocessing logic from the specification.
    Discretizes temperature and calculates all feasible (order, group, temp) tuples.
    """
    print("\nBuilding advanced feasibility set...")
    
    feasible_combinations = []
    
    # Calculate days based on degree-days and water temperature
    days_to_ready = params["DaysToReadyDD"] / params["WaterTemp"]
    shelf_days = params["ShelfLifeDD"] / params["WaterTemp"]

    for o_idx, order in orders_df.iterrows():
        for g_idx, group in fish_groups_df.iterrows():
            
            # 1. Find temperature overlap (using degree-days as a proxy for temperature)
            t_low = max(group['MinTemp_prod'], order['MinTemp_customer'])
            t_high = min(group['MaxTemp_prod'], order['MaxTemp_customer'])

            # If no overlap, this pairing is impossible
            if t_low >= t_high:
                continue

            # 2. Discretize temperature candidates (in degree-days)
            temp_candidates = np.linspace(t_low, t_high, params["TemperatureGridPoints"])

            for temp_candidate_dd in temp_candidates:
                # This logic calculates a single window based on the group's latest strip date.
                # A more complex model could vary this based on the temp_candidate.
                strip_stop_date = pd.to_datetime(group['StrippingStopDate'])
                
                ready_date = strip_stop_date + timedelta(days=days_to_ready)
                expiry_date = ready_date + timedelta(days=shelf_days)
                
                required_delivery_date = pd.to_datetime(order['DeliveryDate'])

                # 3. Check if the delivery date is within the window
                if ready_date <= required_delivery_date <= expiry_date:
                    feasible_combinations.append({
                        'OrderNr': order['OrderNr'],
                        'Group': group['Site_Broodst_Season'],
                        'TemperatureDD': round(temp_candidate_dd, 2),
                        'DeliveryDate': required_delivery_date.date(),
                        'Volume': order['Volume']
                    })

    print(f"Found {len(feasible_combinations)} feasible (order, group, temp) combinations.")
    if not feasible_combinations:
        return pd.DataFrame()
        
    return pd.DataFrame(feasible_combinations)


# ==========================================
# EXAMPLE DATA
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

# ==========================================
# DASH APP LAYOUT
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Batch-Based Egg Allocation Pilot", className="text-center my-4 p-3 text-white bg-primary rounded"), width=12)),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Input Data")),
                dbc.CardBody([
                    html.H5("Fish Groups"),
                    dash_table.DataTable(id='fish-groups-table', data=fish_groups.to_dict('records'), style_table={'overflowX': 'auto'}),
                    html.Hr(),
                    html.H5("Orders"),
                    dash_table.DataTable(id='orders-table', data=orders.to_dict('records'), style_table={'overflowX': 'auto'}),
                ])
            ]),
        ], width=12)
    ], className="mb-4"),
    dbc.Row(dbc.Col(dbc.Button("Run Allocation and Generate Report", id="run-button", color="success", size="lg", className="w-100"), width=12), className="mb-4"),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="results-output"))
], fluid=True)


# ==========================================
# DASH APP CALLBACK
# ==========================================
@app.callback(
    Output("results-output", "children"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True
)
def run_allocation_report(n_clicks):
    # This callback still uses the OLD logic for now.
    # We will integrate the new logic in the next step.
    all_batches = [generate_batches(group) for _, group in fish_groups.iterrows()]
    batches_df = pd.concat(all_batches, ignore_index=True)
    results = solve_allocation(orders, batches_df)
    fig = create_visualization(batches_df, results)
    summary_stats = html.Div([
        dbc.Row([
            dbc.Col(html.P(f"Total Orders: {len(orders)}"), width=3),
            dbc.Col(html.P(f"Successfully Assigned: {results['Success'].sum()}"), width=3),
            dbc.Col(html.P(f"Unassigned: {(~results['Success']).sum()}"), width=3),
            dbc.Col(html.P(f"Success Rate: {results['Success'].mean() * 100:.1f}%"), width=3),
        ])
    ])
    utilization_data = []
    for _, batch in batches_df.iterrows():
        assigned = results[results['AssignedBatch'] == batch['BatchID']]
        if not assigned.empty:
            gain_used = assigned[assigned['Product'].isin(['Gain', 'Elite', 'Nucleus'])]['Volume'].sum()
            shield_used = assigned[assigned['Product'] == 'Shield']['Volume'].sum()
            total_cap = batch['GainCapacity'] + batch['ShieldCapacity']
            util_pct = ((gain_used + shield_used) / total_cap * 100) if total_cap > 0 else 0
            utilization_data.append({
                'BatchID': batch['BatchID'],
                'Utilization': f"{util_pct:.1f}%",
                'Gain Used / Capacity': f"{gain_used:,.0f} / {batch['GainCapacity']:,.0f}",
                'Shield Used / Capacity': f"{shield_used:,.0f} / {batch['ShieldCapacity']:,.0f}",
            })
    utilization_df = pd.DataFrame(utilization_data)
    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.H4("Results")),
            dbc.CardBody([
                html.H5("Summary", className="mt-3"),
                summary_stats,
                html.Hr(),
                html.H5("Allocation Results", className="mt-3"),
                dash_table.DataTable(
                    data=results.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Success} = false', 'column_id': 'OrderNr'}, 'backgroundColor': '#ffdddd'},
                        {'if': {'filter_query': '{Success} = true', 'column_id': 'OrderNr'}, 'backgroundColor': '#ddffdd'}
                    ]
                ),
                html.Hr(),
                html.H5("Batch Utilization", className="mt-3"),
                dash_table.DataTable(data=utilization_df.to_dict('records'), style_table={'overflowX': 'auto'}),
                html.Hr(),
                html.H5("Delivery Window Analysis", className="mt-3"),
                dcc.Graph(figure=fig)
            ])
        ])
    ])

# ==========================================
# SCRIPT EXECUTION & TESTING (NEW TEST BLOCK)
# ==========================================
if __name__ == '__main__':
    # This block runs only when you execute the script directly.
    # It allows us to test the new logic without affecting the Dash app.
    
    print("--- RUNNING DEFENSIVE TEST FOR NEW LOGIC ---")
    
    # Use the example data defined in the script to test the new function
    feasible_set_df = build_advanced_feasibility_set(orders, fish_groups, ADVANCED_MODEL_PARAMS)
    
    if not feasible_set_df.empty:
        print("\n--- Feasible Set (First 5 Rows) ---")
        print(feasible_set_df.head())
        
        print("\n--- Feasible Options for Order 1001 ---")
        print(feasible_set_df[feasible_set_df['OrderNr'] == 1001])

        print("\n--- Feasible Options for Order 1002 ---")
        print(feasible_set_df[feasible_set_df['OrderNr'] == 1002])

        print("\n--- Feasible Options for Order 1003 ---")
        print(feasible_set_df[feasible_set_df['OrderNr'] == 1003])
    else:
        print("No feasible combinations were found by the new logic.")

    print("\n--- STARTING DASH APP (The app still uses the OLD logic for now) ---")
    app.run_server(debug=True)
