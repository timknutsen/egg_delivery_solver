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
# PARAMS for Advanced Model
# ==========================================
ADVANCED_MODEL_PARAMS = {
    "TemperatureGridPoints": 5,
    "DaysToReadyDD": 300,
    "ShelfLifeDD": 340,
    "WaterTemp": 8.0
}

# ==========================================
# OLD LOGIC (Kept for the UI for now)
# ==========================================

def generate_batches(fish_group, water_temp=8.0):
    try:
        strip_start = pd.to_datetime(fish_group['StrippingStartDate'])
        strip_stop = pd.to_datetime(fish_group['StrippingStopDate'])
        weeks = pd.date_range(strip_start, strip_stop, freq='W-MON')
        if len(weeks) == 0: weeks = pd.DatetimeIndex([strip_start])
        if len(weeks) > 5: weeks = weeks[:5]
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
            batches.append({'BatchID': f"{fish_group['Site_Broodst_Season']}_W{i+1}", 'Group': fish_group['Site_Broodst_Season'], 'Site': fish_group['Site'], 'StripDate': strip_date, 'MaturationEnd': maturation_end, 'ProductionEnd': production_end, 'GainCapacity': float(fish_group['Gain-eggs']) * weights[i], 'ShieldCapacity': float(fish_group['Shield-eggs']) * weights[i], 'MinTemp_prod': fish_group['MinTemp_prod'], 'MaxTemp_prod': fish_group['MaxTemp_prod'], 'Organic': fish_group['Organic']})
        return pd.DataFrame(batches)
    except Exception as e:
        print(f"Error generating batches: {e}")
        raise

def solve_allocation(orders, batches):
    try:
        feasible = {i: [] for i in orders.index}
        for order_idx, order in orders.iterrows():
            for batch_idx, batch in batches.iterrows():
                if check_feasibility(batch, order):
                    feasible[order_idx].append(batch_idx)
        dummy_idx = len(batches)
        for i in orders.index: feasible[i].append(dummy_idx)
        prob = pl.LpProblem("EggAllocation", pl.LpMinimize)
        x = {(i, j): pl.LpVariable(f"x_{i}_{j}", cat="Binary") for i in orders.index for j in feasible[i]}
        prob += pl.lpSum(1000000 * x[i, j] if j == dummy_idx else 0 for (i, j) in x.keys())
        for i in orders.index: prob += pl.lpSum(x[i, j] for j in feasible[i]) == 1
        for j in batches.index:
            gain_orders = [i for i in orders.index if orders.loc[i, 'Product'] in ['Gain', 'Elite', 'Nucleus'] and j in feasible[i]]
            if gain_orders: prob += pl.lpSum(x[i, j] * orders.loc[i, 'Volume'] for i in gain_orders) <= batches.loc[j, 'GainCapacity']
            shield_orders = [i for i in orders.index if orders.loc[i, 'Product'] == 'Shield' and j in feasible[i]]
            if shield_orders: prob += pl.lpSum(x[i, j] * orders.loc[i, 'Volume'] for i in shield_orders) <= batches.loc[j, 'ShieldCapacity']
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
        results = orders.copy()
        results['AssignedBatch'] = 'UNASSIGNED'; results['Success'] = False
        for i in orders.index:
            for j in feasible[i]:
                if pl.value(x.get((i, j))) and round(pl.value(x.get((i, j)))) == 1:
                    if j != dummy_idx:
                        results.loc[i, 'AssignedBatch'] = batches.loc[j, 'BatchID']; results.loc[i, 'Success'] = True
                    break
        return results
    except Exception as e:
        print(f"Error in solve_allocation: {e}")
        raise

def check_feasibility(batch, order, water_temp=8.0):
    try:
        green_start, green_end = calculate_green_window(batch, order, water_temp)
        if green_start is None: return False
        return green_start <= pd.to_datetime(order['DeliveryDate']) <= green_end
    except Exception as e:
        print(f"Error checking feasibility: {e}")
        return False
        
def calculate_green_window(batch, order, water_temp=8.0):
    try:
        cust_min_days = order['MinTemp_customer'] / water_temp; cust_max_days = order['MaxTemp_customer'] / water_temp
        cust_min_date = batch['StripDate'] + timedelta(days=cust_min_days); cust_max_date = batch['StripDate'] + timedelta(days=cust_max_days)
        green_start = max(batch['MaturationEnd'], cust_min_date); green_end = min(batch['ProductionEnd'], cust_max_date)
        if green_start >= green_end: return None, None
        return green_start, green_end
    except Exception as e:
        print(f"Error calculating green window: {e}")
        return None, None

def create_visualization(batches_df, results):
    try:
        fig = go.Figure()
        groups = batches_df['Group'].unique()
        y_positions = {group: i for i, group in enumerate(groups)}
        y_range = [0, len(groups) - 1 + 0.5] if groups.size > 0 else [-1, 1]
        for _, batch in batches_df.iterrows():
            y_base = y_positions[batch['Group']]; batch_offset = list(batches_df[batches_df['Group'] == batch['Group']].index).index(batch.name) * 0.15; y_pos = y_base + batch_offset
            fig.add_trace(go.Scatter(x=[batch['StripDate'], batch['MaturationEnd']], y=[y_pos, y_pos], mode='lines', line=dict(color='#1f77b4', width=20), name='Maturation', legendgroup='maturation', showlegend=(batch.name == 0), hovertemplate=f"<b>{batch['BatchID']}</b><br>Maturation<br>%{{x}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[batch['MaturationEnd'], batch['ProductionEnd']], y=[y_pos, y_pos], mode='lines', line=dict(color='#ff7f0e', width=20), name='Production Window', legendgroup='production', showlegend=(batch.name == 0), hovertemplate=f"<b>{batch['BatchID']}</b><br>Production<br>%{{x}}<extra></extra>"))
            for order_idx, order in results.iterrows():
                green_start, green_end = calculate_green_window(batch, order)
                if green_start is not None: fig.add_trace(go.Scatter(x=[green_start, green_end], y=[y_pos, y_pos], mode='lines', line=dict(color='#2ca02c', width=15), name='Customer Window', legendgroup='green', showlegend=(batch.name == 0 and order_idx == 0), hovertemplate=f"<b>{batch['BatchID']}</b><br>Green Zone (Order {order['OrderNr']})<br>%{{x}}<extra></extra>"))
        for _, order in results.iterrows():
            delivery_date = pd.to_datetime(order['DeliveryDate']).to_pydatetime(); color = 'green' if order['Success'] else 'red'
            fig.add_vline(x=delivery_date, line_dash="dash", line_color=color, line_width=2)
            fig.add_annotation(x=delivery_date, y=1.05, yref="paper", text=f"Order {order['OrderNr']}", showarrow=False, font=dict(color=color))
        fig.update_layout(title="Batch-Level Delivery Window Analysis", xaxis_title="Date", yaxis_title="Fish Group", yaxis=dict(tickmode='array', tickvals=list(y_positions.values()), ticktext=list(y_positions.keys()), range=y_range), height=600, hovermode='closest', showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    except Exception as e:
        print(f"Error creating visualization: {e}"); traceback.print_exc()
        return go.Figure().update_layout(title_text="Error creating visualization. Check console for details.")

# ==========================================
# ADVANCED LOGIC (NEW)
# ==========================================

def build_advanced_feasibility_set(orders_df, fish_groups_df, params):
    """Creates a DataFrame of all feasible (order, group, temp) combinations."""
    print("\nBuilding advanced feasibility set...")
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
                    feasible_combinations.append({'OrderNr': order['OrderNr'], 'Group': group['Site_Broodst_Season'], 'TemperatureDD': round(temp_candidate_dd, 2), 'DeliveryDate': required_delivery_date.date(), 'Volume': order['Volume']})
    print(f"Found {len(feasible_combinations)} feasible (order, group, temp) combinations.")
    return pd.DataFrame(feasible_combinations) if feasible_combinations else pd.DataFrame()

def solve_advanced_allocation(orders_df, fish_groups_df, feasible_set_df):
    """Solves allocation using the advanced, temperature-discretized model."""
    print("\nSolving with advanced allocation model...")
    if feasible_set_df.empty:
        print("Feasible set is empty. No solution possible.")
        return None

    prob = pl.LpProblem("AdvancedEggAllocation", pl.LpMinimize)
    
    # Create unique identifiers for each feasible choice
    feasible_set_df['id'] = feasible_set_df.index
    
    # Decision Variables
    # y[i] = 1 if feasible choice `i` is selected, 0 otherwise
    y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in feasible_set_df['id']}
    
    # z[o, g] = 1 if order `o` is assigned to group `g`
    order_group_pairs = feasible_set_df[['OrderNr', 'Group']].drop_duplicates().to_records(index=False)
    z = {(o, g): pl.LpVariable(f"z_{o}_{g}", cat="Binary") for o, g in order_group_pairs}

    # Objective Function: Minimize temperature deviation
    objective_terms = []
    for idx, row in feasible_set_df.iterrows():
        order_info = orders_df[orders_df['OrderNr'] == row['OrderNr']].iloc[0]
        t_mid = (order_info['MinTemp_customer'] + order_info['MaxTemp_customer']) / 2
        cost = abs(row['TemperatureDD'] - t_mid)
        objective_terms.append(cost * y[row['id']])
    prob += pl.lpSum(objective_terms)

    # Constraints
    # C1: Each order must be fully satisfied by exactly one choice
    for order_nr in orders_df['OrderNr']:
        choices_for_order = feasible_set_df[feasible_set_df['OrderNr'] == order_nr]['id']
        prob += pl.lpSum(y[i] for i in choices_for_order) == 1

    # C2: Capacity constraint per fish group (simplified for now)
    # This uses the total capacity of the group, not daily/weekly breakdown.
    for g_idx, group in fish_groups_df.iterrows():
        group_name = group['Site_Broodst_Season']
        total_capacity = group['Gain-eggs'] + group['Shield-eggs'] # Simplified capacity
        
        choices_in_group = feasible_set_df[feasible_set_df['Group'] == group_name]
        if not choices_in_group.empty:
            prob += pl.lpSum(y[i] * choices_in_group.loc[i, 'Volume'] for i in choices_in_group['id']) <= total_capacity

    # Solve
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    print(f"Solver status: {pl.LpStatus[prob.status]}")

    # Extract results
    results = []
    for i, var in y.items():
        if pl.value(var) and round(pl.value(var)) == 1:
            results.append(feasible_set_df.loc[i])
            
    return pd.DataFrame(results)

# ==========================================
# EXAMPLE DATA & DASH APP LAYOUT (Unchanged)
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
    dbc.Row(dbc.Col(html.H1("Batch-Based Egg Allocation Pilot", className="text-center my-4 p-3 text-white bg-primary rounded"), width=12)),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader(html.H4("Input Data")), dbc.CardBody([html.H5("Fish Groups"), dash_table.DataTable(id='fish-groups-table', data=fish_groups.to_dict('records'), style_table={'overflowX': 'auto'}), html.Hr(), html.H5("Orders"), dash_table.DataTable(id='orders-table', data=orders.to_dict('records'), style_table={'overflowX': 'auto'})])])], width=12)], className="mb-4"),
    dbc.Row(dbc.Col(dbc.Button("Run Allocation and Generate Report", id="run-button", color="success", size="lg", className="w-100"), width=12), className="mb-4"),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="results-output"))
], fluid=True)

@app.callback(Output("results-output", "children"), Input("run-button", "n_clicks"), prevent_initial_call=True)
def run_allocation_report(n_clicks):
    all_batches = [generate_batches(group) for _, group in fish_groups.iterrows()]
    batches_df = pd.concat(all_batches, ignore_index=True)
    results = solve_allocation(orders, batches_df)
    fig = create_visualization(batches_df, results)
    summary_stats = html.Div([dbc.Row([dbc.Col(html.P(f"Total Orders: {len(orders)}"), width=3), dbc.Col(html.P(f"Successfully Assigned: {results['Success'].sum()}"), width=3), dbc.Col(html.P(f"Unassigned: {(~results['Success']).sum()}"), width=3), dbc.Col(html.P(f"Success Rate: {results['Success'].mean() * 100:.1f}%"), width=3)])])
    utilization_data = []
    for _, batch in batches_df.iterrows():
        assigned = results[results['AssignedBatch'] == batch['BatchID']]
        if not assigned.empty:
            gain_used = assigned[assigned['Product'].isin(['Gain', 'Elite', 'Nucleus'])]['Volume'].sum(); shield_used = assigned[assigned['Product'] == 'Shield']['Volume'].sum(); total_cap = batch['GainCapacity'] + batch['ShieldCapacity']; util_pct = ((gain_used + shield_used) / total_cap * 100) if total_cap > 0 else 0
            utilization_data.append({'BatchID': batch['BatchID'], 'Utilization': f"{util_pct:.1f}%", 'Gain Used / Capacity': f"{gain_used:,.0f} / {batch['GainCapacity']:,.0f}", 'Shield Used / Capacity': f"{shield_used:,.0f} / {batch['ShieldCapacity']:,.0f}"})
    utilization_df = pd.DataFrame(utilization_data)
    return html.Div([dbc.Card([dbc.CardHeader(html.H4("Results")), dbc.CardBody([html.H5("Summary", className="mt-3"), summary_stats, html.Hr(), html.H5("Allocation Results", className="mt-3"), dash_table.DataTable(data=results.to_dict('records'), style_table={'overflowX': 'auto'}, style_data_conditional=[{'if': {'filter_query': '{Success} = false', 'column_id': 'OrderNr'}, 'backgroundColor': '#ffdddd'}, {'if': {'filter_query': '{Success} = true', 'column_id': 'OrderNr'}, 'backgroundColor': '#ddffdd'}]), html.Hr(), html.H5("Batch Utilization", className="mt-3"), dash_table.DataTable(data=utilization_df.to_dict('records'), style_table={'overflowX': 'auto'}), html.Hr(), html.H5("Delivery Window Analysis", className="mt-3"), dcc.Graph(figure=fig)])])])

# ==========================================
# SCRIPT EXECUTION & TESTING (UPDATED TEST BLOCK)
# ==========================================
if __name__ == '__main__':
    print("--- RUNNING DEFENSIVE TEST FOR NEW LOGIC ---")
    
    # Step 1: Build the feasibility set
    feasible_set_df = build_advanced_feasibility_set(orders, fish_groups, ADVANCED_MODEL_PARAMS)
    
    # Step 2: Run the new advanced solver
    advanced_results_df = solve_advanced_allocation(orders, fish_groups, feasible_set_df)
    
    if advanced_results_df is not None:
        print("\n--- Advanced Solver Results ---")
        print(advanced_results_df.to_string())
    else:
        print("\n--- Advanced Solver found no solution. ---")


    print("\n--- STARTING DASH APP (The app still uses the OLD logic for now) ---")
    app.run_server(debug=True)
