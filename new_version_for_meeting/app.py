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
# PARAMS & CONFIGURATION
# ==========================================
WATER_TEMP_C = 8.0

ADVANCED_MODEL_PARAMS = {
    "TemperatureGridPoints": 5,  # Kept for compatibility, but not used in new logic
}

# ==========================================
# EXAMPLE DATA (with Celsius temperatures)
# ==========================================
fish_groups_data = pd.DataFrame([
    {'Site': 'Hemne', 'Site_Broodst_Season': 'Hemne_Normal_24/25', 'StrippingStartDate': '2024-09-01', 'StrippingStopDate': '2024-09-22', 'MinTemp_C': 1, 'MaxTemp_C': 8, 'Gain-eggs': 5000000.0, 'Shield-eggs': 0.0, 'Organic': False},
    {'Site': 'Vestseøra', 'Site_Broodst_Season': 'Vestseøra_Organic_24/25', 'StrippingStartDate': '2024-08-25', 'StrippingStopDate': '2024-09-15', 'MinTemp_C': 1, 'MaxTemp_C': 8, 'Gain-eggs': 3000000.0, 'Shield-eggs': 2000000.0, 'Organic': True}
])

orders_data = pd.DataFrame([
    {'OrderNr': 1001, 'DeliveryDate': '2024-11-15', 'Product': 'Gain', 'Volume': 800000.0, 'MinTemp_C': 2, 'MaxTemp_C': 6},
    {'OrderNr': 1002, 'DeliveryDate': '2024-11-28', 'Product': 'Gain', 'Volume': 1200000.0, 'MinTemp_C': 2, 'MaxTemp_C': 6},
    {'OrderNr': 1003, 'DeliveryDate': '2024-12-10', 'Product': 'Shield', 'Volume': 600000.0, 'MinTemp_C': 2, 'MaxTemp_C': 6}
])

# ==========================================
# HELPER & LOGIC FUNCTIONS
# ==========================================

def preprocess_data(orders_df, groups_df, water_temp):
    """Calculates degree-days from Celsius temperatures."""
    DD_TO_MATURE = 300  # Degree-days needed for minimum maturation
    
    g_df = groups_df.copy()
    # MinTemp_prod: minimum degree-days before eggs are viable
    # MaxTemp_prod: maximum degree-days before eggs expire
    g_df['MinTemp_prod'] = g_df['MinTemp_C'] * (DD_TO_MATURE / 1)  # Scale based on 1°C baseline
    g_df['MaxTemp_prod'] = g_df['MaxTemp_C'] * (DD_TO_MATURE / 1)  # = 2400 DD at 8°C max
    
    o_df = orders_df.copy()
    # Customer requirements in degree-days
    # MinTemp_customer: minimum DD eggs must have reached
    # MaxTemp_customer: maximum DD eggs can have
    o_df['MinTemp_customer'] = o_df['MinTemp_C'] * (DD_TO_MATURE / 1)  # 2°C -> 600 DD
    o_df['MaxTemp_customer'] = o_df['MaxTemp_C'] * (DD_TO_MATURE / 1)  # 6°C -> 1800 DD
    
    return o_df, g_df


def generate_weekly_batches(fish_groups_df, water_temp):
    """
    Breaks down each group into weekly batches with a normal distribution of capacity.
    Each batch gets its own strip date, maturation window, and capacity allocation.
    """
    all_batches = []
    for _, group in fish_groups_df.iterrows():
        strip_start = pd.to_datetime(group['StrippingStartDate'])
        strip_stop = pd.to_datetime(group['StrippingStopDate'])
        
        # Generate weekly strip dates
        weeks = pd.date_range(strip_start, strip_stop, freq='W-MON')
        if len(weeks) == 0: 
            weeks = pd.DatetimeIndex([strip_start])
        
        # Normal distribution weights for capacity allocation
        n = len(weeks)
        indices = np.arange(n)
        weights = np.exp(-0.5 * ((indices - (n - 1) / 2) / max(n / 4, 1)) ** 2)
        weights = weights / weights.sum()

        for i, strip_date in enumerate(weeks):
            # Calculate maturation and expiry based on THIS batch's strip date
            maturation_days = group['MinTemp_prod'] / water_temp
            production_days = group['MaxTemp_prod'] / water_temp
            
            all_batches.append({
                'BatchID': f"{group['Site_Broodst_Season']}_Uke_{i+1}",
                'Group': group['Site_Broodst_Season'],
                'Site': group['Site'],
                'StripDate': strip_date,
                'MaturationEnd': strip_date + timedelta(days=maturation_days),
                'ProductionEnd': strip_date + timedelta(days=production_days),
                'GainCapacity': float(group['Gain-eggs']) * weights[i],
                'ShieldCapacity': float(group['Shield-eggs']) * weights[i],
                'Organic': group['Organic'],
                'MinTemp_prod': group['MinTemp_prod'],
                'MaxTemp_prod': group['MaxTemp_prod'],
            })
    
    return pd.DataFrame(all_batches)


def build_advanced_feasibility_set(orders_df, batches_df, params, water_temp):
    """
    Creates a DataFrame of all feasible (order, batch) combinations.
    
    FIXED: Now works at BATCH level instead of group level.
    Each batch has its own strip date, so each batch has its own delivery window.
    """
    feasible_combinations = []
    
    for _, order in orders_df.iterrows():
        delivery_date = pd.to_datetime(order['DeliveryDate'])
        
        # Calculate customer's required maturation window in DAYS
        # Customer says: "I want eggs that are between X and Y degree-days old"
        cust_min_days = order['MinTemp_customer'] / water_temp
        cust_max_days = order['MaxTemp_customer'] / water_temp
        
        for _, batch in batches_df.iterrows():
            # For THIS specific batch, calculate when eggs meet customer requirements
            # Based on THIS batch's strip date (not the group's dates)
            
            # Customer window: when eggs from this batch are in customer's DD range
            customer_window_start = batch['StripDate'] + timedelta(days=cust_min_days)
            customer_window_end = batch['StripDate'] + timedelta(days=cust_max_days)
            
            # Production window: when eggs are viable (not immature, not expired)
            production_ready = batch['MaturationEnd']    # Eggs ready (min DD reached)
            production_expiry = batch['ProductionEnd']   # Eggs expire (max DD exceeded)
            
            # VALID delivery window = intersection of customer & production windows
            valid_start = max(customer_window_start, production_ready)
            valid_end = min(customer_window_end, production_expiry)
            
            # Check if this batch can fulfill this order
            if valid_start <= delivery_date <= valid_end:
                # Calculate degree-days the eggs will have at delivery
                days_since_strip = (delivery_date - batch['StripDate']).days
                degree_days_at_delivery = days_since_strip * water_temp
                
                feasible_combinations.append({
                    'OrderNr': order['OrderNr'],
                    'BatchID': batch['BatchID'],
                    'Group': batch['Group'],
                    'Site': batch['Site'],
                    'StripDate': batch['StripDate'],
                    'DeliveryDate': delivery_date,
                    'DegreeDaysAtDelivery': round(degree_days_at_delivery, 2),
                    'Volume': order['Volume'],
                    'Product': order['Product'],
                    'ValidWindowStart': valid_start,
                    'ValidWindowEnd': valid_end,
                    'BatchGainCapacity': batch['GainCapacity'],
                    'BatchShieldCapacity': batch['ShieldCapacity'],
                    'BatchTotalCapacity': batch['GainCapacity'] + batch['ShieldCapacity'],
                })
    
    return pd.DataFrame(feasible_combinations)


def solve_advanced_allocation(orders_df, batches_df, feasible_set_df):
    """
    Solves allocation using linear programming at BATCH level.
    
    FIXED: 
    - Capacity constraints are now per BATCH (not per group)
    - Objective minimizes degree-days (FIFO-like: prefer fresher eggs)
    """
    if feasible_set_df.empty:
        unallocated = orders_df[['OrderNr']].copy()
        unallocated['Reason'] = 'No feasible batch found within delivery window'
        return pd.DataFrame(), unallocated

    prob = pl.LpProblem("BatchLevelEggAllocation", pl.LpMinimize)
    
    # Reset index to ensure consistent indexing
    feasible_set_df = feasible_set_df.reset_index(drop=True)
    feasible_set_df['id'] = feasible_set_df.index
    
    # Decision variables: binary variable for each feasible (order, batch) combination
    y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in feasible_set_df['id']}
    
    # ===========================================
    # OBJECTIVE: Minimize degree-days at delivery
    # ===========================================
    # Lower degree-days = eggs are younger = more shelf life = FIFO-like behavior
    # This naturally prioritizes batches with eggs that will expire soonest
    objective_terms = []
    for idx, row in feasible_set_df.iterrows():
        cost = row['DegreeDaysAtDelivery']
        objective_terms.append(cost * y[row['id']])
    prob += pl.lpSum(objective_terms)

    # ===========================================
    # CONSTRAINT 1: Each order assigned to exactly one batch
    # ===========================================
    for order_nr in orders_df['OrderNr'].unique():
        choices_for_order = feasible_set_df[feasible_set_df['OrderNr'] == order_nr]['id'].tolist()
        if len(choices_for_order) > 0:
            prob += pl.lpSum(y[i] for i in choices_for_order) == 1
        # If no choices exist, order will be unallocated

    # ===========================================
    # CONSTRAINT 2: Each BATCH's capacity cannot be exceeded
    # ===========================================
    for batch_id in batches_df['BatchID'].unique():
        choices_in_batch = feasible_set_df[feasible_set_df['BatchID'] == batch_id]
        
        if not choices_in_batch.empty:
            # Get this batch's capacity
            batch_info = batches_df[batches_df['BatchID'] == batch_id].iloc[0]
            batch_capacity = batch_info['GainCapacity'] + batch_info['ShieldCapacity']
            
            # Sum of volumes assigned to this batch must not exceed its capacity
            prob += pl.lpSum(
                y[i] * choices_in_batch.loc[i, 'Volume'] 
                for i in choices_in_batch['id'].tolist()
            ) <= batch_capacity

    # ===========================================
    # SOLVE
    # ===========================================
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    
    # ===========================================
    # EXTRACT RESULTS
    # ===========================================
    allocated_rows = []
    for i, var in y.items():
        val = pl.value(var)
        if val is not None and round(val) == 1:
            allocated_rows.append(feasible_set_df.loc[i])
    
    allocated_df = pd.DataFrame(allocated_rows) if allocated_rows else pd.DataFrame()
    
    # Find unallocated orders
    allocated_order_nrs = set(allocated_df['OrderNr'].unique()) if not allocated_df.empty else set()
    all_order_nrs = set(orders_df['OrderNr'].unique())
    unallocated_order_nrs = all_order_nrs - allocated_order_nrs
    
    unallocated_df = orders_df[orders_df['OrderNr'].isin(unallocated_order_nrs)].copy()
    if not unallocated_df.empty:
        # Add reason for each unallocated order
        reasons = []
        for order_nr in unallocated_df['OrderNr']:
            if order_nr not in feasible_set_df['OrderNr'].values:
                reasons.append('No batch has valid delivery window for this date')
            else:
                reasons.append('Capacity exceeded in all feasible batches')
        unallocated_df['Reason'] = reasons
    
    return allocated_df, unallocated_df


def create_batch_level_gantt_chart(batches_df, orders_df, allocated_df, water_temp):
    """
    Generates the detailed batch-level Gantt chart.
    Shows maturation period (blue), production window (red), and customer window (green).
    """
    try:
        fig = go.Figure()
        
        # Create y-positions for each batch
        batch_ids = batches_df['BatchID'].tolist()
        y_positions = {batch_id: i for i, batch_id in enumerate(batch_ids)}

        for _, batch in batches_df.iterrows():
            y_pos = y_positions[batch['BatchID']]
            
            # 1. BLUE BAR: Maturation Period (eggs not ready yet)
            fig.add_trace(go.Scatter(
                x=[batch['StripDate'], batch['MaturationEnd']], 
                y=[y_pos, y_pos], 
                mode='lines', 
                line=dict(color='#1f77b4', width=20), 
                name='Modningstid (eggs maturing)', 
                legendgroup='Modningstid', 
                showlegend=(y_pos == 0),
                hovertemplate=(
                    f"<b>{batch['BatchID']}</b><br>"
                    f"Modningstid (Blue)<br>"
                    f"Strip: {batch['StripDate'].strftime('%Y-%m-%d')}<br>"
                    f"Ready: {batch['MaturationEnd'].strftime('%Y-%m-%d')}<br>"
                    f"<extra></extra>"
                )
            ))
            
            # 2. RED BAR: Production Window (eggs viable, 1-8°C tolerance)
            fig.add_trace(go.Scatter(
                x=[batch['MaturationEnd'], batch['ProductionEnd']], 
                y=[y_pos, y_pos], 
                mode='lines', 
                line=dict(color='#d62728', width=20), 
                name='Leveringsvindu produksjon (1-8°C)', 
                legendgroup='Leveringsvindu', 
                showlegend=(y_pos == 0),
                hovertemplate=(
                    f"<b>{batch['BatchID']}</b><br>"
                    f"Produksjonsvindu (Red)<br>"
                    f"From: {batch['MaturationEnd'].strftime('%Y-%m-%d')}<br>"
                    f"Until: {batch['ProductionEnd'].strftime('%Y-%m-%d')}<br>"
                    f"<extra></extra>"
                )
            ))

            # 3. GREEN BAR: Customer Window (stricter 2-6°C requirement)
            # Calculate for each order's temperature requirements
            green_shown = False
            for _, order in orders_df.iterrows():
                cust_min_days = order['MinTemp_customer'] / water_temp
                cust_max_days = order['MaxTemp_customer'] / water_temp
                
                cust_start = batch['StripDate'] + timedelta(days=cust_min_days)
                cust_end = batch['StripDate'] + timedelta(days=cust_max_days)
                
                # Green bar is intersection of customer window and production window
                green_start = max(batch['MaturationEnd'], cust_start)
                green_end = min(batch['ProductionEnd'], cust_end)

                if green_start < green_end:
                    fig.add_trace(go.Scatter(
                        x=[green_start, green_end], 
                        y=[y_pos, y_pos], 
                        mode='lines', 
                        line=dict(color='#2ca02c', width=12), 
                        name='Kundekrav (2-6°C)', 
                        legendgroup='Kundekrav', 
                        showlegend=(y_pos == 0 and not green_shown),
                        hovertemplate=(
                            f"<b>{batch['BatchID']}</b><br>"
                            f"Kundevindu (Green) for Order {order['OrderNr']}<br>"
                            f"From: {green_start.strftime('%Y-%m-%d')}<br>"
                            f"Until: {green_end.strftime('%Y-%m-%d')}<br>"
                            f"<extra></extra>"
                        )
                    ))
                    green_shown = True
                    break  # Only show one green bar per batch (they're the same for same temp requirements)

        # 4. PURPLE VERTICAL LINES: Allocated delivery dates
        if not allocated_df.empty:
            for _, allocation in allocated_df.iterrows():
                delivery_date = pd.to_datetime(allocation['DeliveryDate'])
                batch_id = allocation['BatchID']
                
                # Add vertical line at delivery date
                fig.add_vline(
                    x=delivery_date, 
                    line_dash="dash", 
                    line_color="purple", 
                    line_width=2
                )
                
                # Add annotation at top
                fig.add_annotation(
                    x=delivery_date,
                    y=1.05,
                    yref="paper",
                    text=f"Ordre {allocation['OrderNr']} → {batch_id.split('_')[-1]}",
                    showarrow=False,
                    font=dict(color="purple", size=10, family="Arial Black"),
                    textangle=-45
                )
                
                # Add marker on the specific batch that was assigned
                if batch_id in y_positions:
                    fig.add_trace(go.Scatter(
                        x=[delivery_date],
                        y=[y_positions[batch_id]],
                        mode='markers',
                        marker=dict(color='purple', size=15, symbol='diamond'),
                        name=f'Tildeling Ordre {allocation["OrderNr"]}',
                        showlegend=False,
                        hovertemplate=(
                            f"<b>TILDELT</b><br>"
                            f"Ordre: {allocation['OrderNr']}<br>"
                            f"Batch: {batch_id}<br>"
                            f"Levering: {delivery_date.strftime('%Y-%m-%d')}<br>"
                            f"Døgngrader: {allocation['DegreeDaysAtDelivery']}<br>"
                            f"<extra></extra>"
                        )
                    ))

        fig.update_layout(
            title="Detaljert Tidslinje per Produksjonsbatch (Batch-Level Allocation)",
            xaxis_title="Dato", 
            yaxis_title="Batch",
            yaxis=dict(
                tickmode='array', 
                tickvals=list(y_positions.values()), 
                ticktext=list(y_positions.keys()),
                autorange="reversed"  # First batch at top
            ),
            height=max(400, len(batch_ids) * 60),  # Dynamic height based on batch count
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            margin=dict(r=200)  # Make room for legend
        )
        return fig
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
        return go.Figure().update_layout(title_text=f"Error creating visualization: {str(e)}")


# ==========================================
# DASH APP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("🐟 Avansert Eggallokeringsplanlegger (Batch-Level)", 
                            className="text-center my-4 p-3 text-white bg-primary rounded"))),
    
    dbc.Alert([
        html.H4("Slik fungerer appen (OPPDATERT: Batch-Level Allokering)", className="alert-heading"),
        html.P([
            "Appen tildeler nå ordrer til ", html.Strong("individuelle ukentlige batcher"), 
            " i stedet for hele fiskegrupper. Hver batch har sin egen strippdato og leveringsvindu."
        ]),
        html.Hr(),
        html.H5("Fargekoder i grafen:"),
        html.Ul([
            html.Li([html.Strong("Blå", style={"color": "#1f77b4"}), " = Modningstid (eggs not ready yet)"]),
            html.Li([html.Strong("Rød", style={"color": "#d62728"}), " = Produksjonsvindu (1-8°C, eggs viable)"]),
            html.Li([html.Strong("Grønn", style={"color": "#2ca02c"}), " = Kundevindu (2-6°C, stricter requirement)"]),
            html.Li([html.Strong("Lilla linje/diamant", style={"color": "purple"}), " = Tildelt leveringsdato"]),
        ]),
        html.Hr(),
        html.H5("Nøkkelendringer fra forrige versjon:"),
        html.Ul([
            html.Li("✅ Allokering skjer nå på BATCH-nivå (ikke gruppe-nivå)"),
            html.Li("✅ Hver batch har sin egen kapasitet (normalfordelt)"),
            html.Li("✅ Leveringsvindu beregnes fra batchens strippdato"),
            html.Li("✅ FIFO-lignende: Prioriterer lavere døgngrader (friskere rogn)"),
        ]),
        html.P(f"Vanntemperatur: {WATER_TEMP_C}°C (brukes til å konvertere døgngrader ↔ dager)"),
    ], color="info"),
    
    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader(html.H4("Input Data")), 
        dbc.CardBody([
            html.H5("Fiskegrupper (Produksjon)"), 
            dash_table.DataTable(
                id='fish-groups-table', 
                data=fish_groups_data.to_dict('records'),
                columns=[{"name": col, "id": col} for col in fish_groups_data.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'minWidth': '80px'},
                style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'}
            ), 
            html.Hr(), 
            html.H5("Ordrer (Kunder)"), 
            dash_table.DataTable(
                id='orders-table', 
                data=orders_data.to_dict('records'),
                columns=[{"name": col, "id": col} for col in orders_data.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'minWidth': '80px'},
                style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'}
            )
        ])
    ])])], className="mb-4"),
    
    dbc.Row(dbc.Col(dbc.Button("🚀 Kjør Batch-Level Allokering", id="run-button", color="success", size="lg", className="w-100")), className="mb-4"),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="results-output"))
], fluid=True)


@app.callback(
    Output("results-output", "children"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True
)
def run_advanced_allocation_report(n_clicks):
    try:
        # Step 1: Preprocess data (convert temps to degree-days)
        orders, fish_groups = preprocess_data(orders_data, fish_groups_data, WATER_TEMP_C)
        
        # Step 2: Generate weekly batches with individual strip dates and capacities
        batches_df = generate_weekly_batches(fish_groups, WATER_TEMP_C)
        
        # Step 3: Build feasibility set at BATCH level
        feasible_set_df = build_advanced_feasibility_set(orders, batches_df, ADVANCED_MODEL_PARAMS, WATER_TEMP_C)
        
        # Step 4: Solve allocation at BATCH level
        allocated_df, unallocated_df = solve_advanced_allocation(orders, batches_df, feasible_set_df)

        # ==========================================
        # BUILD OUTPUT DISPLAY
        # ==========================================
        
        # Summary statistics
        total_orders = len(orders)
        assigned_count = len(allocated_df)
        total_batches = len(batches_df)
        feasible_combinations = len(feasible_set_df)
        
        summary_stats = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{total_orders}", className="text-center"),
                    html.P("Totalt ordrer", className="text-center text-muted")
                ])
            ], color="primary", outline=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{assigned_count}", className="text-center text-success"),
                    html.P("Tildelt", className="text-center text-muted")
                ])
            ], color="success", outline=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{total_orders - assigned_count}", className="text-center text-danger"),
                    html.P("Ikke tildelt", className="text-center text-muted")
                ])
            ], color="danger", outline=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{total_batches}", className="text-center"),
                    html.P("Batcher generert", className="text-center text-muted")
                ])
            ], color="info", outline=True), width=3),
        ], className="mb-4")

        # Batch capacity table
        batch_summary = batches_df[['BatchID', 'Group', 'StripDate', 'MaturationEnd', 'ProductionEnd', 'GainCapacity', 'ShieldCapacity']].copy()
        batch_summary['StripDate'] = batch_summary['StripDate'].dt.strftime('%Y-%m-%d')
        batch_summary['MaturationEnd'] = batch_summary['MaturationEnd'].dt.strftime('%Y-%m-%d')
        batch_summary['ProductionEnd'] = batch_summary['ProductionEnd'].dt.strftime('%Y-%m-%d')
        batch_summary['GainCapacity'] = batch_summary['GainCapacity'].apply(lambda x: f"{x:,.0f}")
        batch_summary['ShieldCapacity'] = batch_summary['ShieldCapacity'].apply(lambda x: f"{x:,.0f}")

        # Allocation results table
        if not allocated_df.empty:
            results_for_display = allocated_df[['OrderNr', 'BatchID', 'Group', 'DegreeDaysAtDelivery', 'DeliveryDate', 'Volume']].copy()
            results_for_display['DeliveryDate'] = pd.to_datetime(results_for_display['DeliveryDate']).dt.strftime('%Y-%m-%d')
            results_for_display['Volume'] = results_for_display['Volume'].apply(lambda x: f"{x:,.0f}")
            results_for_display.rename(columns={
                'BatchID': 'Tildelt Batch',
                'Group': 'Gruppe', 
                'DegreeDaysAtDelivery': 'Døgngrader ved levering', 
                'DeliveryDate': 'Leveringsdato', 
                'Volume': 'Volum'
            }, inplace=True)
        else:
            results_for_display = pd.DataFrame()
        
        # Add unallocated orders
        if not unallocated_df.empty:
            unallocated_display = unallocated_df[['OrderNr', 'Reason']].copy()
            unallocated_display['Tildelt Batch'] = 'IKKE TILDELT'
            unallocated_display['Gruppe'] = '-'
            unallocated_display['Døgngrader ved levering'] = '-'
            unallocated_display['Leveringsdato'] = unallocated_df['DeliveryDate']
            unallocated_display['Volum'] = unallocated_df['Volume'].apply(lambda x: f"{x:,.0f}")
            unallocated_display.rename(columns={'Reason': 'Årsak'}, inplace=True)
            
            if not results_for_display.empty:
                results_for_display = pd.concat([results_for_display, unallocated_display[results_for_display.columns.tolist() + ['Årsak']]], ignore_index=True)
            else:
                results_for_display = unallocated_display

        # Create Gantt chart
        fig = create_batch_level_gantt_chart(batches_df, orders, allocated_df, WATER_TEMP_C)

        # Feasibility debug info
        if not feasible_set_df.empty:
            feasibility_summary = feasible_set_df.groupby('OrderNr').agg({
                'BatchID': 'count',
                'DegreeDaysAtDelivery': ['min', 'max']
            }).reset_index()
            feasibility_summary.columns = ['OrderNr', 'Antall mulige batcher', 'Min DD', 'Max DD']
        else:
            feasibility_summary = pd.DataFrame({'Info': ['Ingen mulige kombinasjoner funnet']})

        return html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("📈 Resultater fra Batch-Level Allokering")),
                dbc.CardBody([
                    html.H5("Sammendrag", className="mt-3"), 
                    summary_stats, 
                    html.Hr(),
                    
                    html.H5("Genererte Batcher", className="mt-3"),
                    dbc.Alert("Hver fiskegruppe er delt opp i ukentlige batcher med normalfordelt kapasitet.", color="secondary"),
                    dash_table.DataTable(
                        data=batch_summary.to_dict('records'),
                        columns=[{"name": col, "id": col} for col in batch_summary.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
                    ),
                    html.Hr(),
                    
                    html.H5("Mulige Tildelinger per Ordre", className="mt-3"),
                    dbc.Alert("Viser hvor mange batcher som kan oppfylle hver ordre basert på leveringsvindu.", color="secondary"),
                    dash_table.DataTable(
                        data=feasibility_summary.to_dict('records'),
                        columns=[{"name": col, "id": col} for col in feasibility_summary.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': '#6c757d', 'color': 'white', 'fontWeight': 'bold'},
                    ),
                    html.Hr(),
                    
                    html.H5("Ordretildelinger", className="mt-3"),
                    dash_table.DataTable(
                        data=results_for_display.to_dict('records') if not results_for_display.empty else [],
                        columns=[{"name": col, "id": col} for col in results_for_display.columns] if not results_for_display.empty else [],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{Tildelt Batch} = "IKKE TILDELT"'}, 'backgroundColor': '#ffdddd'},
                            {'if': {'filter_query': '{Tildelt Batch} != "IKKE TILDELT"'}, 'backgroundColor': '#ddffdd'}
                        ]
                    ), 
                    html.Hr(),
                    
                    html.H5("Detaljert Tidslinjeanalyse", className="mt-3"),
                    dbc.Alert([
                        html.P([
                            html.Strong("Slik leser du grafen: "),
                            "Hver rad er en ukentlig batch. ",
                            "Leveringsdatoen (lilla linje) må treffe det ", 
                            html.Strong("grønne området", style={"color": "#2ca02c"}),
                            " for at tildelingen skal være gyldig."
                        ]),
                        html.P([
                            html.Strong("Lilla diamant (◆)"), 
                            " viser hvilken spesifikk batch som ble tildelt hver ordre."
                        ])
                    ], color="secondary"),
                    dcc.Graph(figure=fig)
                ])
            ])
        ])
    
    except Exception as e:
        traceback.print_exc()
        return dbc.Alert([
            html.H4("Feil oppstod"),
            html.P(str(e)),
            html.Pre(traceback.format_exc())
        ], color="danger")


if __name__ == '__main__':
    app.run(debug=True, port=8051)
