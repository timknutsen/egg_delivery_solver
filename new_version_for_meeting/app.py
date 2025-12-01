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
    "TemperatureGridPoints": 5,
}

# ==========================================
# EXAMPLE DATA - With Locked/Preferred columns
# ==========================================
fish_groups_data = pd.DataFrame([
    {
        'Site': 'Hemne', 
        'Site_Broodst_Season': 'Hemne_Normal_24/25', 
        'StrippingStartDate': '2024-09-01', 
        'StrippingStopDate': '2024-09-22', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 5000000.0, 
        'Shield-eggs': 0.0, 
        'Organic': False
    },
    {
        'Site': 'Vestseøra', 
        'Site_Broodst_Season': 'Vestseøra_Organic_24/25', 
        'StrippingStartDate': '2024-08-25', 
        'StrippingStopDate': '2024-09-15', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 3000000.0, 
        'Shield-eggs': 2000000.0, 
        'Organic': True
    }
])

orders_data = pd.DataFrame([
    {
        'OrderNr': 1001, 
        'DeliveryDate': '2024-11-15', 
        'Product': 'Gain', 
        'Volume': 800000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hemne',
        'PreferredGroup': None,
    },
    {
        'OrderNr': 1002, 
        'DeliveryDate': '2024-11-28', 
        'Product': 'Gain', 
        'Volume': 1200000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    {
        'OrderNr': 1003, 
        'DeliveryDate': '2024-12-10', 
        'Product': 'Shield', 
        'Volume': 600000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'LockedSite': None,
        'LockedGroup': 'Vestseøra_Organic_24/25',
        'PreferredSite': None,
        'PreferredGroup': None,
    },
])

# ==========================================
# HELPER & LOGIC FUNCTIONS
# ==========================================

def preprocess_data(orders_df, groups_df, water_temp):
    """Calculates degree-days from Celsius temperatures."""
    DD_TO_MATURE = 300
    
    g_df = groups_df.copy()
    g_df['MinTemp_prod'] = g_df['MinTemp_C'] * (DD_TO_MATURE / 1)
    g_df['MaxTemp_prod'] = g_df['MaxTemp_C'] * (DD_TO_MATURE / 1)
    
    o_df = orders_df.copy()
    o_df['MinTemp_customer'] = o_df['MinTemp_C'] * (DD_TO_MATURE / 1)
    o_df['MaxTemp_customer'] = o_df['MaxTemp_C'] * (DD_TO_MATURE / 1)
    
    return o_df, g_df


def generate_weekly_batches(fish_groups_df, water_temp):
    """
    Breaks down each group into weekly batches with normal distribution of capacity.
    Each batch gets its own strip date, maturation window, and capacity allocation.
    """
    all_batches = []
    for _, group in fish_groups_df.iterrows():
        strip_start = pd.to_datetime(group['StrippingStartDate'])
        strip_stop = pd.to_datetime(group['StrippingStopDate'])
        
        weeks = pd.date_range(strip_start, strip_stop, freq='W-MON')
        if len(weeks) == 0: 
            weeks = pd.DatetimeIndex([strip_start])
        
        n = len(weeks)
        indices = np.arange(n)
        weights = np.exp(-0.5 * ((indices - (n - 1) / 2) / max(n / 4, 1)) ** 2)
        weights = weights / weights.sum()

        for i, strip_date in enumerate(weeks):
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
    Creates feasible (order, batch) combinations.
    
    Handles:
    - Locked (hard constraints): Filters out non-matching batches
    - Preferred (soft constraints): Adds preference bonus for optimization
    """
    feasible_combinations = []
    
    for _, order in orders_df.iterrows():
        delivery_date = pd.to_datetime(order['DeliveryDate'])
        cust_min_days = order['MinTemp_customer'] / water_temp
        cust_max_days = order['MaxTemp_customer'] / water_temp
        
        for _, batch in batches_df.iterrows():
            
            # =============================================
            # HARD CONSTRAINTS (Locked) - Filter out invalid
            # =============================================
            locked_site = order.get('LockedSite')
            if pd.notna(locked_site) and str(locked_site).strip() != '':
                if batch['Site'] != locked_site:
                    continue
            
            locked_group = order.get('LockedGroup')
            if pd.notna(locked_group) and str(locked_group).strip() != '':
                if batch['Group'] != locked_group:
                    continue
            
            # =============================================
            # Delivery window check
            # =============================================
            customer_window_start = batch['StripDate'] + timedelta(days=cust_min_days)
            customer_window_end = batch['StripDate'] + timedelta(days=cust_max_days)
            production_ready = batch['MaturationEnd']
            production_expiry = batch['ProductionEnd']
            
            valid_start = max(customer_window_start, production_ready)
            valid_end = min(customer_window_end, production_expiry)
            
            if valid_start <= delivery_date <= valid_end:
                days_since_strip = (delivery_date - batch['StripDate']).days
                degree_days_at_delivery = days_since_strip * water_temp
                
                # =============================================
                # SOFT CONSTRAINTS (Preferred) - Calculate bonus
                # =============================================
                preference_bonus = 0
                preference_matched = []
                
                preferred_site = order.get('PreferredSite')
                if pd.notna(preferred_site) and str(preferred_site).strip() != '':
                    if batch['Site'] == preferred_site:
                        preference_bonus -= 1000
                        preference_matched.append(f"Site={preferred_site}")
                
                preferred_group = order.get('PreferredGroup')
                if pd.notna(preferred_group) and str(preferred_group).strip() != '':
                    if batch['Group'] == preferred_group:
                        preference_bonus -= 1000
                        preference_matched.append(f"Group={preferred_group}")
                
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
                    'PreferenceBonus': preference_bonus,
                    'PreferenceMatched': ', '.join(preference_matched) if preference_matched else '',
                    'LockedSite': str(locked_site) if pd.notna(locked_site) else '',
                    'LockedGroup': str(locked_group) if pd.notna(locked_group) else '',
                })
    
    return pd.DataFrame(feasible_combinations)


def solve_advanced_allocation(orders_df, batches_df, feasible_set_df):
    """
    Solves allocation using linear programming at BATCH level.
    
    Objective: Minimize (DegreeDays + PreferenceBonus)
    - DegreeDays: Lower = fresher eggs (FIFO)
    - PreferenceBonus: Negative bonus for preferred matches
    
    Constraints:
    - Each order assigned to exactly one batch
    - Batch capacity cannot be exceeded
    """
    if feasible_set_df.empty:
        unallocated = orders_df[['OrderNr']].copy()
        unallocated['Reason'] = 'No feasible batch (check Locked constraints and delivery window)'
        return pd.DataFrame(), unallocated

    prob = pl.LpProblem("BatchLevelEggAllocation", pl.LpMinimize)
    
    feasible_set_df = feasible_set_df.reset_index(drop=True)
    feasible_set_df['id'] = feasible_set_df.index
    
    y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in feasible_set_df['id']}
    
    # =============================================
    # OBJECTIVE: DegreeDays + PreferenceBonus
    # =============================================
    objective_terms = []
    for idx, row in feasible_set_df.iterrows():
        cost = row['DegreeDaysAtDelivery'] + row['PreferenceBonus']
        objective_terms.append(cost * y[row['id']])
    prob += pl.lpSum(objective_terms)

    # =============================================
    # CONSTRAINT 1: Each order gets exactly one batch
    # =============================================
    for order_nr in orders_df['OrderNr'].unique():
        choices_for_order = feasible_set_df[feasible_set_df['OrderNr'] == order_nr]['id'].tolist()
        if len(choices_for_order) > 0:
            prob += pl.lpSum(y[i] for i in choices_for_order) == 1

    # =============================================
    # CONSTRAINT 2: Batch capacity
    # =============================================
    for batch_id in batches_df['BatchID'].unique():
        choices_in_batch = feasible_set_df[feasible_set_df['BatchID'] == batch_id]
        if not choices_in_batch.empty:
            batch_info = batches_df[batches_df['BatchID'] == batch_id].iloc[0]
            batch_capacity = batch_info['GainCapacity'] + batch_info['ShieldCapacity']
            prob += pl.lpSum(
                y[i] * choices_in_batch.loc[i, 'Volume'] 
                for i in choices_in_batch['id'].tolist()
            ) <= batch_capacity

    # =============================================
    # SOLVE
    # =============================================
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    
    # =============================================
    # EXTRACT RESULTS
    # =============================================
    allocated_rows = []
    for i, var in y.items():
        val = pl.value(var)
        if val is not None and round(val) == 1:
            allocated_rows.append(feasible_set_df.loc[i])
    
    allocated_df = pd.DataFrame(allocated_rows) if allocated_rows else pd.DataFrame()
    
    # Find unallocated with detailed reasons
    allocated_order_nrs = set(allocated_df['OrderNr'].unique()) if not allocated_df.empty else set()
    all_order_nrs = set(orders_df['OrderNr'].unique())
    unallocated_order_nrs = all_order_nrs - allocated_order_nrs
    
    unallocated_df = orders_df[orders_df['OrderNr'].isin(unallocated_order_nrs)].copy()
    if not unallocated_df.empty:
        reasons = []
        for _, order in unallocated_df.iterrows():
            order_nr = order['OrderNr']
            if order_nr not in feasible_set_df['OrderNr'].values:
                locked_site = order.get('LockedSite', '')
                locked_group = order.get('LockedGroup', '')
                if pd.notna(locked_site) and str(locked_site).strip() != '':
                    reasons.append(f'LockedSite={locked_site}: No valid batch at this site for delivery date')
                elif pd.notna(locked_group) and str(locked_group).strip() != '':
                    reasons.append(f'LockedGroup={locked_group}: No valid batch in this group for delivery date')
                else:
                    reasons.append('No batch has valid delivery window for this date')
            else:
                reasons.append('Capacity exceeded in all feasible batches')
        unallocated_df['Reason'] = reasons
    
    return allocated_df, unallocated_df


def create_batch_level_gantt_chart(batches_df, orders_df, allocated_df, water_temp):
    """
    Generates Gantt chart showing:
    - Blue: Maturation period
    - Red: Production window (1-8°C)
    - Green: Customer window (2-6°C)
    - Purple: Allocated deliveries with preference info
    """
    try:
        fig = go.Figure()
        batch_ids = batches_df['BatchID'].tolist()
        y_positions = {batch_id: i for i, batch_id in enumerate(batch_ids)}

        for _, batch in batches_df.iterrows():
            y_pos = y_positions[batch['BatchID']]
            
            # Blue: Maturation period
            fig.add_trace(go.Scatter(
                x=[batch['StripDate'], batch['MaturationEnd']], 
                y=[y_pos, y_pos], 
                mode='lines', 
                line=dict(color='#1f77b4', width=20), 
                name='Modningstid', 
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
            
            # Red: Production window
            fig.add_trace(go.Scatter(
                x=[batch['MaturationEnd'], batch['ProductionEnd']], 
                y=[y_pos, y_pos], 
                mode='lines', 
                line=dict(color='#d62728', width=20), 
                name='Produksjonsvindu (1-8°C)', 
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

            # Green: Customer window
            green_shown = False
            for _, order in orders_df.iterrows():
                cust_min_days = order['MinTemp_customer'] / water_temp
                cust_max_days = order['MaxTemp_customer'] / water_temp
                cust_start = batch['StripDate'] + timedelta(days=cust_min_days)
                cust_end = batch['StripDate'] + timedelta(days=cust_max_days)
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
                            f"Kundevindu (Green)<br>"
                            f"From: {green_start.strftime('%Y-%m-%d')}<br>"
                            f"Until: {green_end.strftime('%Y-%m-%d')}<br>"
                            f"<extra></extra>"
                        )
                    ))
                    green_shown = True
                    break

        # Purple: Allocated deliveries
        if not allocated_df.empty:
            for _, allocation in allocated_df.iterrows():
                delivery_date = pd.to_datetime(allocation['DeliveryDate'])
                batch_id = allocation['BatchID']
                
                fig.add_vline(
                    x=delivery_date, 
                    line_dash="dash", 
                    line_color="purple", 
                    line_width=2
                )
                
                fig.add_annotation(
                    x=delivery_date,
                    y=1.05,
                    yref="paper",
                    text=f"Ordre {allocation['OrderNr']}",
                    showarrow=False,
                    font=dict(color="purple", size=10, family="Arial Black"),
                    textangle=-45
                )
                
                if batch_id in y_positions:
                    pref_info = allocation.get('PreferenceMatched', '')
                    locked_info = []
                    if allocation.get('LockedSite', ''):
                        locked_info.append(f"LockedSite={allocation['LockedSite']}")
                    if allocation.get('LockedGroup', ''):
                        locked_info.append(f"LockedGroup={allocation['LockedGroup']}")
                    
                    hover_text = (
                        f"<b>TILDELT</b><br>"
                        f"Ordre: {allocation['OrderNr']}<br>"
                        f"Batch: {batch_id}<br>"
                        f"Site: {allocation['Site']}<br>"
                        f"Levering: {delivery_date.strftime('%Y-%m-%d')}<br>"
                        f"Døgngrader: {allocation['DegreeDaysAtDelivery']}<br>"
                    )
                    if locked_info:
                        hover_text += f"Locked: {', '.join(locked_info)}<br>"
                    if pref_info:
                        hover_text += f"Preference oppfylt: {pref_info}<br>"
                    hover_text += "<extra></extra>"
                    
                    fig.add_trace(go.Scatter(
                        x=[delivery_date],
                        y=[y_positions[batch_id]],
                        mode='markers',
                        marker=dict(color='purple', size=15, symbol='diamond'),
                        name=f'Ordre {allocation["OrderNr"]}',
                        showlegend=False,
                        hovertemplate=hover_text
                    ))

        fig.update_layout(
            title="Batch-Level Tidslinje med Locked/Preferred Constraints",
            xaxis_title="Dato", 
            yaxis_title="Batch",
            yaxis=dict(
                tickmode='array', 
                tickvals=list(y_positions.values()), 
                ticktext=list(y_positions.keys()),
                autorange="reversed"
            ),
            height=max(400, len(batch_ids) * 60),
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            margin=dict(r=200)
        )
        return fig
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
        return go.Figure().update_layout(title_text=f"Error: {str(e)}")


# ==========================================
# DASH APP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(
        "🐟 Eggallokering med Locked/Preferred Constraints", 
        className="text-center my-4 p-3 text-white bg-primary rounded"
    ))),
    
    dbc.Alert([
        html.H4("Slik fungerer appen", className="alert-heading"),
        html.P([
            "Appen tildeler ordrer til ukentlige produksjonsbatcher basert på leveringsvindu, ",
            "kapasitet, og brukerens constraints."
        ]),
        html.Hr(),
        html.H5("Constraint-typer:"),
        html.Ul([
            html.Li([
                html.Strong("Locked (Hard): "), 
                "Ordren MÅ tildeles spesifisert Site/Group. Hvis umulig → ordre avvises."
            ]),
            html.Li([
                html.Strong("Preferred (Soft): "), 
                "Ordren BØR tildeles spesifisert Site/Group. Optimaliserer prøver, men kan velge annen."
            ]),
        ]),
        html.Hr(),
        html.H5("Fargekoder i grafen:"),
        html.Ul([
            html.Li([html.Strong("Blå", style={"color": "#1f77b4"}), " = Modningstid (eggs maturing)"]),
            html.Li([html.Strong("Rød", style={"color": "#d62728"}), " = Produksjonsvindu (1-8°C)"]),
            html.Li([html.Strong("Grønn", style={"color": "#2ca02c"}), " = Kundevindu (2-6°C)"]),
            html.Li([html.Strong("Lilla ◆", style={"color": "purple"}), " = Tildelt leveringsdato"]),
        ]),
        html.P(f"Vanntemperatur: {WATER_TEMP_C}°C"),
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
            html.H5("Ordrer (med Locked/Preferred)"), 
            dash_table.DataTable(
                id='orders-table', 
                data=orders_data.to_dict('records'),
                columns=[{"name": col, "id": col} for col in orders_data.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'minWidth': '80px'},
                style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{LockedSite} != "" || {LockedGroup} != ""'},
                        'backgroundColor': '#ffe6e6'
                    },
                    {
                        'if': {'filter_query': '{PreferredSite} != "" || {PreferredGroup} != ""'},
                        'backgroundColor': '#e6ffe6'
                    },
                ]
            ),
            html.Small([
                html.Span("🔴 Rosa = Locked constraint  ", style={"color": "#cc0000"}),
                html.Span("🟢 Grønn = Preferred constraint", style={"color": "#006600"}),
            ], className="text-muted")
        ])
    ])])], className="mb-4"),
    
    dbc.Row(dbc.Col(dbc.Button(
        "🚀 Kjør Allokering", 
        id="run-button", 
        color="success", 
        size="lg", 
        className="w-100"
    )), className="mb-4"),
    
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="results-output"))
], fluid=True)


@app.callback(
    Output("results-output", "children"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True
)
def run_advanced_allocation_report(n_clicks):
    try:
        # Step 1: Preprocess data
        orders, fish_groups = preprocess_data(orders_data, fish_groups_data, WATER_TEMP_C)
        
        # Step 2: Generate weekly batches
        batches_df = generate_weekly_batches(fish_groups, WATER_TEMP_C)
        
        # Step 3: Build feasibility set (with Locked/Preferred handling)
        feasible_set_df = build_advanced_feasibility_set(orders, batches_df, ADVANCED_MODEL_PARAMS, WATER_TEMP_C)
        
        # Step 4: Solve allocation
        allocated_df, unallocated_df = solve_advanced_allocation(orders, batches_df, feasible_set_df)

        # ==========================================
        # BUILD OUTPUT
        # ==========================================
        total_orders = len(orders)
        assigned_count = len(allocated_df)
        total_batches = len(batches_df)
        
        # Count preference matches
        pref_matched_count = 0
        if not allocated_df.empty and 'PreferenceMatched' in allocated_df.columns:
            pref_matched_count = len(allocated_df[allocated_df['PreferenceMatched'] != ''])
        
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
                    html.H4(f"{pref_matched_count}", className="text-center text-info"),
                    html.P("Preferanser oppfylt", className="text-center text-muted")
                ])
            ], color="info", outline=True), width=3),
        ], className="mb-4")

        # Batch capacity table
        batch_summary = batches_df[['BatchID', 'Group', 'Site', 'StripDate', 'MaturationEnd', 'ProductionEnd', 'GainCapacity', 'ShieldCapacity']].copy()
        batch_summary['StripDate'] = batch_summary['StripDate'].dt.strftime('%Y-%m-%d')
        batch_summary['MaturationEnd'] = batch_summary['MaturationEnd'].dt.strftime('%Y-%m-%d')
        batch_summary['ProductionEnd'] = batch_summary['ProductionEnd'].dt.strftime('%Y-%m-%d')
        batch_summary['GainCapacity'] = batch_summary['GainCapacity'].apply(lambda x: f"{x:,.0f}")
        batch_summary['ShieldCapacity'] = batch_summary['ShieldCapacity'].apply(lambda x: f"{x:,.0f}")

        # Allocation results
        if not allocated_df.empty:
            results_for_display = allocated_df[['OrderNr', 'BatchID', 'Group', 'Site', 'DegreeDaysAtDelivery', 'DeliveryDate', 'Volume', 'PreferenceMatched']].copy()
            results_for_display['DeliveryDate'] = pd.to_datetime(results_for_display['DeliveryDate']).dt.strftime('%Y-%m-%d')
            results_for_display['Volume'] = results_for_display['Volume'].apply(lambda x: f"{x:,.0f}")
            results_for_display.rename(columns={
                'BatchID': 'Tildelt Batch',
                'Group': 'Gruppe',
                'Site': 'Anlegg',
                'DegreeDaysAtDelivery': 'Døgngrader', 
                'DeliveryDate': 'Leveringsdato', 
                'Volume': 'Volum',
                'PreferenceMatched': 'Preferanse oppfylt'
            }, inplace=True)
        else:
            results_for_display = pd.DataFrame()
        
        # Add unallocated
        if not unallocated_df.empty:
            unallocated_display = unallocated_df[['OrderNr', 'Reason']].copy()
            unallocated_display['Tildelt Batch'] = 'IKKE TILDELT'
            unallocated_display['Gruppe'] = '-'
            unallocated_display['Anlegg'] = '-'
            unallocated_display['Døgngrader'] = '-'
            unallocated_display['Leveringsdato'] = unallocated_df['DeliveryDate']
            unallocated_display['Volum'] = unallocated_df['Volume'].apply(lambda x: f"{x:,.0f}")
            unallocated_display['Preferanse oppfylt'] = '-'
            unallocated_display.rename(columns={'Reason': 'Årsak'}, inplace=True)
            
            cols = ['OrderNr', 'Tildelt Batch', 'Gruppe', 'Anlegg', 'Døgngrader', 'Leveringsdato', 'Volum', 'Preferanse oppfylt', 'Årsak']
            if not results_for_display.empty:
                results_for_display['Årsak'] = ''
                results_for_display = pd.concat([results_for_display, unallocated_display[cols]], ignore_index=True)
            else:
                results_for_display = unallocated_display[cols]

        # Gantt chart
        fig = create_batch_level_gantt_chart(batches_df, orders, allocated_df, WATER_TEMP_C)

        # Feasibility summary
        if not feasible_set_df.empty:
            feasibility_summary = feasible_set_df.groupby('OrderNr').agg({
                'BatchID': 'count',
                'DegreeDaysAtDelivery': ['min', 'max'],
                'PreferenceBonus': 'min'
            }).reset_index()
            feasibility_summary.columns = ['OrderNr', 'Mulige batcher', 'Min DD', 'Max DD', 'Beste bonus']
        else:
            feasibility_summary = pd.DataFrame({'Info': ['Ingen mulige kombinasjoner funnet']})

        return html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("📈 Allokeringsresultater")),
                dbc.CardBody([
                    html.H5("Sammendrag"), 
                    summary_stats, 
                    html.Hr(),
                    
                    html.H5("Genererte Batcher"),
                    dbc.Alert("Hver fiskegruppe deles i ukentlige batcher med normalfordelt kapasitet.", color="secondary"),
                    dash_table.DataTable(
                        data=batch_summary.to_dict('records'),
                        columns=[{"name": col, "id": col} for col in batch_summary.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
                    ),
                    html.Hr(),
                    
                    html.H5("Mulige Tildelinger per Ordre"),
                    dash_table.DataTable(
                        data=feasibility_summary.to_dict('records'),
                        columns=[{"name": col, "id": col} for col in feasibility_summary.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': '#6c757d', 'color': 'white', 'fontWeight': 'bold'},
                    ),
                    html.Hr(),
                    
                    html.H5("Ordretildelinger"),
                    dash_table.DataTable(
                        data=results_for_display.to_dict('records') if not results_for_display.empty else [],
                        columns=[{"name": col, "id": col} for col in results_for_display.columns] if not results_for_display.empty else [],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{Tildelt Batch} = "IKKE TILDELT"'}, 'backgroundColor': '#ffdddd'},
                            {'if': {'filter_query': '{Tildelt Batch} != "IKKE TILDELT"'}, 'backgroundColor': '#ddffdd'},
                            {'if': {'filter_query': '{Preferanse oppfylt} != "" && {Preferanse oppfylt} != "-"'}, 'backgroundColor': '#d4edda'},
                        ]
                    ), 
                    html.Hr(),
                    
                    html.H5("Tidslinje"),
                    dbc.Alert([
                        html.P([
                            "Leveringsdato (lilla) må treffe ", 
                            html.Strong("grønt område", style={"color": "#2ca02c"}),
                            " for gyldig tildeling. Hover over ◆ for detaljer om constraints."
                        ]),
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
    import os
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, host='0.0.0.0', port=port)
