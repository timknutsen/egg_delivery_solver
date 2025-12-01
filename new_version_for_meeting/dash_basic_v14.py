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
# EXAMPLE DATA (with Celsius temperatures)
# ==========================================
fish_groups_data = pd.DataFrame([
    {'Site': 'Hemne', 'Site_Broodst_Season': 'Hemne_Normal_24/25', 'StrippingStartDate': '2024-09-01', 'StrippingStopDate': '2024-09-22', 'MinTemp_C': 1, 'MaxTemp_C': 8, 'Gain-eggs': 5000000.0, 'Shield-eggs': 0.0, 'Organic': False},
    {'Site': 'Vestseøra', 'Site_Broodst_Season': 'Vestseøra_Organic_24/25', 'StrippingStartDate': '2024-08-25', 'StrippingStopDate': '2024-09-15', 'MinTemp_C': 1, 'MaxTemp_C': 8, 'Gain-eggs': 3000000.0, 'Shield-eggs': 2000000.0, 'Organic': True}
])

orders_data = pd.DataFrame([
    {'OrderNr': 1001, 'DeliveryDate': '2024-11-15', 'Product': 'Elite', 'Volume': 800000.0, 'MinTemp_C': 2, 'MaxTemp_C': 6},
    {'OrderNr': 1002, 'DeliveryDate': '2024-11-28', 'Product': 'Gain', 'Volume': 1200000.0, 'MinTemp_C': 2, 'MaxTemp_C': 6},
    {'OrderNr': 1003, 'DeliveryDate': '2024-12-10', 'Product': 'Shield', 'Volume': 600000.0, 'MinTemp_C': 2, 'MaxTemp_C': 6}
])

# ==========================================
# HELPER & LOGIC FUNCTIONS
# ==========================================

def preprocess_data(orders_df, groups_df, water_temp):
    """Calculates degree-days from Celsius temperatures."""
    DD_TO_MATURE = 300
    
    g_df = groups_df.copy()
    g_df['MinTemp_prod'] = DD_TO_MATURE
    g_df['MaxTemp_prod'] = g_df['MaxTemp_C'] * (DD_TO_MATURE / g_df['MinTemp_C'])
    
    o_df = orders_df.copy()
    o_df['MinTemp_customer'] = o_df['MinTemp_C'] / g_df['MinTemp_C'].iloc[0] * DD_TO_MATURE
    o_df['MaxTemp_customer'] = o_df['MaxTemp_C'] / g_df['MinTemp_C'].iloc[0] * DD_TO_MATURE
    
    return o_df, g_df
    
def generate_weekly_batches(fish_groups_df, water_temp):
    """Breaks down each group into weekly batches with a normal distribution of capacity."""
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
                'StripDate': strip_date,
                'MaturationEnd': strip_date + timedelta(days=maturation_days),
                'ProductionEnd': strip_date + timedelta(days=production_days),
                'GainCapacity': float(group['Gain-eggs']) * weights[i],
                'ShieldCapacity': float(group['Shield-eggs']) * weights[i],
            })
    return pd.DataFrame(all_batches)

def build_advanced_feasibility_set(orders_df, fish_groups_df, params, water_temp):
    """Creates a DataFrame of all feasible (order, group, temp) combinations."""
    feasible_combinations = []
    for o_idx, order in orders_df.iterrows():
        for g_idx, group in fish_groups_df.iterrows():
            t_low = max(group['MinTemp_prod'], order['MinTemp_customer'])
            t_high = min(group['MaxTemp_prod'], order['MaxTemp_customer'])
            if t_low >= t_high: 
                continue
            
            temp_candidates = np.linspace(t_low, t_high, params["TemperatureGridPoints"])
            for temp_candidate_dd in temp_candidates:
                days_to_ready = group['MinTemp_prod'] / water_temp
                days_to_expiry = group['MaxTemp_prod'] / water_temp
                
                strip_stop_date = pd.to_datetime(group['StrippingStopDate'])
                ready_date = strip_stop_date + timedelta(days=days_to_ready)
                expiry_date = pd.to_datetime(group['StrippingStartDate']) + timedelta(days=days_to_expiry)
                
                required_delivery_date = pd.to_datetime(order['DeliveryDate'])
                
                if ready_date <= required_delivery_date <= expiry_date:
                    feasible_combinations.append({
                        'OrderNr': order['OrderNr'], 
                        'Group': group['Site_Broodst_Season'],
                        'TemperatureDD': round(temp_candidate_dd, 2),
                        'DeliveryDate': required_delivery_date.date(), 
                        'Volume': order['Volume']
                    })
    return pd.DataFrame(feasible_combinations)

def solve_advanced_allocation(orders_df, fish_groups_df, feasible_set_df):
    """Solves allocation using the advanced, temperature-discretized model."""
    if feasible_set_df.empty:
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

    for _, group in fish_groups_df.iterrows():
        group_name = group['Site_Broodst_Season']
        total_capacity = group['Gain-eggs'] + group['Shield-eggs']
        choices_in_group = feasible_set_df[feasible_set_df['Group'] == group_name]
        if not choices_in_group.empty:
            prob += pl.lpSum(y[i] * choices_in_group.loc[i, 'Volume'] for i in choices_in_group['id']) <= total_capacity

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    
    allocated_orders = [feasible_set_df.loc[i] for i, var in y.items() if pl.value(var) and round(pl.value(var)) == 1]
    allocated_df = pd.DataFrame(allocated_orders)
    
    allocated_order_nrs = allocated_df['OrderNr'].unique() if not allocated_df.empty else []
    unallocated_df = orders_df[~orders_df['OrderNr'].isin(allocated_order_nrs)].copy()
    unallocated_df['Reason'] = 'Could not be assigned (capacity or other constraints)'
    return allocated_df, unallocated_df

def create_batch_level_gantt_chart(batches_df, orders_df, allocated_df, water_temp):
    """Generates the detailed batch-level Gantt chart as specified in the diagram."""
    try:
        fig = go.Figure()
        y_positions = {batch_id: i for i, batch_id in enumerate(batches_df['BatchID'].unique())}

        for _, batch in batches_df.iterrows():
            y_pos = y_positions[batch['BatchID']]
            
            # 1. Blue Bar (Modningstid - Maturation Period)
            fig.add_trace(go.Scatter(
                x=[batch['StripDate'], batch['MaturationEnd']], 
                y=[y_pos, y_pos], 
                mode='lines', 
                line=dict(color='#1f77b4', width=20), 
                name='Modningstid', 
                legendgroup='Modningstid', 
                showlegend=(y_pos==0),
                hovertemplate=f"<b>{batch['BatchID']}</b><br>Modningstid<br>%{{x}}<extra></extra>"
            ))
            
            # 2. Red Bar (Leveringsvindu - Production Window based on 1-8°C)
            fig.add_trace(go.Scatter(
                x=[batch['MaturationEnd'], batch['ProductionEnd']], 
                y=[y_pos, y_pos], 
                mode='lines', 
                line=dict(color='#d62728', width=20), 
                name='Leveringsvindu produksjon (1-8°C)', 
                legendgroup='Leveringsvindu', 
                showlegend=(y_pos==0),
                hovertemplate=f"<b>{batch['BatchID']}</b><br>Produksjonsvindu<br>%{{x}}<extra></extra>"
            ))

            # 3. Green Bar (Leveringsvindu - Customer Window based on 2-6°C)
            for _, order in orders_df.iterrows():
                cust_maturation_days = order['MinTemp_customer'] / water_temp
                cust_production_days = order['MaxTemp_customer'] / water_temp
                cust_start = batch['StripDate'] + timedelta(days=cust_maturation_days)
                cust_end = batch['StripDate'] + timedelta(days=cust_production_days)
                
                green_start = max(batch['MaturationEnd'], cust_start)
                green_end = min(batch['ProductionEnd'], cust_end)

                if green_start < green_end:
                    fig.add_trace(go.Scatter(
                        x=[green_start, green_end], 
                        y=[y_pos, y_pos], 
                        mode='lines', 
                        line=dict(color='#2ca02c', width=13), 
                        name='Kundekrav (2-6°C)', 
                        legendgroup='Kundekrav', 
                        showlegend=(y_pos==0),
                        hovertemplate=f"<b>{batch['BatchID']}</b><br>Grønt vindu (Ordre {order['OrderNr']})<br>%{{x}}<extra></extra>"
                    ))

        # 4. Vertical Lines for Allocated Orders
        if not allocated_df.empty:
            for _, allocation in allocated_df.iterrows():
                delivery_date = pd.to_datetime(allocation['DeliveryDate']).to_pydatetime()
                
                # Add the vertical line
                fig.add_vline(
                    x=delivery_date, 
                    line_dash="dash", 
                    line_color="purple", 
                    line_width=2
                )
                
                # Add the annotation separately to avoid Plotly bug
                fig.add_annotation(
                    x=delivery_date,
                    y=1.05,
                    yref="paper",
                    text=f"Ordre {allocation['OrderNr']}",
                    showarrow=False,
                    font=dict(color="purple", size=12, family="Arial Black")
                )

        fig.update_layout(
            title="Detaljert Tidslinje per Produksjonsbatch",
            xaxis_title="Dato", 
            yaxis_title="Batch",
            yaxis=dict(
                tickmode='array', 
                tickvals=list(y_positions.values()), 
                ticktext=list(y_positions.keys())
            ),
            height=600, 
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
        return go.Figure().update_layout(title_text="Error creating visualization.")

# ==========================================
# DASH APP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("🐟 Avansert Eggallokeringsplanlegger", className="text-center my-4 p-3 text-white bg-primary rounded"))),
    
    dbc.Alert([
        html.H4("Slik fungerer appen", className="alert-heading"),
        html.P([
            "Appen kalkulerer leveringsvinduer basert på temperaturkrav. ",
            html.Strong("Blå"), " = Modningstid. ",
            html.Strong("Rød", style={"color": "#d62728"}), " = Produksjonsvindu (1-8°C). ",
            html.Strong("Grønn", style={"color": "#2ca02c"}), " = Kundekrav (2-6°C). ",
            html.Strong("Lilla linje", style={"color": "purple"}), " = Tildelt leveringsdato."
        ]),
        html.P(f"Alle kalkulasjoner bruker en standard vanntemperatur på {WATER_TEMP_C}°C for å konvertere mellom temperaturkrav og dager."),
        html.Hr(),
        html.H5("Forklaring av input-data:"),
        html.Ul([
            html.Li([html.Strong("MinTemp_C / MaxTemp_C:"), " Temperaturområde i Celsius. Produksjon: 1-8°C (bred toleranse). Kunde: 2-6°C (smalere krav)."]),
            html.Li([html.Strong("StrippingStartDate / StopDate:"), " Tidsvindu for når gyting/stripping skjer."]),
            html.Li([html.Strong("Gain-eggs / Shield-eggs:"), " Produksjonskapasitet i antall egg."])
        ])
    ], color="info"),
    
    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader(html.H4("Input Data")), 
        dbc.CardBody([
            html.H5("Fiskegrupper (Produksjon)"), 
            dash_table.DataTable(
                id='fish-groups-table', 
                data=fish_groups_data.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'}
            ), 
            html.Hr(), 
            html.H5("Ordrer (Kunder)"), 
            dash_table.DataTable(
                id='orders-table', 
                data=orders_data.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'}
            )
        ])
    ])])], className="mb-4"),
    
    dbc.Row(dbc.Col(dbc.Button("🚀 Kjør Avansert Allokering", id="run-button", color="success", size="lg", className="w-100")), className="mb-4"),
    dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="results-output"))
], fluid=True)

@app.callback(
    Output("results-output", "children"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True
)
def run_advanced_allocation_report(n_clicks):
    orders, fish_groups = preprocess_data(orders_data, fish_groups_data, WATER_TEMP_C)
    
    feasible_set_df = build_advanced_feasibility_set(orders, fish_groups, ADVANCED_MODEL_PARAMS, WATER_TEMP_C)
    allocated_df, unallocated_df = solve_advanced_allocation(orders, fish_groups, feasible_set_df)

    total_orders = len(orders)
    assigned_count = len(allocated_df)
    summary_stats = html.Div([dbc.Row([
        dbc.Col(html.P(f"📊 Totalt antall ordrer: {total_orders}"), width=4),
        dbc.Col(html.P(f"✅ Vellykket tildelt: {assigned_count}"), className="text-success", width=4),
        dbc.Col(html.P(f"❌ Ikke tildelt: {total_orders - assigned_count}"), className="text-danger", width=4),
    ])])

    batches_df = generate_weekly_batches(fish_groups, WATER_TEMP_C)
    fig = create_batch_level_gantt_chart(batches_df, orders, allocated_df, WATER_TEMP_C)

    results_for_display = allocated_df[['OrderNr', 'Group', 'TemperatureDD', 'DeliveryDate', 'Volume']].copy() if not allocated_df.empty else pd.DataFrame()
    if not results_for_display.empty:
        results_for_display.rename(columns={
            'Group': 'Tildelt Gruppe', 
            'TemperatureDD': 'Optimal Temp (GD)', 
            'DeliveryDate': 'Leveringsdato', 
            'Volume': 'Volum'
        }, inplace=True)
    
    if not unallocated_df.empty:
        unallocated_display = unallocated_df[['OrderNr', 'Reason']].copy()
        unallocated_display.rename(columns={'Reason': 'Årsak'}, inplace=True)
        unallocated_display['Tildelt Gruppe'] = 'IKKE TILDELT'
        results_for_display = pd.concat([results_for_display, unallocated_display], ignore_index=True)

    return html.Div([dbc.Card([
        dbc.CardHeader(html.H4("📈 Resultater fra Avansert Allokering")),
        dbc.CardBody([
            html.H5("Sammendrag", className="mt-3"), 
            summary_stats, 
            html.Hr(),
            html.H5("Ordretildelinger", className="mt-3"),
            dash_table.DataTable(
                data=results_for_display.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Tildelt Gruppe} = "IKKE TILDELT"'}, 'backgroundColor': '#ffdddd'},
                    {'if': {'filter_query': '{Tildelt Gruppe} != "IKKE TILDELT"'}, 'backgroundColor': '#ddffdd'}
                ]
            ), 
            html.Hr(),
            html.H5("Detaljert Tidslinjeanalyse", className="mt-3"),
            dbc.Alert([
                html.P("Grafen viser alle teoretisk mulige leveringsvinduer (grønne barer) og de endelige tildelingene valgt av optimeringsmotoren (lilla linjer)."),
                html.P("Hvis leveringsdato faller innenfor det grønne området → Success ✅")
            ], color="secondary"),
            dcc.Graph(figure=fig)
        ])
    ])])

if __name__ == '__main__':
    app.run_server(debug=True)
