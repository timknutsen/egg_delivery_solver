"""
DASH APP
========
UI-layout og callbacks.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import traceback
import os

from config import FISH_GROUPS, ORDERS, WATER_TEMP_C
from logic import run_allocation

# ==========================================
# APP SETUP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# ==========================================
# LAYOUT
# ==========================================
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(
        "🐟 Eggallokering", 
        className="text-center my-4 p-3 text-white bg-primary rounded"
    ))),
    
    dbc.Alert([
        html.H5("Constraint-typer:"),
        html.Ul([
            html.Li([html.Strong("RequireOrganic: "), "MÅ ha organic egg (hard)"]),
            html.Li([html.Strong("LockedSite/Group: "), "MÅ tildeles spesifisert (hard)"]),
            html.Li([html.Strong("PreferredSite/Group: "), "BØR tildeles (soft)"]),
        ]),
        html.P(f"Vanntemperatur: {WATER_TEMP_C}°C | 🌿 = Organic"),
    ], color="info"),
    
    dbc.Card([
        dbc.CardHeader("Input Data"),
        dbc.CardBody([
            html.H5("Fiskegrupper"),
            dash_table.DataTable(
                id='fish-table',
                data=FISH_GROUPS.to_dict('records'),
                columns=[{"name": c, "id": c} for c in FISH_GROUPS.columns],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#007bff', 'color': 'white'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Organic} = true'}, 'backgroundColor': '#d4edda'},
                ],
            ),
            html.Hr(),
            html.H5("Ordrer"),
            dash_table.DataTable(
                id='orders-table',
                data=ORDERS.to_dict('records'),
                columns=[{"name": c, "id": c} for c in ORDERS.columns],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#007bff', 'color': 'white'},
                style_data_conditional=[
                    {'if': {'filter_query': '{RequireOrganic} = true'}, 'backgroundColor': '#d4edda'},
                    {'if': {'filter_query': '{LockedSite} != ""'}, 'backgroundColor': '#f8d7da'},
                    {'if': {'filter_query': '{LockedGroup} != ""'}, 'backgroundColor': '#f8d7da'},
                ],
            ),
            html.Small([
                html.Span("🟢 Grønn = Organic  ", style={"color": "#155724"}),
                html.Span("🔴 Rosa = Locked constraint", style={"color": "#721c24"}),
            ], className="text-muted")
        ])
    ], className="mb-4"),
    
    dbc.Button("🚀 Kjør Allokering", id="run-btn", color="success", size="lg", className="w-100 mb-4"),
    
    dcc.Loading(html.Div(id="output"))
], fluid=True)


# ==========================================
# CALLBACK
# ==========================================
@app.callback(Output("output", "children"), Input("run-btn", "n_clicks"), prevent_initial_call=True)
def on_run(n):
    try:
        result = run_allocation(FISH_GROUPS, ORDERS)
        results_df = result['results']
        
        # Teller
        total = len(results_df)
        allocated = len(results_df[results_df['BatchID'] != 'IKKE TILDELT'])
        not_allocated = total - allocated
        organic_orders = len(results_df[results_df['RequireOrganic'] == '✓'])
        
        # Formater for visning
        display_df = results_df.copy()
        display_df['DeliveryDate'] = pd.to_datetime(display_df['DeliveryDate']).dt.strftime('%Y-%m-%d')
        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
        
        # Velg kolonner for visning
        display_cols = ['OrderNr', 'Customer', 'BatchID', 'Site', 'Organic', 'DegreeDays', 
                        'DeliveryDate', 'Volume', 'Product', 'RequireOrganic', 'PreferenceMatched', 'Reason']
        display_df = display_df[display_cols]
        
        return html.Div([
            dbc.Card([
                dbc.CardHeader("Resultater"),
                dbc.CardBody([
                    # Sammendrag
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{total}", className="text-center"),
                                html.P("Totalt", className="text-center text-muted")
                            ])
                        ], color="primary", outline=True), width=3),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{allocated}", className="text-center text-success"),
                                html.P("Tildelt", className="text-center text-muted")
                            ])
                        ], color="success", outline=True), width=3),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{not_allocated}", className="text-center text-danger"),
                                html.P("Ikke tildelt", className="text-center text-muted")
                            ])
                        ], color="danger", outline=True), width=3),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{organic_orders}", className="text-center text-success"),
                                html.P("Organic-krav", className="text-center text-muted")
                            ])
                        ], color="success", outline=True), width=3),
                    ], className="mb-4"),
                    
                    html.Hr(),
                    
                    # Resultat-tabell
                    html.H5("Alle ordrer (tildelt og ikke tildelt)"),
                    dash_table.DataTable(
                        data=display_df.to_dict('records'),
                        columns=[{"name": c, "id": c} for c in display_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_data_conditional=[
                            # Ikke tildelt = rød bakgrunn
                            {
                                'if': {'filter_query': '{BatchID} = "IKKE TILDELT"'},
                                'backgroundColor': '#f8d7da',
                                'color': '#721c24'
                            },
                            # Tildelt = grønn bakgrunn
                            {
                                'if': {'filter_query': '{BatchID} != "IKKE TILDELT"'},
                                'backgroundColor': '#d4edda',
                            },
                            # Organic ordre
                            {
                                'if': {'filter_query': '{RequireOrganic} = "✓"'},
                                'fontWeight': 'bold'
                            },
                            # Preferanse oppfylt
                            {
                                'if': {
                                    'filter_query': '{PreferenceMatched} != ""',
                                    'column_id': 'PreferenceMatched'
                                },
                                'backgroundColor': '#cce5ff',
                                'color': '#004085'
                            },
                        ],
                        filter_action="native",
                        sort_action="native",
                        page_size=20,
                    ),
                    
                    html.Hr(),
                    
                    # Gantt-chart
                    html.H5("Tidslinje"),
                    dbc.Alert([
                        html.Span("🌿 = Organic batch | "),
                        html.Span("◆ = Tildelt ordre | "),
                        html.Span("Lilla linje = Leveringsdato"),
                    ], color="secondary"),
                    dcc.Graph(figure=result['chart'])
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


# ==========================================
# RUN
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, host='0.0.0.0', port=port)
