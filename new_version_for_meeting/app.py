"""
DASH APP
========
UI-layout og callbacks.
Inkluderer:
- Visning av mulige grupper per ordre
- Eksport av eksempel inputfiler (kombinert og kun ordrer)
- Opplasting av inputfiler (kombinert og kun ordrer)
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import traceback
import os
import base64

from config import FISH_GROUPS, ORDERS, WATER_TEMP_C
from logic import (
    run_allocation, 
    generate_example_excel, 
    generate_orders_example_excel,
    parse_uploaded_excel,
    parse_orders_excel
)

# ==========================================
# APP SETUP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# Store for uploaded data
uploaded_data = {'fish_groups': None, 'orders': None}

# ==========================================
# LAYOUT
# ==========================================
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(
        "🐟 Eggallokering", 
        className="text-center my-4 p-3 text-white bg-primary rounded"
    ))),
    
    # Info-boks
    dbc.Alert([
        html.H5("Constraint-typer:"),
        html.Ul([
            html.Li([html.Strong("RequireOrganic: "), "MÅ ha organic egg (hard)"]),
            html.Li([html.Strong("LockedSite/Group: "), "MÅ tildeles spesifisert (hard)"]),
            html.Li([html.Strong("PreferredSite/Group: "), "BØR tildeles (soft)"]),
        ]),
        html.P(f"Vanntemperatur: {WATER_TEMP_C}°C | 🌿 = Organic"),
    ], color="info"),
    
    # Fil-opplasting og eksport
    dbc.Card([
        dbc.CardHeader("📁 Data Import/Export"),
        dbc.CardBody([
            # Rad 1: Kombinert fil (fiskegrupper + ordrer)
            html.H6("Komplett inputfil (Fiskegrupper + Ordrer)", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button("📤 Last opp komplett fil", color="primary", outline=True, size="sm"),
                        multiple=False,
                        accept='.xlsx,.xls'
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Button("📥 Last ned eksempel (komplett)", id="download-btn", color="secondary", outline=True, size="sm"),
                    dcc.Download(id="download-example"),
                ], width=4),
                dbc.Col([
                    html.Div(id='upload-status'),
                ], width=4),
            ], className="mb-3"),
            
            html.Hr(),
            
            # Rad 2: Kun ordrer
            html.H6("Kun ordrer (bruker default fiskegrupper)", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id='upload-orders-only',
                        children=dbc.Button("📤 Last opp ordrefil", color="info", outline=True, size="sm"),
                        multiple=False,
                        accept='.xlsx,.xls'
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Button("📥 Last ned eksempel (ordrer)", id="download-orders-btn", color="secondary", outline=True, size="sm"),
                    dcc.Download(id="download-orders-example"),
                ], width=4),
                dbc.Col([
                    html.Div(id='upload-orders-status'),
                ], width=4),
            ], className="mb-3"),
            
            html.Hr(),
            
            # Reset-knapp
            dbc.Button("🔄 Tilbakestill til default data", id="reset-btn", color="warning", outline=True, size="sm"),
            html.Div(id='reset-status', className="mt-2"),
            
            html.Hr(),
            html.Small([
                html.Strong("Komplett fil: "), "Excel med ark 'Fiskegrupper' og 'Ordrer'", html.Br(),
                html.Strong("Ordrefil: "), "Excel med kun ordrer (bruker innebygde fiskegrupper)"
            ], className="text-muted"),
        ])
    ], className="mb-4"),
    
    # Input data visning
    dbc.Card([
        dbc.CardHeader("Input Data"),
        dbc.CardBody([
            html.H5("Fiskegrupper"),
            html.Div(id='fish-table-container'),
            html.Hr(),
            html.H5("Ordrer"),
            html.Div(id='orders-table-container'),
            html.Small([
                html.Span("🟢 Grønn = Organic  ", style={"color": "#155724"}),
                html.Span("🔴 Rosa = Locked constraint", style={"color": "#721c24"}),
            ], className="text-muted")
        ])
    ], className="mb-4"),
    
    # Kjør-knapp
    dbc.Button("🚀 Kjør Allokering", id="run-btn", color="success", size="lg", className="w-100 mb-4"),
    
    # Resultater
    dcc.Loading(html.Div(id="output")),
    
    # Hidden store for data
    dcc.Store(id='data-store', data={'use_uploaded': False, 'orders_only': False})
], fluid=True)


# ==========================================
# CALLBACKS
# ==========================================

# Vis tabeller ved oppstart/endring
@app.callback(
    [Output('fish-table-container', 'children'),
     Output('orders-table-container', 'children')],
    [Input('data-store', 'data')]
)
def update_tables(store_data):
    fish_df = FISH_GROUPS
    orders_df = ORDERS
    
    if store_data.get('use_uploaded'):
        if store_data.get('orders_only'):
            # Kun ordrer lastet opp - bruk default fiskegrupper
            if uploaded_data['orders'] is not None:
                orders_df = uploaded_data['orders']
        else:
            # Komplett fil lastet opp
            if uploaded_data['fish_groups'] is not None:
                fish_df = uploaded_data['fish_groups']
            if uploaded_data['orders'] is not None:
                orders_df = uploaded_data['orders']
    
    fish_table = dash_table.DataTable(
        data=fish_df.to_dict('records'),
        columns=[{"name": c, "id": c} for c in fish_df.columns],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#007bff', 'color': 'white'},
        style_data_conditional=[
            {'if': {'filter_query': '{Organic} = true || {Organic} = True'}, 'backgroundColor': '#d4edda'},
        ],
        page_size=10,
    )
    
    orders_table = dash_table.DataTable(
        data=orders_df.to_dict('records'),
        columns=[{"name": c, "id": c} for c in orders_df.columns],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#007bff', 'color': 'white'},
        style_data_conditional=[
            {'if': {'filter_query': '{RequireOrganic} = true || {RequireOrganic} = True'}, 'backgroundColor': '#d4edda'},
            {'if': {'filter_query': '{LockedSite} != ""'}, 'backgroundColor': '#f8d7da'},
            {'if': {'filter_query': '{LockedGroup} != ""'}, 'backgroundColor': '#f8d7da'},
        ],
        page_size=10,
    )
    
    return fish_table, orders_table


# Last ned komplett eksempel
@app.callback(
    Output("download-example", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_example(n_clicks):
    excel_bytes = generate_example_excel()
    return dcc.send_bytes(excel_bytes, "eggallokering_eksempel.xlsx")


# Last ned ordre-eksempel
@app.callback(
    Output("download-orders-example", "data"),
    Input("download-orders-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_orders_example(n_clicks):
    excel_bytes = generate_orders_example_excel()
    return dcc.send_bytes(excel_bytes, "ordrer_eksempel.xlsx")


# Håndter komplett fil-opplasting
@app.callback(
    [Output('upload-status', 'children'),
     Output('data-store', 'data', allow_duplicate=True)],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        return "", {'use_uploaded': False, 'orders_only': False}
    
    fish_groups, orders, error = parse_uploaded_excel(contents, filename)
    
    if error:
        return dbc.Alert(error, color="danger", className="py-1 px-2 mb-0"), {'use_uploaded': False, 'orders_only': False}
    
    # Lagre i global store
    uploaded_data['fish_groups'] = fish_groups
    uploaded_data['orders'] = orders
    
    return (
        dbc.Alert(f"✅ {filename}: {len(fish_groups)} grupper, {len(orders)} ordrer", color="success", className="py-1 px-2 mb-0"),
        {'use_uploaded': True, 'orders_only': False}
    )


# Håndter ordre-fil opplasting
@app.callback(
    [Output('upload-orders-status', 'children'),
     Output('data-store', 'data', allow_duplicate=True)],
    Input('upload-orders-only', 'contents'),
    State('upload-orders-only', 'filename'),
    prevent_initial_call=True
)
def handle_orders_upload(contents, filename):
    if contents is None:
        return "", {'use_uploaded': False, 'orders_only': False}
    
    orders, error = parse_orders_excel(contents, filename)
    
    if error:
        return dbc.Alert(error, color="danger", className="py-1 px-2 mb-0"), {'use_uploaded': False, 'orders_only': False}
    
    # Lagre kun ordrer (fiskegrupper = default)
    uploaded_data['fish_groups'] = None
    uploaded_data['orders'] = orders
    
    return (
        dbc.Alert(f"✅ {filename}: {len(orders)} ordrer (bruker default grupper)", color="success", className="py-1 px-2 mb-0"),
        {'use_uploaded': True, 'orders_only': True}
    )


# Reset til default data
@app.callback(
    [Output('reset-status', 'children'),
     Output('data-store', 'data', allow_duplicate=True),
     Output('upload-status', 'children', allow_duplicate=True),
     Output('upload-orders-status', 'children', allow_duplicate=True)],
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_data(n_clicks):
    uploaded_data['fish_groups'] = None
    uploaded_data['orders'] = None
    
    return (
        dbc.Alert("✅ Tilbakestilt til default data", color="info", className="py-1 px-2 mb-0"),
        {'use_uploaded': False, 'orders_only': False},
        "",
        ""
    )


# Kjør allokering
@app.callback(
    Output("output", "children"),
    Input("run-btn", "n_clicks"),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def on_run(n_clicks, store_data):
    try:
        # Velg data basert på hva som er lastet opp
        if store_data.get('use_uploaded'):
            if store_data.get('orders_only'):
                # Kun ordrer lastet opp
                fish_groups = FISH_GROUPS
                orders = uploaded_data['orders'] if uploaded_data['orders'] is not None else ORDERS
            else:
                # Komplett fil lastet opp
                fish_groups = uploaded_data['fish_groups'] if uploaded_data['fish_groups'] is not None else FISH_GROUPS
                orders = uploaded_data['orders'] if uploaded_data['orders'] is not None else ORDERS
        else:
            fish_groups = FISH_GROUPS
            orders = ORDERS
        
        result = run_allocation(fish_groups, orders)
        results_df = result['results']
        possible_groups_df = result['possible_groups']
        
        # Tellere
        total = len(results_df)
        allocated = len(results_df[results_df['BatchID'] != 'IKKE TILDELT'])
        not_allocated = total - allocated
        organic_orders = len(results_df[results_df['RequireOrganic'] == '✓'])
        
        # Formater resultater
        display_df = results_df.copy()
        display_df['DeliveryDate'] = pd.to_datetime(display_df['DeliveryDate']).dt.strftime('%Y-%m-%d')
        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
        
        display_cols = ['OrderNr', 'Customer', 'BatchID', 'Site', 'Organic', 'DegreeDays', 
                        'DeliveryDate', 'Volume', 'Product', 'RequireOrganic', 'PreferenceMatched', 'Reason']
        display_df = display_df[display_cols]
        
        # Formater mulige grupper
        possible_display = possible_groups_df.copy()
        possible_display['Volume'] = possible_display['Volume'].apply(lambda x: f"{x:,.0f}")
        
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
                    
                    # Mulige grupper per ordre
                    html.H5("📋 Mulige grupper per ordre (for manuell vurdering)"),
                    dbc.Alert("Viser alle mulige grupper/batcher hver ordre kan tildeles. Nyttig for manuell gjennomgang.", color="secondary"),
                    dash_table.DataTable(
                        data=possible_display.to_dict('records'),
                        columns=[{"name": c, "id": c} for c in possible_display.columns],
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#6c757d', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{AntallMuligeBatcher} = 0'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
                            {'if': {'filter_query': '{AntallMuligeBatcher} = 1'}, 'backgroundColor': '#fff3cd'},
                            {'if': {'filter_query': '{AntallMuligeBatcher} > 5'}, 'backgroundColor': '#d4edda'},
                        ],
                        filter_action="native",
                        sort_action="native",
                        page_size=10,
                    ),
                    html.Small([
                        html.Span("🔴 Rød = Ingen muligheter  ", style={"color": "#721c24"}),
                        html.Span("🟡 Gul = Kun 1 mulighet  ", style={"color": "#856404"}),
                        html.Span("🟢 Grønn = Flere muligheter", style={"color": "#155724"}),
                    ], className="text-muted d-block mb-4"),
                    
                    html.Hr(),
                    
                    # Resultat-tabell
                    html.H5("📊 Allokeringsresultater"),
                    dash_table.DataTable(
                        data=display_df.to_dict('records'),
                        columns=[{"name": c, "id": c} for c in display_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{BatchID} = "IKKE TILDELT"'},
                                'backgroundColor': '#f8d7da',
                                'color': '#721c24'
                            },
                            {
                                'if': {'filter_query': '{BatchID} != "IKKE TILDELT"'},
                                'backgroundColor': '#d4edda',
                            },
                            {
                                'if': {'filter_query': '{RequireOrganic} = "✓"'},
                                'fontWeight': 'bold'
                            },
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
                    html.H5("📅 Tidslinje"),
                    dbc.Alert([
                        html.Span("🌿 = Organic batch | "),
                        html.Span("◆ = Tildelt ordre | "),
                        html.Span("Lilla linje = Leveringsdato"),
                    ], color="secondary"),
                    dcc.Graph(figure=result['chart']),
                    
                    html.Hr(),
                    
                    # Eksporter resultater
                    html.H5("💾 Eksporter resultater"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("📥 Last ned resultater (Excel)", id="export-results-btn", color="primary", outline=True),
                            dcc.Download(id="download-results"),
                        ], width=4),
                    ]),
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


# Eksporter resultater
@app.callback(
    Output("download-results", "data"),
    Input("export-results-btn", "n_clicks"),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def export_results(n_clicks, store_data):
    try:
        # Velg data basert på hva som er lastet opp
        if store_data.get('use_uploaded'):
            if store_data.get('orders_only'):
                fish_groups = FISH_GROUPS
                orders = uploaded_data['orders'] if uploaded_data['orders'] is not None else ORDERS
            else:
                fish_groups = uploaded_data['fish_groups'] if uploaded_data['fish_groups'] is not None else FISH_GROUPS
                orders = uploaded_data['orders'] if uploaded_data['orders'] is not None else ORDERS
        else:
            fish_groups = FISH_GROUPS
            orders = ORDERS
        
        result = run_allocation(fish_groups, orders)
        
        # Lag Excel med flere ark
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Resultater
            results_export = result['results'].copy()
            results_export['DeliveryDate'] = pd.to_datetime(results_export['DeliveryDate']).dt.strftime('%Y-%m-%d')
            results_export.to_excel(writer, sheet_name='Allokeringsresultater', index=False)
            
            # Mulige grupper
            result['possible_groups'].to_excel(writer, sheet_name='Mulige grupper per ordre', index=False)
            
            # Batcher
            batches_export = result['batches'].copy()
            batches_export['StripDate'] = batches_export['StripDate'].dt.strftime('%Y-%m-%d')
            batches_export['MaturationEnd'] = batches_export['MaturationEnd'].dt.strftime('%Y-%m-%d')
            batches_export['ProductionEnd'] = batches_export['ProductionEnd'].dt.strftime('%Y-%m-%d')
            batches_export.to_excel(writer, sheet_name='Genererte batcher', index=False)
            
            # Input data
            fish_groups.to_excel(writer, sheet_name='Input - Fiskegrupper', index=False)
            orders.to_excel(writer, sheet_name='Input - Ordrer', index=False)
        
        output.seek(0)
        return dcc.send_bytes(output.getvalue(), "allokeringsresultater.xlsx")
        
    except Exception as e:
        print(f"Export error: {e}")
        return None


# ==========================================
# RUN
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, host='0.0.0.0', port=port)
