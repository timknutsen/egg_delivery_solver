import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import plotly.graph_objects as go
import random
import datetime
import time
import os
from optimization import binary_allocation, partial_allocation

# Sample data
salmon_producers = ["Mowi ASA", "Lerøy Seafood Group", "SalMar ASA", "Cermaq Group ASA", "Bakkafrost"]
product_types = ["Shield", "Gain (Premium)"]

orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 5, 6],
    "CustomerID": ["Mowi ASA", "SalMar ASA", "Bakkafrost", "Mowi ASA", "SalMar ASA"],
    "OrderedEggs": [50000, 40000, 50000, 60000, 70000],
    "Product": ["Shield", "Shield", "Shield", "Gain (Premium)", "Shield"],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-22", "2025-03-25"]
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["Early (mid-September)", "Natural (November)", "Natural (November)", "Late (December-January)", "Late (December-January)"],
    "ProducedEggs": [100000, 100000, 50000, 60000, 42832],
    "Location": ["Steigen", "Hemne", "Erfjord", "Steigen", "Tingvoll"],
    "Product": ["Shield", "Gain (Premium)", "Shield", "Gain (Premium)", "Gain (Premium)"],
    "StartSaleDate": ["2025-02-15", "2025-02-15", "2025-03-01", "2025-02-20", "2025-12-20"],
    "ExpireDate": ["2025-04-15", "2025-04-20", "2025-04-30", "2025-04-25", "2026-02-08"]
})

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Simplified styles
styles = {
    'container': {'font-family': 'Arial', 'padding': '20px'},
    'section': {'margin-bottom': '20px', 'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px', 'background-color': '#f9f9f9'},
    'button': {'background-color': '#4CAF50', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'margin-top': '10px'},
    'run_button': {'background-color': '#008CBA', 'color': 'white', 'padding': '12px 25px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'margin-top': '15px', 'font-size': '16px'},
    'status_bar': {'padding': '10px', 'margin-top': '15px', 'border-radius': '4px', 'font-weight': 'bold', 'text-align': 'center', 'background-color': '#f0f0f0'},
    'graph': {'height': '400px'},
    'tabs': {'margin-top': '10px'}
}

# App layout
app.layout = html.Div(style=styles['container'], children=[
    html.H1("Roe Allocation Dashboard", style={'text-align': 'center', 'margin-bottom': '20px'}),

    # Status Bar
    html.Div(id='status-bar', style=styles['status_bar'], children="Ready"),
    
    # Inventory Analytics Dashboard
    html.Div(style={**styles['section'], 'background-color': 'white'}, children=[
        html.Div(style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}, children=[
            html.Div(style={'background-color': '#7B2CBF', 'width': '40px', 'height': '40px', 'border-radius': '50%', 'margin-right': '10px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}, children=[
                html.Div(style={'border': '6px solid white', 'border-left-color': 'transparent', 'width': '20px', 'height': '20px', 'border-radius': '50%'})
            ]),
            html.H2("Inventory Analytics", style={'margin': '0', 'font-weight': '500'})
        ]),
        
        html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '15px'}, children=[
            # Total Inventory
            html.Div(style={'flex': '1', 'padding': '20px', 'background-color': '#f9f9f9', 'border-radius': '8px', 'margin-right': '10px'}, children=[
                html.H3("Total Inventory", style={'margin-top': '0', 'font-weight': '500', 'color': '#333'}),
                html.Div(id="total-inventory-value", style={'font-size': '36px', 'font-weight': 'bold', 'color': '#4169E1', 'margin-bottom': '10px'}),
                html.Div(id="total-inventory-subtitle", style={'color': '#666'})
            ]),
            
            # Total Orders
            html.Div(style={'flex': '1', 'padding': '20px', 'background-color': '#f9f9f9', 'border-radius': '8px', 'margin-right': '10px'}, children=[
                html.H3("Total Orders", style={'margin-top': '0', 'font-weight': '500', 'color': '#333'}),
                html.Div(id="total-orders-value", style={'font-size': '36px', 'font-weight': 'bold', 'color': '#4CAF50', 'margin-bottom': '10px'}),
                html.Div(id="total-orders-subtitle", style={'color': '#666'})
            ]),
            
            # Inventory Balance
            html.Div(style={'flex': '1', 'padding': '20px', 'background-color': '#f9f9f9', 'border-radius': '8px'}, children=[
                html.H3("Inventory Balance", style={'margin-top': '0', 'font-weight': '500', 'color': '#333'}),
                html.Div(id="inventory-balance-value", style={'font-size': '36px', 'font-weight': 'bold', 'margin-bottom': '10px'}),
                html.Div(id="inventory-balance-subtitle", style={'color': '#666'})
            ])
        ])
    ]),

    # Customer Orders
    html.Div(style=styles['section'], children=[
        html.H3("Customer Orders"),
        dash_table.DataTable(
            id='orders-table',
            columns=[
                {"name": "Order ID", "id": "OrderID", "type": "numeric"},
                {"name": "Customer", "id": "CustomerID"},
                {"name": "Eggs Ordered", "id": "OrderedEggs", "type": "numeric"},
                {"name": "Product", "id": "Product"},
                {"name": "Delivery Date", "id": "DeliveryDate"}
            ],
            data=orders_data.to_dict('records'),
            editable=True,
            row_deletable=True,
            style_table={'overflowX': 'auto'}
        ),
        html.Button("Add Order", id='add-order-button', style=styles['button'])
    ]),

    # Available Roe
    html.Div(style=styles['section'], children=[
        html.H3("Available Roe"),
        dash_table.DataTable(
            id='roe-table',
            columns=[
                {"name": "Broodstock Group", "id": "BroodstockGroup"},
                {"name": "Produced Eggs", "id": "ProducedEggs", "type": "numeric"},
                {"name": "Location", "id": "Location"},
                {"name": "Product", "id": "Product"},
                {"name": "Start Sale Date", "id": "StartSaleDate"},
                {"name": "Expiry Date", "id": "ExpireDate"}
            ],
            data=roe_data.to_dict('records'),
            editable=True,
            row_deletable=True,
            style_table={'overflowX': 'auto'}
        ),
        html.Button("Add Roe", id='add-roe-button', style=styles['button'])
    ]),

    # Optimization Settings
    html.Div(style=styles['section'], children=[
        html.H3("Optimization Settings"),
        html.Div(style={'margin-bottom': '15px'}, children=[
            html.Label("Constraints:"),
            dcc.Checklist(
                id='constraints-checklist',
                options=[
                    {'label': 'Product Match', 'value': 'product_match'},
                    {'label': 'Date Constraints', 'value': 'date_constraints'},
                    {'label': 'NST Priority', 'value': 'nst_priority'}
                ],
                value=['product_match', 'date_constraints']  # Default matches screenshot
            )
        ]),
        html.Div(style={'margin-bottom': '15px'}, children=[
            html.Label("Allocation Method:"),
            dcc.RadioItems(
                id='allocation-method',
                options=[
                    {'label': 'Binary', 'value': 'binary'},
                    {'label': 'Partial', 'value': 'partial'}
                ],
                value='binary'  # Default matches screenshot
            )
        ]),
        html.Div(children=[
            html.Label("Order Priority:"),
            dcc.RadioItems(
                id='order-priority',
                options=[
                    {'label': 'Chronological', 'value': 'chronological'},
                    {'label': 'Maximize Allocation', 'value': 'maximize'}
                ],
                value='chronological'  # Default matches screenshot
            )
        ]),
        html.Button("Run Solver", id='run-solver-button', style=styles['run_button'])
    ]),

    # Results
    html.Div(style=styles['section'], children=[
        html.H3("Allocation Results"),
        html.Div(id='computation-info', style={'font-style': 'italic', 'margin-bottom': '10px'}),
        dash_table.DataTable(
            id='allocation-table',
            style_table={'overflowX': 'auto'}
        ),
        html.Div(id='unfulfilled-orders', style={'margin-top': '15px'})
    ]),

    # Visualization with Tabs
    html.Div(style=styles['section'], children=[
        html.H3("Visualization"),
        dcc.Tabs(style=styles['tabs'], children=[
            dcc.Tab(label="Broodstock Utilization", children=[
                dcc.Graph(id='utilization-graph', style=styles['graph'])
            ]),
            dcc.Tab(label="Timeline View", children=[
                dcc.Graph(id='timeline-graph', style=styles['graph'])
            ])
        ])
    ]),

    # Hidden div to store the calculation results
    dcc.Store(id='calculation-results', data=None)
])

# --- Callbacks ---
@app.callback(
    [Output('total-inventory-value', 'children'),
     Output('total-inventory-subtitle', 'children'),
     Output('total-orders-value', 'children'),
     Output('total-orders-subtitle', 'children'),
     Output('inventory-balance-value', 'children'),
     Output('inventory-balance-value', 'style'),
     Output('inventory-balance-subtitle', 'children'),
     Output('inventory-balance-subtitle', 'style')],
    [Input('roe-table', 'data'),
     Input('orders-table', 'data')]
)
def update_inventory_summary(roe_data, orders_data):
    roe_df = pd.DataFrame(roe_data)
    orders_df = pd.DataFrame(orders_data)
    
    try:
        total_inventory = roe_df["ProducedEggs"].astype(int).sum()
        total_orders = orders_df["OrderedEggs"].astype(int).sum()
        inventory_balance = total_inventory - total_orders
        
        total_inventory_formatted = f"{total_inventory:,}".replace(',', ' ')
        total_orders_formatted = f"{total_orders:,}".replace(',', ' ')
        inventory_balance_formatted = f"{inventory_balance:,}".replace(',', ' ')
        if inventory_balance < 0:
            inventory_balance_formatted = f"−{abs(inventory_balance):,}".replace(',', ' ')
        
        if inventory_balance >= 0:
            balance_color = "#4169E1"
            balance_status = "Available for allocation"
            status_style = {'color': '#666'}
        else:
            balance_color = "#D32F2F"
            balance_status = "Inventory shortage!"
            status_style = {'color': '#D32F2F', 'font-weight': 'bold'}
            
        return (
            html.Span([total_inventory_formatted, html.Span(" eggs", style={'color': '#666', 'font-size': '24px', 'font-weight': 'normal'})]),
            "Across all locations",
            html.Span([total_orders_formatted, html.Span(" eggs", style={'color': '#666', 'font-size': '24px', 'font-weight': 'normal'})]),
            "Pending delivery",
            html.Span([inventory_balance_formatted, html.Span(" eggs", style={'color': '#666', 'font-size': '24px', 'font-weight': 'normal'})]),
            {'font-size': '36px', 'font-weight': 'bold', 'color': balance_color, 'margin-bottom': '10px'},
            balance_status,
            status_style
        )
    except:
        return (
            "0 eggs", "No data", 
            "0 eggs", "No data", 
            "0 eggs", {'color': '#666', 'font-size': '36px', 'font-weight': 'bold', 'margin-bottom': '10px'},
            "No data", {'color': '#666'}
        )

@app.callback(
    Output('orders-table', 'data'),
    Input('add-order-button', 'n_clicks'),
    State('orders-table', 'data'),
    prevent_initial_call=True
)
def add_order_row(n_clicks, existing_data):
    if n_clicks:
        new_order_id = max([row.get('OrderID', 0) for row in existing_data], default=0) + 1
        customer = random.choice(salmon_producers)
        eggs = random.randint(40000, 90000)
        product = random.choice(product_types)
        delivery_date = (datetime.date.today() + datetime.timedelta(days=random.randint(30, 90))).strftime('%Y-%m-%d')
        new_row = {'OrderID': new_order_id, 'CustomerID': customer, 'OrderedEggs': eggs, 'Product': product, 'DeliveryDate': delivery_date}
        existing_data.append(new_row)
        return existing_data
    return existing_data

@app.callback(
    Output('roe-table', 'data'),
    Input('add-roe-button', 'n_clicks'),
    State('roe-table', 'data'),
    prevent_initial_call=True
)
def add_roe_row(n_clicks, existing_data):
    if n_clicks:
        broodstock_groups = ["Early (mid-September)", "Natural (November)", "Late (December-January)"]
        site_locations = ["Profunda", "Hemne", "Tingvoll", "Steigen"]
        
        new_broodstock_group = random.choice(broodstock_groups)
        location = random.choice(site_locations)
        
        current_year = datetime.date.today().year
        if "Early" in new_broodstock_group:
            start_sale_date = datetime.date(current_year, 9, 15) + datetime.timedelta(days=random.randint(0, 10))
            expire_date = start_sale_date + datetime.timedelta(days=random.randint(45, 60))
        elif "Natural" in new_broodstock_group:
            start_sale_date = datetime.date(current_year, 11, 1) + datetime.timedelta(days=random.randint(0, 15))
            expire_date = start_sale_date + datetime.timedelta(days=random.randint(45, 60))
        else:  # Late
            start_sale_date = datetime.date(current_year, 12, 20) + datetime.timedelta(days=random.randint(0, 20))
            expire_date = start_sale_date + datetime.timedelta(days=random.randint(45, 60))
        
        new_row = {
            'BroodstockGroup': new_broodstock_group,
            'ProducedEggs': random.randint(40000, 100000),
            'Location': location,
            'Product': random.choice(product_types),
            'StartSaleDate': start_sale_date.strftime('%Y-%m-%d'),
            'ExpireDate': expire_date.strftime('%Y-%m-%d')
        }
        existing_data.append(new_row)
        return existing_data
    return existing_data

@app.callback(
    [
        Output('status-bar', 'children'),
        Output('status-bar', 'style'),
        Output('computation-info', 'children'),
        Output('calculation-results', 'data')
    ],
    [
        Input('run-solver-button', 'n_clicks'),
        Input('constraints-checklist', 'value'),
        Input('allocation-method', 'value'),
        Input('order-priority', 'value')
    ],
    [
        State('orders-table', 'data'),
        State('roe-table', 'data')
    ],
    prevent_initial_call=True
)
def run_solver(n_clicks, constraints_value, allocation_method, order_priority, orders_data, roe_data):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'run-solver-button':
        try:
            orders_df = pd.DataFrame(orders_data)
            roe_df = pd.DataFrame(roe_data)

            # Assign a unique GroupID to each row in roe_df
            roe_df['GroupID'] = range(len(roe_df))

            orders_df["OrderedEggs"] = pd.to_numeric(orders_df["OrderedEggs"], errors='coerce')
            roe_df["ProducedEggs"] = pd.to_numeric(roe_df["ProducedEggs"], errors='coerce')

            if orders_df.empty or roe_df.empty or orders_df.isnull().any().any() or roe_df.isnull().any().any():
                return "Error: Invalid data", {**styles['status_bar'], 'background-color': '#f8d7da', 'color': '#721c24'}, "", None

            start_time = time.time()

            if allocation_method == "binary":
                allocation_results, unfulfilled_orders = binary_allocation(orders_df, roe_df, constraints_value, order_priority)
            else:
                allocation_results, unfulfilled_orders = partial_allocation(orders_df, roe_df, constraints_value, order_priority)

            computation_time = time.time() - start_time

            utilization_data = []
            allocation_df = pd.DataFrame(allocation_results) if allocation_results else pd.DataFrame()
            for _, group_row in roe_df.iterrows():
                group_id = group_row["GroupID"]
                total_capacity = group_row["ProducedEggs"]
                allocated = allocation_df[allocation_df["GroupID"] == group_id]["AllocatedEggs"].sum() if not allocation_df.empty else 0
                utilization_data.append({
                    "BroodstockGroup": group_row["BroodstockGroup"],
                    "Location": group_row["Location"],
                    "Total": int(total_capacity),
                    "Allocated": int(allocated),
                    "Remaining": int(total_capacity - allocated)
                })

            return (
                "Allocation Complete",
                {**styles['status_bar'], 'background-color': '#d4edda', 'color': '#155724'},
                f"Allocation completed in {computation_time:.2f} seconds using {order_priority} priority",
                {
                    "allocation_results": allocation_results,
                    "unfulfilled_orders": unfulfilled_orders,
                    "utilization_data": utilization_data,
                    "roe_data": roe_df.to_dict('records')  # Include roe data for timeline visualization
                }
            )
        except Exception as e:
            return (
                f"Error: {str(e)}",
                {**styles['status_bar'], 'background-color': '#f8d7da', 'color': '#721c24'},
                f"Error during allocation: {str(e)}",
                None
            )
    return "Ready", styles['status_bar'], "", None

@app.callback(
    [
        Output('allocation-table', 'columns'),
        Output('allocation-table', 'data'),
        Output('unfulfilled-orders', 'children'),
        Output('utilization-graph', 'figure'),
        Output('timeline-graph', 'figure')
    ],
    Input('calculation-results', 'data'),
    prevent_initial_call=True
)
def update_ui_with_results(calculation_results):
    if not calculation_results:
        return [], [], "No results yet", go.Figure(layout={'title': 'No Data'}), go.Figure(layout={'title': 'No Data'})

    try:
        allocation_results = calculation_results.get('allocation_results', [])
        unfulfilled_orders = calculation_results.get('unfulfilled_orders', [])
        utilization_data = calculation_results.get('utilization_data', [])
        roe_data = calculation_results.get('roe_data', [])

        # Allocation Table
        allocation_columns = [
            {"name": i, "id": i, "type": "numeric" if i in ("OrderedEggs", "AllocatedEggs", "FulfillmentPct") else "text"}
            for i in ("OrderID", "CustomerID", "OrderedEggs", "AllocatedEggs", "FulfillmentPct", "Product", "DeliveryDate", "BroodstockGroup", "Location")
        ]
        allocation_data = allocation_results

        # Unfulfilled Orders
        if unfulfilled_orders:
            unfulfilled_text = f"Unfulfilled Orders: {', '.join(map(str, unfulfilled_orders))}"
            unfulfilled_div = html.Div(unfulfilled_text, style={'color': 'red'})
        else:
            unfulfilled_div = html.Div("All orders fulfilled!", style={'color': 'green'})

        # Broodstock Utilization Graph
        fig_utilization = go.Figure()
        if utilization_data:
            df = pd.DataFrame(utilization_data)
            fig_utilization.add_trace(go.Bar(x=df["BroodstockGroup"], y=df["Allocated"], name="Allocated", marker_color="#4CAF50"))
            fig_utilization.add_trace(go.Bar(x=df["BroodstockGroup"], y=df["Remaining"], name="Remaining", marker_color="#ccc"))
            fig_utilization.update_layout(barmode='stack', title="Broodstock Utilization", xaxis_title="Broodstock Group", yaxis_title="Eggs")
            for i, row in df.iterrows():
                utilization_pct = (row["Allocated"] / row["Total"]) * 100 if row["Total"] > 0 else 0
                fig_utilization.add_annotation(
                    x=row["BroodstockGroup"],
                    y=row["Total"] / 2,
                    text=f"{utilization_pct:.1f}%",
                    showarrow=False,
                    font=dict(color="white", size=12)
                )
        else:
            fig_utilization.update_layout(title="No Broodstock Data Available")

        # Timeline Graph
        fig_timeline = go.Figure()
        if allocation_results and roe_data:
            alloc_df = pd.DataFrame(allocation_results)
            roe_df = pd.DataFrame(roe_data)
            alloc_df['DeliveryDate'] = pd.to_datetime(alloc_df['DeliveryDate'])
            roe_df['StartSaleDate'] = pd.to_datetime(roe_df['StartSaleDate'])
            roe_df['ExpireDate'] = pd.to_datetime(roe_df['ExpireDate'])
            
            all_dates = list(alloc_df['DeliveryDate'].tolist())
            all_dates.extend(roe_df['StartSaleDate'].tolist())
            all_dates.extend(roe_df['ExpireDate'].tolist())
            min_date = min(all_dates) - pd.Timedelta(days=3)
            max_date = max(all_dates) + pd.Timedelta(days=3)
            
            for i, row in roe_df.iterrows():
                fill_color = 'rgba(144, 238, 144, 0.3)' if row['Product'] == 'Shield' else 'rgba(135, 206, 250, 0.3)'
                line_color = 'rgba(0, 128, 0, 0.5)' if row['Product'] == 'Shield' else 'rgba(0, 0, 255, 0.5)'
                fig_timeline.add_shape(
                    type="rect",
                    y0=i - 0.4, y1=i + 0.4,
                    x0=row['StartSaleDate'], x1=row['ExpireDate'],
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1),
                    layer="below"
                )
                fig_timeline.add_annotation(
                    y=i,
                    x=min_date - pd.Timedelta(days=2),
                    text=f"{row['BroodstockGroup']}",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    yanchor="middle",
                    xanchor="right"
                )
                fig_timeline.add_annotation(
                    y=i,
                    x=min_date - pd.Timedelta(days=5),
                    text=f"{row['Location']}",
                    showarrow=False,
                    font=dict(size=10),
                    yanchor="middle",
                    xanchor="right"
                )
            
            for product in alloc_df['Product'].unique():
                product_df = alloc_df[alloc_df['Product'] == product]
                group_id_to_position = {row['GroupID']: i for i, row in roe_df.iterrows()}
                color = '#4CAF50' if product == 'Shield' else '#2196F3'
                fig_timeline.add_trace(go.Scatter(
                    y=[group_id_to_position.get(row['GroupID'], 0) for _, row in product_df.iterrows()],
                    x=product_df['DeliveryDate'],
                    mode="markers",
                    marker=dict(size=product_df['AllocatedEggs'] / 5000, color=color, line=dict(width=1, color="black"), opacity=0.8),
                    name=f"{product} Orders",
                    text=[f"Order {row['OrderID']}: {row['AllocatedEggs']:,} eggs<br>Customer: {row['CustomerID']}" for _, row in product_df.iterrows()],
                    hoverinfo="text"
                ))
            
            fig_timeline.update_layout(
                title="Roe Allocation Timeline",
                xaxis=dict(title="Date", type="date"),
                yaxis=dict(title="Broodstock Group", tickmode='array', tickvals=list(range(len(roe_df))), ticktext=[f"{row['BroodstockGroup']}" for _, row in roe_df.iterrows()], showgrid=False),
                height=500,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
        else:
            fig_timeline.update_layout(title="No Allocation Data Available")

        return allocation_columns, allocation_data, unfulfilled_div, fig_utilization, fig_timeline

    except Exception as e:
        print(f"Error updating UI: {e}")
        return [], [], html.Div(f"Error: {str(e)}", style={'color': 'red'}), go.Figure(layout={'title': f'Error: {str(e)}'}), go.Figure(layout={'title': f'Error: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    host = '0.0.0.0'
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    print(f"Starting server on {host}:{port} with debug={debug}")
    app.run_server(host=host, port=port, debug=debug)
