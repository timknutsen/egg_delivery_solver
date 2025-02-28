import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import pandas as pd
import plotly.graph_objects as go
import pulp
import os

# Simplified sample data with essential columns only
orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 4, 5],
    "CustomerID": ["C1", "C2", "C3", "mowi", "NST"],
    "OrderedEggs": [50000, 70000, 60000, 80000, 45000],
    "Product": ["Standard", "Premium", "Standard", "Premium", "Standard"],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-25", "2025-03-12"]
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["A", "B", "C", "D"],
    "ProducedEggs": [100000, 120000, 90000, 80000],
    "Location": ["Steigen", "Hemne", "Erfjord", "Steigen"],
    "Product": ["Standard", "Premium", "Standard", "Premium"],
    "StartSaleDate": ["2025-02-15", "2025-02-20", "2025-02-25", "2025-03-01"],
    "ExpireDate": ["2025-04-15", "2025-04-20", "2025-04-25", "2025-05-01"]
})

# Initialize Dash app
app = dash.Dash(__name__, meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

# App layout with simplified structure
app.layout = html.Div([
    html.Div([
        html.H1("Roe Allocation Dashboard", className="app-header"),
        html.P("Optimize allocation of roe from broodstock groups to customer orders"),
    ], className="header"),
    
    html.Div([
        html.H3("Customer Orders"),
        dash_table.DataTable(
            id="orders-table",
            columns=[
                {"name": "Order ID", "id": "OrderID"},
                {"name": "Customer", "id": "CustomerID"},
                {"name": "Eggs Ordered", "id": "OrderedEggs", "type": "numeric", "format": {"specifier": ","}},
                {"name": "Product Type", "id": "Product"},
                {"name": "Delivery Date", "id": "DeliveryDate"}
            ],
            data=orders_data.to_dict("records"),
            editable=True,
            row_deletable=True,
            style_table={"overflowX": "auto"},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f5f5f5'}]
        ),
        html.Button("Add Order Row", id="add-order-row", n_clicks=0, className="btn"),
    ], className="table-section"),
    
    html.Div([
        html.H3("Available Roe by Broodstock Group"),
        dash_table.DataTable(
            id="roe-table",
            columns=[
                {"name": "Broodstock Group", "id": "BroodstockGroup"},
                {"name": "Available Eggs", "id": "ProducedEggs", "type": "numeric", "format": {"specifier": ","}},
                {"name": "Location", "id": "Location"},
                {"name": "Product Type", "id": "Product"},
                {"name": "Sale Start", "id": "StartSaleDate"},
                {"name": "Expiry Date", "id": "ExpireDate"}
            ],
            data=roe_data.to_dict("records"),
            editable=True,
            row_deletable=True,
            style_table={"overflowX": "auto"},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f5f5f5'}]
        ),
        html.Button("Add Roe Row", id="add-roe-row", n_clicks=0, className="btn"),
    ], className="table-section"),
    
    html.Div([
        html.H3("Optimization Settings"),
        html.Div([
            html.Label("Apply Constraints:"),
            dcc.Checklist(
                id="constraints-checklist",
                options=[
                    {"label": "Match product types", "value": "product_match"},
                    {"label": "Respect delivery/expiry dates", "value": "date_constraints"},
                    {"label": "NST partners get priority for Steigen roe", "value": "nst_priority"}
                ],
                value=["product_match", "date_constraints", "nst_priority"]
            ),
        ], className="settings-panel"),
        html.Button("Run Allocation Solver", id="run-solver", n_clicks=0, className="run-btn"),
    ], className="optimization-section"),
    
    html.Div([
        html.H3("Allocation Results"),
        dash_table.DataTable(id="allocation-table", style_table={"overflowX": "auto"}),
        html.Div(id="unfulfilled-orders"),
    ], className="results-section"),

    html.Div([
        html.H3("Visualization"),
        dcc.Tabs([
            dcc.Tab(label="Egg Buffer Over Time", children=[dcc.Graph(id="roe-allocation-graph")]),
            dcc.Tab(label="Timeline View", children=[dcc.Graph(id="timeline-graph")]),
        ])
    ], className="visualization-section")
], className="container")

# Callbacks for adding rows
@app.callback(
    Output('orders-table', 'data'),
    Input('add-order-row', 'n_clicks'),
    State('orders-table', 'data'),
    prevent_initial_call=True
)
def add_order_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({"OrderID": max([row["OrderID"] for row in rows], default=0) + 1, "CustomerID": "", "OrderedEggs": 0, "Product": "", "DeliveryDate": ""})
    return rows

@app.callback(
    Output('roe-table', 'data'),
    Input('add-roe-row', 'n_clicks'),
    State('roe-table', 'data'),
    prevent_initial_call=True
)
def add_roe_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({"BroodstockGroup": "", "ProducedEggs": 0, "Location": "", "Product": "", "StartSaleDate": "", "ExpireDate": ""})
    return rows

# Solver callback
@app.callback(
    [Output("allocation-table", "columns"), Output("allocation-table", "data"), 
     Output("roe-allocation-graph", "figure"), Output("timeline-graph", "figure"), 
     Output("unfulfilled-orders", "children")],
    Input("run-solver", "n_clicks"),
    [State("orders-table", "data"), State("roe-table", "data"), State("constraints-checklist", "value")],
    prevent_initial_call=True
)
def run_solver(n_clicks, orders, roe, constraints):
    orders_df = pd.DataFrame(orders)
    roe_df = pd.DataFrame(roe)
    orders_df["OrderedEggs"] = pd.to_numeric(orders_df["OrderedEggs"], errors='coerce')
    roe_df["ProducedEggs"] = pd.to_numeric(roe_df["ProducedEggs"], errors='coerce')

    # Run the solver
    prob = pulp.LpProblem("Roe_Allocation", pulp.LpMaximize)
    allocation_vars = {(order, group): pulp.LpVariable(f"x_{order}_{group}", cat="Binary") 
                       for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"]}

    for order in orders_df["OrderID"]:
        prob += pulp.lpSum(allocation_vars[order, group] for group in roe_df["BroodstockGroup"]) <= 1

    for group in roe_df["BroodstockGroup"]:
        prob += pulp.lpSum(allocation_vars[order, group] * orders_df.loc[orders_df["OrderID"] == order, "OrderedEggs"].values[0] 
                           for order in orders_df["OrderID"]) <= roe_df.loc[roe_df["BroodstockGroup"] == group, "ProducedEggs"].values[0]

    if "product_match" in constraints:
        for order in orders_df["OrderID"]:
            order_product = orders_df.loc[orders_df["OrderID"] == order, "Product"].values[0]
            for group in roe_df["BroodstockGroup"]:
                group_product = roe_df.loc[roe_df["BroodstockGroup"] == group, "Product"].values[0]
                if order_product != group_product:
                    prob += allocation_vars[order, group] == 0

    if "date_constraints" in constraints:
        for order in orders_df["OrderID"]:
            order_delivery = pd.to_datetime(orders_df.loc[orders_df["OrderID"] == order, "DeliveryDate"].values[0])
            for group in roe_df["BroodstockGroup"]:
                group_start = pd.to_datetime(roe_df.loc[roe_df["BroodstockGroup"] == group, "StartSaleDate"].values[0])
                group_expire = pd.to_datetime(roe_df.loc[roe_df["BroodstockGroup"] == group, "ExpireDate"].values[0])
                if order_delivery < group_start or order_delivery > group_expire:
                    prob += allocation_vars[order, group] == 0

    if "nst_priority" in constraints:
        steigen_groups = roe_df[roe_df["Location"] == "Steigen"]["BroodstockGroup"].tolist()
        nst_orders = orders_df[orders_df["CustomerID"] == "NST"]["OrderID"].tolist()
        for group in steigen_groups:
            for order in orders_df["OrderID"]:
                if order not in nst_orders:
                    prob += allocation_vars[order, group] <= pulp.lpSum(allocation_vars[nst_order, group] 
                                                                        for nst_order in nst_orders)

    prob += pulp.lpSum(allocation_vars[order, group] for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Collect allocation results
    allocation_results = []
    for (order, group), var in allocation_vars.items():
        if var.varValue == 1:
            order_row = orders_df[orders_df["OrderID"] == order].iloc[0]
            group_row = roe_df[roe_df["BroodstockGroup"] == group].iloc[0]
            allocation_results.append({
                "OrderID": order,
                "CustomerID": order_row["CustomerID"],
                "OrderedEggs": order_row["OrderedEggs"],
                "Product": order_row["Product"],
                "DeliveryDate": order_row["DeliveryDate"],
                "BroodstockGroup": group,
                "Location": group_row["Location"]
            })

    allocation_df = pd.DataFrame(allocation_results)
    fulfilled_orders = allocation_df["OrderID"].unique() if not allocation_df.empty else []
    unfulfilled_orders = orders_df[~orders_df["OrderID"].isin(fulfilled_orders)]

    # Calculate weekly buffer for the new graph
    # Step 1: Determine the range of weeks
    all_dates = []
    for _, row in roe_df.iterrows():
        all_dates.extend([pd.to_datetime(row['StartSaleDate']), pd.to_datetime(row['ExpireDate'])])
    for _, row in orders_df.iterrows():
        all_dates.append(pd.to_datetime(row['DeliveryDate']))
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')  # Weekly intervals starting on Mondays
    week_labels = [f"{d.year}-{d.isocalendar().week}" for d in weeks]

    # Step 2: Calculate available roe per week (Egg Buffer)
    egg_buffer = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        available_roe = 0
        for _, row in roe_df.iterrows():
            start_sale = pd.to_datetime(row['StartSaleDate'])
            expire = pd.to_datetime(row['ExpireDate'])
            if start_sale <= week_end and expire >= week_start:
                available_roe += row['ProducedEggs']
        egg_buffer.append(available_roe / 1_000_000)  # Convert to millions

    # Step 3: Calculate allocated roe per week
    allocated_roe_weekly = [0] * len(weeks)
    if not allocation_df.empty:
        for _, row in allocation_df.iterrows():
            delivery_date = pd.to_datetime(row['DeliveryDate'])
            for i, week_start in enumerate(weeks):
                week_end = week_start + pd.Timedelta(days=6)
                if week_start <= delivery_date <= week_end:
                    allocated_roe_weekly[i] += row['OrderedEggs'] / 1_000_000  # Convert to millions
                    break

    # Step 4: Calculate remaining buffer
    remaining_buffer = [max(egg - alloc, 0) for egg, alloc in zip(egg_buffer, allocated_roe_weekly)]

    # Step 5: Calculate Buffer % (assume "last period" is the first week's buffer)
    last_period_buffer = remaining_buffer[0] if remaining_buffer else 1  # Avoid division by zero
    buffer_percent = [((buf / last_period_buffer) * 100) if last_period_buffer != 0 else 0 for buf in remaining_buffer]

    # Step 6: Calculate Buffer on 2025-02-04
    ref_date = pd.to_datetime("2025-02-04")
    ref_buffer = 0
    for i, week_start in enumerate(weeks):
        week_end = week_start + pd.Timedelta(days=6)
        if week_start <= ref_date <= week_end:
            ref_buffer = remaining_buffer[i]
            break

    # Create the new Egg Buffer graph
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=week_labels, y=egg_buffer, fill='tozeroy', name="Egg Buffer", fillcolor='rgba(255, 165, 0, 0.5)', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=week_labels, y=remaining_buffer, name="Buffer", line=dict(color='black')))
    fig1.add_trace(go.Scatter(x=week_labels, y=[ref_buffer] * len(week_labels), name="Buffer 04.02.25", line=dict(color='green', dash='dash')))
    fig1.add_trace(go.Scatter(x=week_labels, y=buffer_percent, name="Buffer %", line=dict(color='green'), yaxis="y2"))

    fig1.update_layout(
        title="Egg Buffer & Buffer % with Last Period Buffer",
        xaxis_title="Week",
        yaxis=dict(title="Egg Buffer (Millions)", side="left"),
        yaxis2=dict(title="Buffer %", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    # Timeline graph (unchanged from original)
    fig2 = go.Figure()
    roe_df['StartSaleDate'] = pd.to_datetime(roe_df['StartSaleDate'])
    roe_df['ExpireDate'] = pd.to_datetime(roe_df['ExpireDate'])
    orders_df['DeliveryDate'] = pd.to_datetime(orders_df['DeliveryDate'])

    for _, row in roe_df.iterrows():
        fig2.add_trace(go.Bar(x=[row['ExpireDate'] - row['StartSaleDate']], y=[row['BroodstockGroup']], orientation='h', 
                              base=[row['StartSaleDate']], marker=dict(color='rgba(200, 200, 200, 0.3)'), 
                              name=f"{row['BroodstockGroup']} Available", text=f"{row['BroodstockGroup']}: {row['ProducedEggs']:,} eggs<br>{row['Product']}", 
                              hoverinfo="text", showlegend=False))

    if not allocation_df.empty:
        for customer in allocation_df['CustomerID'].unique():
            customer_orders = allocation_df[allocation_df['CustomerID'] == customer]
            for _, row in customer_orders.iterrows():
                delivery_date = pd.to_datetime(row['DeliveryDate'])
                fig2.add_trace(go.Scatter(x=[delivery_date], y=[row['BroodstockGroup']], mode="markers", marker=dict(size=12, color='#1f77b4'), 
                                          name=customer, text=f"Order {row['OrderID']}: {row['OrderedEggs']:,} eggs<br>Customer: {row['CustomerID']}<br>Product: {row['Product']}", 
                                          hoverinfo="text", showlegend=True))

    fig2.update_layout(title="Roe Allocation Timeline", xaxis_title="Date", yaxis_title="Broodstock Group", 
                       xaxis_type="date", yaxis_autorange="reversed", height=500, showlegend=True)

    unfulfilled_html = html.Div([html.H4("Unfulfilled Orders", style={"color": "red"}), 
                                 html.P(f"There are {len(unfulfilled_orders)} orders that could not be fulfilled:")]) if len(unfulfilled_orders) > 0 else html.P("All orders have been fulfilled!", style={"color": "green"})

    columns = [{"name": col, "id": col} for col in allocation_df.columns] if not allocation_df.empty else []
    return columns, allocation_df.to_dict("records"), fig1, fig2, unfulfilled_html

# CSS styling (unchanged from original)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Roe Allocation Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .header {
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .app-header {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .table-section {
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }
            .btn {
                margin-top: 10px;
                padding: 8px 12px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 4px;
                cursor: pointer;
            }
            .btn:hover {
                background-color: #e9ecef;
            }
            .run-btn {
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 20px;
            }
            .run-btn:hover {
                background-color: #0069d9;
            }
            .settings-panel {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .settings-panel label {
                display: block;
                margin-top: 10px;
                margin-bottom: 5px;
                font-weight: bold;
            }
            .results-section, .visualization-section {
                margin-top: 30px;
            }
            .dash-table-container {
                max-width: 100%;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)
