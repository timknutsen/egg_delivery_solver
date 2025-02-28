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
            dcc.Tab(label="Roe Allocation", children=[dcc.Graph(id="roe-allocation-graph")]),
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

    if not allocation_df.empty:
        used_eggs = allocation_df.groupby("BroodstockGroup")["OrderedEggs"].sum().reset_index().rename(columns={"OrderedEggs": "UsedEggs"})
        allocation_viz = pd.merge(roe_df, used_eggs, on="BroodstockGroup", how="left").fillna(0)
        allocation_viz["RemainingEggs"] = allocation_viz["ProducedEggs"] - allocation_viz["UsedEggs"]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=allocation_viz["BroodstockGroup"], y=allocation_viz["UsedEggs"], name="Used Eggs", marker_color='#1f77b4'))
        fig1.add_trace(go.Bar(x=allocation_viz["BroodstockGroup"], y=allocation_viz["RemainingEggs"], name="Remaining Eggs", marker_color='#2ca02c'))
        fig1.update_layout(title="Roe Allocation by Broodstock Group", xaxis_title="Broodstock Group", yaxis_title="Eggs", barmode='stack')

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
    else:
        fig1 = go.Figure().update_layout(title="No allocation data available")
        fig2 = go.Figure().update_layout(title="No timeline data available")

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
