import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
from datetime import datetime
import numpy as np
import os  # Added os import for environment variables

# Enhanced sample data with more realistic fields
orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 4, 5],
    "CustomerID": ["C1", "C2", "C3", "C4", "NST"],  # NST is a special partner
    "OrderedEggs": [50000, 70000, 60000, 80000, 45000],
    "Product": ["Standard", "Premium", "Standard", "Premium", "Standard"],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-25", "2025-03-12"],
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["A", "B", "C", "D"],
    "ProducedEggs": [100000, 120000, 90000, 80000],
    "Location": ["Steigen", "Hemne", "Erfjord", "Steigen"],
    "Product": ["Standard", "Premium", "Standard", "Premium"],
    "StartSaleDate": ["2025-02-15", "2025-02-20", "2025-02-25", "2025-03-01"],
    "ExpireDate": ["2025-04-15", "2025-04-20", "2025-04-25", "2025-05-01"],
})

# Initialize Dash app with a better theme
app = dash.Dash(__name__, 
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

# App layout with improved UI
app.layout = html.Div([
    html.Div([
        html.H1("Roe Allocation Dashboard", className="app-header"),
        html.P("Optimize allocation of roe from broodstock groups to customer orders"),
    ], className="header"),
    
    html.Div([
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
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f5f5f5'
                    }
                ]
            ),
            html.Button("Add Order Row", id="add-order-row", n_clicks=0, className="btn"),
        ], className="table-container"),
        
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
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f5f5f5'
                    }
                ]
            ),
            html.Button("Add Roe Row", id="add-roe-row", n_clicks=0, className="btn"),
        ], className="table-container"),
    ], className="data-tables"),
    
    html.Div([
        html.H3("Optimization Settings"),
        html.Div([
            html.Label("Optimization Objective:"),
            dcc.Dropdown(
                id="objective-dropdown",
                options=[
                    {"label": "Maximize fulfilled orders", "value": "max_orders"},
                    {"label": "Minimize waste", "value": "min_waste"},
                    {"label": "Prioritize NST partners", "value": "priority_nst"}
                ],
                value="max_orders"
            ),
            html.Label("Apply Constraints:"),
            dcc.Checklist(
                id="constraints-checklist",
                options=[
                    {"label": "NST partners get priority for Steigen roe", "value": "nst_priority"},
                    {"label": "Match product types", "value": "product_match"},
                    {"label": "Respect delivery/expiry dates", "value": "date_constraints"}
                ],
                value=["nst_priority", "product_match"]
            ),
        ], className="settings-panel"),
        
        html.Button("Run Allocation Solver", id="run-solver", n_clicks=0, className="run-btn"),
    ], className="optimization-section"),
    
    html.Div([
        html.H3("Allocation Results"),
        dash_table.DataTable(
            id="allocation-table", 
            style_table={"overflowX": "auto"},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f5f5f5'
                }
            ]
        ),
        html.Div(id="unfulfilled-orders"),
    ], className="results-section"),

    html.Div([
        html.H3("Visualization"),
        dcc.Tabs([
            dcc.Tab(label="Roe Allocation", children=[
                dcc.Graph(id="roe-allocation-graph")
            ]),
            dcc.Tab(label="Timeline View", children=[
                dcc.Graph(id="timeline-graph")
            ]),
        ])
    ], className="visualization-section")
], className="container")

# Callbacks for adding rows to tables
@app.callback(
    Output('orders-table', 'data'),
    Input('add-order-row', 'n_clicks'),
    State('orders-table', 'data'),
    prevent_initial_call=True
)
def add_order_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({
            "OrderID": max([row["OrderID"] for row in rows], default=0) + 1,
            "CustomerID": "",
            "OrderedEggs": 0,
            "Product": "",
            "DeliveryDate": ""
        })
    return rows

@app.callback(
    Output('roe-table', 'data'),
    Input('add-roe-row', 'n_clicks'),
    State('roe-table', 'data'),
    prevent_initial_call=True
)
def add_roe_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({
            "BroodstockGroup": "",
            "ProducedEggs": 0,
            "Location": "",
            "Product": "",
            "StartSaleDate": "",
            "ExpireDate": ""
        })
    return rows

@app.callback(
    [Output("allocation-table", "columns"), 
     Output("allocation-table", "data"),
     Output("roe-allocation-graph", "figure"),
     Output("timeline-graph", "figure"),
     Output("unfulfilled-orders", "children")],
    Input("run-solver", "n_clicks"),
    [State("orders-table", "data"),
     State("roe-table", "data"),
     State("objective-dropdown", "value"),
     State("constraints-checklist", "value")],
    prevent_initial_call=True
)
def run_solver(n_clicks, orders, roe, objective, constraints):
    # Convert to DataFrames
    orders_df = pd.DataFrame(orders)
    roe_df = pd.DataFrame(roe)
    
    # Convert numeric columns
    orders_df["OrderedEggs"] = pd.to_numeric(orders_df["OrderedEggs"])
    roe_df["ProducedEggs"] = pd.to_numeric(roe_df["ProducedEggs"])
    
    # Define LP problem
    prob = pulp.LpProblem("Roe_Allocation", pulp.LpMaximize)
    
    # Variables: Assign each order to a broodstock group
    allocation_vars = {(order, group): pulp.LpVariable(f"x_{order}_{group}", cat="Binary")
                       for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"]}
    
    # Constraint: Each order must be assigned to at most one group
    for order in orders_df["OrderID"]:
        prob += pulp.lpSum(allocation_vars[order, group] for group in roe_df["BroodstockGroup"]) <= 1

    # Constraint: Each group's allocated eggs cannot exceed its production capacity
    for group in roe_df["BroodstockGroup"]:
        prob += pulp.lpSum(allocation_vars[order, group] * orders_df.loc[orders_df["OrderID"] == order, "OrderedEggs"].values[0]
                           for order in orders_df["OrderID"]) <= roe_df.loc[roe_df["BroodstockGroup"] == group, "ProducedEggs"].values[0]
    
    # Additional constraints based on user selection
    if "product_match" in constraints:
        for order in orders_df["OrderID"]:
            order_product = orders_df.loc[orders_df["OrderID"] == order, "Product"].values[0]
            for group in roe_df["BroodstockGroup"]:
                group_product = roe_df.loc[roe_df["BroodstockGroup"] == group, "Product"].values[0]
                if order_product != group_product:
                    prob += allocation_vars[order, group] == 0
    
    if "nst_priority" in constraints:
        # NST partners get priority for Steigen roe
        steigen_groups = roe_df[roe_df["Location"] == "Steigen"]["BroodstockGroup"].tolist()
        nst_orders = orders_df[orders_df["CustomerID"] == "NST"]["OrderID"].tolist()
        
        # Add a constraint that gives NST priority for Steigen roe
        for group in steigen_groups:
            for order in orders_df["OrderID"]:
                if order not in nst_orders:
                    # Non-NST orders can only use Steigen roe if there's enough for NST
                    nst_demand = sum(orders_df.loc[orders_df["OrderID"].isin(nst_orders), "OrderedEggs"])
                    steigen_capacity = sum(roe_df.loc[roe_df["BroodstockGroup"].isin(steigen_groups), "ProducedEggs"])
                    
                    if nst_demand > 0 and nst_demand <= steigen_capacity:
                        # Add a constraint that ensures NST orders are fulfilled first
                        prob += allocation_vars[order, group] <= pulp.lpSum(
                            allocation_vars[nst_order, steigen_group] 
                            for nst_order in nst_orders 
                            for steigen_group in steigen_groups
                        ) / len(nst_orders)
    
    if "date_constraints" in constraints:
        for order in orders_df["OrderID"]:
            order_delivery = pd.to_datetime(orders_df.loc[orders_df["OrderID"] == order, "DeliveryDate"].values[0])
            for group in roe_df["BroodstockGroup"]:
                group_start = pd.to_datetime(roe_df.loc[roe_df["BroodstockGroup"] == group, "StartSaleDate"].values[0])
                group_expire = pd.to_datetime(roe_df.loc[roe_df["BroodstockGroup"] == group, "ExpireDate"].values[0])
                
                # Can't use roe before it's available or after it expires
                if order_delivery < group_start or order_delivery > group_expire:
                    prob += allocation_vars[order, group] == 0
    
    # Set objective based on user selection
    if objective == "max_orders":
        # Maximize the number of assigned orders
        prob += pulp.lpSum(allocation_vars[order, group] for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])
    elif objective == "min_waste":
        # Minimize unused roe
        total_roe = sum(roe_df["ProducedEggs"])
        used_roe = pulp.lpSum(allocation_vars[order, group] * orders_df.loc[orders_df["OrderID"] == order, "OrderedEggs"].values[0]
                             for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])
        prob += used_roe  # Maximize used roe = minimize waste
    elif objective == "priority_nst":
        # Prioritize NST orders with a higher weight
        nst_orders = orders_df[orders_df["CustomerID"] == "NST"]["OrderID"].tolist()
        prob += pulp.lpSum(2 * allocation_vars[order, group] for order in nst_orders for group in roe_df["BroodstockGroup"]) + \
                pulp.lpSum(allocation_vars[order, group] for order in orders_df["OrderID"] if order not in nst_orders for group in roe_df["BroodstockGroup"])

    # Solve problem
    from pulp import GLPK_CMD
    prob.solve(GLPK_CMD(msg=False))

    # Collect results
    allocation_results = []
    for (order, group), var in allocation_vars.items():
        if var.varValue == 1:
            order_row = orders_df[orders_df["OrderID"] == order].iloc[0]
            group_row = roe_df[roe_df["BroodstockGroup"] == group].iloc[0]
            allocation_results.append({
                "OrderID": order,
                "CustomerID": order_row["CustomerID"],
                "OrderedEggs": order_row["OrderedEggs"],
                "BroodstockGroup": group,
                "Location": group_row["Location"],
                "Product": order_row["Product"],
                "DeliveryDate": order_row["DeliveryDate"]
            })

    allocation_df = pd.DataFrame(allocation_results)
    
    # Find unfulfilled orders
    fulfilled_orders = allocation_df["OrderID"].unique() if not allocation_df.empty else []
    unfulfilled_orders = orders_df[~orders_df["OrderID"].isin(fulfilled_orders)]
    
    # Create allocation visualization
    if not allocation_df.empty:
        # Calculate used eggs per broodstock group
        used_eggs = allocation_df.groupby("BroodstockGroup")["OrderedEggs"].sum().reset_index()
        used_eggs = used_eggs.rename(columns={"OrderedEggs": "UsedEggs"})
        
        # Merge with roe data to get total available
        allocation_viz = pd.merge(roe_df, used_eggs, on="BroodstockGroup", how="left")
        allocation_viz["UsedEggs"] = allocation_viz["UsedEggs"].fillna(0)
        allocation_viz["RemainingEggs"] = allocation_viz["ProducedEggs"] - allocation_viz["UsedEggs"]
        
        # Create stacked bar chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=allocation_viz["BroodstockGroup"],
            y=allocation_viz["UsedEggs"],
            name="Used Eggs",
            marker_color='#1f77b4'
        ))
        fig1.add_trace(go.Bar(
            x=allocation_viz["BroodstockGroup"],
            y=allocation_viz["RemainingEggs"],
            name="Remaining Eggs",
            marker_color='#2ca02c'
        ))
        
        fig1.update_layout(
            title="Roe Allocation by Broodstock Group",
            xaxis_title="Broodstock Group",
            yaxis_title="Eggs",
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create timeline visualization
        fig2 = go.Figure()
        
        # Add roe availability periods
        for _, row in roe_df.iterrows():
            fig2.add_trace(go.Bar(
                x=[row["BroodstockGroup"]],
                y=[1],
                base=0,
                marker=dict(
                    color='rgba(55, 128, 191, 0.3)',
                    line=dict(color='rgba(55, 128, 191, 0.7)', width=2)
                ),
                name=f"{row['BroodstockGroup']} Available",
                text=f"{row['ProducedEggs']:,} eggs",
                hoverinfo="text",
                showlegend=False
            ))
        
        # Add order allocations
        if not allocation_df.empty:
            for _, row in allocation_df.iterrows():
                fig2.add_trace(go.Scatter(
                    x=[row["BroodstockGroup"]],
                    y=[0.5],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=15,
                        color="red"
                    ),
                    name=f"Order {row['OrderID']}",
                    text=f"Order {row['OrderID']}: {row['OrderedEggs']:,} eggs for {row['CustomerID']}",
                    hoverinfo="text",
                    showlegend=False
                ))
        
        fig2.update_layout(
            title="Roe Allocation Timeline",
            xaxis_title="Broodstock Group",
            yaxis=dict(
                showticklabels=False,
                showgrid=False
            ),
            height=300
        )
    else:
        # Create empty figures if no allocation
        fig1 = go.Figure()
        fig1.update_layout(title="No allocation data available")
        
        fig2 = go.Figure()
        fig2.update_layout(title="No timeline data available")
    
    # Create unfulfilled orders message
    if len(unfulfilled_orders) > 0:
        unfulfilled_html = html.Div([
            html.H4("Unfulfilled Orders", style={"color": "red"}),
            html.P(f"There are {len(unfulfilled_orders)} orders that could not be fulfilled:"),
            html.Ul([
                html.Li(f"Order {row['OrderID']}: {row['OrderedEggs']:,} eggs for {row['CustomerID']} ({row['Product']})")
                for _, row in unfulfilled_orders.iterrows()
            ])
        ])
    else:
        unfulfilled_html = html.P("All orders have been fulfilled!", style={"color": "green"})

    # Return results
    columns = [{"name": col, "id": col} for col in allocation_df.columns] if not allocation_df.empty else []
    return (columns, 
            allocation_df.to_dict("records") if not allocation_df.empty else [], 
            fig1, 
            fig2,
            unfulfilled_html)

# Add some CSS for better styling
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
            .data-tables {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }
            .table-container {
                flex: 1;
                min-width: 300px;
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

# Run the Dash app with updated configuration
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)
