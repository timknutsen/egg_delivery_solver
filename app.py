import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
from datetime import datetime
import os  # Added os import for environment variables

# Enhanced sample data with additional fields
orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 4, 5],
    "CustomerID": ["C1", "C2", "C3", "mowi", "NST"],  # Updated C4 to mowi
    "OrderedEggs": [50000, 70000, 60000, 80000, 45000],
    "Product": ["Standard", "Premium", "Standard", "Premium", "Standard"],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-25", "2025-03-12"],
    "GeographicPreference": ["Steigen", "Hemne", "Erfjord", "Steigen", "Steigen"],  # Example geographic preference
    "CoolingPreference": [False, True, False, True, False]  # Example cooling preference
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["A", "B", "C", "D"],
    "ProducedEggs": [100000, 120000, 90000, 80000],
    "Location": ["Steigen", "Hemne", "Erfjord", "Steigen"],
    "Product": ["Standard", "Premium", "Standard", "Premium"],
    "StartSaleDate": ["2025-02-15", "2025-02-20", "2025-02-25", "2025-03-01"],
    "ExpireDate": ["2025-04-15", "2025-04-20", "2025-04-25", "2025-05-01"],
    "BroadfishCycle": ["Cycle1", "Cycle2", "Cycle1", "Cycle2"],  # New field for Broadfish Cycle
    "GainPercentage": [80, 60, 90, 70],  # New field for sustainability (GAIN %)
    "QualityScore": [85, 90, 75, 88],  # New field for quality
    "PDStatus": ["PD-Free", "PD-Free", "PD-Present", "PD-Free"],  # New field for screening status (PD)
    "CoolingCapacity": [True, True, False, True]  # New field for cooling capacity
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
                    {"name": "Delivery Date", "id": "DeliveryDate"},
                    {"name": "Geographic Preference", "id": "GeographicPreference"},
                    {"name": "Cooling Preference", "id": "CoolingPreference", "type": "boolean"}
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
                    {"name": "Expiry Date", "id": "ExpireDate"},
                    {"name": "Broadfish Cycle", "id": "BroadfishCycle"},
                    {"name": "Gain %", "id": "GainPercentage", "type": "numeric", "format": {"specifier": ".0f"}},
                    {"name": "Quality Score", "id": "QualityScore", "type": "numeric", "format": {"specifier": ".0f"}},
                    {"name": "PD Status", "id": "PDStatus"},
                    {"name": "Cooling Capacity", "id": "CoolingCapacity", "type": "boolean"}
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
                    {"label": "Prioritize NST partners", "value": "priority_nst"},
                    {"label": "Increase Broodstock Groups", "value": "increase_groups"},
                    {"label": "Decrease Broodstock Groups", "value": "decrease_groups"}
                ],
                value="max_orders"
            ),
            html.Label("Apply Constraints:"),
            dcc.Checklist(
                id="constraints-checklist",
                options=[
                    {"label": "NST partners get priority for Steigen roe", "value": "nst_priority"},
                    {"label": "Match product types", "value": "product_match"},
                    {"label": "Respect delivery/expiry dates", "value": "date_constraints"},
                    {"label": "Geographic preference", "value": "geo_preference"},
                    {"label": "Minimum Gain % (70%)", "value": "gain_constraint"},
                    {"label": "Minimum Quality Score (80)", "value": "quality_constraint"},
                    {"label": "PD-Free only", "value": "pd_free"},
                    {"label": "Cooling capacity match", "value": "cooling_match"},
                    {"label": "Temperature Treatment Scenario", "value": "temp_treatment"}
                ],
                value=["nst_priority", "product_match"]
            ),
            html.Label("Temperature Treatment (days added to storage):"),
            dcc.Slider(
                id="temp-treatment-slider",
                min=0,
                max=30,
                step=5,
                value=0,
                marks={i: f'{i}d' for i in range(0, 31, 5)}
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
            "DeliveryDate": "",
            "GeographicPreference": "",
            "CoolingPreference": False
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
            "ExpireDate": "",
            "BroadfishCycle": "",
            "GainPercentage": 0,
            "QualityScore": 0,
            "PDStatus": "",
            "CoolingCapacity": False
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
     State("constraints-checklist", "value"),
     State("temp-treatment-slider", "value")],
    prevent_initial_call=True
)
def run_solver(n_clicks, orders, roe, objective, constraints, temp_treatment_days):
    # Convert to DataFrames
    orders_df = pd.DataFrame(orders)
    roe_df = pd.DataFrame(roe)
    
    # Convert numeric columns
    orders_df["OrderedEggs"] = pd.to_numeric(orders_df["OrderedEggs"], errors='coerce')
    roe_df["ProducedEggs"] = pd.to_numeric(roe_df["ProducedEggs"], errors='coerce')
    roe_df["GainPercentage"] = pd.to_numeric(roe_df["GainPercentage"], errors='coerce')
    roe_df["QualityScore"] = pd.to_numeric(roe_df["QualityScore"], errors='coerce')

    # Adjust roe availability based on temperature treatment (simplified simulation)
    if "temp_treatment" in constraints:
        roe_df["AdjustedProducedEggs"] = roe_df["ProducedEggs"] * (1 - (temp_treatment_days / 30) * 0.1)  # 10% reduction per 30 days
    else:
        roe_df["AdjustedProducedEggs"] = roe_df["ProducedEggs"]

    # Define LP problem
    prob = pulp.LpProblem("Roe_Allocation", pulp.LpMaximize)
    
    # Variables: Assign each order to a broodstock group
    allocation_vars = {(order, group): pulp.LpVariable(f"x_{order}_{group}", cat="Binary")
                       for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"]}
    
    # Constraint: Each order must be assigned to at most one group
    for order in orders_df["OrderID"]:
        prob += pulp.lpSum(allocation_vars[order, group] for group in roe_df["BroodstockGroup"]) <= 1

    # Constraint: Each group's allocated eggs cannot exceed its adjusted production capacity
    for group in roe_df["BroodstockGroup"]:
        prob += pulp.lpSum(allocation_vars[order, group] * orders_df.loc[orders_df["OrderID"] == order, "OrderedEggs"].values[0]
                           for order in orders_df["OrderID"]) <= roe_df.loc[roe_df["BroodstockGroup"] == group, "AdjustedProducedEggs"].values[0]
    
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
        
        for group in steigen_groups:
            for order in orders_df["OrderID"]:
                if order not in nst_orders:
                    nst_demand = sum(orders_df.loc[orders_df["OrderID"].isin(nst_orders), "OrderedEggs"])
                    steigen_capacity = sum(roe_df.loc[roe_df["BroodstockGroup"].isin(steigen_groups), "AdjustedProducedEggs"])
                    
                    if nst_demand > 0 and nst_demand <= steigen_capacity:
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
                
                if order_delivery < group_start or order_delivery > group_expire:
                    prob += allocation_vars[order, group] == 0
    
    if "geo_preference" in constraints:
        for order in orders_df["OrderID"]:
            order_geo = orders_df.loc[orders_df["OrderID"] == order, "GeographicPreference"].values[0]
            for group in roe_df["BroodstockGroup"]:
                group_location = roe_df.loc[roe_df["BroodstockGroup"] == group, "Location"].values[0]
                if order_geo and order_geo != group_location:
                    prob += allocation_vars[order, group] == 0
    
    if "gain_constraint" in constraints:
        for group in roe_df["BroodstockGroup"]:
            if roe_df.loc[roe_df["BroodstockGroup"] == group, "GainPercentage"].values[0] < 70:
                for order in orders_df["OrderID"]:
                    prob += allocation_vars[order, group] == 0
    
    if "quality_constraint" in constraints:
        for group in roe_df["BroodstockGroup"]:
            if roe_df.loc[roe_df["BroodstockGroup"] == group, "QualityScore"].values[0] < 80:
                for order in orders_df["OrderID"]:
                    prob += allocation_vars[order, group] == 0
    
    if "pd_free" in constraints:
        for group in roe_df["BroodstockGroup"]:
            if roe_df.loc[roe_df["BroodstockGroup"] == group, "PDStatus"].values[0] != "PD-Free":
                for order in orders_df["OrderID"]:
                    prob += allocation_vars[order, group] == 0
    
    if "cooling_match" in constraints:
        for order in orders_df["OrderID"]:
            order_cooling = orders_df.loc[orders_df["OrderID"] == order, "CoolingPreference"].values[0]
            for group in roe_df["BroodstockGroup"]:
                group_cooling = roe_df.loc[roe_df["BroodstockGroup"] == group, "CoolingCapacity"].values[0]
                if order_cooling and not group_cooling:
                    prob += allocation_vars[order, group] == 0

    # Set objective based on user selection
    if objective == "max_orders":
        # Maximize the number of assigned orders
        prob += pulp.lpSum(allocation_vars[order, group] for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])
    elif objective == "min_waste":
        # Minimize unused roe
        total_roe = sum(roe_df["AdjustedProducedEggs"])
        used_roe = pulp.lpSum(allocation_vars[order, group] * orders_df.loc[orders_df["OrderID"] == order, "OrderedEggs"].values[0]
                             for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])
        prob += used_roe  # Maximize used roe = minimize waste
    elif objective == "priority_nst":
        # Prioritize NST orders with a higher weight
        nst_orders = orders_df[orders_df["CustomerID"] == "NST"]["OrderID"].tolist()
        prob += pulp.lpSum(2 * allocation_vars[order, group] for order in nst_orders for group in roe_df["BroodstockGroup"]) + \
                pulp.lpSum(allocation_vars[order, group] for order in orders_df["OrderID"] if order not in nst_orders for group in roe_df["BroodstockGroup"])
    elif objective == "increase_groups":
        # Increase allocation to groups with higher GainPercentage or QualityScore
        prob += pulp.lpSum(allocation_vars[order, group] * (roe_df.loc[roe_df["BroodstockGroup"] == group, "GainPercentage"].values[0] + 
                                                            roe_df.loc[roe_df["BroodstockGroup"] == group, "QualityScore"].values[0])
                           for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])
    elif objective == "decrease_groups":
        # Decrease allocation to groups with lower GainPercentage or QualityScore
        prob += pulp.lpSum(allocation_vars[order, group] * (100 - (roe_df.loc[roe_df["BroodstockGroup"] == group, "GainPercentage"].values[0] + 
                                                                   roe_df.loc[roe_df["BroodstockGroup"] == group, "QualityScore"].values[0]) / 2)
                           for order in orders_df["OrderID"] for group in roe_df["BroodstockGroup"])

    # Solve problem using CBC solver
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

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
                "DeliveryDate": order_row["DeliveryDate"],
                "BroadfishCycle": group_row["BroadfishCycle"],
                "GainPercentage": group_row["GainPercentage"],
                "QualityScore": group_row["QualityScore"]
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
        allocation_viz["RemainingEggs"] = allocation_viz["AdjustedProducedEggs"] - allocation_viz["UsedEggs"]
        
        # Create stacked bar chart for Roe Allocation
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
        
        # Create improved timeline visualization
        fig2 = go.Figure()

        # Convert dates to datetime for plotting
        roe_df['StartSaleDate'] = pd.to_datetime(roe_df['StartSaleDate'])
        roe_df['ExpireDate'] = pd.to_datetime(roe_df['ExpireDate'])
        orders_df['DeliveryDate'] = pd.to_datetime(orders_df['DeliveryDate'])

        # Define color mapping for broodstock groups
        color_map = {
            'A': 'rgba(100, 149, 237, 0.6)',  # Cornflower blue
            'B': 'rgba(144, 238, 144, 0.6)',  # Light green
            'C': 'rgba(255, 182, 193, 0.6)',  # Light pink
            'D': 'rgba(255, 218, 185, 0.6)'   # Peach
        }

        # Add roe availability periods with better colors
        for _, row in roe_df.iterrows():
            group = row['BroodstockGroup']
            color = color_map.get(group, 'rgba(55, 128, 191, 0.6)')
            
            fig2.add_trace(go.Bar(
                x=[row['ExpireDate'] - row['StartSaleDate']],  # Duration
                y=[group],
                orientation='h',
                base=[row['StartSaleDate']],  # Start date
                marker=dict(
                    color=color,
                    line=dict(color='rgba(50, 50, 50, 0.8)', width=1)
                ),
                name=f"{group} Available",
                text=f"{group}: {row['AdjustedProducedEggs']:,} eggs<br>{row['Product']}<br>Quality: {row['QualityScore']}",
                hoverinfo="text",
                showlegend=True
            ))

        # Add order delivery points with better visibility
        if not allocation_df.empty:
            for _, row in allocation_df.iterrows():
                delivery_date = pd.to_datetime(row['DeliveryDate'])
                fig2.add_trace(go.Scatter(
                    x=[delivery_date],
                    y=[row['BroodstockGroup']],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=12,
                        color="red",
                        line=dict(color='black', width=1)
                    ),
                    name=f"Order {row['OrderID']}",
                    text=f"Order {row['OrderID']}: {row['OrderedEggs']:,} eggs<br>Customer: {row['CustomerID']}<br>Product: {row['Product']}",
                    hoverinfo="text",
                    showlegend=False
                ))

        # Calculate date range with a small buffer (10% on each side)
        all_dates = []
        for _, row in roe_df.iterrows():
            all_dates.extend([row['StartSaleDate'], row['ExpireDate']])
        for _, row in orders_df.iterrows():
            all_dates.append(pd.to_datetime(row['DeliveryDate']))

        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = (max_date - min_date).days
        buffer_days = max(date_range * 0.1, 5)  # At least 5 days buffer

        # Update layout for timeline with improved date range
        fig2.update_layout(
            title="Roe Allocation Timeline",
            xaxis_title="Date",
            yaxis_title="Broodstock Group",
            xaxis=dict(
                type="date",
                tickformat="%Y-%m-%d",
                range=[
                    (min_date - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d'),
                    (max_date + pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
                ]
            ),
            yaxis=dict(
                autorange="reversed",  # Reverse y-axis to show A at top, D at bottom
                categoryorder='array',
                categoryarray=['A', 'B', 'C', 'D']
            ),
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            hovermode="closest"
        )

        # Add a grid for better readability
        fig2.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)'
        )

        fig2.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)'
        )

        # Add annotations for broodstock group details
        for i, group in enumerate(['A', 'B', 'C', 'D']):
            group_data = roe_df[roe_df['BroodstockGroup'] == group]
            if not group_data.empty:
                fig2.add_annotation(
                    x=min_date - pd.Timedelta(days=buffer_days/2),
                    y=group,
                    text=f"{group} Available",
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="right"
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
