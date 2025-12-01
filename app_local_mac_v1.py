import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import plotly.graph_objects as go
import pulp
import os
import random
import datetime
import time

# Sample data (replace with your actual data loading)
salmon_producers = ["Mowi ASA", "Lerøy Seafood Group", "SalMar ASA", "Cermaq Group ASA", "Bakkafrost"]
product_types = ["Shield", "Gain (Premium)"]

orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 4, 5, 6],
    "CustomerID": ["Mowi ASA", "SalMar ASA", "Bakkafrost", "Lerøy Seafood Group", "Mowi ASA", "SalMar ASA"],
    "OrderedEggs": [50000, 40000, 50000, 80000, 60000, 70000],
    "Product": ["Shield", "Shield", "Shield", "Gain (Premium)", "Gain (Premium)", "Shield"],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-15", "2025-03-22", "2025-03-25"]
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["A", "B", "C", "D"],
    "ProducedEggs": [100000, 100000, 50000, 60000],
    "Location": ["Steigen", "Hemne", "Erfjord", "Steigen"],
    "Product": ["Shield", "Gain (Premium)", "Shield", "Gain (Premium)"],
    "StartSaleDate": ["2025-02-15", "2025-02-15", "2025-03-01", "2025-02-20"],
    "ExpireDate": ["2025-04-15", "2025-04-20", "2025-04-30", "2025-04-25"]
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
    'graph': {'height': '400px'}
}

# App layout
app.layout = html.Div(style=styles['container'], children=[
    html.H1("Roe Allocation Dashboard", style={'text-align': 'center', 'margin-bottom': '20px'}),

    # Status Bar
    html.Div(id='status-bar', style=styles['status_bar'], children="Ready"),

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
                value=['product_match', 'date_constraints']
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
                value='partial'
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
                value='chronological'
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

    # Broodstock Utilization Graph
    html.Div(style=styles['section'], children=[
        html.H3("Broodstock Utilization"),
        dcc.Graph(id='utilization-graph', style=styles['graph'])
    ]),

    # Hidden div to store the calculation results
    dcc.Store(id='calculation-results', data=None)
])

# --- Callbacks ---
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
        delivery_date = (datetime.date.today() + datetime.timedelta(days=random.randint(30, 90))).strftime('%Y-%m-%d')  # Future date
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
        new_broodstock_group = chr(65 + len(existing_data))  # A, B, C...
        new_row = {
            'BroodstockGroup': new_broodstock_group,
            'ProducedEggs': 50000,
            'Location': 'New Location',
            'Product': random.choice(product_types),
            'StartSaleDate': (datetime.date.today() + datetime.timedelta(days=10)).strftime('%Y-%m-%d'),
            'ExpireDate': (datetime.date.today() + datetime.timedelta(days=60)).strftime('%Y-%m-%d')
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
        Input('constraints-checklist', 'value'), # Added as Input
        Input('allocation-method', 'value'), # Added as Input
        Input('order-priority', 'value') # Added as Input
    ],
    [
        State('orders-table', 'data'),
        State('roe-table', 'data'),
    ],
    prevent_initial_call=True
)
def run_solver(n_clicks, constraints_value, allocation_method, order_priority, orders_data, roe_data):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'run-solver-button' and not n_clicks: # Handle initial button load
        return "Ready", styles['status_bar'], "", None
    if trigger_id == 'constraints-checklist' and not constraints_value: # Handle initial constraints load
        return "Ready", styles['status_bar'], "", None
    if trigger_id == 'allocation-method' and not allocation_method: # Handle initial allocation method load
        return "Ready", styles['status_bar'], "", None
    if trigger_id == 'order-priority' and not order_priority: # Handle initial order priority load
        return "Ready", styles['status_bar'], "", None


    try:
        # Dataframe creation and type conversion
        orders_df = pd.DataFrame(orders_data)
        roe_df = pd.DataFrame(roe_data)

        orders_df["OrderedEggs"] = pd.to_numeric(orders_df["OrderedEggs"], errors='coerce')
        roe_df["ProducedEggs"] = pd.to_numeric(roe_df["ProducedEggs"], errors='coerce')

        # Basic Validation - Check if data is valid and present
        if orders_df.empty or roe_df.empty or orders_df.isnull().any().any() or roe_df.isnull().any().any():
            return "Error: Please ensure orders and roe data are valid.", {**styles['status_bar'], 'background-color': '#f8d7da', 'color': '#721c24'}, "", None

        allocation_results = []  # List to hold allocation results
        unfulfilled_orders = []

        if allocation_method == "binary":
            # --- Binary Allocation Logic (Pulp Solver) ---
            prob = pulp.LpProblem("Roe_Allocation_Binary", pulp.LpMaximize)

            # Decision variables: x[order_id, broodstock_group] is 1 if order is allocated to group, 0 otherwise
            allocation_vars = pulp.LpVariable.dicts(
                "Allocate",
                [(order_id, group) for order_id in orders_df['OrderID'] for group in roe_df['BroodstockGroup']],
                cat='Binary'
            )

            # Objective function: Maximize the number of allocated orders (or total allocated eggs in binary case - let's maximize orders for now)
            prob += pulp.lpSum(allocation_vars[order_id, group] for order_id in orders_df['OrderID'] for group in roe_df['BroodstockGroup']), "Maximize Allocated Orders"

            # Constraints:

            # 1. Each order can be assigned to at most one broodstock group
            for order_id in orders_df['OrderID']:
                prob += pulp.lpSum(allocation_vars[order_id, group] for group in roe_df['BroodstockGroup']) <= 1, f"One_Group_Per_Order_{order_id}"

            # 2. Broodstock group capacity constraint
            for group in roe_df['BroodstockGroup']:
                group_capacity = roe_df[roe_df['BroodstockGroup'] == group]['ProducedEggs'].iloc[0]
                prob += pulp.lpSum(allocation_vars[order_id, group] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                                for order_id in orders_df['OrderID']) <= group_capacity, f"Group_Capacity_{group}"

            # 3. Product type constraint
            if 'product_match' in constraints_value:
                for order_id in orders_df['OrderID']:
                    order_product = orders_df[orders_df['OrderID'] == order_id]['Product'].iloc[0]
                    for group in roe_df['BroodstockGroup']:
                        group_product = roe_df[roe_df['BroodstockGroup'] == group]['Product'].iloc[0]
                        if order_product != group_product:
                            prob += allocation_vars[order_id, group] == 0, f"Product_Match_{order_id}_{group}"

            # 4. Delivery/Expiry date constraints
            if 'date_constraints' in constraints_value:
                for order_id in orders_df['OrderID']:
                    delivery_date = pd.to_datetime(orders_df[orders_df['OrderID'] == order_id]['DeliveryDate'].iloc[0])
                    for group in roe_df['BroodstockGroup']:
                        start_sale_date = pd.to_datetime(roe_df[roe_df['BroodstockGroup'] == group]['StartSaleDate'].iloc[0])
                        expire_date = pd.to_datetime(roe_df[roe_df['BroodstockGroup'] == group]['ExpireDate'].iloc[0])
                        if delivery_date < start_sale_date or delivery_date > expire_date:
                            prob += allocation_vars[order_id, group] == 0, f"Date_Constraint_{order_id}_{group}"

            # 5. NST Priority (Example - needs refinement based on actual priority rules)
            if 'nst_priority' in constraints_value:
                nst_customers = [cust for cust in orders_df['CustomerID'].unique() if "North Sea Traders" in cust]
                nst_orders_ids = orders_df[orders_df['CustomerID'].isin(nst_customers)]['OrderID'].tolist()
                steigen_groups = roe_df[roe_df['Location'] == 'Steigen']['BroodstockGroup'].tolist()

                for group in steigen_groups:
                    for order_id in orders_df['OrderID']:
                        if order_id not in nst_orders_ids: # Non-NST orders for Steigen groups
                            # Prioritize NST orders for Steigen groups - this is a simplification
                             prob += pulp.lpSum(allocation_vars[nst_order_id, group] for nst_order_id in nst_orders_ids if nst_order_id in orders_df['OrderID'].tolist()) >= allocation_vars[order_id, group] , f"NST_Priority_{order_id}_{group}"


            # Solve the problem
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10) # Using CBC solver, adjust as needed
            prob.solve(solver)

            # Extract allocation results
            if prob.status == pulp.LpStatusOptimal:
                for order_id, group in allocation_vars:
                    if allocation_vars[(order_id, group)].varValue > 0.5: # Check if binary variable is close to 1
                        order_row = orders_df[orders_df['OrderID'] == order_id].iloc[0]
                        group_row = roe_df[roe_df['BroodstockGroup'] == group].iloc[0]
                        allocation_results.append({
                            "OrderID": order_id,
                            "CustomerID": order_row["CustomerID"],
                            "OrderedEggs": int(order_row["OrderedEggs"]),
                            "AllocatedEggs": int(order_row["OrderedEggs"]), # Binary allocation - full order or none
                            "FulfillmentPct": 100.0,
                            "Product": order_row["Product"],
                            "DeliveryDate": order_row["DeliveryDate"],
                            "BroodstockGroup": group,
                            "Location": group_row["Location"]
                        })
                # Unfulfilled orders - orders not in allocation_results
                allocated_order_ids = {res['OrderID'] for res in allocation_results}
                unfulfilled_orders = [order_id for order_id in orders_df['OrderID'] if order_id not in allocated_order_ids]
            else:
                unfulfilled_orders = orders_df['OrderID'].tolist() # If not optimal, consider all unfulfilled
                print("Solver did not find optimal solution for binary allocation.") # Add more sophisticated handling if needed


        elif allocation_method == "partial":
            # --- Partial Fulfillment Logic (Simplified - as before) ---
            remaining_roe = roe_df.copy()
            remaining_roe['RemainingEggs'] = remaining_roe['ProducedEggs']

            for _, order in orders_df.iterrows():
                order_id = order['OrderID']
                customer = order['CustomerID']
                eggs_needed = order['OrderedEggs']
                product = order['Product']
                delivery_date = pd.to_datetime(order['DeliveryDate'])
                eggs_allocated = 0
                group_allocations = []

                compatible_groups = []
                for _, group in remaining_roe.iterrows():
                    if 'product_match' in constraints_value and group['Product'] != product:
                        continue
                    if 'date_constraints' in constraints_value:
                        start_date = pd.to_datetime(group['StartSaleDate'])
                        end_date = pd.to_datetime(group['ExpireDate'])
                        if delivery_date < start_date or delivery_date > end_date:
                            continue
                    compatible_groups.append({
                        "BroodstockGroup": group["BroodstockGroup"],
                        "RemainingEggs": group["RemainingEggs"],
                        "Location": group["Location"]
                    })

                compatible_groups = sorted(compatible_groups, key=lambda x: x['RemainingEggs'], reverse=True)

                for group_info in compatible_groups:
                    group_id = group_info['BroodstockGroup']
                    available_eggs = group_info['RemainingEggs']
                    allocation_amount = min(eggs_needed - eggs_allocated, available_eggs)

                    if allocation_amount > 0:
                        group_allocations.append({
                            "BroodstockGroup": group_id,
                            "AllocatedEggs": int(allocation_amount),
                            "Location": group_info["Location"]
                        })
                        eggs_allocated += allocation_amount
                        remaining_roe.loc[remaining_roe['BroodstockGroup'] == group_id, 'RemainingEggs'] -= allocation_amount
                        if eggs_allocated >= eggs_needed:
                            break

                if eggs_allocated > 0:
                    fulfillment_pct = (eggs_allocated / eggs_needed) * 100
                    for group_alloc in group_allocations:
                        allocation_results.append({
                            "OrderID": order_id,
                            "CustomerID": customer,
                            "OrderedEggs": int(eggs_needed),
                            "AllocatedEggs": int(group_alloc["AllocatedEggs"]),
                            "FulfillmentPct": fulfillment_pct,
                            "Product": product,
                            "DeliveryDate": order["DeliveryDate"],
                            "BroodstockGroup": group_alloc["BroodstockGroup"],
                            "Location": group_alloc["Location"]
                        })
                else:
                    unfulfilled_orders.append(order_id)


        # --- Prepare results
        allocation_df = pd.DataFrame(allocation_results)
        computation_time = 0  # Replace with actual calculation time if needed

        # Prepare data for the utilization graph
        utilization_data = []
        for _, group_row in roe_df.iterrows():
            group_id = group_row["BroodstockGroup"]
            total_capacity = group_row["ProducedEggs"]
            allocated = 0
            if not allocation_df.empty:
                allocated = allocation_df[allocation_df["BroodstockGroup"] == group_id]["AllocatedEggs"].sum()
            utilization_data.append({
                "BroodstockGroup": group_id,
                "Total": int(total_capacity),
                "Allocated": int(allocated),
                "Remaining": int(total_capacity - allocated)
            })

        # Return results
        return (
            "Allocation Complete",
            {**styles['status_bar'], 'background-color': '#d4edda', 'color': '#155724'},
            f"Allocation completed in {computation_time} seconds",
            {
                "allocation_results": allocation_results,
                "unfulfilled_orders": unfulfilled_orders,
                "utilization_data": utilization_data
            }
        )

    except Exception as e:
        return (
            f"Error: {str(e)}",
            {**styles['status_bar'], 'background-color': '#f8d7da', 'color': '#721c24'},
            f"Error during allocation: {str(e)}",
            None
        )

@app.callback(
    [
        Output('allocation-table', 'columns'),
        Output('allocation-table', 'data'),
        Output('unfulfilled-orders', 'children'),
        Output('utilization-graph', 'figure')
    ],
    Input('calculation-results', 'data'),
    prevent_initial_call=True
)
def update_ui_with_results(calculation_results):
    if not calculation_results:
        return [], [], "No results yet", go.Figure(layout={'title': 'No Data'})

    try:
        allocation_results = calculation_results.get('allocation_results', [])
        unfulfilled_orders = calculation_results.get('unfulfilled_orders', [])
        utilization_data = calculation_results.get('utilization_data', [])

        # Allocation Table
        allocation_columns = [
            {"name": i, "id": i, "type": "numeric" if i in ("OrderedEggs", "AllocatedEggs", "FulfillmentPct") else "text"}
            for i in ("OrderID", "CustomerID", "OrderedEggs", "AllocatedEggs", "FulfillmentPct", "Product", "DeliveryDate", "BroodstockGroup", "Location")
        ]
        allocation_data = allocation_results

        # Unfulfilled Orders
        if unfulfilled_orders:
            unfulfilled_text = f"Unfulfilled Orders (Binary Allocation - Full order or none): {', '.join(map(str, unfulfilled_orders))}"
            unfulfilled_div = html.Div(unfulfilled_text, style={'color': 'red'})
        else:
            unfulfilled_div = html.Div("All possible orders fulfilled (Binary Allocation)!", style={'color': 'green'})

        # Broodstock Utilization Graph
        fig = go.Figure()
        if utilization_data:
            df = pd.DataFrame(utilization_data)
            fig.add_trace(go.Bar(x=df["BroodstockGroup"], y=df["Allocated"], name="Allocated", marker_color="#4CAF50"))
            fig.add_trace(go.Bar(x=df["BroodstockGroup"], y=df["Remaining"], name="Remaining", marker_color="#ccc"))
            fig.update_layout(barmode='stack', title="Broodstock Utilization", xaxis_title="Broodstock Group", yaxis_title="Eggs")
            # Add percentage labels
            for i, row in df.iterrows():
                utilization_pct = (row["Allocated"] / row["Total"]) * 100 if row["Total"] > 0 else 0
                fig.add_annotation(
                    x=row["BroodstockGroup"],
                    y=row["Total"] / 2,  # Position in the middle of the bar
                    text=f"{utilization_pct:.1f}%",
                    showarrow=False,
                    font=dict(color="white", size=12)  # Adjust font size as needed
                )
        else:
            fig.update_layout(title="No Broodstock Data Available")

        return allocation_columns, allocation_data, unfulfilled_div, fig

    except Exception as e:
        print(f"Error updating UI: {e}")
        return [], [], html.Div(f"Error: {str(e)}", style={'color': 'red'}), go.Figure(layout={'title': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run_server(debug=True)
