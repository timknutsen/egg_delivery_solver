import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import pulp

# Sample data
orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 4],
    "CustomerID": ["C1", "C2", "C3", "C4"],
    "OrderedEggs": [50000, 70000, 60000, 80000],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-25"],
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["A", "B", "C"],
    "ProducedEggs": [100000, 120000, 90000],
    "Location": ["Steigen", "Hemne", "Erfjord"],
})

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Roe Allocation Dashboard"),
    
    # Orders Table
    html.H3("Customer Orders"),
    dash_table.DataTable(
        id="orders-table",
        columns=[{"name": col, "id": col} for col in orders_data.columns],
        data=orders_data.to_dict("records"),
        style_table={"overflowX": "auto"}
    ),
    
    # Roe Production Table
    html.H3("Available Roe by Broodstock Group"),
    dash_table.DataTable(
        id="roe-table",
        columns=[{"name": col, "id": col} for col in roe_data.columns],
        data=roe_data.to_dict("records"),
        style_table={"overflowX": "auto"}
    ),
    
    # Solve Button
    html.Button("Run Allocation Solver", id="run-solver", n_clicks=0),
    
    # Allocation Results
    html.H3("Allocation Results"),
    dash_table.DataTable(id="allocation-table", style_table={"overflowX": "auto"}),

    # Visualization
    dcc.Graph(id="roe-availability-graph")
])

@app.callback(
    [Output("allocation-table", "columns"), Output("allocation-table", "data"),
     Output("roe-availability-graph", "figure")],
    Input("run-solver", "n_clicks"),
    prevent_initial_call=True
)
def run_solver(n_clicks):
    # Define LP problem
    prob = pulp.LpProblem("Roe_Allocation", pulp.LpMaximize)
    
    # Variables: Assign each order to a broodstock group
    allocation_vars = {(order, group): pulp.LpVariable(f"x_{order}_{group}", cat="Binary")
                       for order in orders_data["OrderID"] for group in roe_data["BroodstockGroup"]}
    
    # Constraint: Each order must be assigned to exactly one group
    for order in orders_data["OrderID"]:
        prob += pulp.lpSum(allocation_vars[order, group] for group in roe_data["BroodstockGroup"]) == 1

    # Constraint: Each group's allocated eggs cannot exceed its production capacity
    for group in roe_data["BroodstockGroup"]:
        prob += pulp.lpSum(allocation_vars[order, group] * orders_data.loc[orders_data["OrderID"] == order, "OrderedEggs"].values[0]
                           for order in orders_data["OrderID"]) <= roe_data.loc[roe_data["BroodstockGroup"] == group, "ProducedEggs"].values[0]

    # Objective: Maximize the number of assigned orders
    prob += pulp.lpSum(allocation_vars[order, group] for order in orders_data["OrderID"] for group in roe_data["BroodstockGroup"])

    # Solve problem
    # Use GLPK solver which we just installed via conda
    from pulp import GLPK_CMD
    prob.solve(GLPK_CMD(msg=True))

    # Collect results
    allocation_results = []
    for (order, group), var in allocation_vars.items():
        if var.varValue == 1:
            allocation_results.append({"OrderID": order, "BroodstockGroup": group})

    allocation_df = pd.DataFrame(allocation_results)

    # Visualization of roe allocation
    fig = px.bar(roe_data, x="BroodstockGroup", y="ProducedEggs", title="Roe Availability",
                 labels={"ProducedEggs": "Available Roe"}, text="ProducedEggs")

    return ([{"name": col, "id": col} for col in allocation_df.columns], allocation_df.to_dict("records"), fig)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
