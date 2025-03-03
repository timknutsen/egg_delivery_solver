import pulp
import pandas as pd
from typing import List, Tuple

def add_one_group_per_order_constraints(prob: pulp.LpProblem, order_to_group_assignments: dict, orders_df: pd.DataFrame, roe_df: pd.DataFrame) -> None:
    """Ensure each order is assigned to at most one group."""
    for order_id in orders_df['OrderID']:
        prob += pulp.lpSum(order_to_group_assignments[order_id, group_id] for group_id in roe_df['GroupID']) <= 1, f"One_Group_Per_Order_{order_id}"

def add_group_capacity_constraints(prob: pulp.LpProblem, order_to_group_assignments: dict, orders_df: pd.DataFrame, roe_df: pd.DataFrame) -> None:
    """Ensure allocations don’t exceed group capacity."""
    for _, row in roe_df.iterrows():
        group_id = row['GroupID']
        group_capacity = row['ProducedEggs']
        prob += pulp.lpSum(order_to_group_assignments[order_id, group_id] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                           for order_id in orders_df['OrderID']) <= group_capacity, f"Group_Capacity_{group_id}"

def add_product_match_constraints(prob: pulp.LpProblem, order_to_group_assignments: dict, orders_df: pd.DataFrame, roe_df: pd.DataFrame) -> None:
    """Ensure product match between order and group."""
    for order_id in orders_df['OrderID']:
        order_product = orders_df[orders_df['OrderID'] == order_id]['Product'].iloc[0]
        for _, row in roe_df.iterrows():
            group_id = row['GroupID']
            group_product = row['Product']
            if order_product != group_product:
                prob += order_to_group_assignments[order_id, group_id] == 0, f"Product_Match_{order_id}_{group_id}"

def add_date_constraints(prob: pulp.LpProblem, order_to_group_assignments: dict, orders_df: pd.DataFrame, roe_df: pd.DataFrame) -> None:
    """Ensure delivery date is within the group's sale window."""
    for order_id in orders_df['OrderID']:
        delivery_date = pd.to_datetime(orders_df[orders_df['OrderID'] == order_id]['DeliveryDate'].iloc[0])
        for _, row in roe_df.iterrows():
            group_id = row['GroupID']
            start_sale_date = pd.to_datetime(row['StartSaleDate'])
            expire_date = pd.to_datetime(row['ExpireDate'])
            if delivery_date < start_sale_date or delivery_date > expire_date:
                prob += order_to_group_assignments[order_id, group_id] == 0, f"Date_Constraint_{order_id}_{group_id}"

def binary_allocation(orders_df: pd.DataFrame, roe_df: pd.DataFrame, constraints_value: List[str], order_priority: str) -> Tuple[List[dict], List[int]]:
    """
    Perform binary allocation using PuLP optimizer with unique GroupID.

    Args:
        orders_df (pd.DataFrame): Orders data with columns like OrderID, OrderedEggs, etc.
        roe_df (pd.DataFrame): Roe data with columns like GroupID, ProducedEggs, etc.
        constraints_value (List[str]): List of active constraints (e.g., ['product_match', 'date_constraints'])
        order_priority (str): Priority type, either 'chronological' or 'maximize'

    Returns:
        Tuple[List[dict], List[int]]: (allocation_results, unfulfilled_orders)
    """
    # Set up the LP problem to maximize allocation
    prob = pulp.LpProblem("Roe_Allocation_Binary", pulp.LpMaximize)

    # Define binary variables: 1 if order is assigned to group, 0 otherwise
    order_to_group_assignments = pulp.LpVariable.dicts(
        "Allocate",
        [(order_id, group_id) for order_id in orders_df['OrderID'] for group_id in roe_df['GroupID']],
        cat='Binary'
    )

    # Set objective based on priority
    if order_priority == 'chronological':
        # Prioritize earlier orders by assigning higher weights
        sorted_orders = orders_df.sort_values('DeliveryDate')
        weights = {order_id: len(sorted_orders) - i for i, order_id in enumerate(sorted_orders['OrderID'])}
        prob += pulp.lpSum(order_to_group_assignments[order_id, group_id] * weights[order_id]
                           for order_id in orders_df['OrderID']
                           for group_id in roe_df['GroupID']), "Maximize_Chronological_Orders"
    else:  # 'maximize'
        # Maximize total eggs allocated
        prob += pulp.lpSum(order_to_group_assignments[order_id, group_id] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                           for order_id in orders_df['OrderID']
                           for group_id in roe_df['GroupID']), "Maximize_Total_Eggs"

    # Add constraints
    add_one_group_per_order_constraints(prob, order_to_group_assignments, orders_df, roe_df)
    add_group_capacity_constraints(prob, order_to_group_assignments, orders_df, roe_df)
    if 'product_match' in constraints_value:
        add_product_match_constraints(prob, order_to_group_assignments, orders_df, roe_df)
    if 'date_constraints' in constraints_value:
        add_date_constraints(prob, order_to_group_assignments, orders_df, roe_df)
    if 'nst_priority' in constraints_value:
        # TODO: Implement NST priority logic when required
        pass

    # Solve the problem with a time limit
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    prob.solve(solver)

    allocation_results = []
    if prob.status == pulp.LpStatusOptimal:
        for (order_id, group_id), var in order_to_group_assignments.items():
            if var.varValue > 0.5:
                order_row = orders_df[orders_df['OrderID'] == order_id].iloc[0]
                group_row = roe_df[roe_df['GroupID'] == group_id].iloc[0]
                allocation_results.append({
                    "OrderID": order_id,
                    "CustomerID": order_row["CustomerID"],
                    "OrderedEggs": int(order_row["OrderedEggs"]),
                    "AllocatedEggs": int(order_row["OrderedEggs"]),
                    "FulfillmentPct": 100.0,
                    "Product": order_row["Product"],
                    "DeliveryDate": order_row["DeliveryDate"],
                    "BroodstockGroup": group_row["BroodstockGroup"],
                    "Location": group_row["Location"],
                    "GroupID": group_id
                })
        allocated_order_ids = {res['OrderID'] for res in allocation_results}
        unfulfilled_orders = [order_id for order_id in orders_df['OrderID'] if order_id not in allocated_order_ids]
    else:
        unfulfilled_orders = orders_df['OrderID'].tolist()

    return allocation_results, unfulfilled_orders

def partial_allocation(orders_df: pd.DataFrame, roe_df: pd.DataFrame, constraints_value: List[str], order_priority: str) -> Tuple[List[dict], List[int]]:
    """
    Placeholder for partial allocation functionality.

    Args:
        orders_df (pd.DataFrame): Orders data with columns like OrderID, OrderedEggs, etc.
        roe_df (pd.DataFrame): Roe data with columns like GroupID, ProducedEggs, etc.
        constraints_value (List[str]): List of active constraints (e.g., ['product_match', 'date_constraints'])
        order_priority (str): Priority type, either 'chronological' or 'maximize'

    Returns:
        Tuple[List[dict], List[int]]: (allocation_results, unfulfilled_orders)
    """
    # TODO: Implement partial allocation in future iterations
    return [], orders_df['OrderID'].tolist()  # Stub implementation
