import pulp
import pandas as pd

def binary_allocation(orders_df, roe_df, constraints_value, order_priority):
    """
    Perform binary allocation using PuLP optimizer with unique GroupID.
    
    Args:
        orders_df (DataFrame): Orders data
        roe_df (DataFrame): Roe data with 'GroupID'
        constraints_value (list): Active constraints (e.g., ['product_match', 'date_constraints'])
        order_priority (str): 'chronological' or 'maximize'
        
    Returns:
        tuple: (allocation_results, unfulfilled_orders)
    """
    prob = pulp.LpProblem("Roe_Allocation_Binary", pulp.LpMaximize)
    
    # Define allocation variables with (order_id, group_id) tuples
    allocation_vars = pulp.LpVariable.dicts(
        "Allocate",
        [(order_id, group_id) for order_id in orders_df['OrderID'] for group_id in roe_df['GroupID']],
        cat='Binary'
    )

    # Objective function: Chronological priority (as per screenshot)
    if order_priority == 'chronological':
        sorted_orders = orders_df.sort_values('DeliveryDate')
        weights = {order_id: len(sorted_orders) - i for i, order_id in enumerate(sorted_orders['OrderID'])}
        prob += pulp.lpSum(allocation_vars[order_id, group_id] * weights[order_id] 
                           for order_id in orders_df['OrderID'] 
                           for group_id in roe_df['GroupID']), "Maximize_Chronological_Orders"
    else:  # 'maximize'
        prob += pulp.lpSum(allocation_vars[order_id, group_id] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                           for order_id in orders_df['OrderID'] 
                           for group_id in roe_df['GroupID']), "Maximize_Total_Eggs"

    # Constraint: Each order assigned to at most one group
    for order_id in orders_df['OrderID']:
        prob += pulp.lpSum(allocation_vars[order_id, group_id] for group_id in roe_df['GroupID']) <= 1, f"One_Group_Per_Order_{order_id}"

    # Constraint: Group capacity
    for _, row in roe_df.iterrows():
        group_id = row['GroupID']
        group_capacity = row['ProducedEggs']
        prob += pulp.lpSum(allocation_vars[order_id, group_id] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                           for order_id in orders_df['OrderID']) <= group_capacity, f"Group_Capacity_{group_id}"

    # Constraint: Product Match (enabled in screenshot)
    if 'product_match' in constraints_value:
        for order_id in orders_df['OrderID']:
            order_product = orders_df[orders_df['OrderID'] == order_id]['Product'].iloc[0]
            for _, row in roe_df.iterrows():
                group_id = row['GroupID']
                group_product = row['Product']
                if order_product != group_product:
                    prob += allocation_vars[order_id, group_id] == 0, f"Product_Match_{order_id}_{group_id}"

    # Constraint: Date Constraints (enabled in screenshot)
    if 'date_constraints' in constraints_value:
        for order_id in orders_df['OrderID']:
            delivery_date = pd.to_datetime(orders_df[orders_df['OrderID'] == order_id]['DeliveryDate'].iloc[0])
            for _, row in roe_df.iterrows():
                group_id = row['GroupID']
                start_sale_date = pd.to_datetime(row['StartSaleDate'])
                expire_date = pd.to_datetime(row['ExpireDate'])
                if delivery_date < start_sale_date or delivery_date > expire_date:
                    prob += allocation_vars[order_id, group_id] == 0, f"Date_Constraint_{order_id}_{group_id}"

    # NST Priority (disabled in screenshot, so no action needed)
    if 'nst_priority' in constraints_value:
        pass  # Placeholder for future implementation

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    prob.solve(solver)

    allocation_results = []
    if prob.status == pulp.LpStatusOptimal:
        for (order_id, group_id), var in allocation_vars.items():
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

def partial_allocation(orders_df, roe_df, constraints_value, order_priority):
    """
    Placeholder for partial allocation (not used per screenshot settings).
    """
    return [], orders_df['OrderID'].tolist()  # Stub implementation
