import pulp
import pandas as pd

def binary_allocation(orders_df, roe_df, constraints_value, order_priority):
    """
    Perform binary allocation using PuLP optimizer.
    
    Args:
        orders_df (DataFrame): Orders data
        roe_df (DataFrame): Roe data
        constraints_value (list): Active constraints
        order_priority (str): 'chronological' or 'maximize'
        
    Returns:
        tuple: (allocation_results, unfulfilled_orders)
    """
    prob = pulp.LpProblem("Roe_Allocation_Binary", pulp.LpMaximize)
    allocation_vars = pulp.LpVariable.dicts(
        "Allocate",
        [(order_id, group) for order_id in orders_df['OrderID'] for group in roe_df['BroodstockGroup']],
        cat='Binary'
    )

    # Objective function
    if order_priority == 'chronological':
        sorted_orders = orders_df.sort_values('DeliveryDate')
        weights = {order_id: len(sorted_orders) - i for i, order_id in enumerate(sorted_orders['OrderID'])}
        prob += pulp.lpSum(allocation_vars[order_id, group] * weights[order_id] 
                           for order_id in orders_df['OrderID'] 
                           for group in roe_df['BroodstockGroup']), "Maximize Chronological Orders"
    else:  # 'maximize'
        prob += pulp.lpSum(allocation_vars[order_id, group] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                           for order_id in orders_df['OrderID'] 
                           for group in roe_df['BroodstockGroup']), "Maximize Total Eggs"

    # Constraints
    for order_id in orders_df['OrderID']:
        prob += pulp.lpSum(allocation_vars[order_id, group] for group in roe_df['BroodstockGroup']) <= 1, f"One_Group_Per_Order_{order_id}"

    for group in roe_df['BroodstockGroup']:
        group_capacity = roe_df[roe_df['BroodstockGroup'] == group]['ProducedEggs'].iloc[0]
        prob += pulp.lpSum(allocation_vars[order_id, group] * orders_df[orders_df['OrderID'] == order_id]['OrderedEggs'].iloc[0]
                           for order_id in orders_df['OrderID']) <= group_capacity, f"Group_Capacity_{group}"

    if 'product_match' in constraints_value:
        for order_id in orders_df['OrderID']:
            order_product = orders_df[orders_df['OrderID'] == order_id]['Product'].iloc[0]
            for group in roe_df['BroodstockGroup']:
                group_product = roe_df[roe_df['BroodstockGroup'] == group]['Product'].iloc[0]
                if order_product != group_product:
                    prob += allocation_vars[order_id, group] == 0, f"Product_Match_{order_id}_{group}"

    if 'date_constraints' in constraints_value:
        for order_id in orders_df['OrderID']:
            delivery_date = pd.to_datetime(orders_df[orders_df['OrderID'] == order_id]['DeliveryDate'].iloc[0])
            for group in roe_df['BroodstockGroup']:
                start_sale_date = pd.to_datetime(roe_df[roe_df['BroodstockGroup'] == group]['StartSaleDate'].iloc[0])
                expire_date = pd.to_datetime(roe_df[roe_df['BroodstockGroup'] == group]['ExpireDate'].iloc[0])
                if delivery_date < start_sale_date or delivery_date > expire_date:
                    prob += allocation_vars[order_id, group] == 0, f"Date_Constraint_{order_id}_{group}"

    # NST Priority (soft constraint - could be improved)
    if 'nst_priority' in constraints_value:
        nst_customers = [cust for cust in orders_df['CustomerID'].unique() if "North Sea Traders" in cust]
        nst_orders_ids = orders_df[orders_df['CustomerID'].isin(nst_customers)]['OrderID'].tolist()
        steigen_groups = roe_df[roe_df['Location'] == 'Steigen']['BroodstockGroup'].tolist()
        # Placeholder for NST priority logic

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    prob.solve(solver)

    allocation_results = []
    if prob.status == pulp.LpStatusOptimal:
        for order_id, group in allocation_vars:
            if allocation_vars[(order_id, group)].varValue > 0.5:
                order_row = orders_df[orders_df['OrderID'] == order_id].iloc[0]
                group_row = roe_df[roe_df['BroodstockGroup'] == group].iloc[0]
                allocation_results.append({
                    "OrderID": order_id,
                    "CustomerID": order_row["CustomerID"],
                    "OrderedEggs": int(order_row["OrderedEggs"]),
                    "AllocatedEggs": int(order_row["OrderedEggs"]),
                    "FulfillmentPct": 100.0,
                    "Product": order_row["Product"],
                    "DeliveryDate": order_row["DeliveryDate"],
                    "BroodstockGroup": group,
                    "Location": group_row["Location"]
                })
        allocated_order_ids = {res['OrderID'] for res in allocation_results}
        unfulfilled_orders = [order_id for order_id in orders_df['OrderID'] if order_id not in allocated_order_ids]
    else:
        unfulfilled_orders = orders_df['OrderID'].tolist()

    return allocation_results, unfulfilled_orders

def partial_allocation(orders_df, roe_df, constraints_value, order_priority):
    """
    Perform partial allocation using a greedy algorithm.
    
    Args:
        orders_df (DataFrame): Orders data
        roe_df (DataFrame): Roe data
        constraints_value (list): Active constraints
        order_priority (str): 'chronological' or 'maximize'
        
    Returns:
        tuple: (allocation_results, unfulfilled_orders)
    """
    remaining_roe = roe_df.copy()
    remaining_roe['RemainingEggs'] = remaining_roe['ProducedEggs']

    if order_priority == 'chronological':
        sorted_orders = orders_df.sort_values('DeliveryDate')
    else:  # 'maximize'
        sorted_orders = orders_df.sort_values('OrderedEggs')

    allocation_results = []
    unfulfilled_orders = []

    for _, order in sorted_orders.iterrows():
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

    return allocation_results, unfulfilled_orders
