import pandas as pd

print("=== DIAGNOSTIC: Current Degree-Days Values ===\n")

# Load fish groups
df_fish = pd.read_csv('new_version_for_meeting/example_data/fish_groups_example_updated.csv')
print("Fish Groups degree-days:")
print(df_fish[['Site_Broodst_Season', 'MinTemp_prod', 'MaxTemp_prod']].head(10))
print(f"\nFish MinTemp_prod range: {df_fish['MinTemp_prod'].min()} - {df_fish['MinTemp_prod'].max()}")
print(f"Fish MaxTemp_prod range: {df_fish['MaxTemp_prod'].min()} - {df_fish['MaxTemp_prod'].max()}")

# Load orders
df_orders = pd.read_csv('new_version_for_meeting/example_data/orders_example_updated.csv')
print("\n\nOrders degree-days:")
print(df_orders[['OrderNr', 'DeliveryDate', 'MinTemp_customer', 'MaxTemp_customer']].head(10))
print(f"\nOrders MinTemp_customer range: {df_orders['MinTemp_customer'].min()} - {df_orders['MinTemp_customer'].max()}")
print(f"Orders MaxTemp_customer range: {df_orders['MaxTemp_customer'].min()} - {df_orders['MaxTemp_customer'].max()}")

# Test one specific example
print("\n\n=== TEST CASE ===")
print("Order 7003: DeliveryDate=2024-09-20, MinTemp_customer=400, MaxTemp_customer=560")
print("\nFish groups available around that date:")
df_fish['StrippingStartDate'] = pd.to_datetime(df_fish['StrippingStartDate'])
df_fish['StrippingStopDate'] = pd.to_datetime(df_fish['StrippingStopDate'])
delivery = pd.to_datetime('2024-09-20')

for idx, row in df_fish.iterrows():
    days_from_start = (delivery - row['StrippingStartDate']).days
    days_from_stop = (delivery - row['StrippingStopDate']).days
    dd_start = days_from_start * 8
    dd_stop = days_from_stop * 8
    
    if days_from_start > 0 and days_from_stop < 100:  # Reasonable window
        print(f"\n{row['Site_Broodst_Season']}:")
        print(f"  Stripping: {row['StrippingStartDate'].date()} to {row['StrippingStopDate'].date()}")
        print(f"  Prod limits: {row['MinTemp_prod']} - {row['MaxTemp_prod']} DD")
        print(f"  DD at delivery: {dd_start} (from start) to {dd_stop} (from stop)")
        print(f"  Order needs: 400-560 DD")
