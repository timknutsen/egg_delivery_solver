import pandas as pd

# Fix fish groups - add MinTemp_prod and MaxTemp_prod columns
print("Fixing fish_groups_example_updated.csv...")
df_fish = pd.read_csv('new_version_for_meeting/example_data/fish_groups_example_updated.csv')

# Insert the new columns after 'Organic' (position 7)
df_fish.insert(7, 'MinTemp_prod', 300)
df_fish.insert(8, 'MaxTemp_prod', 500)

# Save back
df_fish.to_csv('new_version_for_meeting/example_data/fish_groups_example_updated.csv', index=False)
print(f"✅ Added MinTemp_prod and MaxTemp_prod columns to fish groups ({len(df_fish)} rows)")

# Fix orders - rename MinTemp/MaxTemp to MinTemp_customer/MaxTemp_customer
print("\nFixing orders_example_updated.csv...")
df_orders = pd.read_csv('new_version_for_meeting/example_data/orders_example_updated.csv')

# Rename columns
df_orders.rename(columns={
    'MinTemp': 'MinTemp_customer', 
    'MaxTemp': 'MaxTemp_customer'
}, inplace=True)

# Save back
df_orders.to_csv('new_version_for_meeting/example_data/orders_example_updated.csv', index=False)
print(f"✅ Renamed MinTemp/MaxTemp to MinTemp_customer/MaxTemp_customer in orders ({len(df_orders)} rows)")

print("\n🎉 All files updated successfully!")
print("\nNew column structure:")
print(f"Fish groups columns: {list(df_fish.columns)}")
print(f"Orders columns: {list(df_orders.columns)}")
