import pandas as pd

print("Quick fix: Adjusting MaxTemp_prod to accommodate customer needs...\n")

# Load fish groups
df_fish = pd.read_csv('new_version_for_meeting/example_data/fish_groups_example_updated.csv')

# Current: MaxTemp_prod = 448 (too low)
# Customers need: up to 640 DD
# Solution: Set MaxTemp_prod = 700 (gives headroom)

df_fish['MinTemp_prod'] = 200  # Keep current minimum
df_fish['MaxTemp_prod'] = 700  # Increase to accommodate all customer needs

df_fish.to_csv('new_version_for_meeting/example_data/fish_groups_example_updated.csv', index=False)

print(f"✅ Updated fish groups:")
print(f"   MinTemp_prod: 200 DD (eggs become sellable)")
print(f"   MaxTemp_prod: 700 DD (safe upper limit)")

# Verify customer ranges are within production limits
df_orders = pd.read_csv('new_version_for_meeting/example_data/orders_example_updated.csv')
print(f"\n✅ Customer requirements:")
print(f"   MinTemp_customer: {df_orders['MinTemp_customer'].min():.0f} - {df_orders['MinTemp_customer'].max():.0f} DD")
print(f"   MaxTemp_customer: {df_orders['MaxTemp_customer'].min():.0f} - {df_orders['MaxTemp_customer'].max():.0f} DD")

print(f"\n✅ Validation:")
max_customer = df_orders['MaxTemp_customer'].max()
max_production = 700
if max_customer <= max_production:
    print(f"   ✓ All customer requirements ({max_customer} DD) fit within production limits ({max_production} DD)")
else:
    print(f"   ✗ ERROR: Some customers need {max_customer} DD but production only supports {max_production} DD")

print("\n🚀 Run the solver again - should work now!")
