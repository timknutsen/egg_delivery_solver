import pandas as pd

# Sample data
salmon_producers = ["Mowi ASA", "Lerøy Seafood Group", "SalMar ASA", "Cermaq Group ASA", "Bakkafrost"]
product_types = ["Shield", "Gain (Premium)"]

orders_data = pd.DataFrame({
    "OrderID": [1, 2, 3, 5, 6],
    "CustomerID": ["Mowi ASA", "SalMar ASA", "Bakkafrost", "Mowi ASA", "SalMar ASA"],
    "OrderedEggs": [50000, 40000, 50000, 60000, 70000],
    "Product": ["Shield", "Shield", "Shield", "Gain (Premium)", "Shield"],
    "DeliveryDate": ["2025-03-10", "2025-03-15", "2025-03-20", "2025-03-22", "2025-03-25"]
})

roe_data = pd.DataFrame({
    "BroodstockGroup": ["Early (mid-September)", "Natural (November)", "Natural (November)", "Late (December-January)", "Late (December-January)"],
    "ProducedEggs": [100000, 100000, 50000, 60000, 42832],
    "Location": ["Steigen", "Hemne", "Erfjord", "Steigen", "Tingvoll"],
    "Product": ["Shield", "Gain (Premium)", "Shield", "Gain (Premium)", "Gain (Premium)"],
    "StartSaleDate": ["2025-02-15", "2025-02-15", "2025-03-01", "2025-02-20", "2025-12-20"],
    "ExpireDate": ["2025-04-15", "2025-04-20", "2025-04-30", "2025-04-25", "2026-02-08"]
})
