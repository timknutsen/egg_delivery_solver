"""
KONFIGURASJON OG DATA
=====================
Endre denne filen for å:
- Justere parametere
- Oppdatere eksempeldata
- Legge til nye fiskegrupper/ordrer
"""

import pandas as pd

# ==========================================
# PARAMETERE
# ==========================================
WATER_TEMP_C = 8.0
DD_TO_MATURE = 300
PREFERENCE_BONUS = -1000

# ==========================================
# FISKEGRUPPER (PRODUKSJON)
# ==========================================
FISH_GROUPS = pd.DataFrame([
    # Hemne - Hovedanlegg
    {
        'Site': 'Hemne', 
        'Site_Broodst_Season': 'Hemne_Normal_24/25', 
        'StrippingStartDate': '2024-09-01', 
        'StrippingStopDate': '2024-09-28', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 8000000.0, 
        'Shield-eggs': 2000000.0, 
        'Organic': False
    },
    {
        'Site': 'Hemne', 
        'Site_Broodst_Season': 'Hemne_Early_24/25', 
        'StrippingStartDate': '2024-08-15', 
        'StrippingStopDate': '2024-08-31', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 3000000.0, 
        'Shield-eggs': 1000000.0, 
        'Organic': False
    },
    {
        'Site': 'Hemne', 
        'Site_Broodst_Season': 'Hemne_Late_24/25', 
        'StrippingStartDate': '2024-10-01', 
        'StrippingStopDate': '2024-10-20', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 4000000.0, 
        'Shield-eggs': 1500000.0, 
        'Organic': False
    },
    # Vestseøra - Organic
    {
        'Site': 'Vestseøra', 
        'Site_Broodst_Season': 'Vestseøra_Organic_24/25', 
        'StrippingStartDate': '2024-08-25', 
        'StrippingStopDate': '2024-09-15', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 3000000.0, 
        'Shield-eggs': 2000000.0, 
        'Organic': True
    },
    # Hønsvikgulen - Lerøy-dedikert
    {
        'Site': 'Hønsvikgulen', 
        'Site_Broodst_Season': 'Hønsvikgulen_Normal_24/25', 
        'StrippingStartDate': '2024-09-05', 
        'StrippingStopDate': '2024-09-25', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 6000000.0, 
        'Shield-eggs': 3000000.0, 
        'Organic': False
    },
    {
        'Site': 'Hønsvikgulen', 
        'Site_Broodst_Season': 'Hønsvikgulen_Late_24/25', 
        'StrippingStartDate': '2024-10-05', 
        'StrippingStopDate': '2024-10-25', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 4000000.0, 
        'Shield-eggs': 2000000.0, 
        'Organic': False
    },
    # Ny: Hemne Organic
    {
        'Site': 'Hemne', 
        'Site_Broodst_Season': 'Hemne_Organic_24/25', 
        'StrippingStartDate': '2024-09-10', 
        'StrippingStopDate': '2024-09-30', 
        'MinTemp_C': 1, 
        'MaxTemp_C': 8, 
        'Gain-eggs': 2000000.0, 
        'Shield-eggs': 1000000.0, 
        'Organic': True
    },
])

# ==========================================
# ORDRER (KUNDER)
# ==========================================
ORDERS = pd.DataFrame([
    # Lerøy - Store ordrer, foretrekker Hønsvikgulen
    {
        'OrderNr': 1001, 
        'Customer': 'Lerøy Midt',
        'DeliveryDate': '2024-11-10', 
        'Product': 'Gain', 
        'Volume': 1500000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hønsvikgulen',
        'PreferredGroup': None,
    },
    {
        'OrderNr': 1002, 
        'Customer': 'Lerøy Midt',
        'DeliveryDate': '2024-11-25', 
        'Product': 'Gain', 
        'Volume': 2000000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': 'Hønsvikgulen',
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    {
        'OrderNr': 1003, 
        'Customer': 'Lerøy Sjøtroll',
        'DeliveryDate': '2024-12-05', 
        'Product': 'Shield', 
        'Volume': 800000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hønsvikgulen',
        'PreferredGroup': None,
    },
    # Mowi - Flere lokasjoner
    {
        'OrderNr': 2001, 
        'Customer': 'Mowi Region Midt',
        'DeliveryDate': '2024-11-15', 
        'Product': 'Gain', 
        'Volume': 1200000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hemne',
        'PreferredGroup': None,
    },
    {
        'OrderNr': 2002, 
        'Customer': 'Mowi Region Midt',
        'DeliveryDate': '2024-12-01', 
        'Product': 'Gain', 
        'Volume': 1800000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 7,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hemne',
        'PreferredGroup': None,
    },
    {
        'OrderNr': 2003, 
        'Customer': 'Mowi Region Nord',
        'DeliveryDate': '2024-12-15', 
        'Product': 'Shield', 
        'Volume': 600000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    # SalMar - Organic-krav
    {
        'OrderNr': 3001, 
        'Customer': 'SalMar Organic',
        'DeliveryDate': '2024-11-20', 
        'Product': 'Gain', 
        'Volume': 900000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': True,  # KREVER ORGANIC
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    {
        'OrderNr': 3002, 
        'Customer': 'SalMar Farming',
        'DeliveryDate': '2024-11-28', 
        'Product': 'Gain', 
        'Volume': 1100000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': 'Hemne_Normal_24/25',
    },
    # Grieg Seafood
    {
        'OrderNr': 4001, 
        'Customer': 'Grieg Seafood',
        'DeliveryDate': '2024-10-25', 
        'Product': 'Gain', 
        'Volume': 700000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    {
        'OrderNr': 4002, 
        'Customer': 'Grieg Seafood',
        'DeliveryDate': '2024-11-05', 
        'Product': 'Shield', 
        'Volume': 500000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 5,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    # NRS (Norway Royal Salmon) - Organic
    {
        'OrderNr': 5001, 
        'Customer': 'NRS Organic',
        'DeliveryDate': '2024-12-10', 
        'Product': 'Gain', 
        'Volume': 800000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': True,  # KREVER ORGANIC
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hemne',  # Foretrekker Hemne Organic
        'PreferredGroup': None,
    },
    {
        'OrderNr': 5002, 
        'Customer': 'NRS',
        'DeliveryDate': '2024-12-20', 
        'Product': 'Gain', 
        'Volume': 1000000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': 'Hemne_Late_24/25',
    },
    # Små kunder - Tidlig levering
    {
        'OrderNr': 6001, 
        'Customer': 'Nordlaks',
        'DeliveryDate': '2024-10-20', 
        'Product': 'Gain', 
        'Volume': 400000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    {
        'OrderNr': 6002, 
        'Customer': 'Nova Sea Organic',
        'DeliveryDate': '2024-11-01', 
        'Product': 'Shield', 
        'Volume': 350000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': True,  # KREVER ORGANIC
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Vestseøra',
        'PreferredGroup': None,
    },
    # Sen levering
    {
        'OrderNr': 7001, 
        'Customer': 'Cermaq',
        'DeliveryDate': '2025-01-10', 
        'Product': 'Gain', 
        'Volume': 1600000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 7,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': None,
        'PreferredGroup': None,
    },
    {
        'OrderNr': 7002, 
        'Customer': 'Cermaq',
        'DeliveryDate': '2025-01-20', 
        'Product': 'Shield', 
        'Volume': 700000.0, 
        'MinTemp_C': 2, 
        'MaxTemp_C': 6,
        'RequireOrganic': False,
        'LockedSite': None,
        'LockedGroup': None,
        'PreferredSite': 'Hønsvikgulen',
        'PreferredGroup': None,
    },
])
