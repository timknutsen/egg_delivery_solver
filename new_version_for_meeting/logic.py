"""
FORRETNINGSLOGIKK
=================
Denne filen inneholder all logikk for:
- Preprocessing av data
- Generering av batcher (Nå med oppslagstabell for vekst)
- Feasibility-sjekk (inkl. Organic)
- Optimalisering/allokering
- Visualisering
- Eksport av eksempeldata

OPPDATERT: 
- Bruker nå Grading Progress Table for nøyaktig beregning av ModStart/ModStop.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import pulp as pl
from pulp import PULP_CBC_CMD
import io
import math

# Importerer konfigurasjon (brukes fortsatt til visse konverteringer og fallback)
from config import WATER_TEMP_C, DD_TO_MATURE, PREFERENCE_BONUS

# Straff for ordrer som ikke blir tildelt selv om de har muligheter.
NOT_ALLOCATED_PENALTY = 100_000

REQUIRED_FISH_COLUMNS = [
    "Site",
    "Site_Broodst_Season",
    "StrippingStartDate",
    "StrippingStopDate",
    "MinTemp_C",
    "MaxTemp_C",
    "Gain-eggs",
    "Shield-eggs",
    "Organic",
]

REQUIRED_ORDER_COLUMNS = [
    "OrderNr",
    "DeliveryDate",
    "Product",
    "Volume",
    "MinTemp_C",
    "MaxTemp_C",
    "RequireOrganic",
    "LockedSite",
    "LockedGroup",
    "PreferredSite",
    "PreferredGroup",
]

PRECOMPUTED_BATCH_COLUMNS = [
    "Site",
    "Site_Broodst_Season",
    "BatchID",
    "Produksjonvolum",
    "SalesStartWeek",
    "SaleStopWeek",
]

ORDER_DEFAULTS = {
    "Customer": "",
    "Product": "Gain",
    "MinTemp_C": WATER_TEMP_C,
    "MaxTemp_C": WATER_TEMP_C,
    "RequireOrganic": False,
    "LockedSite": "",
    "LockedGroup": "",
    "PreferredSite": "",
    "PreferredGroup": "",
}


def _validate_required_columns(df, required_columns, label):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return (
            f"Mangler kolonner i '{label}': {', '.join(missing)}. "
            f"Forventet kolonner inkluderer: {', '.join(required_columns)}"
        )
    return None


def _has_required_columns(df, required_columns):
    return all(c in df.columns for c in required_columns)


def _normalize_bool_value(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "ja", "y", "x", "✓"}
    return bool(value)


def normalize_orders_input(orders_df):
    """Fyller inn valgfrie ordre-kolonner slik at forenklede filer kan kjøres."""
    orders = orders_df.copy()
    for column, default in ORDER_DEFAULTS.items():
        if column not in orders.columns:
            orders[column] = default
        else:
            orders[column] = orders[column].fillna(default)

    orders["DeliveryDate"] = pd.to_datetime(orders["DeliveryDate"])
    orders["Volume"] = pd.to_numeric(orders["Volume"])
    orders["Product"] = orders["Product"].astype(str).str.strip().replace("", "Gain")
    orders["RequireOrganic"] = orders["RequireOrganic"].apply(_normalize_bool_value)
    return orders


def is_precomputed_batch_input(fish_groups_df):
    """Sjekker om Fiskegrupper-arket allerede inneholder ferdige batchvinduer."""
    return _has_required_columns(fish_groups_df, PRECOMPUTED_BATCH_COLUMNS)


def _iso_week_to_monday(week_value):
    week_int = int(week_value)
    year = week_int // 100
    week = week_int % 100
    return pd.Timestamp.fromisocalendar(year, week, 1)


def _iso_week_to_sunday(week_value):
    return _iso_week_to_monday(week_value) + pd.Timedelta(days=6)


def build_precomputed_batches(fish_groups_df):
    """Konverterer ferdige batchvinduer fra Excel til batch-formatet allokeringen bruker."""
    batches = fish_groups_df.copy()
    if "Organic" not in batches.columns:
        batches["Organic"] = False
    batches["Organic"] = batches["Organic"].apply(_normalize_bool_value)
    batches["MaturationEnd"] = batches["SalesStartWeek"].apply(_iso_week_to_monday)
    batches["ProductionEnd"] = batches["SaleStopWeek"].apply(_iso_week_to_sunday)
    batches["StripDate"] = batches["MaturationEnd"]
    batches["GainCapacity"] = pd.to_numeric(batches["Produksjonvolum"])
    batches["ShieldCapacity"] = pd.to_numeric(batches["Produksjonvolum"])
    batches["Group"] = batches["Site_Broodst_Season"]
    batches["CalcInfo"] = (
        "Precomputed sales weeks "
        + batches["SalesStartWeek"].astype(str)
        + "-"
        + batches["SaleStopWeek"].astype(str)
    )
    return batches[
        [
            "BatchID",
            "Group",
            "Site",
            "StripDate",
            "MaturationEnd",
            "ProductionEnd",
            "GainCapacity",
            "ShieldCapacity",
            "Organic",
            "CalcInfo",
        ]
    ]

# ==========================================
# 0. VEKSTTABELL (GRADING CONFIG)
# ==========================================

# Tabell fra Excel-modellen: 
# Key = Temperatur (grader C), Value = { "80": dager til 80%, "95": dager til 95% }
GRADING_TABLE = {
    1: {"80": 169, "95": 200},
    2: {"80": 136, "95": 162},
    3: {"80": 112, "95": 133},
    4: {"80": 93,  "95": 111},
    5: {"80": 79,  "95": 93},
    6: {"80": 67,  "95": 79},
    7: {"80": 57,  "95": 68},
    8: {"80": 50,  "95": 59},
}

def calculate_grading_days(temp_c, threshold_key):
    """
    Interpolerer antall dager basert på temperatur og terskel (80% eller 95%).
    F.eks: 3.18 grader interpoleres mellom gradering 3 og 4.
    """
    # Vi håndterer temp litt utenfor 1-8 ved å klemme, 
    # men for MaxTemp > 8 gjør vi en ekstrapolering basert på stigningen 7->8.
    
    t = float(temp_c)
    
    # Hvis temperatur er innenfor tabellens kjerneområde (1-8)
    if 1.0 <= t <= 8.0:
        floor_t = int(np.floor(t))
        ceil_t = int(np.ceil(t))
        
        val_floor = GRADING_TABLE[floor_t][threshold_key]
        
        if floor_t == ceil_t:
            return val_floor
            
        val_ceil = GRADING_TABLE[ceil_t][threshold_key]
        fraction = t - floor_t
        
        # Lineær interpolering: Start + (differanse * brøkdel)
        # Siden høyere temp = lavere dager, blir (val_ceil - val_floor) negativt.
        return val_floor + (fraction * (val_ceil - val_floor))

    # Hvis temperatur > 8 (For varmt - går fortere)
    elif t > 8.0:
        # Ekstrapolerer med stigningstallet mellom 7 og 8
        val_7 = GRADING_TABLE[7][threshold_key]
        val_8 = GRADING_TABLE[8][threshold_key]
        slope_per_degree = val_8 - val_7 # Negativt tall
        
        extra_temp = t - 8.0
        # Beregn dager, men ha en minimumssperre på f.eks 10 dager for sikkerhet
        return max(10, val_8 + (extra_temp * slope_per_degree))

    # Hvis temperatur < 1 (For kaldt - går saktere)
    else:
        # Ekstrapolerer "bakover" fra 1 til 2
        val_1 = GRADING_TABLE[1][threshold_key]
        val_2 = GRADING_TABLE[2][threshold_key]
        slope_per_degree = val_1 - val_2 # Positiv forskjell (dager øker mot kaldt)
        
        missing_temp = 1.0 - t
        # Legg til dager
        return val_1 + (missing_temp * slope_per_degree)


def calculate_formula_days(temp_c, threshold_key):
    """
    Beregner dager fra temperatur basert på klekkekalkulator-formelen fra Excel.
    threshold_key styrer andel: "80" -> 0.80, "95" -> 0.95.
    """
    threshold_ratio_map = {"80": 0.80, "95": 0.95}
    if threshold_key not in threshold_ratio_map:
        raise ValueError(f"Ugyldig threshold_key={threshold_key}. Bruk '80' eller '95'.")

    t = float(temp_c)
    if t <= -11.0:
        raise ValueError("Temperatur må være > -11.0 for formelmodellen.")

    threshold_ratio = threshold_ratio_map[threshold_key]
    exponent = (2.6562 * math.log10(t + 11.0)) - 5.1908
    days = threshold_ratio / (10 ** exponent)
    return max(1.0, days)


# ==========================================
# 1. PREPROCESSING
# ==========================================
def preprocess_data(orders_df, groups_df):
    """
    Forbereder data. 
    Beholder konvertering til døgngrader for kundesiden (orders),
    men selve batch-produksjonen styres nå av generate_weekly_batches med tabelloppslag.
    """
    g_df = groups_df.copy()
    # Disse kolonnene brukes kanskje ikke lenger kritisk for batch-tid, 
    # men beholdes for referanse
    g_df["MinTemp_prod"] = g_df["MinTemp_C"] * DD_TO_MATURE
    g_df["MaxTemp_prod"] = g_df["MaxTemp_C"] * DD_TO_MATURE

    o_df = orders_df.copy()
    o_df["MinTemp_customer"] = o_df["MinTemp_C"] * DD_TO_MATURE
    o_df["MaxTemp_customer"] = o_df["MaxTemp_C"] * DD_TO_MATURE

    return o_df, g_df


# ==========================================
# 2. BATCH-GENERERING
# ==========================================
def generate_weekly_batches(fish_groups_df, growth_model="table"):
    """
    Deler fiskegrupper inn i ukentlige batcher.
    growth_model:
    - "table": bruker GRADING_TABLE interpolasjon.
    - "formula": bruker klekkekalkulator-formelen.
    """
    all_batches = []

    for _, group in fish_groups_df.iterrows():
        strip_start = pd.to_datetime(group["StrippingStartDate"])
        strip_stop = pd.to_datetime(group["StrippingStopDate"])

        # Hent temperaturer fra input
        min_temp = group["MinTemp_C"]
        max_temp = group["MaxTemp_C"]

        # --- BEREGNING AV SALGSVINDU (DAGER) ---
        # 1. Start av vindu (Early/ModStart): raskeste vekst (MaxTemp), 80%.
        # 2. Slutt av vindu (Late/ModStop): sakteste vekst (MinTemp), 95%.
        if growth_model == "table":
            early_days = calculate_grading_days(max_temp, "80")
            late_days = calculate_grading_days(min_temp, "95")
        elif growth_model == "formula":
            early_days = calculate_formula_days(max_temp, "80")
            late_days = calculate_formula_days(min_temp, "95")
        else:
            raise ValueError(
                f"Ugyldig growth_model={growth_model}. Bruk 'table' eller 'formula'."
            )

        # Generer uker
        weeks = pd.date_range(strip_start, strip_stop, freq="W-MON")
        if len(weeks) == 0:
            weeks = pd.DatetimeIndex([strip_start])

        n = len(weeks)
        indices = np.arange(n)
        # Vekter volumet med en "bell curve" over strykeperioden
        weights = np.exp(-0.5 * ((indices - (n - 1) / 2) / max(n / 4, 1)) ** 2)
        weights = weights / weights.sum()

        for i, strip_date in enumerate(weeks):
            
            # Beregn datoer
            maturation_end = strip_date + timedelta(days=early_days)
            production_end = strip_date + timedelta(days=late_days)

            all_batches.append(
                {
                    "BatchID": f"{group['Site_Broodst_Season']}_Uke_{i+1}",
                    "Group": group["Site_Broodst_Season"],
                    "Site": group["Site"],
                    "StripDate": strip_date,
                    "MaturationEnd": maturation_end,   # Vindu start
                    "ProductionEnd": production_end,   # Vindu slutt
                    "GainCapacity": float(group["Gain-eggs"]) * weights[i],
                    "ShieldCapacity": float(group["Shield-eggs"]) * weights[i],
                    "Organic": group["Organic"],
                    # Debug info (kan fjernes senere)
                    "CalcInfo": (
                        f"Model:{growth_model} | "
                        f"MaxT:{max_temp}->{int(early_days)}d | MinT:{min_temp}->{int(late_days)}d"
                    ),
                }
            )

    return pd.DataFrame(all_batches)


# ==========================================
# 3. FEASIBILITY-SJEKK
# ==========================================
def _week_start(ts):
    """Returnerer mandag i uken til datoen."""
    ts = pd.to_datetime(ts)
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def _is_delivery_in_window(delivery_date, maturation_end, production_end, window_mode):
    """Sjekker om leveringsdato er innenfor batch-vindu (dag- eller uke-nivå)."""
    delivery_date = pd.to_datetime(delivery_date)
    maturation_end = pd.to_datetime(maturation_end)
    production_end = pd.to_datetime(production_end)

    if window_mode == "day":
        return maturation_end <= delivery_date <= production_end

    if window_mode == "week":
        delivery_week = _week_start(delivery_date)
        maturation_week = _week_start(maturation_end)
        production_week = _week_start(production_end)
        return maturation_week <= delivery_week <= production_week

    raise ValueError(f"Ugyldig window_mode={window_mode}. Bruk 'day' eller 'week'.")


def _window_mask(delivery_date, batches_df, window_mode):
    delivery_date = pd.to_datetime(delivery_date)
    if window_mode == "day":
        return (batches_df["MaturationEnd"] <= delivery_date) & (
            delivery_date <= batches_df["ProductionEnd"]
        )

    if window_mode == "week":
        delivery_week = _week_start(delivery_date)
        maturation_weeks = batches_df["_MaturationWeek"]
        production_weeks = batches_df["_ProductionWeek"]
        return (maturation_weeks <= delivery_week) & (delivery_week <= production_weeks)

    raise ValueError(f"Ugyldig window_mode={window_mode}. Bruk 'day' eller 'week'.")


def build_feasibility_set(orders_df, batches_df, window_mode="week"):
    """
    Finner alle gyldige (ordre, batch) kombinasjoner.
    Sjekker om kundens leveringsdato er innenfor det biologiske vinduet.
    window_mode='week' betyr at sjekk gjøres på ukenivå (mandag-søndag).
    """
    feasible = []
    batches_df = batches_df.copy()
    batches_df["MaturationEnd"] = pd.to_datetime(batches_df["MaturationEnd"])
    batches_df["ProductionEnd"] = pd.to_datetime(batches_df["ProductionEnd"])
    batches_df["StripDate"] = pd.to_datetime(batches_df["StripDate"])
    if window_mode == "week":
        batches_df["_MaturationWeek"] = batches_df["MaturationEnd"].apply(_week_start)
        batches_df["_ProductionWeek"] = batches_df["ProductionEnd"].apply(_week_start)

    for _, order in orders_df.iterrows():
        delivery_date = pd.to_datetime(order["DeliveryDate"])
        candidates = batches_df[_window_mask(delivery_date, batches_df, window_mode)]

        require_organic = order.get("RequireOrganic", False)
        if require_organic:
            candidates = candidates[candidates["Organic"]]

        locked_site = order.get("LockedSite")
        if pd.notna(locked_site) and str(locked_site).strip():
            candidates = candidates[candidates["Site"] == locked_site]

        locked_group = order.get("LockedGroup")
        if pd.notna(locked_group) and str(locked_group).strip():
            candidates = candidates[candidates["Group"] == locked_group]

        capacity_col = _capacity_column_for_product(order.get("Product", "Gain"))
        candidates = candidates[candidates[capacity_col] >= float(order["Volume"])]

        pref_site = order.get("PreferredSite")
        pref_group = order.get("PreferredGroup")

        for _, batch in candidates.iterrows():
            days_since_strip = (delivery_date - batch["StripDate"]).days
            dd_at_delivery = days_since_strip * WATER_TEMP_C
            bonus = 0
            pref_matched = []

            if pd.notna(pref_site) and str(pref_site).strip() and batch["Site"] == pref_site:
                bonus += PREFERENCE_BONUS
                pref_matched.append(f"Site={pref_site}")

            if pd.notna(pref_group) and str(pref_group).strip() and batch["Group"] == pref_group:
                bonus += PREFERENCE_BONUS
                pref_matched.append(f"Group={pref_group}")

            feasible.append(
                {
                    "OrderNr": order["OrderNr"],
                    "Customer": order.get("Customer", ""),
                    "BatchID": batch["BatchID"],
                    "Group": batch["Group"],
                    "Site": batch["Site"],
                    "Organic": batch["Organic"],
                    "StripDate": batch["StripDate"],
                    "DeliveryDate": delivery_date,
                    "DegreeDays": round(dd_at_delivery, 2),
                    "Volume": order["Volume"],
                    "Product": order["Product"],
                    "RequireOrganic": require_organic,
                    "PreferenceBonus": bonus,
                    "PreferenceMatched": ", ".join(pref_matched),
                    # Kapasiteter for debugging / constraints
                    "GainCapacity": batch["GainCapacity"],
                    "ShieldCapacity": batch["ShieldCapacity"],
                }
            )

    return pd.DataFrame(feasible)


# ==========================================
# 3b. MULIGE GRUPPER PER ORDRE
# ==========================================
def get_possible_groups_per_order(orders_df, feasible_df):
    """
    Returnerer en oversikt over mulige grupper/batcher per ordre.
    Nyttig for manuell vurdering.
    """
    summary = []

    for _, order in orders_df.iterrows():
        order_nr = order["OrderNr"]
        
        if feasible_df.empty:
             order_feasible = pd.DataFrame()
        else:
             order_feasible = feasible_df[feasible_df["OrderNr"] == order_nr]

        if order_feasible.empty:
            summary.append(
                {
                    "OrderNr": order_nr,
                    "Customer": order.get("Customer", ""),
                    "DeliveryDate": order["DeliveryDate"],
                    "Volume": order["Volume"],
                    "RequireOrganic": "✓" if order.get("RequireOrganic", False) else "",
                    "AntallMuligeBatcher": 0,
                    "MuligeGrupper": "INGEN",
                    "MuligeSites": "INGEN",
                    "MinDegreeDays": "-",
                    "MaxDegreeDays": "-",
                }
            )
        else:
            unique_groups = order_feasible["Group"].unique()
            unique_sites = order_feasible["Site"].unique()

            summary.append(
                {
                    "OrderNr": order_nr,
                    "Customer": order.get("Customer", ""),
                    "DeliveryDate": order["DeliveryDate"],
                    "Volume": order["Volume"],
                    "RequireOrganic": "✓" if order.get("RequireOrganic", False) else "",
                    "AntallMuligeBatcher": len(order_feasible),
                    "MuligeGrupper": ", ".join(unique_groups),
                    "MuligeSites": ", ".join(unique_sites),
                    "MinDegreeDays": order_feasible["DegreeDays"].min(),
                    "MaxDegreeDays": order_feasible["DegreeDays"].max(),
                }
            )

    return pd.DataFrame(summary)


# ==========================================
# 4. OPTIMALISERING
# ==========================================
def solve_allocation(orders_df, batches_df, feasible_df):
    """
    Løser allokeringsproblemet med lineær programmering.
    Prioriterer å tildele flest mulig ordrer, deretter minimere DD/treffe preferanser.
    """
    all_orders = orders_df[
        ["OrderNr", "Customer", "DeliveryDate", "Volume", "Product", "RequireOrganic"]
    ].copy()
    all_orders["DeliveryDate"] = pd.to_datetime(all_orders["DeliveryDate"])

    if feasible_df.empty:
        # Hvis ingenting er mulig, returner alt som IKKE TILDELT
        all_orders["BatchID"] = "IKKE TILDELT"
        all_orders["Site"] = "-"
        all_orders["Organic"] = "-"
        all_orders["DegreeDays"] = "-"
        all_orders["PreferenceMatched"] = ""
        all_orders["Reason"] = "Ingen gyldig batch funnet"
        all_orders["RequireOrganic"] = all_orders["RequireOrganic"].apply(
            lambda x: "✓" if x else ""
        )
        return all_orders, pd.DataFrame()

    prob = pl.LpProblem("EggAllocation", pl.LpMinimize)
    feasible_df = feasible_df.reset_index(drop=True)
    feasible_df["id"] = feasible_df.index

    # Beslutningsvariabler for (ordre, batch)
    y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in feasible_df["id"]}

    # Slack-variabler for ordrer som har muligheter, men som eventuelt ikke kan tildeles pga kapasitet
    order_nrs_with_options = feasible_df["OrderNr"].unique().tolist()
    slack = {
        o: pl.LpVariable(f"slack_{o}", cat="Binary") for o in order_nrs_with_options
    }

    # Objektfunksjon:
    # 1. Minimerer (DegreeDays - Bonus). (Vil velge best match).
    # 2. Legger til ENORM straff for slack. (Vil unngå slack for enhver pris -> Maksimerer tildeling).
    prob += pl.lpSum(
        (row["DegreeDays"] - row["PreferenceBonus"]) * y[row["id"]]
        for _, row in feasible_df.iterrows()
    ) + pl.lpSum(NOT_ALLOCATED_PENALTY * slack[o] for o in order_nrs_with_options)

    # Constraint 1: Hver ordre med muligheter får enten nøyaktig én tildeling, eller slack = 1
    for order_nr in order_nrs_with_options:
        choices = feasible_df[feasible_df["OrderNr"] == order_nr]["id"].tolist()
        prob += pl.lpSum(y[i] for i in choices) + slack[order_nr] == 1

    # Constraint 2: Kapasitet for Gain per batch
    for batch_id in batches_df["BatchID"].unique():
        batch_feasible_gain = feasible_df[
            (feasible_df["BatchID"] == batch_id)
            & (feasible_df["Product"].str.lower() == "gain")
        ]
        if not batch_feasible_gain.empty:
            gain_cap = batches_df[batches_df["BatchID"] == batch_id].iloc[0][
                "GainCapacity"
            ]
            prob += (
                pl.lpSum(
                    y[i] * feasible_df.loc[i, "Volume"]
                    for i in batch_feasible_gain["id"]
                )
                <= gain_cap
            )

    # Constraint 3: Kapasitet for Shield per batch
    for batch_id in batches_df["BatchID"].unique():
        batch_feasible_shield = feasible_df[
            (feasible_df["BatchID"] == batch_id)
            & (feasible_df["Product"].str.lower() == "shield")
        ]
        if not batch_feasible_shield.empty:
            shield_cap = batches_df[batches_df["BatchID"] == batch_id].iloc[0][
                "ShieldCapacity"
            ]
            prob += (
                pl.lpSum(
                    y[i] * feasible_df.loc[i, "Volume"]
                    for i in batch_feasible_shield["id"]
                )
                <= shield_cap
            )

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=60))

    allocated_ids = [
        i for i, v in y.items() if pl.value(v) and round(pl.value(v)) == 1
    ]
    allocated_orders = (
        set(feasible_df.loc[allocated_ids, "OrderNr"]) if allocated_ids else set()
    )

    # Finn hvilke ordrer som brukte slack (for bedre årsaksforklaring)
    slack_used = {
        o for o, v in slack.items() if pl.value(v) and round(pl.value(v)) == 1
    }

    results = []
    for _, order in orders_df.iterrows():
        order_nr = order["OrderNr"]

        if order_nr in allocated_orders:
            alloc_row = feasible_df[
                (feasible_df["OrderNr"] == order_nr)
                & (feasible_df["id"].isin(allocated_ids))
            ].iloc[0]
            results.append(
                {
                    "OrderNr": order_nr,
                    "Customer": order.get("Customer", ""),
                    "BatchID": alloc_row["BatchID"],
                    "Site": alloc_row["Site"],
                    "Organic": "✓" if alloc_row["Organic"] else "",
                    "DegreeDays": alloc_row["DegreeDays"],
                    "DeliveryDate": alloc_row["DeliveryDate"],
                    "Volume": order["Volume"],
                    "Product": order["Product"],
                    "RequireOrganic": "✓" if order.get("RequireOrganic", False) else "",
                    "PreferenceMatched": alloc_row["PreferenceMatched"],
                    "Reason": "",
                }
            )
        else:
            reason = _get_unallocated_reason(order, feasible_df, slack_used)
            results.append(
                {
                    "OrderNr": order_nr,
                    "Customer": order.get("Customer", ""),
                    "BatchID": "IKKE TILDELT",
                    "Site": "-",
                    "Organic": "-",
                    "DegreeDays": "-",
                    "DeliveryDate": pd.to_datetime(order["DeliveryDate"]),
                    "Volume": order["Volume"],
                    "Product": order["Product"],
                    "RequireOrganic": "✓" if order.get("RequireOrganic", False) else "",
                    "PreferenceMatched": "",
                    "Reason": reason,
                }
            )

    results_df = pd.DataFrame(results)
    allocated_df = results_df[results_df["BatchID"] != "IKKE TILDELT"]

    return results_df, allocated_df


def _capacity_column_for_product(product):
    return "ShieldCapacity" if str(product).strip().lower() == "shield" else "GainCapacity"


def solve_allocation_greedy(orders_df, batches_df, feasible_df):
    """
    Rask deterministisk allokering for store, ferdig beregnede batchplaner.
    Tildeler én ordre om gangen til beste batch med ledig kapasitet.
    """
    if feasible_df.empty:
        return solve_allocation(orders_df, batches_df, feasible_df)

    remaining = {}
    for _, batch in batches_df.iterrows():
        remaining[(batch["BatchID"], "GainCapacity")] = float(batch["GainCapacity"])
        remaining[(batch["BatchID"], "ShieldCapacity")] = float(batch["ShieldCapacity"])

    allocated_rows = {}
    feasible_df = feasible_df.copy()
    order_sort = orders_df[["OrderNr", "DeliveryDate", "Volume"]].copy()
    order_sort["DeliveryDate"] = pd.to_datetime(order_sort["DeliveryDate"])
    option_counts = feasible_df.groupby("OrderNr").size().rename("OptionCount")
    order_sort = order_sort.join(option_counts, on="OrderNr")
    order_sort["OptionCount"] = order_sort["OptionCount"].fillna(0)
    order_sort = order_sort.sort_values(
        ["OptionCount", "Volume", "DeliveryDate", "OrderNr"],
        ascending=[True, False, True, True],
    )

    for _, order_ref in order_sort.iterrows():
        order_nr = order_ref["OrderNr"]
        options = feasible_df[feasible_df["OrderNr"] == order_nr].copy()
        if options.empty:
            continue

        product = options.iloc[0]["Product"]
        capacity_col = _capacity_column_for_product(product)
        volume = float(options.iloc[0]["Volume"])
        options["RemainingCapacity"] = options["BatchID"].apply(
            lambda batch_id: remaining.get((batch_id, capacity_col), 0.0)
        )
        options = options[options["RemainingCapacity"] >= volume]
        if options.empty:
            continue

        options["PostAllocationCapacity"] = options["RemainingCapacity"] - volume
        options = options.sort_values(
            ["PreferenceBonus", "PostAllocationCapacity", "DegreeDays"],
            ascending=[False, False, True],
        )
        chosen = options.iloc[0]
        remaining[(chosen["BatchID"], capacity_col)] -= volume
        allocated_rows[order_nr] = chosen

    results = []
    slack_used = set(feasible_df["OrderNr"].unique()) - set(allocated_rows)
    for _, order in orders_df.iterrows():
        order_nr = order["OrderNr"]
        if order_nr in allocated_rows:
            alloc_row = allocated_rows[order_nr]
            results.append(
                {
                    "OrderNr": order_nr,
                    "Customer": order.get("Customer", ""),
                    "BatchID": alloc_row["BatchID"],
                    "Site": alloc_row["Site"],
                    "Organic": "✓" if alloc_row["Organic"] else "",
                    "DegreeDays": alloc_row["DegreeDays"],
                    "DeliveryDate": alloc_row["DeliveryDate"],
                    "Volume": order["Volume"],
                    "Product": order["Product"],
                    "RequireOrganic": "✓" if order.get("RequireOrganic", False) else "",
                    "PreferenceMatched": alloc_row["PreferenceMatched"],
                    "Reason": "",
                }
            )
        else:
            reason = _get_unallocated_reason(order, feasible_df, slack_used)
            results.append(
                {
                    "OrderNr": order_nr,
                    "Customer": order.get("Customer", ""),
                    "BatchID": "IKKE TILDELT",
                    "Site": "-",
                    "Organic": "-",
                    "DegreeDays": "-",
                    "DeliveryDate": pd.to_datetime(order["DeliveryDate"]),
                    "Volume": order["Volume"],
                    "Product": order["Product"],
                    "RequireOrganic": "✓" if order.get("RequireOrganic", False) else "",
                    "PreferenceMatched": "",
                    "Reason": reason,
                }
            )

    results_df = pd.DataFrame(results)
    allocated_df = results_df[results_df["BatchID"] != "IKKE TILDELT"]
    return results_df, allocated_df


def _get_unallocated_reason(order, feasible_df, slack_used=None):
    """
    Hjelpefunksjon for å forklare hvorfor en ordre ikke fikk tildeling.
    """
    order_nr = order["OrderNr"]

    # 1. Ingen mulige batcher i det hele tatt (Datoproblemer, constraint problemer)
    if order_nr not in feasible_df["OrderNr"].values:
        if order.get("RequireOrganic", False):
            return "Ingen organic batch med gyldig vindu"
        if pd.notna(order.get("LockedSite")) and str(order.get("LockedSite")).strip():
            return f"LockedSite={order['LockedSite']}: Ingen gyldig batch"
        if pd.notna(order.get("LockedGroup")) and str(order.get("LockedGroup")).strip():
            return f"LockedGroup={order['LockedGroup']}: Ingen gyldig batch"
        return "Ingen batch med gyldig leveringsvindu"

    # 2. Hadde muligheter, men ble ikke tildelt (Kapasitetsproblemer)
    if slack_used is not None and order_nr in slack_used:
        n_options = len(feasible_df[feasible_df["OrderNr"] == order_nr])
        product = order.get("Product", "ukjent")
        return f"Kapasitet overskredet (hadde {n_options} teknisk mulige {product}-batcher)"

    return "Kapasitet overskredet"


# ==========================================
# 5. VISUALISERINGSDATA
# ==========================================
def _period_start(value, grain):
    date = pd.to_datetime(value).normalize()
    if grain == "month":
        return date.replace(day=1)
    return date - pd.Timedelta(days=int(date.weekday()))


def _period_label(value, grain):
    date = pd.to_datetime(value)
    if grain == "month":
        return date.strftime("%Y-%m")
    iso = date.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def _period_end(value, grain):
    start = pd.to_datetime(value)
    if grain == "month":
        return start + pd.offsets.MonthEnd(0)
    return start + pd.Timedelta(days=6)


def _select_period_grain(delivery_dates):
    if delivery_dates.empty:
        return "week"
    week_starts = delivery_dates.apply(lambda d: _period_start(d, "week"))
    return "month" if week_starts.nunique() > 78 else "week"


def _volume_tone(utilization, volume):
    if volume <= 0:
        return "empty"
    if utilization is None:
        return "active"
    if utilization >= 1:
        return "over"
    if utilization >= 0.75:
        return "high"
    if utilization >= 0.4:
        return "medium"
    return "low"


def create_planning_overview_data(
    batches_df, results_df, window_mode="week", max_sites=40, exception_limit=12
):
    """Lager lettvekts visualiseringsdata aggregert per lokasjon og periode."""
    results = results_df.copy()
    batches = batches_df.copy()
    if results.empty:
        return {
            "type": "planning_overview",
            "period_grain": "week",
            "periods": [],
            "sites": [],
            "cells": [],
            "period_totals": [],
            "exceptions": [],
            "summary": {
                "assigned_orders": 0,
                "unallocated_orders": 0,
                "assigned_volume": 0.0,
                "unallocated_volume": 0.0,
                "hidden_sites": 0,
            },
        }

    results["DeliveryDate"] = pd.to_datetime(results["DeliveryDate"])
    results["Volume"] = pd.to_numeric(results["Volume"], errors="coerce").fillna(0.0)
    grain = _select_period_grain(results["DeliveryDate"])

    results["_PeriodStart"] = results["DeliveryDate"].apply(lambda d: _period_start(d, grain))
    results["_Period"] = results["_PeriodStart"].apply(lambda d: _period_label(d, grain))
    is_assigned = results["BatchID"] != "IKKE TILDELT"
    assigned = results[is_assigned].copy()
    unallocated = results[~is_assigned].copy()
    if not assigned.empty:
        assigned["_PeriodStart"] = assigned["DeliveryDate"].apply(lambda d: _period_start(d, grain))
        assigned["_Period"] = assigned["_PeriodStart"].apply(lambda d: _period_label(d, grain))

    period_starts = sorted(results["_PeriodStart"].dropna().unique())
    periods = [_period_label(p, grain) for p in period_starts]
    period_lookup = dict(zip(periods, period_starts))

    if assigned.empty:
        site_order = []
    else:
        site_order = (
            assigned.groupby("Site", dropna=False)["Volume"]
            .sum()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
    visible_sites = site_order[:max_sites]
    hidden_sites = max(0, len(site_order) - len(visible_sites))

    if "GainCapacity" in batches.columns:
        batches["GainCapacity"] = pd.to_numeric(batches["GainCapacity"], errors="coerce").fillna(0.0)
    if "ShieldCapacity" in batches.columns:
        batches["ShieldCapacity"] = pd.to_numeric(batches["ShieldCapacity"], errors="coerce").fillna(0.0)
    batches["MaturationEnd"] = pd.to_datetime(batches["MaturationEnd"])
    batches["ProductionEnd"] = pd.to_datetime(batches["ProductionEnd"])
    capacity_by_site_period = {}
    for site in visible_sites:
        site_batches = batches[batches["Site"].astype(str) == site]
        for period, start in period_lookup.items():
            end = _period_end(start, grain)
            active = site_batches[
                (site_batches["MaturationEnd"] <= end)
                & (site_batches["ProductionEnd"] >= start)
            ]
            capacity = float(active.get("GainCapacity", pd.Series(dtype=float)).sum())
            capacity_by_site_period[(site, period)] = capacity

    assigned_grouped = (
        assigned.groupby(["Site", "_Period"], dropna=False)
        .agg(volume=("Volume", "sum"), orders=("OrderNr", "count"))
        .reset_index()
        if not assigned.empty
        else pd.DataFrame(columns=["Site", "_Period", "volume", "orders"])
    )
    assigned_lookup = {
        (str(row["Site"]), row["_Period"]): row
        for _, row in assigned_grouped.iterrows()
    }

    cells = []
    for site in visible_sites:
        for period in periods:
            row = assigned_lookup.get((site, period))
            volume = float(row["volume"]) if row is not None else 0.0
            orders = int(row["orders"]) if row is not None else 0
            capacity = capacity_by_site_period.get((site, period), 0.0)
            utilization = (volume / capacity) if capacity > 0 and volume > 0 else None
            cells.append(
                {
                    "site": site,
                    "period": period,
                    "volume": volume,
                    "orders": orders,
                    "capacity": capacity,
                    "utilization": None if utilization is None else round(utilization, 3),
                    "tone": _volume_tone(utilization, volume),
                }
            )

    site_rows = []
    for site in visible_sites:
        site_assigned = assigned[assigned["Site"].astype(str) == site]
        site_rows.append(
            {
                "site": site,
                "assigned_orders": int(len(site_assigned)),
                "assigned_volume": float(site_assigned["Volume"].sum()),
            }
        )

    unallocated_by_period = (
        unallocated.groupby("_Period")
        .agg(volume=("Volume", "sum"), orders=("OrderNr", "count"))
        .to_dict("index")
        if not unallocated.empty
        else {}
    )
    assigned_by_period = (
        assigned.groupby("_Period")
        .agg(volume=("Volume", "sum"), orders=("OrderNr", "count"))
        .to_dict("index")
        if not assigned.empty
        else {}
    )
    period_totals = []
    for period in periods:
        assigned_period = assigned_by_period.get(period, {})
        unallocated_period = unallocated_by_period.get(period, {})
        period_totals.append(
            {
                "period": period,
                "assigned_volume": float(assigned_period.get("volume", 0.0)),
                "assigned_orders": int(assigned_period.get("orders", 0)),
                "unallocated_volume": float(unallocated_period.get("volume", 0.0)),
                "unallocated_orders": int(unallocated_period.get("orders", 0)),
            }
        )

    exceptions = []
    exception_cols = ["OrderNr", "DeliveryDate", "Volume", "Product", "LockedSite", "Reason"]
    for _, row in unallocated.head(exception_limit).iterrows():
        exceptions.append(
            {
                col: (
                    row[col].strftime("%Y-%m-%d")
                    if col == "DeliveryDate"
                    else row.get(col, "")
                )
                for col in exception_cols
                if col in row.index
            }
        )

    return {
        "type": "planning_overview",
        "period_grain": grain,
        "window_mode": window_mode,
        "periods": periods,
        "sites": site_rows,
        "cells": cells,
        "period_totals": period_totals,
        "exceptions": exceptions,
        "summary": {
            "assigned_orders": int(len(assigned)),
            "unallocated_orders": int(len(unallocated)),
            "assigned_volume": float(assigned["Volume"].sum()) if not assigned.empty else 0.0,
            "unallocated_volume": float(unallocated["Volume"].sum()) if not unallocated.empty else 0.0,
            "hidden_sites": int(hidden_sites),
        },
    }


# ==========================================
# 6. EKSPORT FUNKSJONER (Uendret logikk)
# ==========================================
def generate_example_excel():
    """Genrerer eksempel Excel-fil."""
    from config import FISH_GROUPS, ORDERS

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        FISH_GROUPS.to_excel(writer, sheet_name="Fiskegrupper", index=False)
        ORDERS.to_excel(writer, sheet_name="Ordrer", index=False)
    output.seek(0)
    return output.getvalue()


def generate_orders_example_excel():
    """Genererer eksempel for kun ordrer."""
    from config import ORDERS
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        ORDERS.to_excel(writer, sheet_name='Ordrer', index=False)
    output.seek(0)
    return output.getvalue()


def parse_orders_excel(contents, filename):
    import base64

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            excel_file = io.BytesIO(decoded)
            try:
                orders = pd.read_excel(excel_file, sheet_name='Ordrer')
            except:
                excel_file.seek(0)
                orders = pd.read_excel(excel_file, sheet_name=0)
            orders = normalize_orders_input(orders)
            validation_error = _validate_required_columns(
                orders, REQUIRED_ORDER_COLUMNS, "Ordrer"
            )
            if validation_error:
                return None, validation_error
            return orders, None
        else:
            return None, "Feil filformat"
    except Exception as e:
        return None, str(e)

def parse_uploaded_excel(contents, filename):
    import base64

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            excel_file = io.BytesIO(decoded)
            fish_groups = pd.read_excel(excel_file, sheet_name="Fiskegrupper")
            orders = pd.read_excel(excel_file, sheet_name="Ordrer")
            if is_precomputed_batch_input(fish_groups):
                fish_validation_error = None
            else:
                fish_validation_error = _validate_required_columns(
                    fish_groups, REQUIRED_FISH_COLUMNS, "Fiskegrupper"
                )
            if fish_validation_error:
                return None, None, fish_validation_error
            orders = normalize_orders_input(orders)
            orders_validation_error = _validate_required_columns(
                orders, REQUIRED_ORDER_COLUMNS, "Ordrer"
            )
            if orders_validation_error:
                return None, None, orders_validation_error
            return fish_groups, orders, None
        else:
            return None, None, "Feil filformat"
    except Exception as e:
        return None, None, str(e)


# ==========================================
# 7. HOVEDFUNKSJON
# ==========================================
def run_allocation(
    fish_groups_df, orders_df, window_mode="week", growth_model="table"
):
    """Kjører hele allokeringsprosessen og returnerer resultater."""
    orders = normalize_orders_input(orders_df)
    uses_precomputed_batches = is_precomputed_batch_input(fish_groups_df)
    if uses_precomputed_batches:
        batches = build_precomputed_batches(fish_groups_df)
    else:
        orders, groups = preprocess_data(orders, fish_groups_df)
        batches = generate_weekly_batches(groups, growth_model=growth_model)
    feasible = build_feasibility_set(orders, batches, window_mode=window_mode)
    possible_groups = get_possible_groups_per_order(orders, feasible)
    results_df, allocated_df = solve_allocation(orders, batches, feasible)
    overview = create_planning_overview_data(batches, results_df, window_mode=window_mode)

    return {
        "batches": batches,
        "feasible": feasible,
        "possible_groups": possible_groups,
        "results": results_df,
        "allocated": allocated_df,
        "visualization": overview,
    }
