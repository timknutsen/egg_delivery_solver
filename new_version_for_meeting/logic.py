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
import plotly.graph_objects as go
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


def _validate_required_columns(df, required_columns, label):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return (
            f"Mangler kolonner i '{label}': {', '.join(missing)}. "
            f"Forventet kolonner inkluderer: {', '.join(required_columns)}"
        )
    return None

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


def build_feasibility_set(orders_df, batches_df, window_mode="week"):
    """
    Finner alle gyldige (ordre, batch) kombinasjoner.
    Sjekker om kundens leveringsdato er innenfor det biologiske vinduet.
    window_mode='week' betyr at sjekk gjøres på ukenivå (mandag-søndag).
    """
    feasible = []

    for _, order in orders_df.iterrows():
        delivery_date = pd.to_datetime(order["DeliveryDate"])
        
        for _, batch in batches_df.iterrows():
            # HARD CONSTRAINTS
            require_organic = order.get("RequireOrganic", False)
            if require_organic and not batch["Organic"]:
                continue

            locked_site = order.get("LockedSite")
            if pd.notna(locked_site) and str(locked_site).strip():
                if batch["Site"] != locked_site:
                    continue

            locked_group = order.get("LockedGroup")
            if pd.notna(locked_group) and str(locked_group).strip():
                if batch["Group"] != locked_group:
                    continue

            # --- SJEKK LEVERINGSVINDU ---
            # Batch har et absolutt vindu: [MaturationEnd, ProductionEnd]
            # Kunden ønsker levering på 'DeliveryDate'.
            
            # Sjekk 1: Er leveringsdatoen fysisk mulig for batchen?
            # Kan kjøres enten på dag eller uke-nivå.
            if not _is_delivery_in_window(
                delivery_date,
                batch["MaturationEnd"],
                batch["ProductionEnd"],
                window_mode=window_mode,
            ):
                continue
            
            # Sjekk 2 (Valgfri, men god): Kundens temperaturkrav vs Faktisk tid
            # "Hvis jeg leverer på denne datoen, hvor mange døgngrader har fisken fått?"
            days_since_strip = (delivery_date - batch["StripDate"]).days
            dd_at_delivery = days_since_strip * WATER_TEMP_C # Forenklet DD beregning ved sjø

            # SOFT CONSTRAINTS (Preferanser)
            bonus = 0
            pref_matched = []

            pref_site = order.get("PreferredSite")
            if pd.notna(pref_site) and str(pref_site).strip():
                if batch["Site"] == pref_site:
                    bonus += PREFERENCE_BONUS
                    pref_matched.append(f"Site={pref_site}")

            pref_group = order.get("PreferredGroup")
            if pd.notna(pref_group) and str(pref_group).strip():
                if batch["Group"] == pref_group:
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
# 5. VISUALISERING
# ==========================================
def create_gantt_chart(batches_df, orders_df, allocated_df):
    """Lager Gantt-chart med batcher for Modning og Produksjon."""
    fig = go.Figure()
    batch_ids = batches_df["BatchID"].tolist()
    y_pos = {bid: i for i, bid in enumerate(batch_ids)}

    for _, batch in batches_df.iterrows():
        yp = y_pos[batch["BatchID"]]
        organic_label = " 🌿" if batch["Organic"] else ""

        # Modningsfase (blå) - Frem til salgsstart
        fig.add_trace(
            go.Scatter(
                x=[batch["StripDate"], batch["MaturationEnd"]],
                y=[yp, yp],
                mode="lines",
                line=dict(color="#1f77b4", width=20),
                name="Modning (før salg)",
                legendgroup="mod",
                showlegend=(yp == 0),
                hovertemplate=f"<b>{batch['BatchID']}{organic_label}</b><br>Modning<extra></extra>",
            )
        )

        # Salgsvindu (rød) - Fra start til slutt
        fig.add_trace(
            go.Scatter(
                x=[batch["MaturationEnd"], batch["ProductionEnd"]],
                y=[yp, yp],
                mode="lines",
                line=dict(color="#d62728", width=20),
                name="Salgsvindu",
                legendgroup="prod",
                showlegend=(yp == 0),
                hovertemplate=f"<b>{batch['BatchID']}{organic_label}</b><br>Salgsvindu<br>Start: {batch['MaturationEnd'].strftime('%d.%m')}<br>Slutt: {batch['ProductionEnd'].strftime('%d.%m')}<extra></extra>",
            )
        )

    if not allocated_df.empty:
        for _, alloc in allocated_df.iterrows():
            if alloc["BatchID"] == "IKKE TILDELT":
                continue

            dd = pd.to_datetime(alloc["DeliveryDate"])
            bid = alloc["BatchID"]

            # Lilla markør for tildeling
            fig.add_vline(x=dd, line_dash="dash", line_color="purple", line_width=1, opacity=0.3)

            if bid in y_pos:
                pref = (
                    f"<br>Pref: {alloc['PreferenceMatched']}"
                    if alloc.get("PreferenceMatched")
                    else ""
                )
                organic = "<br>🌿 Organic" if alloc.get("Organic") == "✓" else ""

                fig.add_trace(
                    go.Scatter(
                        x=[dd],
                        y=[y_pos[bid]],
                        mode="markers",
                        marker=dict(color="purple", size=10, symbol="diamond"),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Ordre {alloc['OrderNr']}</b><br>"
                            f"Kunde: {alloc.get('Customer', '')}<br>"
                            f"Volume: {alloc.get('Volume', '')}<br>"
                            f"Dato: {dd.strftime('%Y-%m-%d')}{organic}{pref}<extra></extra>"
                        ),
                    )
                )

    y_labels = []
    for bid in batch_ids:
        batch = batches_df[batches_df["BatchID"] == bid].iloc[0]
        label = f"🌿 {bid}" if batch["Organic"] else bid
        y_labels.append(label)

    fig.update_layout(
        title="Produksjonsplan og Salgsvindu",
        xaxis_title="Dato",
        yaxis_title="Batch",
        yaxis=dict(
            tickmode="array",
            tickvals=list(y_pos.values()),
            ticktext=y_labels,
            autorange="reversed",
        ),
        height=max(400, len(batch_ids) * 50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(r=150),
    )
    return fig


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
            fish_validation_error = _validate_required_columns(
                fish_groups, REQUIRED_FISH_COLUMNS, "Fiskegrupper"
            )
            if fish_validation_error:
                return None, None, fish_validation_error
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
    orders, groups = preprocess_data(orders_df, fish_groups_df)
    batches = generate_weekly_batches(groups, growth_model=growth_model)
    feasible = build_feasibility_set(orders, batches, window_mode=window_mode)
    possible_groups = get_possible_groups_per_order(orders, feasible)
    results_df, allocated_df = solve_allocation(orders, batches, feasible)
    chart = create_gantt_chart(batches, orders, allocated_df)

    return {
        "batches": batches,
        "feasible": feasible,
        "possible_groups": possible_groups,
        "results": results_df,
        "allocated": allocated_df,
        "chart": chart,
    }
