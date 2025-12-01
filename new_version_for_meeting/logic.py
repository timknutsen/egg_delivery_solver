"""
FORRETNINGSLOGIKK
=================
Denne filen inneholder all logikk for:
- Preprocessing av data
- Generering av batcher
- Feasibility-sjekk (inkl. Organic)
- Optimalisering/allokering
- Visualisering
- Eksport av eksempeldata
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import pulp as pl
from pulp import PULP_CBC_CMD
import io

from config import WATER_TEMP_C, DD_TO_MATURE, PREFERENCE_BONUS


# ==========================================
# 1. PREPROCESSING
# ==========================================
def preprocess_data(orders_df, groups_df):
    """Konverterer Celsius-temperaturer til døgngrader."""
    g_df = groups_df.copy()
    g_df['MinTemp_prod'] = g_df['MinTemp_C'] * DD_TO_MATURE
    g_df['MaxTemp_prod'] = g_df['MaxTemp_C'] * DD_TO_MATURE
    
    o_df = orders_df.copy()
    o_df['MinTemp_customer'] = o_df['MinTemp_C'] * DD_TO_MATURE
    o_df['MaxTemp_customer'] = o_df['MaxTemp_C'] * DD_TO_MATURE
    
    return o_df, g_df


# ==========================================
# 2. BATCH-GENERERING
# ==========================================
def generate_weekly_batches(fish_groups_df):
    """Deler fiskegrupper inn i ukentlige batcher med normalfordelt kapasitet."""
    all_batches = []
    
    for _, group in fish_groups_df.iterrows():
        strip_start = pd.to_datetime(group['StrippingStartDate'])
        strip_stop = pd.to_datetime(group['StrippingStopDate'])
        
        weeks = pd.date_range(strip_start, strip_stop, freq='W-MON')
        if len(weeks) == 0: 
            weeks = pd.DatetimeIndex([strip_start])
        
        n = len(weeks)
        indices = np.arange(n)
        weights = np.exp(-0.5 * ((indices - (n - 1) / 2) / max(n / 4, 1)) ** 2)
        weights = weights / weights.sum()

        for i, strip_date in enumerate(weeks):
            maturation_days = group['MinTemp_prod'] / WATER_TEMP_C
            production_days = group['MaxTemp_prod'] / WATER_TEMP_C
            
            all_batches.append({
                'BatchID': f"{group['Site_Broodst_Season']}_Uke_{i+1}",
                'Group': group['Site_Broodst_Season'],
                'Site': group['Site'],
                'StripDate': strip_date,
                'MaturationEnd': strip_date + timedelta(days=maturation_days),
                'ProductionEnd': strip_date + timedelta(days=production_days),
                'GainCapacity': float(group['Gain-eggs']) * weights[i],
                'ShieldCapacity': float(group['Shield-eggs']) * weights[i],
                'Organic': group['Organic'],
            })
    
    return pd.DataFrame(all_batches)


# ==========================================
# 3. FEASIBILITY-SJEKK
# ==========================================
def build_feasibility_set(orders_df, batches_df):
    """
    Finner alle gyldige (ordre, batch) kombinasjoner.
    """
    feasible = []
    
    for _, order in orders_df.iterrows():
        delivery_date = pd.to_datetime(order['DeliveryDate'])
        cust_min_days = order['MinTemp_customer'] / WATER_TEMP_C
        cust_max_days = order['MaxTemp_customer'] / WATER_TEMP_C
        
        for _, batch in batches_df.iterrows():
            
            # HARD CONSTRAINTS
            require_organic = order.get('RequireOrganic', False)
            if require_organic and not batch['Organic']:
                continue
            
            locked_site = order.get('LockedSite')
            if pd.notna(locked_site) and str(locked_site).strip():
                if batch['Site'] != locked_site:
                    continue
            
            locked_group = order.get('LockedGroup')
            if pd.notna(locked_group) and str(locked_group).strip():
                if batch['Group'] != locked_group:
                    continue
            
            # Leveringsvindu
            cust_start = batch['StripDate'] + timedelta(days=cust_min_days)
            cust_end = batch['StripDate'] + timedelta(days=cust_max_days)
            valid_start = max(cust_start, batch['MaturationEnd'])
            valid_end = min(cust_end, batch['ProductionEnd'])
            
            if not (valid_start <= delivery_date <= valid_end):
                continue
            
            days_since_strip = (delivery_date - batch['StripDate']).days
            dd_at_delivery = days_since_strip * WATER_TEMP_C
            
            # SOFT CONSTRAINTS
            bonus = 0
            pref_matched = []
            
            pref_site = order.get('PreferredSite')
            if pd.notna(pref_site) and str(pref_site).strip():
                if batch['Site'] == pref_site:
                    bonus += PREFERENCE_BONUS
                    pref_matched.append(f"Site={pref_site}")
            
            pref_group = order.get('PreferredGroup')
            if pd.notna(pref_group) and str(pref_group).strip():
                if batch['Group'] == pref_group:
                    bonus += PREFERENCE_BONUS
                    pref_matched.append(f"Group={pref_group}")
            
            feasible.append({
                'OrderNr': order['OrderNr'],
                'Customer': order.get('Customer', ''),
                'BatchID': batch['BatchID'],
                'Group': batch['Group'],
                'Site': batch['Site'],
                'Organic': batch['Organic'],
                'StripDate': batch['StripDate'],
                'DeliveryDate': delivery_date,
                'DegreeDays': round(dd_at_delivery, 2),
                'Volume': order['Volume'],
                'Product': order['Product'],
                'RequireOrganic': require_organic,
                'PreferenceBonus': bonus,
                'PreferenceMatched': ', '.join(pref_matched),
                'BatchCapacity': batch['GainCapacity'] + batch['ShieldCapacity'],
            })
    
    return pd.DataFrame(feasible)


# ==========================================
# 3b. MULIGE GRUPPER PER ORDRE (NY)
# ==========================================
def get_possible_groups_per_order(orders_df, feasible_df):
    """
    Returnerer en oversikt over mulige grupper/batcher per ordre.
    Nyttig for manuell vurdering.
    """
    summary = []
    
    for _, order in orders_df.iterrows():
        order_nr = order['OrderNr']
        order_feasible = feasible_df[feasible_df['OrderNr'] == order_nr]
        
        if order_feasible.empty:
            summary.append({
                'OrderNr': order_nr,
                'Customer': order.get('Customer', ''),
                'DeliveryDate': order['DeliveryDate'],
                'Volume': order['Volume'],
                'RequireOrganic': '✓' if order.get('RequireOrganic', False) else '',
                'AntallMuligeBatcher': 0,
                'MuligeGrupper': 'INGEN',
                'MuligeSites': 'INGEN',
                'MinDegreeDays': '-',
                'MaxDegreeDays': '-',
            })
        else:
            unique_groups = order_feasible['Group'].unique()
            unique_sites = order_feasible['Site'].unique()
            
            summary.append({
                'OrderNr': order_nr,
                'Customer': order.get('Customer', ''),
                'DeliveryDate': order['DeliveryDate'],
                'Volume': order['Volume'],
                'RequireOrganic': '✓' if order.get('RequireOrganic', False) else '',
                'AntallMuligeBatcher': len(order_feasible),
                'MuligeGrupper': ', '.join(unique_groups),
                'MuligeSites': ', '.join(unique_sites),
                'MinDegreeDays': order_feasible['DegreeDays'].min(),
                'MaxDegreeDays': order_feasible['DegreeDays'].max(),
            })
    
    return pd.DataFrame(summary)


# ==========================================
# 4. OPTIMALISERING
# ==========================================
def solve_allocation(orders_df, batches_df, feasible_df):
    """Løser allokeringsproblemet med lineær programmering."""
    
    all_orders = orders_df[['OrderNr', 'Customer', 'DeliveryDate', 'Volume', 'Product', 'RequireOrganic']].copy()
    all_orders['DeliveryDate'] = pd.to_datetime(all_orders['DeliveryDate'])
    
    if feasible_df.empty:
        all_orders['BatchID'] = 'IKKE TILDELT'
        all_orders['Site'] = '-'
        all_orders['Organic'] = '-'
        all_orders['DegreeDays'] = '-'
        all_orders['PreferenceMatched'] = ''
        all_orders['Reason'] = 'Ingen gyldig batch funnet'
        return all_orders, all_orders

    prob = pl.LpProblem("EggAllocation", pl.LpMinimize)
    feasible_df = feasible_df.reset_index(drop=True)
    feasible_df['id'] = feasible_df.index
    
    y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in feasible_df['id']}
    
    prob += pl.lpSum(
        (row['DegreeDays'] + row['PreferenceBonus']) * y[row['id']] 
        for _, row in feasible_df.iterrows()
    )

    for order_nr in orders_df['OrderNr'].unique():
        choices = feasible_df[feasible_df['OrderNr'] == order_nr]['id'].tolist()
        if choices:
            prob += pl.lpSum(y[i] for i in choices) <= 1

    for batch_id in batches_df['BatchID'].unique():
        batch_rows = feasible_df[feasible_df['BatchID'] == batch_id]
        if not batch_rows.empty:
            cap = batches_df[batches_df['BatchID'] == batch_id].iloc[0]
            cap = cap['GainCapacity'] + cap['ShieldCapacity']
            prob += pl.lpSum(
                y[i] * batch_rows.loc[i, 'Volume'] for i in batch_rows['id']
            ) <= cap

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    
    allocated_ids = [i for i, v in y.items() if pl.value(v) and round(pl.value(v)) == 1]
    allocated_orders = set(feasible_df.loc[allocated_ids, 'OrderNr']) if allocated_ids else set()
    
    results = []
    for _, order in orders_df.iterrows():
        order_nr = order['OrderNr']
        
        if order_nr in allocated_orders:
            alloc_row = feasible_df[(feasible_df['OrderNr'] == order_nr) & 
                                    (feasible_df['id'].isin(allocated_ids))].iloc[0]
            results.append({
                'OrderNr': order_nr,
                'Customer': order.get('Customer', ''),
                'BatchID': alloc_row['BatchID'],
                'Site': alloc_row['Site'],
                'Organic': '✓' if alloc_row['Organic'] else '',
                'DegreeDays': alloc_row['DegreeDays'],
                'DeliveryDate': alloc_row['DeliveryDate'],
                'Volume': order['Volume'],
                'Product': order['Product'],
                'RequireOrganic': '✓' if order.get('RequireOrganic', False) else '',
                'PreferenceMatched': alloc_row['PreferenceMatched'],
                'Reason': '',
            })
        else:
            reason = _get_unallocated_reason(order, feasible_df)
            results.append({
                'OrderNr': order_nr,
                'Customer': order.get('Customer', ''),
                'BatchID': 'IKKE TILDELT',
                'Site': '-',
                'Organic': '-',
                'DegreeDays': '-',
                'DeliveryDate': pd.to_datetime(order['DeliveryDate']),
                'Volume': order['Volume'],
                'Product': order['Product'],
                'RequireOrganic': '✓' if order.get('RequireOrganic', False) else '',
                'PreferenceMatched': '',
                'Reason': reason,
            })
    
    results_df = pd.DataFrame(results)
    allocated_df = results_df[results_df['BatchID'] != 'IKKE TILDELT']
    
    return results_df, allocated_df


def _get_unallocated_reason(order, feasible_df):
    """Finner årsak til at ordre ikke ble tildelt."""
    order_nr = order['OrderNr']
    
    if order_nr not in feasible_df['OrderNr'].values:
        if order.get('RequireOrganic', False):
            return 'Ingen organic batch med gyldig vindu'
        if pd.notna(order.get('LockedSite')) and str(order.get('LockedSite')).strip():
            return f"LockedSite={order['LockedSite']}: Ingen gyldig batch"
        if pd.notna(order.get('LockedGroup')) and str(order.get('LockedGroup')).strip():
            return f"LockedGroup={order['LockedGroup']}: Ingen gyldig batch"
        return 'Ingen batch med gyldig leveringsvindu'
    else:
        return 'Kapasitet overskredet i alle gyldige batcher'


# ==========================================
# 5. VISUALISERING
# ==========================================
def create_gantt_chart(batches_df, orders_df, allocated_df):
    """Lager Gantt-chart med batcher og tildelinger."""
    fig = go.Figure()
    batch_ids = batches_df['BatchID'].tolist()
    y_pos = {bid: i for i, bid in enumerate(batch_ids)}

    for _, batch in batches_df.iterrows():
        yp = y_pos[batch['BatchID']]
        organic_label = " 🌿" if batch['Organic'] else ""
        
        fig.add_trace(go.Scatter(
            x=[batch['StripDate'], batch['MaturationEnd']], y=[yp, yp],
            mode='lines', line=dict(color='#1f77b4', width=20),
            name='Modningstid', legendgroup='mod', showlegend=(yp == 0),
            hovertemplate=f"<b>{batch['BatchID']}{organic_label}</b><br>Modning<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[batch['MaturationEnd'], batch['ProductionEnd']], y=[yp, yp],
            mode='lines', line=dict(color='#d62728', width=20),
            name='Produksjonsvindu', legendgroup='prod', showlegend=(yp == 0),
            hovertemplate=f"<b>{batch['BatchID']}{organic_label}</b><br>Produksjon<extra></extra>"
        ))

        for _, order in orders_df.iterrows():
            cust_min = order['MinTemp_customer'] / WATER_TEMP_C
            cust_max = order['MaxTemp_customer'] / WATER_TEMP_C
            g_start = max(batch['MaturationEnd'], batch['StripDate'] + timedelta(days=cust_min))
            g_end = min(batch['ProductionEnd'], batch['StripDate'] + timedelta(days=cust_max))
            
            if g_start < g_end:
                fig.add_trace(go.Scatter(
                    x=[g_start, g_end], y=[yp, yp],
                    mode='lines', line=dict(color='#2ca02c', width=12),
                    name='Kundevindu', legendgroup='kunde', showlegend=(yp == 0),
                    hovertemplate=f"<b>{batch['BatchID']}{organic_label}</b><br>Kundevindu<extra></extra>"
                ))
                break

    if not allocated_df.empty:
        for _, alloc in allocated_df.iterrows():
            if alloc['BatchID'] == 'IKKE TILDELT':
                continue
                
            dd = pd.to_datetime(alloc['DeliveryDate'])
            bid = alloc['BatchID']
            
            fig.add_vline(x=dd, line_dash="dash", line_color="purple", line_width=2)
            
            if bid in y_pos:
                pref = f"<br>Pref: {alloc['PreferenceMatched']}" if alloc.get('PreferenceMatched') else ""
                organic = "<br>🌿 Organic" if alloc.get('Organic') == '✓' else ""
                
                fig.add_trace(go.Scatter(
                    x=[dd], y=[y_pos[bid]],
                    mode='markers', marker=dict(color='purple', size=15, symbol='diamond'),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Ordre {alloc['OrderNr']}</b><br>"
                        f"Kunde: {alloc.get('Customer', '')}<br>"
                        f"Batch: {bid}<br>"
                        f"DD: {alloc['DegreeDays']}{organic}{pref}<extra></extra>"
                    )
                ))

    y_labels = []
    for bid in batch_ids:
        batch = batches_df[batches_df['BatchID'] == bid].iloc[0]
        label = f"🌿 {bid}" if batch['Organic'] else bid
        y_labels.append(label)

    fig.update_layout(
        title="Batch-tidslinje med tildelinger",
        xaxis_title="Dato", yaxis_title="Batch",
        yaxis=dict(tickmode='array', tickvals=list(y_pos.values()), ticktext=y_labels, autorange="reversed"),
        height=max(400, len(batch_ids) * 50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150)
    )
    return fig


# ==========================================
# 6. EKSPORT FUNKSJONER (NY)
# ==========================================
def generate_example_excel():
    """
    Genererer eksempel Excel-fil med input-data.
    Returnerer bytes som kan lastes ned.
    """
    from config import FISH_GROUPS, ORDERS
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Fiskegrupper
        FISH_GROUPS.to_excel(writer, sheet_name='Fiskegrupper', index=False)
        
        # Ordrer
        ORDERS.to_excel(writer, sheet_name='Ordrer', index=False)
        
        # Instruksjoner
        instructions = pd.DataFrame({
            'Felt': [
                '--- FISKEGRUPPER ---',
                'Site', 'Site_Broodst_Season', 'StrippingStartDate', 'StrippingStopDate',
                'MinTemp_C', 'MaxTemp_C', 'Gain-eggs', 'Shield-eggs', 'Organic',
                '',
                '--- ORDRER ---',
                'OrderNr', 'Customer', 'DeliveryDate', 'Product', 'Volume',
                'MinTemp_C', 'MaxTemp_C', 'RequireOrganic',
                'LockedSite', 'LockedGroup', 'PreferredSite', 'PreferredGroup'
            ],
            'Beskrivelse': [
                '',
                'Anleggets navn (f.eks. Hemne, Vestseøra)',
                'Unik ID for gruppen (f.eks. Hemne_Normal_24/25)',
                'Startdato for stripping (YYYY-MM-DD)',
                'Sluttdato for stripping (YYYY-MM-DD)',
                'Minimum temperatur i °C (typisk 1)',
                'Maksimum temperatur i °C (typisk 8)',
                'Antall Gain-egg i gruppen',
                'Antall Shield-egg i gruppen',
                'True/False - er gruppen organic?',
                '',
                '',
                'Unik ordrenummer',
                'Kundenavn',
                'Ønsket leveringsdato (YYYY-MM-DD)',
                'Produkttype: Gain eller Shield',
                'Antall egg i ordren',
                'Kundens min temperaturkrav (typisk 2)',
                'Kundens max temperaturkrav (typisk 6)',
                'True/False - krever kunden organic?',
                'HARD: Ordre MÅ leveres fra dette anlegget (eller tom)',
                'HARD: Ordre MÅ leveres fra denne gruppen (eller tom)',
                'SOFT: Ordre BØR leveres fra dette anlegget (eller tom)',
                'SOFT: Ordre BØR leveres fra denne gruppen (eller tom)'
            ],
            'Eksempel': [
                '',
                'Hemne',
                'Hemne_Normal_24/25',
                '2024-09-01',
                '2024-09-28',
                '1',
                '8',
                '8000000',
                '2000000',
                'False',
                '',
                '',
                '1001',
                'Lerøy Midt',
                '2024-11-15',
                'Gain',
                '1500000',
                '2',
                '6',
                'False',
                'Hønsvikgulen',
                '',
                'Hemne',
                ''
            ]
        })
        instructions.to_excel(writer, sheet_name='Instruksjoner', index=False)
    
    output.seek(0)
    return output.getvalue()


def parse_uploaded_excel(contents, filename):
    """
    Parser opplastet Excel-fil og returnerer DataFrames.
    """
    import base64
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            excel_file = io.BytesIO(decoded)
            fish_groups = pd.read_excel(excel_file, sheet_name='Fiskegrupper')
            orders = pd.read_excel(excel_file, sheet_name='Ordrer')
            return fish_groups, orders, None
        else:
            return None, None, "Feil filformat. Bruk .xlsx eller .xls"
    except Exception as e:
        return None, None, f"Feil ved parsing av fil: {str(e)}"


# ==========================================
# 7. HOVEDFUNKSJON
# ==========================================
def run_allocation(fish_groups_df, orders_df):
    """Kjører hele allokeringsprosessen og returnerer resultater."""
    orders, groups = preprocess_data(orders_df, fish_groups_df)
    batches = generate_weekly_batches(groups)
    feasible = build_feasibility_set(orders, batches)
    possible_groups = get_possible_groups_per_order(orders, feasible)
    results_df, allocated_df = solve_allocation(orders, batches, feasible)
    chart = create_gantt_chart(batches, orders, allocated_df)
    
    return {
        'batches': batches,
        'feasible': feasible,
        'possible_groups': possible_groups,  # NY
        'results': results_df,
        'allocated': allocated_df,
        'chart': chart
    }
