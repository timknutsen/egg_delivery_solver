"""
DASH APP
========
UI-layout og callbacks.
Inkluderer:
- Visning av mulige grupper per ordre
- Eksport av eksempel inputfiler (kombinert og kun ordrer)
- Opplasting av inputfiler (kombinert og kun ordrer)
"""

import os
import traceback

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from config import FISH_GROUPS, ORDERS, WATER_TEMP_C
from logic import (
    generate_example_excel,
    generate_orders_example_excel,
    parse_orders_excel,
    parse_uploaded_excel,
    run_allocation,
)

# ==========================================
# APP SETUP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Eggallokering"


def _df_to_store_json(df):
    if df is None:
        return None
    return df.to_json(orient="split", date_format="iso")


def _store_json_to_df(payload):
    if payload is None:
        return None
    return pd.read_json(payload, orient="split")


def _resolve_active_data(store_data, uploaded_store):
    fish_groups = FISH_GROUPS
    orders = ORDERS

    uploaded_store = uploaded_store or {}
    uploaded_fish = _store_json_to_df(uploaded_store.get("fish_groups"))
    uploaded_orders = _store_json_to_df(uploaded_store.get("orders"))

    if store_data.get("use_uploaded"):
        if store_data.get("orders_only"):
            if uploaded_orders is not None:
                orders = uploaded_orders
        else:
            if uploaded_fish is not None:
                fish_groups = uploaded_fish
            if uploaded_orders is not None:
                orders = uploaded_orders

    return fish_groups, orders


def _metric_card(title, value, tone="default", detail=None):
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(detail or "", className="metric-detail"),
        ],
        className=f"metric-card metric-{tone}",
    )


def _section_heading(eyebrow, title, description):
    return html.Div(
        [
            html.Div(eyebrow, className="section-eyebrow"),
            html.H2(title, className="section-title"),
            html.P(description, className="section-description"),
        ],
        className="section-heading",
    )


def _build_table(df, header_background, conditional_styles=None, page_size=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": header_background,
            "color": "white",
            "fontWeight": "700",
            "border": "none",
            "padding": "12px 10px",
        },
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "border": "none",
            "backgroundColor": "#f8f6f1",
            "color": "#1f2a2b",
            "fontFamily": '"Avenir Next", "Segoe UI", sans-serif',
            "fontSize": "0.92rem",
        },
        style_data={
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=(conditional_styles or [])
        + [
            {"if": {"row_index": "odd"}, "backgroundColor": "#f2efe8"},
        ],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
    )


def _build_results_figure(figure):
    figure.update_layout(
        title=None,
        paper_bgcolor="#f8f6f1",
        plot_bgcolor="#fffdf8",
        font={"family": '"Avenir Next", "Segoe UI", sans-serif', "color": "#1f2a2b"},
        margin={"l": 20, "r": 20, "t": 24, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.65)",
        },
    )
    figure.update_xaxes(showgrid=True, gridcolor="#d9d1c2", zeroline=False)
    figure.update_yaxes(showgrid=False, zeroline=False)
    return figure


def _format_source_label(store_data):
    if store_data.get("use_uploaded"):
        return "Opplastet ordrefil" if store_data.get("orders_only") else "Opplastet komplett fil"
    return "Default datasett"


def _make_status_panel():
    return html.Div(
        [
            html.Div("Aktiv datakilde", className="status-kicker"),
            html.Div(id="active-source", className="status-source"),
            html.Div(id="active-source-detail", className="status-detail"),
            html.Div(
                [
                    html.Span(f"Vanntemperatur {WATER_TEMP_C}°C", className="status-chip"),
                    html.Span("Organic batches markeres separat", className="status-chip"),
                ],
                className="status-chip-row",
            ),
        ],
        className="status-panel",
    )


def _make_import_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Dataflyt", className="panel-kicker"),
                html.H3("Importer scenario", className="panel-title"),
                html.P(
                    "Last opp et komplett scenario eller bare ordrer. Default datasett fungerer alltid som fallback.",
                    className="panel-copy",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Komplett fil", className="upload-label"),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        [
                                            html.Div("Slipp Excel her", className="upload-title"),
                                            html.Div("eller velg fil", className="upload-subtitle"),
                                        ],
                                        className="upload-surface",
                                    ),
                                    multiple=False,
                                    accept=".xlsx,.xls",
                                    className="upload-wrapper",
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Eksempelfil",
                                            id="download-btn",
                                            color="link",
                                            className="inline-action",
                                        ),
                                        dcc.Download(id="download-example"),
                                    ],
                                    className="inline-action-row",
                                ),
                                html.Div(id="upload-status", className="status-slot"),
                            ],
                            className="upload-card",
                        ),
                        html.Div(
                            [
                                html.Div("Kun ordrer", className="upload-label"),
                                dcc.Upload(
                                    id="upload-orders-only",
                                    children=html.Div(
                                        [
                                            html.Div("Bruk default grupper", className="upload-title"),
                                            html.Div("last opp ordrefil", className="upload-subtitle"),
                                        ],
                                        className="upload-surface upload-surface-alt",
                                    ),
                                    multiple=False,
                                    accept=".xlsx,.xls",
                                    className="upload-wrapper",
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Ordre-eksempel",
                                            id="download-orders-btn",
                                            color="link",
                                            className="inline-action",
                                        ),
                                        dcc.Download(id="download-orders-example"),
                                    ],
                                    className="inline-action-row",
                                ),
                                html.Div(id="upload-orders-status", className="status-slot"),
                            ],
                            className="upload-card",
                        ),
                    ],
                    className="upload-grid",
                ),
                html.Div(
                    [
                        dbc.Button("Tilbakestill", id="reset-btn", color="dark", outline=True),
                        html.Div(id="reset-status", className="status-slot"),
                    ],
                    className="reset-row",
                ),
            ]
        ),
        className="shell-card control-card",
    )


def _make_controls_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Scenario", className="panel-kicker"),
                html.H3("Planleggingsmodus", className="panel-title"),
                html.P(
                    "Velg hvor stramt leveringsvinduet skal tolkes og hvordan vekst beregnes før kjøringen.",
                    className="panel-copy",
                ),
                html.Div(
                    [
                        html.Div("Leveringsvindu", className="control-label"),
                        dbc.RadioItems(
                            id="window-mode",
                            options=[
                                {"label": "Uke", "value": "week"},
                                {"label": "Dag", "value": "day"},
                            ],
                            value="week",
                            persistence=True,
                            persistence_type="local",
                            className="segmented-control",
                            inputClassName="segmented-input",
                            labelClassName="segmented-label",
                            labelCheckedClassName="segmented-label-checked",
                        ),
                        html.Div(
                            "Uke-modus godkjenner ordre og batch i samme mandag-søndag-vindu.",
                            className="control-hint",
                        ),
                    ],
                    className="control-block",
                ),
                html.Div(
                    [
                        html.Div("Vekstmodell", className="control-label"),
                        dbc.RadioItems(
                            id="growth-model",
                            options=[
                                {"label": "Graderingstabell", "value": "table"},
                                {"label": "Klekkekalkulator", "value": "formula"},
                            ],
                            value="table",
                            persistence=True,
                            persistence_type="local",
                            className="segmented-control",
                            inputClassName="segmented-input",
                            labelClassName="segmented-label",
                            labelCheckedClassName="segmented-label-checked",
                        ),
                        html.Div(
                            "Bruk tabell for konservativ planlegging eller formel for mer direkte beregning.",
                            className="control-hint",
                        ),
                    ],
                    className="control-block",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Hard constraints", className="rule-title"),
                                html.Ul(
                                    [
                                        html.Li("RequireOrganic må treffe organic batch"),
                                        html.Li("LockedSite og LockedGroup overstyres aldri"),
                                    ],
                                    className="rule-list",
                                ),
                            ],
                            className="rule-card",
                        ),
                        html.Div(
                            [
                                html.Div("Soft constraints", className="rule-title"),
                                html.Ul(
                                    [
                                        html.Li("PreferredSite og PreferredGroup premieres"),
                                        html.Li("Mulige grupper brukes som manuell støtte"),
                                    ],
                                    className="rule-list",
                                ),
                            ],
                            className="rule-card",
                        ),
                    ],
                    className="rule-grid",
                ),
            ]
        ),
        className="shell-card control-card",
    )


def _make_data_card():
    return dbc.Card(
        dbc.CardBody(
            [
                _section_heading(
                    "Input",
                    "Arbeidsdatasett",
                    "Hold rådata tilgjengelig, men skjerm dem bak faner slik at beslutningsflaten får mer ro.",
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            html.Div(id="fish-table-container", className="table-shell"),
                            label="Fiskegrupper",
                            tabClassName="data-tab",
                            activeTabClassName="data-tab-active",
                        ),
                        dbc.Tab(
                            html.Div(id="orders-table-container", className="table-shell"),
                            label="Ordrer",
                            tabClassName="data-tab",
                            activeTabClassName="data-tab-active",
                        ),
                    ],
                    className="data-tabs",
                ),
                html.Div(
                    [
                        html.Span("Organic markeres grønt", className="legend-chip legend-organic"),
                        html.Span("Låste constraints markeres rødt", className="legend-chip legend-locked"),
                    ],
                    className="legend-row",
                ),
            ]
        ),
        className="shell-card data-card",
    )


app.layout = html.Div(
    [
        html.Div(className="page-glow page-glow-left"),
        html.Div(className="page-glow page-glow-right"),
        dbc.Container(
            [
                html.Header(
                    [
                        html.Div(
                            [
                                html.Div("Egg Delivery Solver", className="hero-kicker"),
                                html.H1("Eggallokering som et planleggingsverktøy", className="hero-title"),
                                html.P(
                                    "Importer scenario, juster biologiske regler og kjør allokering i en arbeidsflate som prioriterer beslutninger fremfor skjemaer.",
                                    className="hero-copy",
                                ),
                            ],
                            className="hero-copy-wrap",
                        ),
                        html.Div(
                            [
                                _make_status_panel(),
                                dbc.Button(
                                    "Kjør allokering",
                                    id="run-btn",
                                    color="dark",
                                    size="lg",
                                    className="run-button",
                                ),
                            ],
                            className="hero-side",
                        ),
                    ],
                    className="hero-shell",
                ),
                html.Main(
                    [
                        html.Section(
                            [
                                _make_import_card(),
                                _make_controls_card(),
                            ],
                            className="control-grid",
                        ),
                        html.Section([_make_data_card()], className="data-section"),
                        html.Section(
                            [
                                _section_heading(
                                    "Output",
                                    "Resultater og konfliktsoner",
                                    "Når modellen kjøres, flyttes fokus hit: dekning, avvik, mulige batcher og tidslinje.",
                                ),
                                dcc.Loading(
                                    html.Div(id="output", className="results-shell"),
                                    color="#d97706",
                                    type="default",
                                ),
                            ],
                            className="results-section",
                        ),
                    ]
                ),
                dcc.Store(id="data-store", data={"use_uploaded": False, "orders_only": False}),
                dcc.Store(
                    id="uploaded-store",
                    storage_type="session",
                    data={"fish_groups": None, "orders": None},
                ),
            ],
            fluid=True,
            className="app-shell",
        ),
    ]
)


# ==========================================
# CALLBACKS
# ==========================================


@app.callback(
    Output("active-source", "children"),
    Output("active-source-detail", "children"),
    Input("data-store", "data"),
)
def update_active_source(store_data):
    label = _format_source_label(store_data)
    if store_data.get("use_uploaded"):
        detail = "Scenarioet bruker opplastede data i denne økten."
    else:
        detail = "Scenarioet bruker de innebygde eksempeldataene."
    return label, detail


@app.callback(
    [Output("fish-table-container", "children"), Output("orders-table-container", "children")],
    [Input("data-store", "data")],
    [State("uploaded-store", "data")],
)
def update_tables(store_data, uploaded_store):
    fish_df, orders_df = _resolve_active_data(store_data, uploaded_store)

    fish_table = _build_table(
        fish_df,
        header_background="#1f4d48",
        conditional_styles=[
            {
                "if": {"filter_query": "{Organic} = true || {Organic} = True"},
                "backgroundColor": "#d9ead7",
                "color": "#18321c",
            },
        ],
        page_size=10,
    )

    orders_table = _build_table(
        orders_df,
        header_background="#8a4b08",
        conditional_styles=[
            {
                "if": {"filter_query": "{RequireOrganic} = true || {RequireOrganic} = True"},
                "backgroundColor": "#d9ead7",
                "color": "#18321c",
            },
            {
                "if": {"filter_query": "{LockedSite} != \"\""},
                "backgroundColor": "#f4d8d0",
                "color": "#4d1f16",
            },
            {
                "if": {"filter_query": "{LockedGroup} != \"\""},
                "backgroundColor": "#f4d8d0",
                "color": "#4d1f16",
            },
        ],
        page_size=10,
    )

    return fish_table, orders_table


@app.callback(
    Output("download-example", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_example(n_clicks):
    excel_bytes = generate_example_excel()
    return dcc.send_bytes(excel_bytes, "eggallokering_eksempel.xlsx")


@app.callback(
    Output("download-orders-example", "data"),
    Input("download-orders-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_orders_example(n_clicks):
    excel_bytes = generate_orders_example_excel()
    return dcc.send_bytes(excel_bytes, "ordrer_eksempel.xlsx")


@app.callback(
    [
        Output("upload-status", "children"),
        Output("data-store", "data", allow_duplicate=True),
        Output("uploaded-store", "data", allow_duplicate=True),
    ],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return "", {"use_uploaded": False, "orders_only": False}, {"fish_groups": None, "orders": None}

    fish_groups, orders, error = parse_uploaded_excel(contents, filename)

    if error:
        return (
            dbc.Alert(error, color="danger", className="status-alert mb-0"),
            {"use_uploaded": False, "orders_only": False},
            {"fish_groups": None, "orders": None},
        )

    return (
        dbc.Alert(
            f"{filename}: {len(fish_groups)} grupper, {len(orders)} ordrer",
            color="success",
            className="status-alert mb-0",
        ),
        {"use_uploaded": True, "orders_only": False},
        {"fish_groups": _df_to_store_json(fish_groups), "orders": _df_to_store_json(orders)},
    )


@app.callback(
    [
        Output("upload-orders-status", "children"),
        Output("data-store", "data", allow_duplicate=True),
        Output("uploaded-store", "data", allow_duplicate=True),
    ],
    Input("upload-orders-only", "contents"),
    State("upload-orders-only", "filename"),
    prevent_initial_call=True,
)
def handle_orders_upload(contents, filename):
    if contents is None:
        return "", {"use_uploaded": False, "orders_only": False}, {"fish_groups": None, "orders": None}

    orders, error = parse_orders_excel(contents, filename)

    if error:
        return (
            dbc.Alert(error, color="danger", className="status-alert mb-0"),
            {"use_uploaded": False, "orders_only": False},
            {"fish_groups": None, "orders": None},
        )

    return (
        dbc.Alert(
            f"{filename}: {len(orders)} ordrer med default fiskegrupper",
            color="success",
            className="status-alert mb-0",
        ),
        {"use_uploaded": True, "orders_only": True},
        {"fish_groups": None, "orders": _df_to_store_json(orders)},
    )


@app.callback(
    [
        Output("reset-status", "children"),
        Output("data-store", "data", allow_duplicate=True),
        Output("upload-status", "children", allow_duplicate=True),
        Output("upload-orders-status", "children", allow_duplicate=True),
        Output("uploaded-store", "data", allow_duplicate=True),
    ],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_data(n_clicks):
    return (
        dbc.Alert("Tilbakestilt til default data", color="info", className="status-alert mb-0"),
        {"use_uploaded": False, "orders_only": False},
        "",
        "",
        {"fish_groups": None, "orders": None},
    )


@app.callback(
    Output("output", "children"),
    Input("run-btn", "n_clicks"),
    State("data-store", "data"),
    State("uploaded-store", "data"),
    State("window-mode", "value"),
    State("growth-model", "value"),
    running=[(Output("run-btn", "disabled"), True, False)],
    prevent_initial_call=True,
)
def on_run(n_clicks, store_data, uploaded_store, window_mode, growth_model):
    try:
        fish_groups, orders = _resolve_active_data(store_data, uploaded_store)

        result = run_allocation(
            fish_groups,
            orders,
            window_mode=window_mode,
            growth_model=growth_model,
        )
        results_df = result["results"]
        possible_groups_df = result["possible_groups"]

        total = len(results_df)
        allocated = len(results_df[results_df["BatchID"] != "IKKE TILDELT"])
        not_allocated = total - allocated
        organic_orders = len(results_df[results_df["RequireOrganic"] == "✓"])
        preference_hits = len(results_df[results_df["PreferenceMatched"].fillna("") != ""])
        coverage_pct = round((allocated / total) * 100) if total else 0

        display_df = results_df.copy()
        display_df["DeliveryDate"] = pd.to_datetime(display_df["DeliveryDate"]).dt.strftime("%Y-%m-%d")
        display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{x:,.0f}")

        display_cols = [
            "OrderNr",
            "Customer",
            "BatchID",
            "Site",
            "Organic",
            "DegreeDays",
            "DeliveryDate",
            "Volume",
            "Product",
            "RequireOrganic",
            "PreferenceMatched",
            "Reason",
        ]
        display_df = display_df[display_cols]

        possible_display = possible_groups_df.copy()
        possible_display["Volume"] = possible_display["Volume"].apply(lambda x: f"{x:,.0f}")

        chart = _build_results_figure(result["chart"])

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Kjøring fullført", className="result-kicker"),
                                html.H3("Allokeringsbilde", className="result-title"),
                                html.P(
                                    f"Aktiv datakilde: {_format_source_label(store_data)}. "
                                    f"Modus: {'Uke' if window_mode == 'week' else 'Dag'}. "
                                    f"Vekstmodell: {growth_model}.",
                                    className="result-copy",
                                ),
                            ],
                            className="result-heading",
                        ),
                        html.Div(
                            [
                                html.Div(f"{coverage_pct}%", className="coverage-value"),
                                dbc.Progress(value=coverage_pct, color="warning", className="coverage-bar"),
                                html.Div("Dekningsgrad", className="coverage-label"),
                            ],
                            className="coverage-panel",
                        ),
                    ],
                    className="result-summary-row",
                ),
                html.Div(
                    [
                        _metric_card("Totalt", f"{total}", tone="ink", detail="ordrer evaluert"),
                        _metric_card("Tildelt", f"{allocated}", tone="forest", detail="ordrer med batch"),
                        _metric_card("Ikke tildelt", f"{not_allocated}", tone="rust", detail="krever manuell vurdering"),
                        _metric_card("Organic-krav", f"{organic_orders}", tone="forest", detail="må treffe organic"),
                        _metric_card("Preferanse-treff", f"{preference_hits}", tone="gold", detail="soft constraints matchet"),
                    ],
                    className="metric-grid",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Mulige grupper per ordre", className="panel-title-small"),
                                html.P(
                                    "Sorter på antall mulige batcher for å finne trange ordrer raskt.",
                                    className="panel-copy-small",
                                ),
                                _build_table(
                                    possible_display,
                                    header_background="#5b6573",
                                    conditional_styles=[
                                        {
                                            "if": {"filter_query": "{AntallMuligeBatcher} = 0"},
                                            "backgroundColor": "#f4d8d0",
                                            "color": "#4d1f16",
                                        },
                                        {
                                            "if": {"filter_query": "{AntallMuligeBatcher} = 1"},
                                            "backgroundColor": "#f6e6b8",
                                            "color": "#5b4200",
                                        },
                                        {
                                            "if": {"filter_query": "{AntallMuligeBatcher} > 5"},
                                            "backgroundColor": "#d9ead7",
                                            "color": "#18321c",
                                        },
                                    ],
                                    page_size=10,
                                ),
                            ],
                            className="result-panel",
                        ),
                        html.Div(
                            [
                                html.Div("Tidslinje", className="panel-title-small"),
                                html.P(
                                    "Tidslinjen er hovedflaten for å lese biologisk vindu, tildeling og konflikter.",
                                    className="panel-copy-small",
                                ),
                                html.Div(
                                    [
                                        html.Span("Organic", className="legend-chip legend-organic"),
                                        html.Span("Tildelt ordre", className="legend-chip legend-neutral"),
                                        html.Span("Leveringsuke i uke-modus", className="legend-chip legend-week"),
                                    ],
                                    className="legend-row",
                                ),
                                dcc.Graph(figure=chart, config={"displayModeBar": False}, className="timeline-graph"),
                            ],
                            className="result-panel timeline-panel",
                        ),
                    ],
                    className="result-two-column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Allokeringsresultater", className="panel-title-small"),
                                html.P(
                                    "Filtrer på `IKKE TILDELT` eller `PreferenceMatched` for å isolere avvik og gode treff.",
                                    className="panel-copy-small",
                                ),
                                _build_table(
                                    display_df,
                                    header_background="#1f4d48",
                                    conditional_styles=[
                                        {
                                            "if": {"filter_query": "{BatchID} = \"IKKE TILDELT\""},
                                            "backgroundColor": "#f4d8d0",
                                            "color": "#4d1f16",
                                        },
                                        {
                                            "if": {"filter_query": "{BatchID} != \"IKKE TILDELT\""},
                                            "backgroundColor": "#d9ead7",
                                            "color": "#18321c",
                                        },
                                        {
                                            "if": {"filter_query": "{RequireOrganic} = \"✓\""},
                                            "fontWeight": "700",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{PreferenceMatched} != \"\"",
                                                "column_id": "PreferenceMatched",
                                            },
                                            "backgroundColor": "#d7e4f4",
                                            "color": "#19324d",
                                        },
                                    ],
                                    page_size=20,
                                ),
                            ],
                            className="result-panel result-table-panel",
                        ),
                        html.Div(
                            [
                                html.Div("Eksport", className="panel-title-small"),
                                html.P(
                                    "Eksporter resultatene med input, genererte batcher og mulighetsmatrise i én arbeidsbok.",
                                    className="panel-copy-small",
                                ),
                                dbc.Button(
                                    "Last ned Excel",
                                    id="export-results-btn",
                                    color="dark",
                                    className="export-button",
                                ),
                                dcc.Download(id="download-results"),
                            ],
                            className="result-panel export-panel",
                        ),
                    ],
                    className="result-bottom-row",
                ),
            ],
            className="results-populated",
        )

    except Exception as e:
        traceback.print_exc()
        return dbc.Alert(
            [
                html.H4("Feil oppstod"),
                html.P(str(e)),
                html.Pre(traceback.format_exc()),
            ],
            color="danger",
            className="status-alert",
        )


@app.callback(
    Output("download-results", "data"),
    Input("export-results-btn", "n_clicks"),
    State("data-store", "data"),
    State("uploaded-store", "data"),
    State("window-mode", "value"),
    State("growth-model", "value"),
    prevent_initial_call=True,
)
def export_results(n_clicks, store_data, uploaded_store, window_mode, growth_model):
    try:
        fish_groups, orders = _resolve_active_data(store_data, uploaded_store)

        result = run_allocation(
            fish_groups,
            orders,
            window_mode=window_mode,
            growth_model=growth_model,
        )

        import io

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            results_export = result["results"].copy()
            results_export["DeliveryDate"] = pd.to_datetime(results_export["DeliveryDate"]).dt.strftime("%Y-%m-%d")
            results_export.to_excel(writer, sheet_name="Allokeringsresultater", index=False)

            result["possible_groups"].to_excel(writer, sheet_name="Mulige grupper per ordre", index=False)

            batches_export = result["batches"].copy()
            batches_export["StripDate"] = batches_export["StripDate"].dt.strftime("%Y-%m-%d")
            batches_export["MaturationEnd"] = batches_export["MaturationEnd"].dt.strftime("%Y-%m-%d")
            batches_export["ProductionEnd"] = batches_export["ProductionEnd"].dt.strftime("%Y-%m-%d")
            batches_export.to_excel(writer, sheet_name="Genererte batcher", index=False)

            fish_groups.to_excel(writer, sheet_name="Input - Fiskegrupper", index=False)
            orders.to_excel(writer, sheet_name="Input - Ordrer", index=False)

        output.seek(0)
        date_tag = pd.Timestamp.today().strftime("%Y-%m-%d")
        filename = f"allokeringsresultater_{window_mode}_{growth_model}_{date_tag}.xlsx"
        return dcc.send_bytes(output.getvalue(), filename)

    except Exception as e:
        print(f"Export error: {e}")
        return None


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8051))
    app.run(debug=False, host="0.0.0.0", port=port)
