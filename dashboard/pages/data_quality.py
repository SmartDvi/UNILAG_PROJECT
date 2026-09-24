"""Data & methodology: coverage, quality and how the numbers are built."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, callback

from dashboard.components import (CARD, chart_card, control_card, data_grid, icon, insight, kpi_card, number_col,
                                  page_header, style_figure)
from dashboard.data import MARKET_INFO, MARKETS, STATE_LIST

dash.register_page(__name__, path="/data", name="Data & Method", order=7, icon="tabler:database",
                   description="Coverage, quality, caveats")

NE_STATES = ["Borno", "Yobe", "Adamawa"]
ne_share = 100 * MARKET_INFO["state"].isin(NE_STATES).mean()

STEPS = [
    ("Ingest", "Raw CSV (17,168 rows, 74 series, 232 months) loaded into DuckDB."),
    ("Clean", "7 aggregate series removed; 67 markets kept. YoY inflation left NULL for each series' first 12 "
              "months (no interpolation)."),
    ("Engineer", "SQL window functions: MoM change and intra-month range per market; state averages; "
                 "state-balanced national index (14 states weighted equally)."),
    ("Analyse", "Policy-period statistics, shock frequency, state premiums, PELT change-points, driver "
                "correlations."),
    ("Forecast", "Next-month log return; chronological train/validation/test split; LSTM ensemble vs naive, "
                 "drift, ARIMA and ridge; Diebold–Mariano tests."),
]

layout = dmc.Stack(
    [
        page_header("Data & Methodology", "What the data covers, how reliable it is, and how the dashboard's "
                    "numbers are built.", "tabler:database"),
        dmc.SimpleGrid(
            [
                kpi_card("Markets", f"{len(MARKET_INFO)}", f"in {MARKET_INFO['state'].nunique()} states",
                         icon_name="tabler:building-store"),
                kpi_card("Months", f"{MARKETS['month'].nunique()}",
                         f"{MARKETS['month'].min():%b %Y} – {MARKETS['month'].max():%b %Y}",
                         color="blue", icon_name="tabler:calendar"),
                kpi_card("North-east share", f"{ne_share:.0f}%", "of markets in Borno, Yobe, Adamawa",
                         color="orange", icon_name="tabler:compass"),
                kpi_card("Trust score range",
                         f"{MARKETS['trust_score'].min():.1f} – {MARKETS['trust_score'].max():.1f}",
                         "on a 0–10 scale: almost constant", color="grape", icon_name="tabler:shield-check"),
            ],
            cols={"base": 1, "sm": 2, "lg": 4},
        ),
        dmc.SimpleGrid(
            [
                insight("Prices are model-based estimates (World Bank Real-Time Prices layout). The modelled Lagos "
                        "price was ₦256/L in June 2023, well below reported official pump prices (about "
                        "₦488–568/L). Read naira levels as levels of this series, not as pump prices.",
                        title="Estimated, not observed", color="red", icon_name="tabler:alert-triangle"),
                insight(f"{ne_share:.0f}% of markets are in three north-eastern states. That is why every national "
                        "figure here uses a state-balanced index: an unweighted market average would mostly "
                        "describe Borno and Yobe.", title="Coverage is regional", color="orange",
                        icon_name="tabler:map-pin"),
            ],
            cols={"base": 1, "md": 2},
        ),
        control_card([
            dmc.MultiSelect(id="dq-states", label="States", data=STATE_LIST, value=[], clearable=True,
                            searchable=True, placeholder="All states", w=360),
            dmc.Select(id="dq-color", label="Colour map by", value="state", allowDeselect=False, w=220,
                       data=[{"value": "state", "label": "State"},
                             {"value": "latest_price", "label": "Latest price (₦/L)"},
                             {"value": "mean_trust", "label": "Mean trust score"}]),
        ]),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("dq-map", "Where the markets are", "Map tiles need an internet connection.",
                                       480), span={"base": 12, "lg": 7}),
                dmc.GridCol(chart_card("dq-bars", "Markets per state", "", 480), span={"base": 12, "lg": 5}),
            ]
        ),
        data_grid(
            "dq-grid",
            [
                {"field": "market", "headerName": "Market", "pinned": "left"},
                {"field": "lga", "headerName": "LGA"},
                {"field": "state", "headerName": "State"},
                number_col("lat", "Lat", ".2f"),
                number_col("lon", "Lon", ".2f"),
                number_col("months", "Months", ",d"),
                number_col("mean_trust", "Mean trust", ".2f"),
                number_col("latest_price", "Latest price (₦/L)"),
            ],
            title="Market register",
            description="Every market in the cleaned data. Each series is complete (232 months).",
        ),
        dmc.Card(
            [
                dmc.Text("Pipeline", fw=600, mb="sm"),
                dmc.Timeline(
                    [dmc.TimelineItem(dmc.Text(text, size="sm", c="dimmed"), title=title,
                                      bullet=icon("tabler:check", 14)) for title, text in STEPS],
                    active=len(STEPS) - 1, bulletSize=22, lineWidth=2,
                ),
                dmc.Text("Full code and interpretation: nigeria_fuel_price.ipynb. Sources: Real-Time Energy "
                         "Prices — Nigeria (HDX); USD/NGN (Investing.com); Brent crude (FRED DCOILBRENTEU).",
                         size="xs", c="dimmed", mt="md"),
            ],
            **CARD,
        ),
    ],
    gap="md",
)


@callback(
    Output("dq-map", "figure"),
    Output("dq-bars", "figure"),
    Output("dq-grid", "rowData"),
    Input("dq-states", "value"),
    Input("dq-color", "value"),
)
def update(states, color_by):
    df = MARKET_INFO if not states else MARKET_INFO[MARKET_INFO["state"].isin(states)]
    geo = df.dropna(subset=["lat", "lon"])
    fig = px.scatter_map(
        geo, lat="lat", lon="lon", color=color_by, hover_name="market",
        hover_data={"state": True, "lga": True, "latest_price": ":,.2f", "lat": False, "lon": False},
        zoom=4.6, center={"lat": 9.6, "lon": 8.3}, map_style="carto-positron",
        color_continuous_scale="Viridis", color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=11))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=0, bgcolor="rgba(255,255,255,0.7)"))

    counts = df.groupby("state").size().sort_values()
    bars = px.bar(x=counts.values, y=counts.index, orientation="h", text=counts.values,
                  color=counts.index.isin(NE_STATES),
                  color_discrete_map={True: "#D85A30", False: "#378ADD"})
    style_figure(bars, legend_top=False).update_layout(showlegend=False, xaxis_title="Markets", yaxis_title=None,
                                                       hovermode="closest")

    rows = df.assign(first_month=df["first_month"].dt.strftime("%Y-%m"),
                     last_month=df["last_month"].dt.strftime("%Y-%m")).to_dict("records")
    return fig, bars, rows
