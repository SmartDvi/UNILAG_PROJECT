"""Price shocks and volatility: how often, how big, and how widespread."""

import dash
import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, callback
from plotly.subplots import make_subplots

from dashboard.components import (add_period_shading, chart_card, control_card, data_grid, date_controls, insight,
                                  kpi_card, number_col, page_header, style_figure)
from dashboard.data import (MARKETS, NATIONAL, PERIOD_SHORT, SPIKE_THRESHOLD, STATE_LIST, STATES, between,
                            parse_range, shock_streak)

dash.register_page(__name__, path="/volatility", name="Price Shocks", order=3, icon="tabler:bolt",
                   description="Cost shocks and MSME risk")

layout = dmc.Stack(
    [
        page_header("Price Shocks & Volatility",
                    "How often petrol prices jump by more than a chosen threshold in a month, and how widely the "
                    "jump is felt across markets. Useful for small-business cost planning.", "tabler:bolt"),
        control_card(
            date_controls("vo")
            + [
                dmc.Select(id="vo-scope", label="Series", value="National index", searchable=True,
                           allowDeselect=False, w=220, data=["National index"] + STATE_LIST,
                           persistence=True, persistence_type="session"),
                dmc.Stack([
                    dmc.Text("Shock threshold (MoM rise)", size="sm", fw=500),
                    dmc.Slider(id="vo-threshold", value=SPIKE_THRESHOLD, min=1, max=15, step=0.5, w=260,
                               marks=[{"value": v, "label": f"{v}%"} for v in (1, 5, 10, 15)], mb="md"),
                ], gap=4),
                dmc.Select(id="vo-window", label="Rolling window", value="6", allowDeselect=False, w=140,
                           data=[{"value": v, "label": f"{v} months"} for v in ("3", "6", "12", "24")]),
            ]
        ),
        dmc.SimpleGrid(id="vo-kpis", cols={"base": 1, "sm": 2, "lg": 4}),
        chart_card("vo-chart", "Month-on-month change, rolling volatility and breadth of shocks",
                   "Top: monthly change (red bars exceed the threshold). Bottom: rolling volatility and the "
                   "share of markets whose own price rose by more than the threshold.", 620),
        dmc.Stack(id="vo-insights"),
        data_grid(
            "vo-grid",
            [
                {"field": "month", "headerName": "Month", "sort": "desc", "pinned": "left"},
                {"field": "period", "headerName": "Period"},
                number_col("price_close", "Price (₦/L)"),
                number_col("mom_pct", "MoM (%)", "+.2f", cellStyle={"color": "#c92a2a", "fontWeight": 600}),
                number_col("breadth", "Markets above threshold (%)", ".0f"),
                {"field": "worst_market", "headerName": "Largest local rise"},
                number_col("worst_rise", "Largest local rise (%)", "+.1f"),
            ],
            title="Shock months",
            description="Months in which the selected series rose by more than the threshold.",
        ),
    ],
    gap="md",
)


def _breadth(threshold, state=None):
    m = MARKETS if state is None else MARKETS[MARKETS["state"] == state]
    m = m.dropna(subset=["mom_change_pct"])
    g = m.groupby("month")
    out = g["mom_change_pct"].apply(lambda s: 100 * (s > threshold).mean()).rename("breadth").to_frame()
    worst = m.loc[g["mom_change_pct"].idxmax()]
    out["worst_market"] = (worst["market"] + " (" + worst["state"] + ")").to_numpy()
    out["worst_rise"] = worst["mom_change_pct"].to_numpy()
    return out.reset_index()


@callback(
    Output("vo-kpis", "children"),
    Output("vo-chart", "figure"),
    Output("vo-grid", "rowData"),
    Output("vo-insights", "children"),
    Input("vo-dates", "value"),
    Input("vo-scope", "value"),
    Input("vo-threshold", "value"),
    Input("vo-window", "value"),
)
def update(dates, scope, threshold, window):
    start, end = parse_range(dates)
    window = int(window)
    state = None if scope == "National index" else scope
    base = NATIONAL if state is None else STATES[STATES["state"] == state]
    series = base[["month", "period", "price_close", "mom_pct"]].copy()
    series["rolling_vol"] = series["mom_pct"].rolling(window).std()
    series = series.merge(_breadth(threshold, state), on="month", how="left")
    df = between(series, start, end).copy()
    df["shock"] = df["mom_pct"] > threshold

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45],
                        specs=[[{}], [{"secondary_y": True}]])
    fig.add_bar(x=df["month"], y=df["mom_pct"], name="MoM change (%)",
                marker_color=np.where(df["shock"], "#D85A30", "#378ADD"), row=1, col=1,
                hovertemplate="%{y:+.2f}%")
    fig.add_hline(y=threshold, line_dash="dot", line_color="#D85A30", row=1, col=1,
                  annotation_text=f"{threshold:g}% threshold", annotation_position="top left")
    fig.add_scatter(x=df["month"], y=df["breadth"], fill="tozeroy", name="Markets above threshold (%)",
                    line=dict(color="rgba(216,90,48,0.6)", width=1), row=2, col=1, hovertemplate="%{y:.0f}%")
    fig.add_scatter(x=df["month"], y=df["rolling_vol"], name=f"{window}-month volatility (%)",
                    line=dict(color="#EF9F27", width=2.5), row=2, col=1, secondary_y=True,
                    hovertemplate="%{y:.2f}%")
    add_period_shading(fig, start, end, row="all", col=1)
    style_figure(fig)
    fig.update_yaxes(title_text="MoM (%)", row=1, col=1)
    fig.update_yaxes(title_text="Markets above threshold (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, secondary_y=True, showgrid=False)

    shocks = df[df["shock"]]
    n, months = len(shocks), df["mom_pct"].notna().sum()
    streak = shock_streak(df["shock"])
    probability = n / months * 100 if months else 0
    kpis = [
        kpi_card("Shock months", f"{n}", f"of {months} months ({probability:.1f}%)", "red", "tabler:bolt"),
        kpi_card("Largest monthly rise", f"{df['mom_pct'].max():+.1f}%",
                 f"{df.loc[df['mom_pct'].idxmax(), 'month']:%b %Y}" if months else "", "orange",
                 "tabler:arrow-big-up"),
        kpi_card("Longest shock streak", f"{streak} month{'s' if streak != 1 else ''}",
                 "consecutive months above threshold", "grape", "tabler:flame"),
        kpi_card("Avg markets hit", f"{df['breadth'].mean():.0f}%",
                 "share of markets above threshold per month", "blue", "tabler:building-store"),
    ]

    by_period = df.groupby("period").agg(months=("mom_pct", "size"), shocks=("shock", "sum"),
                                         breadth=("breadth", "mean"))
    lines = [
        f"{PERIOD_SHORT[p]}: {int(r.shocks)} of {int(r.months)} months "
        f"({100 * r.shocks / r.months:.0f}%), with {r.breadth:.0f}% of markets above the threshold on average."
        for p, r in by_period.iterrows()
    ]
    expected = probability / 100 * 12
    insights = dmc.SimpleGrid(
        [
            insight([dmc.Text(line, size="sm") for line in lines], title="Shock frequency by period",
                    icon_name="tabler:calendar-stats"),
            insight(f"At the frequency seen in this range, a business should plan for about {expected:.1f} "
                    f"month(s) a year in which the {scope.lower() if state is None else scope} price rises by "
                    f"more than {threshold:g}%. Even in calm months, local jumps are common: the largest single-"
                    f"market rise in the range was {df['worst_rise'].max():+.0f}%.",
                    title="Planning rule of thumb", color="orange", icon_name="tabler:briefcase"),
        ],
        cols={"base": 1, "md": 2},
    )

    rows = shocks.assign(month=shocks["month"].dt.strftime("%Y-%m"),
                         period=shocks["period"].map(PERIOD_SHORT))[
        ["month", "period", "price_close", "mom_pct", "breadth", "worst_market", "worst_rise"]].to_dict("records")
    return kpis, fig, rows, insights
