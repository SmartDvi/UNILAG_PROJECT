"""Overview: headline indicators and the national index over time."""

import dash
import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, callback
from plotly.subplots import make_subplots

from dashboard.components import (CARD, add_period_shading, chart_card, control_card, data_grid, date_controls,
                                  fmt_money, fmt_pct, insight, kpi_card, number_col, page_header, style_figure)
from dashboard.data import NATIONAL, PERIOD_SHORT, SPIKE_THRESHOLD, between, parse_range, pct_change

dash.register_page(__name__, path="/", name="Overview", order=0, icon="tabler:layout-dashboard",
                   description="Headline numbers")

OVERLAYS = {
    "usd_ngn": "USD/NGN exchange rate",
    "brent_ngn": "Brent crude in naira (₦/bbl)",
    "brent_usd": "Brent crude (US$/bbl)",
}

layout = dmc.Stack(
    [
        page_header("Overview", "Where petrol prices are now, how they got here, and what changed after subsidy "
                    "removal.", "tabler:layout-dashboard"),
        control_card(
            date_controls("ov")
            + [
                dmc.MultiSelect(
                    id="ov-overlay",
                    label="Overlay drivers (right axis)",
                    data=[{"value": k, "label": v} for k, v in OVERLAYS.items()],
                    value=["usd_ngn"],
                    clearable=True,
                    w=320,
                    persistence=True,
                    persistence_type="session",
                ),
                dmc.Switch(id="ov-log", label="Log scale", checked=False, mb=6),
            ]
        ),
        dmc.SimpleGrid(id="ov-kpis", cols={"base": 1, "sm": 2, "lg": 4}),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("ov-index", "State-balanced petrol price index",
                                       "Close, with the average intra-month low–high band. Shaded areas are "
                                       "policy periods.", 460), span={"base": 12, "lg": 8}),
                dmc.GridCol(dmc.Stack(id="ov-insights"), span={"base": 12, "lg": 4}),
            ]
        ),
        data_grid(
            "ov-period-grid",
            [
                {"field": "Period", "pinned": "left", "minWidth": 200},
                number_col("Months", fmt=",d"),
                number_col("Start (₦/L)"),
                number_col("End (₦/L)"),
                number_col("Change (%)", fmt="+.1f", cellStyle={"fontWeight": 600}),
                number_col("Avg MoM (%)", fmt="+.2f"),
                number_col("Volatility (%)", fmt=".2f"),
                number_col(f"Shock months (>{SPIKE_THRESHOLD:.0f}%)", fmt=",d"),
                number_col("Cross-state CV (%)", fmt=".1f"),
            ],
            title="Policy periods within the selected range",
            description="Volatility is the standard deviation of month-on-month changes. CV is the "
                        "coefficient of variation across states.",
            height=260,
        ),
    ],
    gap="md",
)


@callback(
    Output("ov-kpis", "children"),
    Output("ov-index", "figure"),
    Output("ov-insights", "children"),
    Output("ov-period-grid", "rowData"),
    Input("ov-dates", "value"),
    Input("ov-overlay", "value"),
    Input("ov-log", "checked"),
)
def update(dates, overlays, log_scale):
    start, end = parse_range(dates)
    df = between(NATIONAL, start, end)
    first, last = df.iloc[0], df.iloc[-1]

    yoy = last["yoy_pct"]
    kpis = [
        kpi_card(f"Index, {last['month']:%b %Y}", fmt_money(last["price_close"]) + "/L",
                 f"MoM {fmt_pct(last['mom_pct'])}" + (f" · YoY {fmt_pct(yoy)}" if np.isfinite(yoy) else ""),
                 icon_name="tabler:gas-station"),
        kpi_card("Change over range", fmt_pct(pct_change(first["price_close"], last["price_close"])),
                 f"{fmt_money(first['price_close'])} → {fmt_money(last['price_close'])}",
                 color="orange", icon_name="tabler:trending-up"),
        kpi_card(f"Shock months (>{SPIKE_THRESHOLD:.0f}% MoM)", f"{int((df['mom_pct'] > SPIKE_THRESHOLD).sum())}",
                 f"of {len(df)} months · volatility {df['mom_pct'].std():.2f}%",
                 color="red", icon_name="tabler:bolt"),
        kpi_card("Cross-state dispersion", f"{100 * last['cross_state_cv']:.1f}%",
                 f"CV in {last['month']:%b %Y} · range avg {100 * df['cross_state_cv'].mean():.1f}%",
                 color="violet", icon_name="tabler:map-pins"),
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=df["month"], y=df["price_high"], line_width=0, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=df["month"], y=df["price_low"], line_width=0, fill="tonexty",
                    fillcolor="rgba(55,138,221,0.15)", name="Low–high band", hoverinfo="skip")
    fig.add_scatter(x=df["month"], y=df["price_close"], name="Index (₦/L)", line=dict(color="#1D9E75", width=3),
                    hovertemplate="₦%{y:,.2f}")
    colors = {"usd_ngn": "#7F77DD", "brent_ngn": "#EF9F27", "brent_usd": "#D85A30"}
    for key in overlays or []:
        fig.add_scatter(x=df["month"], y=df[key], name=OVERLAYS[key], secondary_y=True,
                        line=dict(color=colors[key], width=1.6, dash="dot"), hovertemplate="%{y:,.1f}")
    add_period_shading(fig, start, end)
    style_figure(fig, "₦ per litre")
    fig.update_yaxes(type="log" if log_scale else "linear", secondary_y=False)
    fig.update_yaxes(title_text="Driver" if overlays else None, secondary_y=True, showgrid=False)

    rows = []
    for period, g in df.groupby("period"):
        rows.append({
            "Period": PERIOD_SHORT[period],
            "Months": len(g),
            "Start (₦/L)": g["price_close"].iloc[0],
            "End (₦/L)": g["price_close"].iloc[-1],
            "Change (%)": pct_change(g["price_close"].iloc[0], g["price_close"].iloc[-1]),
            "Avg MoM (%)": g["mom_pct"].mean(),
            "Volatility (%)": g["mom_pct"].std(),
            f"Shock months (>{SPIKE_THRESHOLD:.0f}%)": int((g["mom_pct"] > SPIKE_THRESHOLD).sum()),
            "Cross-state CV (%)": 100 * g["cross_state_cv"].mean(),
        })

    fastest = df.loc[df["mom_pct"].idxmax()] if df["mom_pct"].notna().any() else None
    doubling = _months_to_double(df)
    insights = [
        insight(
            [dmc.Text(f"Across {len(df)} months the index moved from {fmt_money(first['price_close'])} to "
                      f"{fmt_money(last['price_close'])} per litre, an average of "
                      f"{df['mom_pct'].mean():+.2f}% a month.", size="sm"),
             dmc.Text(f"At that pace prices double roughly every {doubling}." if doubling else "", size="sm")],
            title="Trend", icon_name="tabler:chart-line"),
    ]
    if fastest is not None:
        insights.append(insight(
            f"The sharpest monthly rise was {fastest['mom_pct']:+.1f}% in {fastest['month']:%B %Y}.",
            title="Biggest jump", color="red", icon_name="tabler:bolt"))
    if len(rows) > 1:
        calmest = min(rows, key=lambda r: r["Volatility (%)"])
        wildest = max(rows, key=lambda r: r["Volatility (%)"])
        insights.append(insight(
            f"Volatility was highest in the {wildest['Period']} period ({wildest['Volatility (%)']:.1f}% a month) "
            f"and lowest in the {calmest['Period']} period ({calmest['Volatility (%)']:.1f}%).",
            title="Regimes", color="orange", icon_name="tabler:activity"))
    insights.append(dmc.Card(dmc.Text(
        "Reading note: the index averages 14 states equally. Prices are model-based estimates and sit "
        "below reported official pump prices after June 2023.", size="xs", c="dimmed"), **CARD))

    return kpis, fig, insights, rows


def _months_to_double(df) -> str | None:
    g = df["mom_pct"].mean() / 100
    if not np.isfinite(g) or g <= 0:
        return None
    months = np.log(2) / np.log(1 + g)
    return f"{months / 12:.1f} years" if months >= 24 else f"{months:.0f} months"
