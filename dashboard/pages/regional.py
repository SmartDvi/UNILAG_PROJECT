"""Regional disparity: which states and markets pay more, and how that changed."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback

from dashboard.components import (chart_card, control_card, data_grid, date_controls, insight, number_col,
                                  page_header, signed_style, style_figure)
from dashboard.data import MARKETS, NATIONAL, STATE_LIST, STATES, between, parse_range
from fuel_forecast.config import SUBSIDY_REMOVAL

dash.register_page(__name__, path="/regional", name="Regional Disparity", order=2, icon="tabler:map-2",
                   description="Who pays more, and where")

METRICS = {
    "premium_pct": "Premium over national index (%)",
    "price_close": "Average price (₦/L)",
    "change_pct": "Price change over the range (%)",
}

layout = dmc.Stack(
    [
        page_header("Regional Disparity", "Price premiums by state and market relative to the national index.",
                    "tabler:map-2"),
        control_card(
            date_controls("rg", default="post")
            + [
                dmc.Select(id="rg-metric", label="Metric", value="premium_pct", allowDeselect=False, w=290,
                           data=[{"value": k, "label": v} for k, v in METRICS.items()]),
                dmc.MultiSelect(id="rg-highlight", label="Highlight states on the map", data=STATE_LIST,
                                value=[], clearable=True, searchable=True, w=300,
                                placeholder="All states"),
            ]
        ),
        dmc.Stack(id="rg-insights"),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("rg-map", "Market map",
                                       "Each dot is a market, coloured by the chosen metric. Map tiles need an "
                                       "internet connection.", 520), span={"base": 12, "lg": 7}),
                dmc.GridCol(chart_card("rg-bars", "State ranking", "", 520), span={"base": 12, "lg": 5}),
            ]
        ),
        chart_card("rg-heatmap", "Premium over the national index, by state and year (%)",
                   "Red means more expensive than the national index. Rows are sorted by the latest year.", 470),
        data_grid(
            "rg-grid",
            [
                {"field": "state", "headerName": "State", "pinned": "left"},
                number_col("markets", "Markets", ",d"),
                number_col("price_close", "Avg price (₦/L)"),
                number_col("premium_pct", "Premium (%)", "+.1f", cellStyle=signed_style("premium_pct")),
                number_col("regulated_premium", "Regulated-era premium (%)", "+.1f",
                           cellStyle=signed_style("regulated_premium")),
                number_col("premium_shift", "Shift vs regulated era (pp)", "+.1f",
                           cellStyle=signed_style("premium_shift")),
                number_col("change_pct", "Price change over range (%)", "+.1f"),
                number_col("volatility", "MoM volatility (%)", ".2f"),
            ],
            title="State summary for the selected range",
            description="'Shift' is the change in premium, in percentage points, compared with 2007 – May 2023.",
            height=520,
            page_size=14,
        ),
    ],
    gap="md",
)

REGULATED_PREMIUM = STATES[STATES["month"] < SUBSIDY_REMOVAL].groupby("state")["premium_pct"].mean()


def state_summary(start, end):
    df = between(STATES, start, end)
    summary = df.groupby("state").agg(
        markets=("markets", "max"),
        price_close=("price_close", "mean"),
        premium_pct=("premium_pct", "mean"),
        first=("price_close", "first"),
        last=("price_close", "last"),
        volatility=("mom_pct", "std"),
    )
    summary["change_pct"] = 100 * (summary["last"] / summary["first"] - 1)
    summary["regulated_premium"] = REGULATED_PREMIUM
    summary["premium_shift"] = summary["premium_pct"] - summary["regulated_premium"]
    return summary.drop(columns=["first", "last"]).reset_index()


@callback(
    Output("rg-map", "figure"),
    Output("rg-bars", "figure"),
    Output("rg-heatmap", "figure"),
    Output("rg-grid", "rowData"),
    Output("rg-insights", "children"),
    Input("rg-dates", "value"),
    Input("rg-metric", "value"),
    Input("rg-highlight", "value"),
)
def update(dates, metric, highlight):
    start, end = parse_range(dates)
    summary = state_summary(start, end)

    # Market-level map
    m = between(MARKETS, start, end).merge(NATIONAL[["month", "price_close"]].rename(columns={"price_close": "nat"}),
                                           on="month")
    m["premium_pct"] = 100 * (m["price_close"] / m["nat"] - 1)
    markets = m.groupby(["geo_id", "market", "lga", "state", "lat", "lon"], as_index=False).agg(
        premium_pct=("premium_pct", "mean"), price_close=("price_close", "mean"),
        first=("price_close", "first"), last=("price_close", "last"))
    markets["change_pct"] = 100 * (markets["last"] / markets["first"] - 1)
    markets = markets.dropna(subset=["lat", "lon"])
    if highlight:
        markets = markets[markets["state"].isin(highlight)]
    diverging = metric == "premium_pct"
    map_fig = px.scatter_map(
        markets, lat="lat", lon="lon", color=metric, size=markets["price_close"].clip(lower=1),
        size_max=16, hover_name="market",
        hover_data={"state": True, "lga": True, "price_close": ":,.2f", "premium_pct": ":+.1f",
                    "change_pct": ":+.1f", "lat": False, "lon": False},
        color_continuous_scale="RdYlGn_r" if diverging else "Viridis",
        color_continuous_midpoint=0 if diverging else None,
        zoom=4.6, center={"lat": 9.6, "lon": 8.3}, map_style="carto-positron",
        labels={metric: METRICS[metric].split(" (")[0]},
    )
    map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar=dict(title="%" if metric != "price_close" else "₦/L", thickness=12))

    # State ranking
    ranked = summary.sort_values(metric)
    bar_colors = (["#D85A30" if v > 0 else "#1D9E75" for v in ranked[metric]] if diverging else "#378ADD")
    fmt = "{:+.1f}%" if metric != "price_close" else "₦{:,.0f}"
    bars = go.Figure(go.Bar(x=ranked[metric], y=ranked["state"], orientation="h", marker_color=bar_colors,
                            name=f"{start:%b %Y} – {end:%b %Y}",
                            text=[fmt.format(v) for v in ranked[metric]], textposition="auto"))
    if diverging:
        bars.add_scatter(x=ranked["regulated_premium"], y=ranked["state"], mode="markers", name="Regulated era",
                         marker=dict(color="black", size=8, symbol="diamond"))
    style_figure(bars, legend_top=True).update_layout(hovermode="closest", xaxis_title=METRICS[metric])

    # Heatmap by year (inside the selected range)
    yearly = between(STATES, start, end).assign(year=lambda d: d["month"].dt.year)
    heat = yearly.pivot_table(index="state", columns="year", values="premium_pct", aggfunc="mean")
    heat = heat.sort_values(heat.columns[-1])
    heat_fig = go.Figure(go.Heatmap(
        z=heat.values, x=[str(c) for c in heat.columns], y=heat.index, colorscale="RdYlGn_r", zmid=0,
        text=heat.round(0).values, texttemplate="%{text:+.0f}", textfont_size=9,
        colorbar=dict(title="%"), hovertemplate="%{y} · %{x}: %{z:+.1f}%<extra></extra>"))
    style_figure(heat_fig, legend_top=False).update_layout(hovermode="closest")

    # Insights
    top = summary.sort_values("premium_pct").iloc[-1]
    low = summary.sort_values("premium_pct").iloc[0]
    riser = summary.sort_values("premium_shift").iloc[-1]
    gap = 100 * (top["price_close"] / low["price_close"] - 1)
    spread_by_year = heat.max() - heat.min()
    insights = dmc.SimpleGrid(
        [
            insight(f"{top['state']} paid {top['premium_pct']:+.1f}% versus the national index and "
                    f"{low['state']} {low['premium_pct']:+.1f}%. On average a litre cost {gap:.0f}% more in "
                    f"{top['state']} than in {low['state']}.",
                    title="Most and least expensive", color="red", icon_name="tabler:map-pin"),
            insight(f"{riser['state']}'s premium moved most compared with the regulated era: "
                    f"{riser['regulated_premium']:+.1f}% → {riser['premium_pct']:+.1f}% "
                    f"({riser['premium_shift']:+.1f} percentage points).",
                    title="Biggest shift", color="orange", icon_name="tabler:arrow-big-up-lines"),
            insight(f"The gap between the dearest and cheapest state was {spread_by_year.iloc[0]:.0f} points in "
                    f"{spread_by_year.index[0]} and {spread_by_year.iloc[-1]:.0f} points in "
                    f"{spread_by_year.index[-1]}.",
                    title="Is the gap widening?", color="violet", icon_name="tabler:arrows-horizontal"),
        ],
        cols={"base": 1, "md": 3},
    )
    return map_fig, bars, heat_fig, summary.round(3).to_dict("records"), insights
