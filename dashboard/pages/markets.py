"""Market explorer: compare states or individual markets over any date range."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback

from dashboard.components import (add_period_shading, chart_card, control_card, data_grid, date_controls, empty_figure,
                                  fmt_money, insight, number_col, page_header, signed_style, style_figure)
from dashboard.data import MARKETS, NATIONAL, STATE_LIST, STATES, between, parse_range, pct_change

dash.register_page(__name__, path="/markets", name="Market Explorer", order=1, icon="tabler:building-store",
                   description="States and markets side by side")

MEASURES = {
    "price_close": ("Closing price", "₦ per litre"),
    "rebased": ("Rebased index (start of range = 100)", "Index"),
    "premium_pct": ("Premium over national index (%)", "%"),
    "yoy_pct": ("Year-on-year change (%)", "%"),
}

layout = dmc.Stack(
    [
        page_header("Market Explorer", "Compare how prices evolved in different states and markets.",
                    "tabler:building-store"),
        control_card(
            date_controls("mk")
            + [
                dmc.SegmentedControl(
                    id="mk-level", value="state", mb=2,
                    data=[{"value": "state", "label": "States"}, {"value": "market", "label": "Markets"}],
                ),
                dmc.MultiSelect(
                    id="mk-states", label="States", data=STATE_LIST, value=["Borno", "Kano", "Lagos"],
                    searchable=True, clearable=True, maxValues=8, w=300,
                    persistence=True, persistence_type="session",
                ),
                dmc.MultiSelect(
                    id="mk-markets", label="Markets (in the selected states)", data=[], value=[],
                    searchable=True, clearable=True, maxValues=8, w=300, placeholder="Pick markets",
                ),
                dmc.Select(
                    id="mk-measure", label="Measure", value="price_close", w=280, allowDeselect=False,
                    data=[{"value": k, "label": v[0]} for k, v in MEASURES.items()],
                ),
                dmc.Switch(id="mk-national", label="Show national index", checked=True, mb=6),
            ]
        ),
        dmc.Stack(id="mk-insights"),
        chart_card("mk-lines", "Price paths", "Each line is a state average or an individual market.", 440),
        dmc.Grid(
            [
                dmc.GridCol(
                    dmc.Stack([
                        dmc.Select(id="mk-candle-series", label="Candlestick for", searchable=True, w=320,
                                   allowDeselect=False),
                        chart_card("mk-candles", "Monthly open / high / low / close",
                                   "Wide candles mean large price swings within the month.", 360),
                    ], gap="xs"),
                    span={"base": 12, "lg": 7},
                ),
                dmc.GridCol(chart_card("mk-ranking", "Change over the selected range", "", 410),
                            span={"base": 12, "lg": 5}),
            ]
        ),
        data_grid(
            "mk-grid",
            [
                {"field": "series", "headerName": "State / market", "pinned": "left", "minWidth": 160},
                {"field": "month", "headerName": "Month", "sort": "desc"},
                number_col("price_open", "Open"),
                number_col("price_high", "High"),
                number_col("price_low", "Low"),
                number_col("price_close", "Close"),
                number_col("mom_pct", "MoM (%)", "+.2f", cellStyle=signed_style("mom_pct")),
                number_col("yoy_pct", "YoY (%)", "+.1f", cellStyle=signed_style("yoy_pct")),
                number_col("premium_pct", "Premium vs national (%)", "+.1f", cellStyle=signed_style("premium_pct")),
            ],
            title="Monthly data for the selection",
            description="Sort, filter (including the floating filter row) and export.",
        ),
    ],
    gap="md",
)


@callback(
    Output("mk-markets", "data"),
    Output("mk-markets", "value"),
    Output("mk-markets", "disabled"),
    Input("mk-states", "value"),
    Input("mk-level", "value"),
    Input("mk-markets", "value"),
)
def market_options(states, level, current):
    states = states or []
    subset = MARKETS[MARKETS["state"].isin(states)][["geo_id", "market", "state"]].drop_duplicates()
    groups = [
        {"group": s, "items": [{"value": r.geo_id, "label": f"{r.market} ({s})"}
                               for r in subset[subset["state"] == s].sort_values("market").itertuples()]}
        for s in states
    ]
    valid = set(subset["geo_id"])
    kept = [m for m in (current or []) if m in valid]
    if level == "market" and not kept:
        kept = [g["items"][0]["value"] for g in groups if g["items"]][:4]
    return groups, kept, level != "market"


def _series(level, states, markets, start, end):
    """Long frame with a 'series' label column for the chosen states or markets."""
    if level == "market":
        df = between(MARKETS[MARKETS["geo_id"].isin(markets or [])], start, end).copy()
        df["series"] = df["market"] + " (" + df["state"] + ")"
        df["mom_pct"] = df["mom_change_pct"]
        df = df.sort_values(["geo_id", "month"])
        full = MARKETS[MARKETS["geo_id"].isin(markets or [])].sort_values(["geo_id", "month"]).copy()
        full["yoy_pct"] = full.groupby("geo_id")["price_close"].pct_change(12) * 100
        full = full.merge(NATIONAL[["month", "price_close"]].rename(columns={"price_close": "national"}), on="month")
        full["premium_pct"] = 100 * (full["price_close"] / full["national"] - 1)
        df = df.merge(full[["geo_id", "month", "yoy_pct", "premium_pct"]], on=["geo_id", "month"])
        key = "geo_id"
    else:
        df = between(STATES[STATES["state"].isin(states or [])], start, end).copy()
        df["series"] = df["state"]
        key = "state"
    if not df.empty:
        df["rebased"] = df.groupby(key)["price_close"].transform(lambda s: 100 * s / s.iloc[0])
    return df


@callback(
    Output("mk-lines", "figure"),
    Output("mk-ranking", "figure"),
    Output("mk-grid", "rowData"),
    Output("mk-insights", "children"),
    Output("mk-candle-series", "data"),
    Output("mk-candle-series", "value"),
    Input("mk-dates", "value"),
    Input("mk-level", "value"),
    Input("mk-states", "value"),
    Input("mk-markets", "value"),
    Input("mk-measure", "value"),
    Input("mk-national", "checked"),
    Input("mk-candle-series", "value"),
)
def update(dates, level, states, markets, measure, show_national, candle_choice):
    start, end = parse_range(dates)
    df = _series(level, states, markets, start, end)
    if df.empty:
        msg = "Select at least one state" if level == "state" else "Select at least one market"
        return empty_figure(msg), empty_figure(msg), [], [], [], None

    label, unit = MEASURES[measure]
    fig = px.line(df, x="month", y=measure, color="series", color_discrete_sequence=px.colors.qualitative.Bold)
    if show_national and measure in ("price_close", "rebased", "yoy_pct"):
        nat = between(NATIONAL, start, end)
        y = {"price_close": nat["price_close"], "yoy_pct": nat["yoy_pct"],
             "rebased": 100 * nat["price_close"] / nat["price_close"].iloc[0]}[measure]
        fig.add_scatter(x=nat["month"], y=y, name="National index", line=dict(color="black", width=2.5, dash="dash"))
    if measure == "premium_pct":
        fig.add_hline(y=0, line_color="black", line_width=1)
    add_period_shading(fig, start, end)
    style_figure(fig, unit)
    fig.update_traces(hovertemplate="%{y:,.2f}")

    change = (df.groupby("series")["price_close"].agg(["first", "last"])
              .assign(change=lambda d: 100 * (d["last"] / d["first"] - 1)).sort_values("change"))
    nat_range = between(NATIONAL, start, end)["price_close"]
    nat_change = pct_change(nat_range.iloc[0], nat_range.iloc[-1])
    rank = go.Figure(go.Bar(x=change["change"], y=change.index, orientation="h",
                            marker_color=["#D85A30" if v > nat_change else "#1D9E75" for v in change["change"]],
                            text=[f"{v:+.0f}%" for v in change["change"]], textposition="auto"))
    rank.add_vline(x=nat_change, line_dash="dash", annotation_text=f"National {nat_change:+.0f}%",
                   annotation_position="bottom right")
    style_figure(rank, legend_top=False).update_layout(hovermode="closest", xaxis_title="% change")

    rows = df.assign(month=df["month"].dt.strftime("%Y-%m"))[
        ["series", "month", "price_open", "price_high", "price_low", "price_close", "mom_pct", "yoy_pct",
         "premium_pct"]].to_dict("records")

    options = sorted(df["series"].unique())
    candle_value = candle_choice if candle_choice in options else options[0]

    top, bottom = change.index[-1], change.index[0]
    latest = df[df["month"] == df["month"].max()].sort_values("price_close")
    insights = dmc.SimpleGrid(
        [
            insight(f"{top} rose most ({change.loc[top, 'change']:+.0f}%) and {bottom} least "
                    f"({change.loc[bottom, 'change']:+.0f}%) between {start:%b %Y} and {end:%b %Y}. "
                    f"The national index changed {nat_change:+.0f}%.",
                    title="Fastest vs slowest", icon_name="tabler:arrows-diff"),
            insight(f"In {latest['month'].max():%b %Y}, {latest['series'].iloc[-1]} was the most expensive "
                    f"({fmt_money(latest['price_close'].iloc[-1])}/L) and {latest['series'].iloc[0]} the "
                    f"cheapest ({fmt_money(latest['price_close'].iloc[0])}/L), a gap of "
                    f"{pct_change(latest['price_close'].iloc[0], latest['price_close'].iloc[-1]):.0f}%.",
                    title="Latest gap", color="orange", icon_name="tabler:scale"),
        ],
        cols={"base": 1, "md": 2},
    )
    return fig, rank, rows, insights, options, candle_value


@callback(
    Output("mk-candles", "figure"),
    Input("mk-candle-series", "value"),
    Input("mk-dates", "value"),
    Input("mk-level", "value"),
    Input("mk-states", "value"),
    Input("mk-markets", "value"),
)
def candles(series, dates, level, states, markets):
    start, end = parse_range(dates)
    df = _series(level, states, markets, start, end)
    df = df[df["series"] == series]
    if df.empty:
        return empty_figure("Choose a series")
    fig = go.Figure(go.Candlestick(x=df["month"], open=df["price_open"], high=df["price_high"],
                                   low=df["price_low"], close=df["price_close"], name=series,
                                   increasing_line_color="#D85A30", decreasing_line_color="#1D9E75"))
    style_figure(fig, "₦ per litre", legend_top=False).update_layout(xaxis_rangeslider_visible=False)
    return fig

