"""Price drivers: how petrol-price changes co-move with the exchange rate and crude oil."""

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback
from plotly.subplots import make_subplots
from scipy import stats

from dashboard.components import (chart_card, control_card, data_grid, date_controls, insight, log_ticks, number_col,
                                  page_header, style_figure)
from dashboard.data import NATIONAL, PERIOD_COLORS, PERIOD_SHORT, STATE_LIST, STATES, between, parse_range

dash.register_page(__name__, path="/drivers", name="Price Drivers", order=5, icon="tabler:currency-dollar",
                   description="Exchange rate and crude oil")

DRIVERS = {
    "usd_ngn": "USD/NGN exchange rate",
    "brent_ngn": "Brent crude in naira",
    "brent_usd": "Brent crude in US dollars",
}
MAX_LAG = 6

layout = dmc.Stack(
    [
        page_header("Price Drivers",
                    "Monthly percentage changes in petrol prices against changes in the exchange rate and crude "
                    "oil. Correlation is not causation: use this to see which signals move together, and when.",
                    "tabler:currency-dollar"),
        control_card(
            date_controls("dr")
            + [
                dmc.Select(id="dr-target", label="Petrol series", value="National index", searchable=True,
                           allowDeselect=False, w=200, data=["National index"] + STATE_LIST),
                dmc.Select(id="dr-driver", label="Driver", value="brent_ngn", allowDeselect=False, w=240,
                           data=[{"value": k, "label": v} for k, v in DRIVERS.items()]),
                dmc.Select(id="dr-lag", label="Driver lead (months)", value="1", allowDeselect=False, w=170,
                           data=[{"value": str(k), "label": f"{k} month{'s' if k != 1 else ''}"}
                                 for k in range(MAX_LAG + 1)]),
                dmc.MultiSelect(id="dr-periods", label="Periods", value=["0", "1", "2"], w=330,
                                data=[{"value": str(k), "label": v} for k, v in PERIOD_SHORT.items()]),
                dmc.Select(id="dr-window", label="Rolling window", value="24", allowDeselect=False, w=140,
                           data=[{"value": v, "label": f"{v} months"} for v in ("12", "24", "36")]),
            ]
        ),
        dmc.Stack(id="dr-insights"),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("dr-scatter", "Petrol change vs driver change",
                                       "One dot per month; lines are least-squares fits per period.", 440),
                            span={"base": 12, "lg": 6}),
                dmc.GridCol(chart_card("dr-lags", "Correlation by lead time",
                                       "How strongly the driver's change k months earlier lines up with this "
                                       "month's petrol change. Filled bars are significant at 5%.", 440),
                            span={"base": 12, "lg": 6}),
            ]
        ),
        chart_card("dr-rolling", "Rolling correlation and cumulative change",
                   "Top: rolling correlation at the selected lead. Bottom: both series rebased to 100.", 520),
        data_grid(
            "dr-grid",
            [
                {"field": "period", "headerName": "Period", "pinned": "left", "minWidth": 200},
                {"field": "driver", "headerName": "Driver"},
                number_col("lag", "Lead (months)", ",d"),
                number_col("months", "Months", ",d"),
                number_col("r", "Correlation r", "+.3f"),
                number_col("p", "p-value", ".3f", cellStyle={"styleConditions": [
                    {"condition": "params.value < 0.05", "style": {"color": "#2b8a3e", "fontWeight": 600}}]}),
                number_col("beta", "Elasticity (β)", "+.3f"),
            ],
            title="Correlation table for all drivers and leads",
            description="β is the slope of petrol % change on driver % change: a 1% driver move goes with β% in "
                        "petrol. p < 0.05 is shown in green.",
            page_size=15,
        ),
    ],
    gap="md",
)


def _frame(target):
    base = NATIONAL[["month", "period", "price_close", "usd_ngn", "brent_usd", "brent_ngn"]].copy()
    if target != "National index":
        state = STATES[STATES["state"] == target][["month", "price_close"]]
        base = base.drop(columns="price_close").merge(state, on="month")
    base["fuel"] = np.log(base["price_close"]).diff() * 100
    for key in DRIVERS:
        change = np.log(base[key]).diff() * 100
        for lag in range(MAX_LAG + 1):
            base[f"{key}_l{lag}"] = change.shift(lag)
    return base


def _corr(df, x):
    d = df[["fuel", x]].dropna()
    if len(d) < 4 or d[x].std() == 0:
        return np.nan, np.nan, np.nan, len(d)
    r, p = stats.pearsonr(d["fuel"], d[x])
    beta = np.polyfit(d[x], d["fuel"], 1)[0]
    return r, p, beta, len(d)


@callback(
    Output("dr-scatter", "figure"),
    Output("dr-lags", "figure"),
    Output("dr-rolling", "figure"),
    Output("dr-grid", "rowData"),
    Output("dr-insights", "children"),
    Input("dr-dates", "value"),
    Input("dr-target", "value"),
    Input("dr-driver", "value"),
    Input("dr-lag", "value"),
    Input("dr-periods", "value"),
    Input("dr-window", "value"),
)
def update(dates, target, driver, lag, periods, window):
    start, end = parse_range(dates)
    lag, window = int(lag), int(window)
    periods = [int(p) for p in (periods or ["0", "1", "2"])]
    full = _frame(target)
    df = between(full, start, end)
    df = df[df["period"].isin(periods)]
    x = f"{driver}_l{lag}"

    scatter = go.Figure()
    for period, g in df.dropna(subset=["fuel", x]).groupby("period"):
        scatter.add_scatter(x=g[x], y=g["fuel"], mode="markers", name=PERIOD_SHORT[period],
                            marker=dict(color=PERIOD_COLORS[period], size=7, opacity=0.75),
                            customdata=g["month"].dt.strftime("%b %Y"),
                            hovertemplate="%{customdata}<br>driver %{x:+.1f}% · petrol %{y:+.1f}%<extra></extra>")
        if len(g) >= 4:
            slope, intercept = np.polyfit(g[x], g["fuel"], 1)
            xs = np.linspace(g[x].min(), g[x].max(), 20)
            scatter.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", showlegend=False,
                                line=dict(color=PERIOD_COLORS[period], width=2))
    style_figure(scatter).update_layout(hovermode="closest",
                                        xaxis_title=f"{DRIVERS[driver]} change {lag}m earlier (%)",
                                        yaxis_title=f"{target} change (%)")

    lag_stats = [_corr(df, f"{driver}_l{k}") for k in range(MAX_LAG + 1)]
    lag_fig = go.Figure(go.Bar(
        x=[f"{k}m" for k in range(MAX_LAG + 1)], y=[s[0] for s in lag_stats],
        marker=dict(color=["#1D9E75" if s[1] < 0.05 else "rgba(29,158,117,0.25)" for s in lag_stats],
                    line=dict(color="#1D9E75", width=1.5)),
        text=[f"{s[0]:+.2f}" if np.isfinite(s[0]) else "" for s in lag_stats], textposition="outside",
        customdata=[s[1] for s in lag_stats], hovertemplate="r = %{y:+.3f}<br>p = %{customdata:.3f}<extra></extra>"))
    lag_fig.add_hline(y=0, line_color="black", line_width=1)
    style_figure(lag_fig, "Correlation r", legend_top=False).update_layout(hovermode="closest",
                                                                           xaxis_title="Driver lead")

    rolled = between(full, start, end)
    rolling_r = rolled["fuel"].rolling(window).corr(rolled[x])
    rebased = rolled.dropna(subset=[driver])
    roll = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    roll.add_scatter(x=rolled["month"], y=rolling_r, name=f"{window}m rolling r", line=dict(color="#7F77DD", width=2.5),
                     row=1, col=1, hovertemplate="%{y:+.2f}")
    roll.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)
    roll.add_scatter(x=rebased["month"], y=100 * rebased["price_close"] / rebased["price_close"].iloc[0],
                     name=f"{target} (rebased)", line=dict(color="#1D9E75", width=2.5), row=2, col=1)
    roll.add_scatter(x=rebased["month"], y=100 * rebased[driver] / rebased[driver].iloc[0],
                     name=f"{DRIVERS[driver]} (rebased)", line=dict(color="#EF9F27", width=2, dash="dot"),
                     row=2, col=1)
    style_figure(roll)
    roll.update_yaxes(title_text="r", range=[-1, 1], row=1, col=1)
    rebased_values = pd.concat([100 * rebased["price_close"] / rebased["price_close"].iloc[0],
                                100 * rebased[driver] / rebased[driver].iloc[0]])
    roll.update_yaxes(title_text="Index (start = 100)", type="log", row=2, col=1, tickformat=",",
                      tickvals=log_ticks(rebased_values.min(), rebased_values.max()))

    rows = []
    for period, g in df.groupby("period"):
        for key, label in DRIVERS.items():
            for k in range(MAX_LAG + 1):
                r, p, beta, n = _corr(g, f"{key}_l{k}")
                rows.append({"period": PERIOD_SHORT[period], "driver": label, "lag": k, "months": n,
                             "r": r, "p": p, "beta": beta})

    best_k = int(np.nanargmax([abs(s[0]) if np.isfinite(s[0]) else -1 for s in lag_stats]))
    r_sel, p_sel, beta_sel, n_sel = lag_stats[lag]
    sig = "statistically significant" if p_sel < 0.05 else "not statistically significant"
    table = pd.DataFrame(rows)
    strongest = table[table["p"] < 0.05].sort_values("r", key=abs, ascending=False).head(1)
    notes = [
        insight(f"At a {lag}-month lead the correlation is r = {r_sel:+.2f} (p = {p_sel:.3f}, {n_sel} months), "
                f"which is {sig}. A 10% move in the driver goes with about {10 * beta_sel:+.1f}% in petrol. "
                f"The strongest lead in this selection is {best_k} month(s) (r = {lag_stats[best_k][0]:+.2f}).",
                title="Selected relationship", icon_name="tabler:link"),
    ]
    if not strongest.empty:
        s = strongest.iloc[0]
        notes.append(insight(f"Across all drivers, leads and periods, the strongest significant link is "
                             f"{s['driver']} leading by {s['lag']} month(s) in the {s['period']} period "
                             f"(r = {s['r']:+.2f}, p = {s['p']:.3f}). With {len(table)} tests, about "
                             f"{0.05 * len(table):.0f} would pass p < 0.05 by chance alone, so treat single "
                             "results with caution.",
                             title="Strongest significant link", color="orange", icon_name="tabler:star"))
    else:
        notes.append(insight("No driver/lead combination is significant at 5% in this selection. Short "
                             "post-subsidy periods leave little statistical power.",
                             title="No significant link", color="gray", icon_name="tabler:info-circle"))
    return scatter, lag_fig, roll, table.round(4).to_dict("records"), dmc.SimpleGrid(notes, cols={"base": 1, "md": 2})
