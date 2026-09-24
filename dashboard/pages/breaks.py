"""Structural breaks: interactive PELT change-point detection."""

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback

from dashboard.components import (chart_card, control_card, data_grid, insight, log_ticks, mark_date, number_col,
                                  page_header, style_figure)
from dashboard.data import NATIONAL, STATE_LIST, pelt_breaks, series_for
from fuel_forecast.config import DOMESTIC_REFINING, SUBSIDY_REMOVAL

dash.register_page(__name__, path="/breaks", name="Structural Breaks", order=4, icon="tabler:git-branch",
                   description="Data-driven regime changes")

PENALTIES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MODELS = {"l2": "Mean shift (L2)", "l1": "Median shift (L1)", "rbf": "Distribution change (RBF kernel)"}
EVENTS = {
    pd.Timestamp("2012-01-01"): "Jan 2012 partial subsidy cut",
    pd.Timestamp("2016-05-01"): "May 2016 price cap ₦145",
    SUBSIDY_REMOVAL: "Jun 2023 subsidy removal",
    DOMESTIC_REFINING: "Sep 2024 Dangote petrol",
}

layout = dmc.Stack(
    [
        page_header("Structural Breaks",
                    "PELT (Killick et al., 2012) finds the months where the level of log prices shifts. Change "
                    "the settings to see how robust the breaks are.", "tabler:git-branch"),
        control_card([
            dmc.Select(id="br-series", label="Series", value="National index", searchable=True,
                       allowDeselect=False, w=220, data=["National index"] + STATE_LIST),
            dmc.Select(id="br-model", label="Cost function", value="l2", allowDeselect=False, w=260,
                       data=[{"value": k, "label": v} for k, v in MODELS.items()]),
            dmc.Stack([
                dmc.Text("Penalty (higher = fewer breaks)", size="sm", fw=500),
                dmc.SegmentedControl(id="br-penalty", value="1.0", size="xs",
                                     data=[{"value": str(p), "label": str(p)} for p in PENALTIES]),
            ], gap=4),
            dmc.NumberInput(id="br-minsize", label="Min. regime length (months)", value=6, min=3, max=36,
                            step=1, w=200),
            dmc.Switch(id="br-events", label="Show policy events", checked=True, mb=6),
        ]),
        dmc.Stack(id="br-insights"),
        chart_card("br-chart", "Detected regimes", "Shaded blocks are regimes; dashed lines are break months.", 470),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("br-sensitivity", "Penalty sensitivity",
                                       "Number of breaks for each penalty. A flat stretch means a stable result.",
                                       330), span={"base": 12, "lg": 5}),
                dmc.GridCol(
                    data_grid(
                        "br-grid",
                        [
                            {"field": "Regime", "pinned": "left", "maxWidth": 110},
                            {"field": "Start"}, {"field": "End"},
                            number_col("Months", fmt=",d"),
                            number_col("Mean price (₦/L)"),
                            number_col("Avg MoM (%)", fmt="+.2f"),
                            number_col("Volatility (%)", fmt=".2f"),
                            number_col("Price change (%)", fmt="+.0f"),
                            number_col("USD/NGN change (%)", fmt="+.0f"),
                        ],
                        title="Regime statistics", height=330, page_size=8,
                    ),
                    span={"base": 12, "lg": 7},
                ),
            ]
        ),
    ],
    gap="md",
)


@callback(
    Output("br-chart", "figure"),
    Output("br-sensitivity", "figure"),
    Output("br-grid", "rowData"),
    Output("br-insights", "children"),
    Input("br-series", "value"),
    Input("br-model", "value"),
    Input("br-penalty", "value"),
    Input("br-minsize", "value"),
    Input("br-events", "checked"),
)
def update(series_key, model, penalty, min_size, show_events):
    penalty = float(penalty)
    min_size = int(min_size or 6)
    s = series_for(series_key)
    breaks = list(pelt_breaks(series_key, model, penalty, min_size))
    fx = NATIONAL.set_index("month")["usd_ngn"]

    bounds = [s.index[0]] + breaks + [s.index[-1] + pd.offsets.MonthBegin(1)]
    palette = ["#378ADD", "#EF9F27", "#D85A30", "#7F77DD", "#1D9E75", "#888780"]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, name=series_key, line=dict(color="#1D9E75", width=2.5),
                               hovertemplate="₦%{y:,.2f}"))
    rows = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg = s[(s.index >= a) & (s.index < b)]
        fig.add_vrect(x0=a, x1=b, fillcolor=palette[i % len(palette)], opacity=0.1, line_width=0)
        fig.add_scatter(x=[seg.index[0], seg.index[-1]], y=[seg.mean()] * 2, mode="lines", showlegend=False,
                        line=dict(color=palette[i % len(palette)], width=2, dash="dot"),
                        hovertemplate=f"Regime {i + 1} mean ₦%{{y:,.0f}}<extra></extra>")
        mom = seg.pct_change() * 100
        fx_seg = fx.reindex(seg.index).dropna()
        rows.append({
            "Regime": i + 1, "Start": f"{seg.index[0]:%b %Y}", "End": f"{seg.index[-1]:%b %Y}",
            "Months": len(seg), "Mean price (₦/L)": seg.mean(), "Avg MoM (%)": mom.mean(),
            "Volatility (%)": mom.std(), "Price change (%)": 100 * (seg.iloc[-1] / seg.iloc[0] - 1),
            "USD/NGN change (%)": 100 * (fx_seg.iloc[-1] / fx_seg.iloc[0] - 1) if len(fx_seg) > 1 else np.nan,
        })
    for b in breaks:
        mark_date(fig, b, f"{b:%b %Y}")
    if show_events:
        for date, label in EVENTS.items():
            mark_date(fig, date, label, color="#c92a2a", dash="dot", vertical_text=True)
    style_figure(fig, "₦ per litre", legend_top=False)
    fig.update_yaxes(type="log", tickvals=log_ticks(s.min(), s.max()), tickformat=",")

    counts = [len(pelt_breaks(series_key, model, p, min_size)) for p in PENALTIES]
    sens = go.Figure(go.Scatter(x=[str(p) for p in PENALTIES], y=counts, mode="lines+markers+text",
                                text=counts, textposition="top center", line=dict(color="#378ADD", width=2),
                                marker=dict(size=[14 if p == penalty else 8 for p in PENALTIES])))
    style_figure(sens, legend_top=False).update_layout(xaxis_title="Penalty", yaxis_title="Breaks found",
                                                       hovermode="closest")

    stable = [p for p, c in zip(PENALTIES, counts) if c == len(breaks)]
    notes = [insight(
        f"{len(breaks)} break(s) found: {', '.join(f'{b:%b %Y}' for b in breaks) or 'none'}. The same number of "
        f"breaks appears for penalties {', '.join(map(str, stable))}"
        + (", so the result is fairly robust." if len(stable) >= 3 else ", so the result is sensitive to tuning."),
        title="Result", icon_name="tabler:git-branch")]
    if breaks:
        nearest = min(breaks, key=lambda b: abs((b - SUBSIDY_REMOVAL).days))
        gap = round((nearest - SUBSIDY_REMOVAL).days / 30.4)
        notes.append(insight(
            f"The break nearest to the June 2023 subsidy removal is {nearest:%B %Y}, "
            f"{abs(gap)} month(s) {'after' if gap > 0 else 'before' if gap < 0 else 'at'} the policy date. "
            "In this estimated series the lasting level shift does not coincide exactly with the policy date.",
            title="Policy date vs data", color="orange", icon_name="tabler:calendar-event"))
    return fig, sens, rows, dmc.SimpleGrid(notes, cols={"base": 1, "md": len(notes)})
