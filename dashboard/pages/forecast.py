"""Forecast models: benchmark comparison and the 3-month outlook."""

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback

from dashboard.components import (CARD, chart_card, control_card, data_grid, empty_figure, fmt_money, icon, insight,
                                  kpi_card, mark_date, number_col, page_header, style_figure)
from dashboard.data import FORECAST, MODEL_NAMES, NATIONAL, PREDICTIONS, forecast_metrics

dash.register_page(__name__, path="/forecast", name="Forecast Models", order=6, icon="tabler:crystal-ball",
                   description="LSTM vs benchmarks, outlook")

NAIVE = next(m for m in MODEL_NAMES if m.startswith("Naive"))
ARIMA = next(m for m in MODEL_NAMES if m.startswith("ARIMA"))
LSTM = next(m for m in MODEL_NAMES if m.startswith("LSTM"))
EVAL = PREDICTIONS[PREDICTIONS["split"] != "train"]
COLORS = dict(zip(MODEL_NAMES, px.colors.qualitative.Set2))
EVAL_START = EVAL["month"].min()

layout = dmc.Stack(
    [
        page_header("Forecast Models",
                    "One-month-ahead forecasts from five models on data they were not trained on, and the "
                    "3-month outlook. Metrics recalculate for whatever window you choose.", "tabler:crystal-ball"),
        control_card([
            dmc.Stack([
                dmc.Text("Evaluation set", size="sm", fw=500),
                dmc.SegmentedControl(id="fc-split", value="test", data=[
                    {"value": "validation", "label": "Validation (May 22 – Apr 24)"},
                    {"value": "test", "label": "Test (May 24 – Apr 26)"},
                    {"value": "both", "label": "Both"},
                ]),
            ], gap=4),
            dmc.DatePickerInput(id="fc-dates", label="Window within the set", type="range", clearable=True,
                                valueFormat="MMM YYYY", w=250, leftSection=icon("tabler:calendar"),
                                minDate=EVAL_START.strftime("%Y-%m-%d"),
                                maxDate=(EVAL["month"].max() + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                                placeholder="Whole set"),
            dmc.MultiSelect(id="fc-models", label="Models on the chart", data=MODEL_NAMES,
                            value=[NAIVE, ARIMA, LSTM], w=420, clearable=False),
            dmc.Select(id="fc-metric", label="Rank by", value="MAPE (%)", allowDeselect=False, w=150,
                       data=["MAPE (%)", "MAE (₦/L)", "RMSE (₦/L)"]),
        ]),
        dmc.Stack(id="fc-insights"),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("fc-lines", "Actual vs one-month-ahead forecasts", "", 430),
                            span={"base": 12, "lg": 7}),
                dmc.GridCol(chart_card("fc-cumulative", "Cumulative absolute error",
                                       "Lower is better. Where lines diverge shows when a model gained or lost "
                                       "ground.", 430), span={"base": 12, "lg": 5}),
            ]
        ),
        data_grid(
            "fc-grid",
            [
                {"field": "Model", "pinned": "left", "minWidth": 200},
                number_col("MAPE (%)", fmt=".3f"),
                number_col("MAE (₦/L)"),
                number_col("RMSE (₦/L)"),
                number_col("Bias (₦/L)", fmt="+.2f"),
                number_col("Direction correct (%)", fmt=".1f"),
                number_col("vs naive (%)", fmt="+.1f", cellStyle={"styleConditions": [
                    {"condition": "params.value < 0", "style": {"color": "#2b8a3e", "fontWeight": 600}},
                    {"condition": "params.value > 0", "style": {"color": "#c92a2a", "fontWeight": 600}}]}),
                number_col("Months", fmt=",d"),
            ],
            title="Accuracy in the selected window",
            description="'vs naive' is the MAPE change relative to the naive forecast (negative = better). "
                        "Significance tests for the full sets are in the notebook (K10).",
            height=340,
        ),
        dmc.Divider(label="3-month outlook", labelPosition="center", my="sm"),
        dmc.SimpleGrid(id="fc-kpis", cols={"base": 1, "sm": 3}),
        dmc.Grid(
            [
                dmc.GridCol(chart_card("fc-outlook", "Forward forecast",
                                       "ARIMA central forecast with 95% interval; LSTM path for comparison.", 400),
                            span={"base": 12, "lg": 8}),
                dmc.GridCol(dmc.Card([
                    dmc.Text("How to read the outlook", fw=600, mb="xs"),
                    dmc.List([
                        dmc.ListItem("ARIMA(1,0,0) is the recommended model: it was the most accurate on the "
                                     "test period and the only one significantly better than naive."),
                        dmc.ListItem("The interval widens quickly: by month 3 it spans about −12% to +14%."),
                        dmc.ListItem("The LSTM path assumes the exchange rate and Brent stay flat, and it "
                                     "underperformed the naive forecast on the test set."),
                        dmc.ListItem("A 'shock' flag appears if a forecast month rises by more than 5%."),
                    ], size="sm", spacing="xs"),
                ], **CARD), span={"base": 12, "lg": 4}),
            ]
        ),
    ],
    gap="md",
)


def _window(split, dates):
    df = EVAL if split == "both" else EVAL[EVAL["split"] == split]
    if dates and len(dates) == 2 and all(dates):
        a, b = (pd.Timestamp(d).to_period("M").to_timestamp() for d in dates)
        df = df[(df["month"] >= a) & (df["month"] <= b)]
    return df


@callback(
    Output("fc-lines", "figure"),
    Output("fc-cumulative", "figure"),
    Output("fc-grid", "rowData"),
    Output("fc-insights", "children"),
    Input("fc-split", "value"),
    Input("fc-dates", "value"),
    Input("fc-models", "value"),
    Input("fc-metric", "value"),
)
def update(split, dates, models, metric):
    df = _window(split, dates)
    models = models or [NAIVE]
    if len(df) < 2:
        msg = "Choose a window with at least two months"
        return empty_figure(msg), empty_figure(msg), [], []

    lines = go.Figure(go.Scatter(x=df["month"], y=df["actual"], name="Actual", mode="lines+markers",
                                 line=dict(color="black", width=3), hovertemplate="₦%{y:,.2f}"))
    cumulative = go.Figure()
    for m in models:
        lines.add_scatter(x=df["month"], y=df[m], name=m, line=dict(color=COLORS[m], width=2),
                          hovertemplate="₦%{y:,.2f}")
        cumulative.add_scatter(x=df["month"], y=(df["actual"] - df[m]).abs().cumsum(), name=m,
                               line=dict(color=COLORS[m], width=2.5), hovertemplate="₦%{y:,.0f}")
    if split == "both":
        boundary = PREDICTIONS.loc[PREDICTIONS["split"] == "test", "month"].min()
        for fig in (lines, cumulative):
            mark_date(fig, boundary, "test starts")
    style_figure(lines, "₦ per litre")
    style_figure(cumulative, "Cumulative |error| (₦/L)")

    table = pd.DataFrame([forecast_metrics(df, m) for m in MODEL_NAMES])
    naive_mape = table.loc[table["Model"] == NAIVE, "MAPE (%)"].iloc[0]
    table["vs naive (%)"] = 100 * (table["MAPE (%)"] / naive_mape - 1)
    table = table.sort_values(metric)

    best = table.iloc[0]
    lstm = table[table["Model"] == LSTM].iloc[0]
    beat_naive = table[table["vs naive (%)"] < 0]["Model"].tolist()
    notes = dmc.SimpleGrid([
        insight(f"{best['Model']} is the most accurate here ({metric} {best[metric]:.2f}), "
                f"{abs(best['vs naive (%)']):.0f}% {'better' if best['vs naive (%)'] < 0 else 'worse'} than naive on "
                f"MAPE over {int(best['Months'])} months.", title="Best in this window", icon_name="tabler:trophy"),
        insight(f"The LSTM ranks {table['Model'].tolist().index(LSTM) + 1} of {len(table)} "
                f"(MAPE {lstm['MAPE (%)']:.2f}% vs naive {naive_mape:.2f}%). "
                + (f"Models beating naive: {', '.join(beat_naive)}." if beat_naive else "No model beats naive here."),
                title="Deep learning vs simple rules", color="orange", icon_name="tabler:brain"),
    ], cols={"base": 1, "md": 2})
    return lines, cumulative, table.round(4).to_dict("records"), notes


@callback(Output("fc-outlook", "figure"), Output("fc-kpis", "children"), Input("fc-split", "value"))
def outlook(_):
    history = NATIONAL[NATIONAL["month"] >= "2024-01-01"]
    last = history.iloc[-1]
    fig = go.Figure()
    fig.add_scatter(x=history["month"], y=history["price_close"], name="Actual", line=dict(color="#1D9E75", width=3))
    band_x = [last["month"], *FORECAST["month"], *FORECAST["month"][::-1], last["month"]]
    band_y = [last["price_close"], *FORECAST["upper_95"], *FORECAST["lower_95"][::-1], last["price_close"]]
    fig.add_scatter(x=band_x, y=band_y, fill="toself", fillcolor="rgba(55,138,221,0.15)", line_width=0, mode="lines",
                    name="95% interval", hoverinfo="skip")
    fig.add_scatter(x=[last["month"], *FORECAST["month"]], y=[last["price_close"], *FORECAST["arima_forecast"]],
                    name="ARIMA forecast", mode="lines+markers", line=dict(color="#378ADD", width=3))
    fig.add_scatter(x=[last["month"], *FORECAST["month"]], y=[last["price_close"], *FORECAST["lstm_forecast"]],
                    name="LSTM path", mode="lines+markers", line=dict(color="#D85A30", width=2, dash="dash"))
    style_figure(fig, "₦ per litre")

    kpis = [
        kpi_card(f"{row.month:%B %Y}", fmt_money(row.arima_forecast) + "/L",
                 f"{row.change_vs_last_pct:+.1f}% vs {last['month']:%b %Y} · 95%: "
                 f"{fmt_money(row.lower_95)} – {fmt_money(row.upper_95)}",
                 color="red" if row.shock_alert == "yes" else "blue",
                 icon_name="tabler:alert-triangle" if row.shock_alert == "yes" else "tabler:calendar-month")
        for row in FORECAST.itertuples()
    ]
    return fig, kpis
