"""Reusable layout pieces: cards, insight callouts, date controls, grids and figure styling."""

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, ctx, dcc, no_update
from dash_iconify import DashIconify

from .data import MAX_MONTH, MIN_MONTH, PERIOD_COLORS, PERIOD_SHORT, period_spans
from fuel_forecast.config import DOMESTIC_REFINING, SUBSIDY_REMOVAL

dmc.add_figure_templates(default="mantine_light")

CARD = {"withBorder": True, "radius": "md", "shadow": "xs", "p": "md"}


def icon(name: str, size: int = 18):
    return DashIconify(icon=name, width=size)


# ── Text blocks ─────────────────────────────────────────────────────────────
def page_header(title: str, description: str, icon_name: str):
    return dmc.Group(
        [
            dmc.ThemeIcon(icon(icon_name, 26), size=48, radius="md", variant="light"),
            dmc.Stack([dmc.Title(title, order=2), dmc.Text(description, c="dimmed", size="sm")], gap=2),
        ],
        mb="md",
        wrap="nowrap",
    )


def insight(children, title: str = "Insight", color: str = "teal", icon_name: str = "tabler:bulb"):
    return dmc.Alert(children, title=title, color=color, icon=icon(icon_name), radius="md", variant="light")


def kpi_card(label: str, value: str, note: str = "", color: str = "teal", icon_name: str = "tabler:gas-station"):
    return dmc.Card(
        dmc.Group(
            [
                dmc.Stack(
                    [
                        dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=700),
                        dmc.Text(value, size="xl", fw=700),
                        dmc.Text(note, size="xs", c="dimmed"),
                    ],
                    gap=2,
                ),
                dmc.ThemeIcon(icon(icon_name, 22), size=42, radius="md", variant="light", color=color),
            ],
            justify="space-between",
            align="flex-start",
            wrap="nowrap",
        ),
        **CARD,
    )


def chart_card(graph_id: str, title: str, description: str = "", height: int = 420):
    return dmc.Card(
        [
            dmc.Text(title, fw=600),
            dmc.Text(description, size="xs", c="dimmed") if description else None,
            dcc.Loading(dcc.Graph(id=graph_id, style={"height": height}, config={"displaylogo": False}), type="dot"),
        ],
        **CARD,
    )


def control_card(children):
    return dmc.Card(dmc.Group(children, align="flex-end", gap="md", grow=False), **CARD, mb="md")


# ── Date range control with quick presets ───────────────────────────────────
QUICK_RANGES = {
    "all": (MIN_MONTH, MAX_MONTH),
    "regulated": (MIN_MONTH, SUBSIDY_REMOVAL - pd.offsets.MonthBegin(1)),
    "post": (SUBSIDY_REMOVAL, MAX_MONTH),
    "refining": (DOMESTIC_REFINING, MAX_MONTH),
    "24m": (MAX_MONTH - pd.DateOffset(months=23), MAX_MONTH),
}


def _range_value(key: str) -> list[str]:
    start, end = QUICK_RANGES[key]
    return [start.strftime("%Y-%m-%d"), (end + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")]


def date_controls(prefix: str, default: str = "all"):
    """A range DatePickerInput plus a SegmentedControl of preset ranges, kept in sync."""
    date_id, quick_id = f"{prefix}-dates", f"{prefix}-quick"

    @callback(
        Output(date_id, "value"),
        Output(quick_id, "value"),
        Input(quick_id, "value"),
        Input(date_id, "value"),
    )
    def _sync(quick, dates):
        if ctx.triggered_id == quick_id and quick in QUICK_RANGES:
            return _range_value(quick), no_update
        matching = next((k for k in QUICK_RANGES if dates == _range_value(k)), "custom")
        return no_update, matching

    return [
        dmc.DatePickerInput(
            id=date_id,
            label="Date range",
            type="range",
            value=_range_value(default),
            minDate=MIN_MONTH.strftime("%Y-%m-%d"),
            maxDate=(MAX_MONTH + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
            valueFormat="MMM YYYY",
            leftSection=icon("tabler:calendar"),
            w=260,
            persistence=True,
            persistence_type="session",
        ),
        dmc.Stack(
            [
                dmc.Text("Quick range", size="sm", fw=500),
                dmc.SegmentedControl(
                    id=quick_id,
                    value=default,
                    size="xs",
                    data=[
                        {"value": "all", "label": "2007–2026"},
                        {"value": "regulated", "label": "Regulated"},
                        {"value": "post", "label": "Post-subsidy"},
                        {"value": "refining", "label": "Domestic refining"},
                        {"value": "24m", "label": "Last 24m"},
                        {"value": "custom", "label": "Custom", "disabled": True},
                    ],
                ),
            ],
            gap=4,
        ),
    ]


# ── Figures ─────────────────────────────────────────────────────────────────
def style_figure(fig: go.Figure, y_title: str = "", legend_top: bool = True) -> go.Figure:
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="") if legend_top else None,
        xaxis_title=None,
    )
    return fig


def add_period_shading(fig: go.Figure, start=None, end=None, **row_col) -> go.Figure:
    """Shade the three policy periods; label them (staggered) on single-panel charts."""
    for i, (first, last, period) in enumerate(period_spans(start, end)):
        fig.add_vrect(x0=first, x1=last, fillcolor=PERIOD_COLORS[period], opacity=0.08, line_width=0, **row_col)
        if not row_col:
            fig.add_annotation(x=first, y=1 - 0.05 * (i % 2), xref="x", yref="paper", text=PERIOD_SHORT[period],
                               showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, color="#666"))
    return fig


def mark_date(fig: go.Figure, x, text: str = "", color: str = "black", dash: str = "dash",
              vertical_text: bool = False, **row_col) -> go.Figure:
    """Vertical line at a date with a label (add_vline's own label fails on date axes)."""
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1, **row_col)
    if text:
        fig.add_annotation(x=x, y=0 if vertical_text else 1, yref="paper", text=text, showarrow=False,
                           textangle=-90 if vertical_text else 0, xanchor="right" if vertical_text else "left",
                           yanchor="bottom", font=dict(size=9 if vertical_text else 10, color=color))
    return fig


def log_ticks(low: float, high: float) -> list[float]:
    """Readable tick values (1, 2, 5 × 10^k) for a log axis spanning low..high."""
    import math
    ticks = []
    for k in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        ticks += [m * 10**k for m in (1, 2, 5) if low <= m * 10**k <= high]
    return ticks or [low, high]


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ── AG Grid ─────────────────────────────────────────────────────────────────
def number_col(field: str, header: str | None = None, fmt: str = ",.2f", **extra) -> dict:
    return {
        "field": field,
        "headerName": header or field,
        "type": "rightAligned",
        "filter": "agNumberColumnFilter",
        "valueFormatter": {"function": f"params.value == null ? '' : d3.format('{fmt}')(params.value)"},
        **extra,
    }


def signed_style(field: str) -> dict:
    """Colour positive values red (more expensive / worse) and negative values green."""
    return {
        "styleConditions": [
            {"condition": f"params.data.{field} > 0", "style": {"color": "#c92a2a", "fontWeight": 600}},
            {"condition": f"params.data.{field} < 0", "style": {"color": "#2b8a3e", "fontWeight": 600}},
        ]
    }


def data_grid(grid_id: str, column_defs: list[dict], row_data=None, page_size: int = 12, height: int = 430,
              title: str = "", description: str = ""):
    """An AG Grid inside a card, with sorting, filtering, pagination and CSV export."""
    button_id = f"{grid_id}-export"

    @callback(Output(grid_id, "exportDataAsCsv"), Input(button_id, "n_clicks"), prevent_initial_call=True)
    def _export(_):
        return True

    grid = dag.AgGrid(
        id=grid_id,
        columnDefs=column_defs,
        rowData=row_data or [],
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "floatingFilter": True, "minWidth": 110},
        columnSize="responsiveSizeToFit",
        dashGridOptions={"pagination": True, "paginationPageSize": page_size, "animateRows": True,
                         "paginationPageSizeSelector": sorted({page_size, 25, 50, 100})},
        csvExportParams={"fileName": f"{grid_id}.csv"},
        className="ag-theme-quartz",
        style={"height": height},
    )
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Stack([dmc.Text(title, fw=600), dmc.Text(description, size="xs", c="dimmed")], gap=0),
                    dmc.Button("Export CSV", id=button_id, size="xs", variant="light",
                               leftSection=icon("tabler:download", 14)),
                ],
                justify="space-between",
                mb="xs",
            ),
            grid,
        ],
        **CARD,
    )


def fmt_money(v: float) -> str:
    return f"₦{v:,.2f}"


def fmt_pct(v: float, signed: bool = True) -> str:
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"
