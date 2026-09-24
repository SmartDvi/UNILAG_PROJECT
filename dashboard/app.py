"""Nigeria Petrol Price Dashboard: a multipage Dash app.

Run after the notebook has produced outputs/tables. Any of these work:

    python main.py --dashboard         # from the project root
    python -m dashboard.app --debug    # from the project root, with hot reload
    python app.py                      # from inside the dashboard/ folder

Then open http://127.0.0.1:8050
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable however the app is started (script or module).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dash  # noqa: E402
import dash_mantine_components as dmc  # noqa: E402
from dash import Dash, Input, Output, callback  # noqa: E402

from dashboard.components import icon  # noqa: E402
from dashboard.data import MAX_MONTH  # noqa: E402

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    title="Nigeria Petrol Price Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server  # for deployment with gunicorn: gunicorn dashboard.app:server

THEME = {
    "primaryColor": "teal",
    "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    "defaultRadius": "md",
}


def navbar():
    pages = sorted(dash.page_registry.values(), key=lambda p: p["order"])
    return dmc.Stack(
        [
            dmc.NavLink(
                id={"type": "nav", "path": page["path"]},
                label=page["name"],
                description=page.get("description"),
                leftSection=icon(page.get("icon", "tabler:point")),
                href=page["path"],
                variant="light",
            )
            for page in pages
        ],
        gap=4,
    )


app.layout = dmc.MantineProvider(
    theme=THEME,
    children=dmc.AppShell(
        [
            dmc.AppShellHeader(
                dmc.Group(
                    [
                        dmc.Group(
                            [
                                dmc.ThemeIcon(icon("tabler:gas-station", 22), size=36, radius="md"),
                                dmc.Stack(
                                    [
                                        dmc.Title("Nigeria Petrol Price Dashboard", order=4),
                                        dmc.Text("Retail petrol prices, 2007 – 2026 · 67 markets in 14 states",
                                                 size="xs", c="dimmed"),
                                    ],
                                    gap=0,
                                ),
                            ],
                        ),
                        dmc.Badge(f"Data to {MAX_MONTH:%b %Y}", variant="light", size="lg"),
                    ],
                    justify="space-between",
                    h="100%",
                    px="md",
                ),
            ),
            dmc.AppShellNavbar(
                dmc.ScrollArea(
                    [
                        navbar(),
                        dmc.Divider(my="md"),
                        dmc.Text(
                            "Figures describe the state-balanced index of the covered markets. "
                            "Prices are model-based estimates, not official pump prices.",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                ),
                p="md",
            ),
            dmc.AppShellMain(dash.page_container),
        ],
        header={"height": 64},
        navbar={"width": 270, "breakpoint": "sm"},
        padding="lg",
    ),
)


@callback(
    Output({"type": "nav", "path": dash.ALL}, "active"),
    Input("_pages_location", "pathname"),
)
def highlight_active_page(pathname):
    return [page["path"] == pathname for page in sorted(dash.page_registry.values(), key=lambda p: p["order"])]


def main():
    parser = argparse.ArgumentParser(description="Run the dashboard.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
