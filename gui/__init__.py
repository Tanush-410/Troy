"""Streamlit dashboard over run logs: read-only discovery, filtering and metrics.

`gui.data` is the whole data layer (runs, dataset, tables, detection, overview
numbers); `gui.filters` holds the pure filter state the app binds its widgets to.
Neither imports streamlit, so both are testable without it.
"""

from gui.data import (
    DashboardData,
    RunInfo,
    discover_runs,
    load_dashboard,
    load_dashboard_from_root,
)
from gui.filters import EpisodeFilters, GlobalFilters

__all__ = [
    "DashboardData",
    "EpisodeFilters",
    "GlobalFilters",
    "RunInfo",
    "discover_runs",
    "load_dashboard",
    "load_dashboard_from_root",
]
