"""Filter state for the dashboard. Pure: no I/O and no streamlit, so the app binds
its widgets to these dataclasses and `gui.data` does the loading.

`GlobalFilters` cut the loaded dataset down to a slice (through
`analysis.tables.Dataset.select`); `EpisodeFilters` only narrow the rows of the
episode table, and never change the numbers on the page.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from analysis.tables import Dataset, Episode

from gui.data import ALL, EpisodeRow, rows_by_harm, rows_by_outcome

RECORD_FIELDS = {"model": "model", "agent": "agent_role", "control": "control_condition",
                 "drift": "drift_condition"}


@dataclass(frozen=True)
class GlobalFilters:
    """The slice of the data every panel shows. "all" (or "") means no choice."""

    model: str = ALL
    agent: str = ALL
    control: str = ALL
    drift: str = ALL
    run: str = ALL

    def active(self) -> dict[str, str]:
        """The record fields worth selecting on, i.e. the ones the user picked."""
        chosen = {"model": self.model, "agent": self.agent, "control": self.control, "drift": self.drift}
        return {RECORD_FIELDS[field]: value for field, value in chosen.items() if value and value != ALL}

    def matches(self, episode: Episode) -> bool:
        record = episode.record
        return self.run in (ALL, record.run_id) and all(
            getattr(record, field) == value for field, value in self.active().items())

    def episodes(self, dataset: Dataset) -> list[Episode]:
        """The episodes of `dataset` in this slice, in dataset order."""
        return [ep for ep in dataset.select(**self.active()) if self.matches(ep)]


@dataclass(frozen=True)
class EpisodeFilters:
    """Column filters of the episode table, applied to rows only."""

    search: str = ""
    outcome: str = ALL
    harm: str = ALL
    min_harm: int = 0
    model: str = ALL
    agent: str = ALL
    control: str = ALL
    drift: str = ALL

    def active(self) -> dict[str, str]:
        """The filters that are set, for listing them above the table."""
        picked = {"search": self.search.strip(), "outcome": self.outcome, "harm": self.harm,
                  "min_harm": str(self.min_harm), "model": self.model, "agent": self.agent,
                  "control": self.control, "drift": self.drift}
        return {k: v for k, v in picked.items() if v and v not in (ALL, "0")}

    def rows(self, rows: Sequence[EpisodeRow]) -> list[EpisodeRow]:
        """The rows of the episode table left after these filters."""
        out = rows_by_outcome(rows, self.outcome)
        out = rows_by_harm(out, harm=self.harm, min_harm=self.min_harm)
        needle = self.search.strip().lower()
        return [r for r in out
                if self.model in (ALL, r.model) and self.agent in (ALL, r.agent)
                and self.control in (ALL, r.control) and self.drift in (ALL, r.drift)
                and (not needle or needle in self.haystack(r))]

    def matches(self, row: EpisodeRow) -> bool:
        return bool(self.rows([row]))

    def haystack(self, row: EpisodeRow) -> str:
        """The text the search box is matched against."""
        return " ".join((row.episode_id, row.run_id, str(row.seed), row.model, row.agent, row.drift,
                         row.control, row.outcome)).lower()
