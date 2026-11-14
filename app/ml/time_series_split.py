"""Chronological, rolling-origin splits for time-series forecasting.

Splits are made on *target dates*: every training label must have been
observed before the first validation label. This prevents the classic
multi-horizon leak where a training row's target date falls inside the
validation window. Random splitting is never used.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TimeSeriesFold:
    name: str
    kind: str  # "validation" or "test"
    train_index: np.ndarray
    validation_index: np.ndarray
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None
    history_days: int = 0

    @property
    def train_rows(self) -> int:
        return int(len(self.train_index))

    @property
    def validation_rows(self) -> int:
        return int(len(self.validation_index))

    def to_dict(self) -> dict[str, Any]:
        def iso(value: pd.Timestamp | None) -> str | None:
            return None if value is None else pd.Timestamp(value).date().isoformat()

        return {
            "name": self.name,
            "kind": self.kind,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "train_start": iso(self.train_start),
            "train_end": iso(self.train_end),
            "validation_start": iso(self.validation_start),
            "validation_end": iso(self.validation_end),
            "history_days": self.history_days,
        }


@dataclass(slots=True)
class SplitPlan:
    folds: list[TimeSeriesFold] = field(default_factory=list)
    final_test: TimeSeriesFold | None = None
    messages: list[str] = field(default_factory=list)
    horizon_days: int = 30
    min_train_days: int = 90

    @property
    def is_valid(self) -> bool:
        return bool(self.folds) and self.final_test is not None

    def describe(self) -> list[dict[str, Any]]:
        rows = [fold.to_dict() for fold in self.folds]
        if self.final_test is not None:
            rows.append(self.final_test.to_dict())
        return rows


def build_rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    horizon_days: int = 30,
    validation_folds: int = 3,
    min_train_days: int = 90,
    min_validation_rows: int = 1,
) -> SplitPlan:
    """Create expanding-window folds plus one untouched final test window.

    The final test window is the last ``horizon_days`` calendar days of target
    dates. Validation folds immediately precede it, each exactly one horizon
    long, so every fold simulates the production 30-day forecast.
    """
    plan = SplitPlan(horizon_days=horizon_days, min_train_days=min_train_days)
    if frame is None or frame.empty:
        plan.messages.append("No supervised rows are available for splitting.")
        return plan
    if "target_date" not in frame.columns:
        plan.messages.append("Supervised rows lack the target_date column.")
        return plan

    target_dates = pd.to_datetime(frame["target_date"]).dt.normalize()
    if target_dates.isna().any():
        plan.messages.append("Some supervised rows have an invalid target_date.")
        return plan
    dates = target_dates.drop_duplicates().sort_values().reset_index(drop=True)
    first_date = dates.iloc[0]
    last_date = dates.iloc[-1]
    span_days = int((last_date - first_date).days) + 1

    required = min_train_days + (validation_folds + 1) * horizon_days
    if span_days < required:
        plan.messages.append(
            f"Target dates span {span_days} days; {required} are required for "
            f"{validation_folds} validation folds, a test window and {min_train_days} training days."
        )
        return plan

    test_start = last_date - pd.Timedelta(days=horizon_days - 1)
    test_mask = (target_dates >= test_start).to_numpy()
    test_index = np.flatnonzero(test_mask)
    if len(test_index) < min_validation_rows:
        plan.messages.append("The final test window contains no supervised rows.")
        return plan
    test_train_mask = (target_dates < test_start).to_numpy()
    test_train_index = np.flatnonzero(test_train_mask)
    if not _train_window_ok(target_dates, test_train_index, test_start, min_train_days):
        plan.messages.append(
            f"The training window before the final test starts on {test_start.date()} and does not "
            f"cover {min_train_days} days."
        )
        return plan
    plan.final_test = _make_fold(
        "final_test",
        "test",
        frame,
        test_train_index,
        test_index,
        test_start,
        last_date,
    )

    # Validation folds walk backwards from the test start.
    fold_end = test_start - pd.Timedelta(days=1)
    for number in range(validation_folds, 0, -1):
        fold_start = fold_end - pd.Timedelta(days=horizon_days - 1)
        train_mask = (target_dates < fold_start).to_numpy()
        validation_mask = ((target_dates >= fold_start) & (target_dates <= fold_end)).to_numpy()
        train_index = np.flatnonzero(train_mask)
        validation_index = np.flatnonzero(validation_mask)
        if (
            len(train_index) == 0
            or len(validation_index) < min_validation_rows
            or not _train_window_ok(target_dates, train_index, fold_start, min_train_days)
        ):
            plan.messages.append(
                f"Validation fold {number} ({fold_start.date()} to {fold_end.date()}) "
                "could not be constructed with enough history."
            )
            break
        plan.folds.append(
            _make_fold(
                f"fold_{number}",
                "validation",
                frame,
                train_index,
                validation_index,
                fold_start,
                fold_end,
            )
        )
        fold_end = fold_start - pd.Timedelta(days=1)

    plan.folds.sort(key=lambda fold: fold.validation_start)  # chronological
    if not plan.folds:
        plan.final_test = None
        plan.messages.append("No valid rolling-origin validation fold could be built.")
        return plan

    plan.messages.append(
        f"{len(plan.folds)} validation folds from {plan.folds[0].validation_start.date()} to "
        f"{plan.folds[-1].validation_end.date()}; untouched final test "
        f"{plan.final_test.validation_start.date()} to {plan.final_test.validation_end.date()}."
    )
    return plan


def select_rows(frame: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    """Positional row selection, preserving the frame's index order."""
    return frame.iloc[index]


def split_by_origin(
    frame: pd.DataFrame,
    *,
    origin_date: dt.date | str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (history, future) around a forecast origin.

    A row is in ``future`` when its target date is after ``origin_date``. This
    is used by notebooks and smoke tests to confirm the leakage guard.
    """
    cut = pd.Timestamp(origin_date).normalize()
    targets = pd.to_datetime(frame["target_date"]).dt.normalize()
    return frame.loc[targets <= cut].copy(), frame.loc[targets > cut].copy()


def _make_fold(
    name: str,
    kind: str,
    frame: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> TimeSeriesFold:
    targets = pd.to_datetime(frame["target_date"])
    train_targets = targets.iloc[train_index]
    train_start = train_targets.min() if len(train_index) else None
    train_end = train_targets.max() if len(train_index) else None
    history_days = (
        int((validation_start - train_start).days) if train_start is not None else 0
    )
    return TimeSeriesFold(
        name=name,
        kind=kind,
        train_index=train_index,
        validation_index=validation_index,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        history_days=history_days,
    )


def _train_window_ok(
    target_dates: pd.Series,
    train_index: np.ndarray,
    validation_start: pd.Timestamp,
    min_train_days: int,
) -> bool:
    if len(train_index) == 0:
        return False
    earliest = target_dates.iloc[train_index].min()
    return int((validation_start - earliest).days) >= min_train_days
