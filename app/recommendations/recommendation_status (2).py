"""Recommendation lifecycle statuses and allowed transitions."""

from __future__ import annotations

from app.core.exceptions import CloudCostError
from app.database.models import RecommendationStatus

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    RecommendationStatus.NEW: {
        RecommendationStatus.UNDER_REVIEW,
        RecommendationStatus.ACCEPTED,
        RecommendationStatus.REJECTED,
    },
    RecommendationStatus.UNDER_REVIEW: {
        RecommendationStatus.ACCEPTED,
        RecommendationStatus.REJECTED,
        RecommendationStatus.NEW,
    },
    RecommendationStatus.ACCEPTED: {RecommendationStatus.COMPLETED, RecommendationStatus.UNDER_REVIEW},
    RecommendationStatus.REJECTED: {RecommendationStatus.UNDER_REVIEW},
    RecommendationStatus.COMPLETED: set(),
}


def validate_transition(current: str, new_status: str) -> None:
    if new_status not in {status.value for status in RecommendationStatus}:
        raise CloudCostError(f"Unknown recommendation status '{new_status}'")
    allowed = ALLOWED_TRANSITIONS.get(str(current), set())
    if new_status not in allowed:
        raise CloudCostError(
            f"Cannot move recommendation from '{current}' to '{new_status}'"
        )


def statuses() -> list[str]:
    return [status.value for status in RecommendationStatus]


def status_label(value: str) -> str:
    labels = {
        "new": "New",
        "under_review": "Under Review",
        "accepted": "Accepted",
        "rejected": "Rejected",
        "completed": "Completed",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())
