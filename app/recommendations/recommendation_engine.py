"""Recommendation engine: runs transparent rules per currency/cost-basis cohort."""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd

from app.core.config import AppConfig, get_app_config
from app.database.models import RecommendationRecord, RecommendationStatus
from app.database.repositories import RecommendationRepository, SettingsRepository
from app.recommendations.rules import RULES, RecommendationDraft, RuleContext
from app.utils.dates import utc_now


def _record_id(draft: RecommendationDraft) -> str:
    parts = [
        draft.rule_id,
        draft.provider or "",
        draft.account or "",
        draft.service or "",
        draft.period_start.isoformat() if draft.period_start else "",
        draft.period_end.isoformat() if draft.period_end else "",
        draft.title,
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"rec-{digest}"


def _as_date(value) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def _to_record(draft: RecommendationDraft) -> RecommendationRecord:
    now = utc_now()
    return RecommendationRecord(
        recommendation_id=_record_id(draft),
        rule_id=draft.rule_id,
        title=draft.title,
        status=RecommendationStatus.NEW,
        provider=draft.provider,
        account=draft.account,
        service=draft.service,
        observation=draft.observation,
        evidence=draft.evidence,
        suggested_action=draft.suggested_action,
        risk=draft.risk,
        estimated_savings=draft.estimated_savings,
        currency=draft.currency,
        priority=draft.priority,
        period_start=_as_date(draft.period_start),
        period_end=_as_date(draft.period_end),
        created_at=now,
        updated_at=now,
        metadata=draft.metadata,
    )


class RecommendationEngine:
    """Generate and persist evidence-based recommendations."""

    def __init__(
        self,
        *,
        repository: RecommendationRepository | None = None,
        settings_repository: SettingsRepository | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or get_app_config()
        self.repository = repository or RecommendationRepository(
            self.config.storage.database_path
        )
        self.settings_repository = settings_repository or SettingsRepository(
            self.config.storage.database_path
        )

    def generate(
        self,
        frame: pd.DataFrame,
        *,
        monthly_budget: float | None = None,
        persist: bool = True,
    ) -> list[RecommendationRecord]:
        if frame is None or frame.empty:
            return []
        if monthly_budget is None:
            monthly_budget = self._budget_from_settings()

        drafts: list[RecommendationDraft] = []
        for (currency, basis), cohort in frame.groupby(["BillingCurrency", "cost_basis"], dropna=False):
            if not currency:
                continue
            context = RuleContext(
                frame=cohort,
                monthly_budget=monthly_budget,
                currency=str(currency),
                cost_basis=str(basis),
            )
            for rule in RULES:
                try:
                    drafts.extend(rule(context))
                except Exception:  # a single bad rule must not break the page
                    continue

        records: list[RecommendationRecord] = []
        seen: set[str] = set()
        for draft in drafts:
            record = _to_record(draft)
            if record.recommendation_id in seen:
                continue
            seen.add(record.recommendation_id)
            records.append(record)

        records.sort(key=lambda item: {"high": 0, "medium": 1, "low": 2}.get(item.priority, 3))
        if persist and records:
            self.repository.upsert_many(records)
        return records

    def list(self, status: str | None = None) -> list[RecommendationRecord]:
        return self.repository.list(status)

    def update_status(self, recommendation_id: str, status: str) -> bool:
        from app.recommendations.recommendation_status import validate_transition

        existing = {record.recommendation_id: record for record in self.repository.list()}
        record = existing.get(recommendation_id)
        if record is None:
            return False
        validate_transition(record.status, status)
        return self.repository.update_status(recommendation_id, status)

    def _budget_from_settings(self) -> float | None:
        raw = self.settings_repository.get("monthly_budget")
        if raw in (None, "", "None"):
            budget = self.config.budgets.monthly_budget
            return float(budget) if budget else None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
