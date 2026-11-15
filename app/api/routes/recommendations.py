"""Recommendation endpoints. Recommendations are rule-based and require review."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import load_costs
from app.api.schemas import RecommendationPatch
from app.core.exceptions import CloudCostError
from app.recommendations.recommendation_engine import RecommendationEngine

router = APIRouter()


def _serialize(record) -> dict:
    return {
        "recommendation_id": record.recommendation_id,
        "rule_id": record.rule_id,
        "title": record.title,
        "ServiceProviderName": record.provider,
        "account": record.account,
        "service": record.service,
        "observation": record.observation,
        "evidence": record.evidence,
        "suggested_action": record.suggested_action,
        "risk": record.risk,
        "estimated_savings": record.estimated_savings,
        "BillingCurrency": record.currency,
        "priority": record.priority,
        "status": str(record.status),
        "period_start": record.period_start.isoformat() if record.period_start else None,
        "period_end": record.period_end.isoformat() if record.period_end else None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "metadata": record.metadata,
    }


@router.get("/recommendations")
def list_recommendations(
    status: str | None = None,
    refresh: bool = False,
) -> dict:
    engine = RecommendationEngine()
    if refresh:
        _, frame = load_costs()
        engine.generate(frame)
    items = [_serialize(record) for record in engine.list(status)]
    return {"rows": len(items), "recommendations": items}


@router.get("/recommendations/{recommendation_id}")
def get_recommendation(recommendation_id: str) -> dict:
    engine = RecommendationEngine()
    for record in engine.list():
        if record.recommendation_id == recommendation_id:
            return _serialize(record)
    raise HTTPException(status_code=404, detail="Recommendation not found")


@router.patch("/recommendations/{recommendation_id}")
def update_recommendation(recommendation_id: str, patch: RecommendationPatch) -> dict:
    engine = RecommendationEngine()
    try:
        updated = engine.update_status(recommendation_id, patch.status)
    except CloudCostError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    for record in engine.list():
        if record.recommendation_id == recommendation_id:
            return _serialize(record)
    raise HTTPException(status_code=404, detail="Recommendation not found")
