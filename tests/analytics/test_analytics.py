from __future__ import annotations

from app.analytics.cost_breakdown import breakdown
from app.analytics.cost_summary import summarize_costs
from app.analytics.cost_trends import cost_trend


def test_cost_summary_and_breakdown(sample_normalized_frame):
    summary = summarize_costs(sample_normalized_frame)
    assert summary["records"] == len(sample_normalized_frame)
    assert (summary["combined_total"] or 0) > 0
    assert summary["is_multi_currency"] is False
    by_provider = breakdown(sample_normalized_frame, "ServiceProviderName")
    assert set(by_provider["ServiceProviderName"]) == {"AWS", "Azure", "GCP"}
    trend = cost_trend(sample_normalized_frame, freq="D", group_by=("ServiceProviderName",))
    assert not trend.empty
    assert "cost" in trend.columns
