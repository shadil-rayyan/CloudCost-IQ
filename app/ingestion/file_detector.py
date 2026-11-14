"""Detection of billing file provider and schema."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.core.exceptions import UnsupportedFormatError
from app.ingestion.aws_parser import AWSCURParser
from app.ingestion.azure_parser import AzureCostParser
from app.ingestion.base_parser import BaseBillingParser, ColumnResolver
from app.ingestion.focus_parser import FOCUSParser
from app.ingestion.gcp_parser import GCPBillingParser

PARSERS: tuple[BaseBillingParser, ...] = (
    FOCUSParser(),
    AWSCURParser(),
    AzureCostParser(),
    GCPBillingParser(),
)

PARSERS_BY_PROVIDER: dict[str, BaseBillingParser] = {
    parser.provider: parser for parser in PARSERS
}

PROVIDER_CHOICES: tuple[str, ...] = ("Auto", "AWS", "Azure", "GCP", "FOCUS")


@dataclass(slots=True)
class DetectedSchema:
    provider: str
    parser: BaseBillingParser
    score: float
    candidates: list[tuple[str, float]]


def get_parser(provider: str) -> BaseBillingParser:
    key = provider if provider in PARSERS_BY_PROVIDER else provider.title()
    if key not in PARSERS_BY_PROVIDER:
        raise UnsupportedFormatError(f"Unsupported provider '{provider}'")
    return PARSERS_BY_PROVIDER[key]


def detect_schema(frame: pd.DataFrame, provider_hint: str | None = None) -> DetectedSchema:
    """Detect the billing schema from columns.

    ``provider_hint`` (``Auto`` or a provider name) lets the user override
    detection; the chosen parser still validates required columns and reports
    a precise error if the file does not match.
    """
    resolver = ColumnResolver(frame.columns)
    scored = [(parser.provider, parser.detect_score(resolver)) for parser in PARSERS]
    candidates = sorted(scored, key=lambda item: item[1], reverse=True)

    hint = (provider_hint or "Auto").strip()
    if hint.lower() not in {"auto", ""}:
        parser = get_parser(hint)
        score = parser.detect_score(resolver)
        return DetectedSchema(
            provider=parser.provider,
            parser=parser,
            score=score,
            candidates=candidates,
        )

    provider, score = candidates[0]
    if score < 0.5:
        description = ", ".join(f"{name}={value:.2f}" for name, value in candidates)
        raise UnsupportedFormatError(
            "Unable to identify a supported billing schema. "
            "Supported formats are AWS CUR 2.0, Azure Cost Management exports, "
            f"GCP detailed billing exports and FOCUS 1.x. Detection scores: {description}"
        )
    return DetectedSchema(
        provider=provider,
        parser=PARSERS_BY_PROVIDER[provider],
        score=score,
        candidates=candidates,
    )
