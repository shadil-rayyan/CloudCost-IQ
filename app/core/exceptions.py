"""Typed exceptions used across CloudCost IQ.

All application errors derive from :class:`CloudCostError` so the UI/API can
distinguish expected failures from programming errors and never leak raw
tracebacks or internal paths to end users.
"""

from __future__ import annotations


class CloudCostError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(CloudCostError):
    """Configuration is missing or invalid."""


class IngestionError(CloudCostError):
    """A billing file could not be ingested."""


class UnsupportedFormatError(IngestionError):
    """The uploaded file format or schema is not supported."""


class FileSafetyError(IngestionError):
    """The uploaded file failed a safety check."""


class SchemaValidationError(IngestionError):
    """A required billing schema column is missing or incompatible."""


class EmptyDataError(IngestionError):
    """The uploaded file contains no usable billing rows."""


class DuplicateImportError(IngestionError):
    """The exact same file has already been imported."""


class ModelNotAvailableError(CloudCostError):
    """No approved model artifact is available for inference."""


class ModelContractError(CloudCostError):
    """A model artifact bundle is incomplete or inconsistent."""


class InsufficientHistoryError(CloudCostError):
    """The dataset is too short for a reliable forecast."""


class CurrencyMismatchError(CloudCostError):
    """Operation refused because incompatible currencies would be combined."""


class CostBasisMismatchError(CloudCostError):
    """Operation refused because cost bases are incompatible."""


class DataQualityError(CloudCostError):
    """Data quality prevented an operation from running safely."""
