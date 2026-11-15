"""FOCUS-specific exceptions."""

from __future__ import annotations


class FocusValidationError(ValueError):
    """Raised when a FOCUS dataset fails validation."""
