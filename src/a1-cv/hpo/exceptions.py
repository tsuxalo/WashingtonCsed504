from __future__ import annotations


class HpoError(RuntimeError):
    """Base error for the additions-only HPO package."""


class ConfigurationError(HpoError):
    """Raised when a study configuration is invalid."""


class SearchSpaceError(ConfigurationError):
    """Raised when a search-space source cannot be normalized."""


class InvalidTrialError(HpoError):
    """Raised before training when a candidate violates a hard constraint."""


class TrialExecutionError(HpoError):
    """Raised when a candidate fails during model training or evaluation."""


class StudyResumeError(HpoError):
    """Raised when persisted state is incompatible with the requested study."""
