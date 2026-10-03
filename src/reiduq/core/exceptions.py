"""Typed domain exceptions. Never raise a bare Exception in this codebase."""


class ReIDUQError(Exception):
    """Base for every error this package raises."""


class ConfigError(ReIDUQError):
    """Malformed, missing, or contradictory configuration."""


class SplitLeakageError(ReIDUQError):
    """Train/calibration/test splits overlap. This invalidates every result."""


class UnsafePathError(ReIDUQError):
    """A path escaped its allowed root."""


class CalibrationNotFittedError(ReIDUQError):
    """transform() was called before fit()."""


class DataIntegrityError(ReIDUQError):
    """Dataset on disk does not match its expected schema."""
