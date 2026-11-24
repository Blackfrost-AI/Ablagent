"""Domain exceptions: unsupported operations and incomplete evidence never pass."""


class AblagentError(Exception):
    """Base class for an expected, reportable domain failure."""

    code = "ablagent_error"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class ValidationError(AblagentError, ValueError):
    """An identity, schema, tensor, or capability contract was not satisfied."""


class PolicyBlocked(AblagentError):
    """A requested eligibility decision lacks required evidence or authority."""
