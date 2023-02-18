"""Exceptions for Critique."""


class CritiqueError(Exception):
    """Base exception for Critique-specific errors."""

    pass


class RequestFailed(CritiqueError):
    """A request to the service failed."""

    pass


class TaskFailed(CritiqueError):
    """Submitted task failed."""

    pass


class Timeout(CritiqueError):
    """Submitted task didn't complete within a specified duration."""

    pass
