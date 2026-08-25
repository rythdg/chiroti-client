"""Exceptions chiroti.ask() and chiroti.models() raise, one per server error kind."""


class ChirotiConnectionError(Exception):
    """The Chiroti server could not be reached."""


class AuthenticationError(Exception):
    """The configured token was missing or rejected."""


class InvalidInputError(Exception):
    """The request itself was invalid (e.g. empty prompt)."""


class ModelNotFoundError(Exception):
    """The requested model name is not served by Chiroti."""


class InferenceError(Exception):
    """The model backend failed to produce a response."""
