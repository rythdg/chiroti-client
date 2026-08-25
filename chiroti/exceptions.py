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


class UnsupportedFeatureError(Exception):
    """The currently hosted model doesn't support a requested feature."""


class OutputValidationError(Exception):
    """The model's response didn't validate against the requested output_format."""

    def __init__(self, message: str, raw_text: str):
        self.raw_text = raw_text
        super().__init__(message)
