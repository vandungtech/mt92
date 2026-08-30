class ControllerError(RuntimeError):
    """Base class for an operator-actionable controller failure."""


class ConfigError(ControllerError):
    """Configuration is missing, malformed, or unsafe."""


class PreflightError(ControllerError):
    """The host cannot safely operate the configured miner."""


class RoundRefused(ControllerError):
    """No trustworthy current submission round could be established."""


class VerificationError(ControllerError):
    """A required postcondition could not be proven."""
