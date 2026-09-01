class ControllerError(RuntimeError):
    """Base class for an operator-actionable controller failure."""


class ConfigError(ControllerError):
    """Configuration is missing, malformed, or unsafe."""


class PreflightError(ControllerError):
    """The host cannot safely operate the configured miner."""


class ArtifactCompetitionBindingError(PreflightError):
    """The artifact is not explicitly bound to the configured competition."""


class RoundRefused(ControllerError):
    """No trustworthy current submission round could be established."""


class RoundNotOpen(RoundRefused):
    """The trusted coordinator has no currently usable submission round."""


class VerificationError(ControllerError):
    """A required postcondition could not be proven."""


class AuthorizationRefused(ControllerError):
    """A transaction fell outside the operator's explicit authorization."""
