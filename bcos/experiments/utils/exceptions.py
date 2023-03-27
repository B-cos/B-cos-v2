__all__ = [
    "AmbiguityError",
    "EMANotFound",
    "MetricsNotFoundError",
    "ModelFactoryDoesNotSupportPretrainedError",
]


class EMANotFound(Exception):
    pass


class MetricsNotFoundError(Exception):
    pass


class ModelFactoryDoesNotSupportPretrainedError(Exception):
    pass


class AmbiguityError(Exception):
    pass
