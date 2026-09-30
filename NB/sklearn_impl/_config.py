"""目标模型所需的 sklearn._config 流程；配置仅作用于本地实现。"""

from contextlib import contextmanager
from contextvars import ContextVar

_DEFAULTS = {
    "assume_finite": False,
    "skip_parameter_validation": False,
    "enable_metadata_routing": False,
    "array_api_dispatch": False,
    "print_changed_only": True,
}
_configuration = ContextVar("nb_configuration", default=_DEFAULTS)


def get_config():
    return _configuration.get().copy()


def set_config(*, assume_finite=None, skip_parameter_validation=None,
               enable_metadata_routing=None, array_api_dispatch=None, print_changed_only=None):
    config = get_config()
    for name, value in locals().copy().items():
        if name in _DEFAULTS and value is not None:
            config[name] = value
    _configuration.set(config)


@contextmanager
def config_context(**settings):
    previous = get_config()
    try:
        set_config(**settings)
        yield
    finally:
        _configuration.set(previous)
