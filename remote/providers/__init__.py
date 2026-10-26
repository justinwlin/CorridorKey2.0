from .base import Provider, SSHTarget
from .runpod import RunpodProvider

PROVIDERS: dict[str, type[Provider]] = {
    RunpodProvider.name: RunpodProvider,
}

__all__ = ["PROVIDERS", "Provider", "SSHTarget"]
