from .base import Provider, SSHTarget
from .runpod import RunpodProvider
from .ssh import SSHProvider

PROVIDERS: dict[str, type[Provider]] = {
    RunpodProvider.name: RunpodProvider,
    SSHProvider.name: SSHProvider,
}

__all__ = ["PROVIDERS", "Provider", "SSHTarget"]
