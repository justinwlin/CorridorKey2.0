from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SSHTarget:
    """Where to connect. Fields left as None fall back to the user's SSH
    config (~/.ssh/config, agent, default port and user)."""

    host: str
    port: int | None = None
    user: str | None = None
    # Private key for this machine. None means the run's throwaway key for
    # providers that install it (`installs_run_key`), else the SSH defaults.
    identity_file: str | None = None


class Provider(ABC):
    """Supplies CUDA Linux machines that are reachable over SSH.

    The runner does everything else (code upload, setup, inference, downloads)
    over SSH, so a new provider only implements create, wait_for_ssh, and
    destroy, and registers itself in remote/providers/__init__.py.
    """

    name: str
    # Code, weights, and job files live under this directory on the machine.
    # A relative path is resolved against the SSH user's home directory.
    # Point it at persistent storage to keep the ~5 GB of weights between runs.
    workdir = "/workspace"
    # Python on the machine that has a CUDA build of PyTorch.
    python = "python"
    # True when create() starts a fresh machine that trusts the run's
    # throwaway key. Its host key is new every time, so it is not checked.
    # False for machines the user already owns: their SSH config, keys, and
    # known_hosts are used as for any other SSH connection.
    installs_run_key = True

    @classmethod
    def add_arguments(cls, group: argparse._ArgumentGroup) -> None:
        """Register provider-specific command-line flags."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @property
    def max_machines(self) -> int | None:
        """Upper limit on machines per run, or None if there is none. Also the
        default for --machines when it is lower than the clip count."""
        return None

    @abstractmethod
    def create(self, name: str, public_key: str) -> str:
        """Start one GPU machine that accepts `public_key` for SSH. Return its id."""

    @abstractmethod
    def wait_for_ssh(self, instance_id: str, timeout: float) -> SSHTarget:
        """Block until the machine has an SSH endpoint and return it."""

    @abstractmethod
    def destroy(self, instance_id: str) -> None:
        """Release the machine so it stops billing. Must be safe to repeat."""
