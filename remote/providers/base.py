from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SSHTarget:
    host: str
    port: int
    user: str = "root"


class Provider(ABC):
    """Rents a CUDA Linux machine that is reachable over SSH, then releases it.

    The runner does everything else (code upload, setup, inference, downloads)
    over SSH, so a new provider only implements create, wait_for_ssh, and
    destroy, and registers itself in remote/providers/__init__.py.
    """

    name: str
    # Code, weights, and job files live under this directory on the machine.
    # Point it at persistent storage to keep the ~5 GB of weights between runs.
    workdir = "/workspace"

    @classmethod
    def add_arguments(cls, group: argparse._ArgumentGroup) -> None:
        """Register provider-specific command-line flags."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def create(self, name: str, public_key: str) -> str:
        """Start one GPU machine that accepts `public_key` for SSH. Return its id."""

    @abstractmethod
    def wait_for_ssh(self, instance_id: str, timeout: float) -> SSHTarget:
        """Block until the machine has a public SSH endpoint and return it."""

    @abstractmethod
    def destroy(self, instance_id: str) -> None:
        """Release the machine so it stops billing. Must be safe to repeat."""
