"""Machines you already have, reached over SSH. Nothing is started or stopped."""

from __future__ import annotations

import threading

from ..base import Provider, SSHTarget


def parse_host(value: str) -> SSHTarget:
    """Parse [user@]host[:port]. The host can be an alias from ~/.ssh/config."""
    user, _, rest = value.rpartition("@")
    host, sep, port = rest.rpartition(":")
    if not sep or not port.isdigit():
        host, port = rest, ""
    if not host:
        raise ValueError(f"bad --ssh_host {value!r}, expected [user@]host[:port]")
    return SSHTarget(host=host, port=int(port) if port else None, user=user or None)


class SSHProvider(Provider):
    name = "ssh"
    installs_run_key = False

    @classmethod
    def add_arguments(cls, group):
        group.add_argument(
            "--ssh_host", action="append", default=None, metavar="[USER@]HOST[:PORT]",
            help="A Linux machine with an NVIDIA GPU. Repeat for several "
                 "machines. Aliases from ~/.ssh/config work.")
        group.add_argument("--ssh_key", default=None,
                           help="Private key to use. Default: your SSH agent and config.")
        group.add_argument("--ssh_python", default="python3",
                           help="Python on the machine with CUDA PyTorch installed, "
                                "e.g. ~/CorridorKey2.0/venv/bin/python. The "
                                "requirements are installed into it.")
        group.add_argument("--ssh_workdir", default="corridorkey-remote",
                           help="Directory for code, weights, and job files. "
                                "Relative paths are under the SSH user's home.")

    def __init__(self, args):
        super().__init__(args)
        if not args.ssh_host:
            raise RuntimeError("--provider ssh needs at least one --ssh_host.")
        self.targets = [parse_host(h) for h in args.ssh_host]
        self.workdir = args.ssh_workdir
        self.python = args.ssh_python
        self._next = 0
        self._lock = threading.Lock()

    @property
    def max_machines(self):
        return len(self.targets)

    def create(self, name, public_key):
        with self._lock:
            if self._next >= len(self.targets):
                raise RuntimeError("more machines requested than --ssh_host values")
            index = self._next
            self._next += 1
        return str(index)

    def wait_for_ssh(self, instance_id, timeout):
        target = self.targets[int(instance_id)]
        if self.args.ssh_key:
            target = SSHTarget(target.host, target.port, target.user, self.args.ssh_key)
        return target

    def destroy(self, instance_id):
        """The machine belongs to the user, so it is left running."""
