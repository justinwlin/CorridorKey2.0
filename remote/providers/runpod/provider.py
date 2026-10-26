"""Runpod GPU pods through the Runpod REST API v2 (https://api.runpod.io/v2)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..base import Provider, SSHTarget

API_URL = "https://api.runpod.io/v2"
# Matches the torch 2.9.1 / CUDA 12.8 combination the README was tested with.
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
# The image's CUDA 12.8 build of PyTorch needs a host driver at least this new.
MIN_CUDA_VERSION = "12.8"
# Tried in order: 48 GB cards first (80/40 Hann windows up to 1440p, see the
# README VRAM table), then 24-32 GB cards, then larger cards. A long list keeps
# runs starting when popular types are sold out.
DEFAULT_GPUS = (
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
)
# Create responses that mean "this GPU type can't be placed, try the next":
# 400 for no capacity, 409 for a conflict, 422 for an unknown or invalid type.
TRY_NEXT_GPU = (400, 409, 422)


class RunpodAPIError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, detail: str):
        super().__init__(f"Runpod API {method} {path} -> {status}: {detail}")
        self.status = status
        self.detail = detail


class RunpodProvider(Provider):
    name = "runpod"

    @classmethod
    def add_arguments(cls, group):
        group.add_argument(
            "--runpod_gpu", action="append", default=None,
            help="GPU type id to rent, e.g. 'NVIDIA GeForce RTX 4090'. Repeat to "
                 "give fallbacks in order. Default: " + ", ".join(DEFAULT_GPUS))
        group.add_argument("--runpod_image", default=DEFAULT_IMAGE,
                           help="Container image. Needs CUDA PyTorch and an SSH "
                                "server that reads PUBLIC_KEY.")
        group.add_argument("--runpod_cloud_type", choices=("SECURE", "COMMUNITY"),
                           default="SECURE")
        group.add_argument(
            "--runpod_network_volume", default=None,
            help="Network volume id to mount at /workspace. The weights are "
                 "downloaded to it once and reused by later runs and by every "
                 "pod in this run. Pods start in the volume's data center.")
        group.add_argument("--runpod_disk_gb", type=int, default=100,
                           help="Container disk in GB. Holds the code, weights, "
                                "clips, outputs, and --low_vram feature files.")

    def __init__(self, args):
        super().__init__(args)
        self.api_key = os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set RUNPOD_API_KEY to a Runpod API key "
                "(https://console.runpod.io/user/settings).")
        self.mounts, self.data_centers = self._storage()
        # The container disk is local to the host, so reading frames and the
        # --low_vram feature files is fast. A network volume is only used to
        # keep the weights between runs.
        self.workdir = "/workspace" if self.mounts else "/corridorkey-remote"

    def _request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            API_URL + path, data=data, method=method,
            # The API's firewall rejects urllib's default User-Agent.
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "User-Agent": "corridorkey-remote"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace").strip()
            try:
                err = json.loads(raw)
                detail = "; ".join([err.get("detail") or ""] + err.get("errors", [])).strip("; ")
            except (ValueError, AttributeError):
                detail = raw
            raise RunpodAPIError(method, path, e.code, detail or raw) from None
        return json.loads(raw) if raw else None

    def _storage(self) -> tuple[dict, list[str]]:
        """Mounts for /workspace, and the data centers they restrict pods to."""
        volume_id = self.args.runpod_network_volume
        if not volume_id:
            return {}, []
        volume = self._request("GET", f"/network-volumes/{volume_id}")
        return ({"network": [{"volumeId": volume_id, "path": "/workspace"}]},
                [volume["dataCenter"]])

    def create(self, name, public_key):
        body = {
            "name": name,
            "image": self.args.runpod_image,
            "cloud": self.args.runpod_cloud_type,
            "ports": ["22/tcp"],
            "disk": self.args.runpod_disk_gb,
            # Setting PUBLIC_KEY directly lets this run's throwaway key in
            # without registering keys on the account.
            "env": {"PUBLIC_KEY": public_key},
        }
        if self.mounts:
            body["mounts"] = self.mounts
        if self.data_centers:
            body["dataCenterIds"] = self.data_centers
        failures = []
        # v2 takes one GPU type per create, so the fallback order is ours.
        for gpu in self.args.runpod_gpu or DEFAULT_GPUS:
            body["gpu"] = {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA_VERSION}
            try:
                pod = self._request("POST", "/pods", body)
            except RunpodAPIError as e:
                if e.status not in TRY_NEXT_GPU:
                    raise
                failures.append(f"{gpu}: {e.detail}")
                continue
            cost = pod.get("cost")
            price = f" at ${cost:.2f}/hr" if cost else ""
            print(f"[runpod] created pod {pod['id']} ({gpu}{price})", flush=True)
            return pod["id"]
        raise RuntimeError("no GPU type could be rented:\n  " + "\n  ".join(failures))

    def wait_for_ssh(self, instance_id, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pod = self._request("GET", f"/pods/{instance_id}")
            if pod.get("status") in ("EXITED", "ERROR", "TERMINATED"):
                raise RuntimeError(f"pod {instance_id} is {pod['status']}")
            # The proxy endpoint only carries an interactive shell, so use the
            # pod's own sshd, which appears once 22/tcp has a public port.
            direct = (pod.get("ssh") or {}).get("direct")
            if direct:
                return SSHTarget(host=direct["host"], port=int(direct["port"]),
                                 user=direct["username"])
            time.sleep(5)
        raise TimeoutError(f"pod {instance_id} had no SSH endpoint after {timeout:.0f}s")

    def destroy(self, instance_id):
        try:
            self._request("DELETE", f"/pods/{instance_id}")
        except RunpodAPIError as e:
            if e.status != 404:
                raise
        print(f"[runpod] terminated pod {instance_id}", flush=True)
