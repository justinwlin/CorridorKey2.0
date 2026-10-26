"""Runpod GPU pods through the Runpod REST API (https://rest.runpod.io/v1)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import Provider, SSHTarget

API_URL = "https://rest.runpod.io/v1"
# Matches the torch 2.9.1 / CUDA 12.8 combination the README was tested with.
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
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
        group.add_argument("--runpod_volume_gb", type=int, default=60,
                           help="Pod volume size when no network volume is "
                                "used. Holds weights, inputs, and outputs.")

    def __init__(self, args):
        super().__init__(args)
        self.api_key = os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set RUNPOD_API_KEY to a Runpod API key "
                "(https://console.runpod.io/user/settings).")

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
            detail = e.read().decode(errors="replace").strip()
            raise RuntimeError(f"Runpod API {method} {path} -> {e.code}: {detail}") from None
        return json.loads(raw) if raw else None

    def create(self, name, public_key):
        args = self.args
        body = {
            "name": name,
            "imageName": args.runpod_image,
            "gpuTypeIds": list(args.runpod_gpu or DEFAULT_GPUS),
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "cloudType": args.runpod_cloud_type,
            "supportPublicIp": True,
            "ports": ["22/tcp"],
            "containerDiskInGb": 30,
            "env": {"PUBLIC_KEY": public_key},
        }
        if args.runpod_network_volume:
            body["networkVolumeId"] = args.runpod_network_volume
        else:
            body["volumeInGb"] = args.runpod_volume_gb
        pod = self._request("POST", "/pods", body)
        gpu = (pod.get("gpu") or {}).get("displayName") or "GPU"
        cost = pod.get("adjustedCostPerHr") or pod.get("costPerHr")
        price = f" at ${cost:.2f}/hr" if cost else ""
        print(f"[runpod] created pod {pod['id']} ({gpu}{price})", flush=True)
        return pod["id"]

    def wait_for_ssh(self, instance_id, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pod = self._request("GET", f"/pods/{instance_id}")
            if pod.get("desiredStatus") in ("EXITED", "TERMINATED"):
                raise RuntimeError(f"pod {instance_id} stopped: {pod.get('lastStatusChange')}")
            port = (pod.get("portMappings") or {}).get("22")
            if pod.get("publicIp") and port:
                return SSHTarget(host=pod["publicIp"], port=int(port), user="root")
            time.sleep(5)
        raise TimeoutError(f"pod {instance_id} had no SSH endpoint after {timeout:.0f}s")

    def destroy(self, instance_id):
        try:
            self._request("DELETE", f"/pods/{instance_id}")
        except RuntimeError as e:
            if "404" not in str(e):
                raise
        print(f"[runpod] terminated pod {instance_id}", flush=True)
