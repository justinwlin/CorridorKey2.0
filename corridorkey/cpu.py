"""CPU thread sizing that respects container CPU limits.

Docker, Kubernetes, and cloud GPU containers often expose every core of the
host while a cgroup quota only grants a few of them. PyTorch sizes its
intra-op pool from the visible cores, so a container granted 8 CPUs on a
96-core host runs 48 threads and spends most of its time throttled. Frame
preprocessing (EXR decode, color conversion) is CPU-bound, so this decides
how fast image-sequence inputs are read.
"""

from __future__ import annotations

import math
import os


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _ancestors(root: str, rel: str):
    """`root/rel` and each parent directory up to `root`."""
    parts = [p for p in rel.strip("/").split("/") if p and p != ".."]
    for i in range(len(parts), -1, -1):
        yield os.path.join(root, *parts[:i])


def _v2_quota(text: str | None) -> float | None:
    try:
        quota, period = text.split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (AttributeError, ValueError, ZeroDivisionError):
        return None


def _v1_quota(quota: str | None, period: str | None) -> float | None:
    try:
        quota_us, period_us = int(quota), int(period)
    except (TypeError, ValueError):
        return None
    return quota_us / period_us if quota_us > 0 and period_us > 0 else None


def cgroup_cpu_limit(proc_cgroup: str = "/proc/self/cgroup",
                     fs_root: str = "/sys/fs/cgroup") -> float | None:
    """CPUs granted by the tightest cgroup quota over this process's cgroup
    and its ancestors, or None when there is no quota (or no cgroups)."""
    limits = []
    for line in (_read(proc_cgroup) or "").splitlines():
        hierarchy, controllers, rel = (line.split(":", 2) + ["", ""])[:3]
        if hierarchy == "0" and controllers == "":
            # cgroup v2: one unified hierarchy mounted at fs_root.
            for d in _ancestors(fs_root, rel):
                limits.append(_v2_quota(_read(os.path.join(d, "cpu.max"))))
        elif "cpu" in controllers.split(","):
            # cgroup v1: the cpu controller is mounted as cpu, cpu,cpuacct,
            # or cpuacct,cpu depending on the distribution.
            for mount in (controllers, "cpu", "cpu,cpuacct", "cpuacct,cpu"):
                mount_root = os.path.join(fs_root, mount)
                if os.path.isdir(mount_root):
                    break
            else:
                continue
            for d in _ancestors(mount_root, rel):
                limits.append(_v1_quota(
                    _read(os.path.join(d, "cpu.cfs_quota_us")),
                    _read(os.path.join(d, "cpu.cfs_period_us"))))
    limits = [x for x in limits if x is not None]
    return min(limits) if limits else None


def visible_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def available_cpus() -> int:
    """CPUs this process can actually use: visible cores capped by the quota."""
    cpus = visible_cpus()
    limit = cgroup_cpu_limit()
    if limit is not None:
        cpus = min(cpus, max(1, math.floor(limit)))
    return cpus


def configure_cpu_threads(threads: int | None = None) -> int:
    """Size torch and OpenCV thread pools to the CPUs this process can use.

    An explicit `threads`, or OMP_NUM_THREADS in the environment, wins.
    """
    import cv2
    import torch

    if threads is None and os.environ.get("OMP_NUM_THREADS"):
        threads = torch.get_num_threads()
    if threads is None:
        threads = available_cpus()
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    return threads
