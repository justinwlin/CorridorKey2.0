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


def _cgroup_v2_limit(path: str = "/sys/fs/cgroup/cpu.max") -> float | None:
    try:
        quota, period = open(path).read().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return int(quota) / int(period)


def _cgroup_v1_limit(root: str = "/sys/fs/cgroup/cpu") -> float | None:
    try:
        quota = int(open(os.path.join(root, "cpu.cfs_quota_us")).read())
        period = int(open(os.path.join(root, "cpu.cfs_period_us")).read())
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def cgroup_cpu_limit() -> float | None:
    """CPUs granted by the cgroup quota, or None when there is no quota."""
    limit = _cgroup_v2_limit()
    return limit if limit is not None else _cgroup_v1_limit()


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
