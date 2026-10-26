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


def _cgroup_mounts(mountinfo: str):
    """(mount root, mount point, is_v2, super options) for each cgroup mount."""
    for line in (_read(mountinfo) or "").splitlines():
        left, _, right = line.partition(" - ")
        fields, fs = left.split(), right.split()
        if len(fields) >= 5 and len(fs) >= 3 and fs[0] in ("cgroup", "cgroup2"):
            yield fields[3], fields[4], fs[0] == "cgroup2", fs[2].split(",")


def _dirs_to_check(rel: str, mount_root: str, mount_point: str):
    """The process cgroup's directory under `mount_point`, then its parents.

    /proc/self/cgroup paths are relative to the hierarchy root, while the
    mount may expose only a subtree (`mount_root`). Without a cgroup
    namespace a container sees e.g. /docker/<id> in both places.
    """
    root = mount_root.rstrip("/")
    if rel == root or rel.startswith(root + "/"):
        rel = rel[len(root):]
    elif root:
        rel = ""  # outside the mounted subtree: only the mount itself is visible
    return _ancestors(mount_point, rel)


def cgroup_cpu_limit(proc_cgroup: str = "/proc/self/cgroup",
                     mountinfo: str = "/proc/self/mountinfo") -> float | None:
    """CPUs granted by the tightest cgroup quota over this process's cgroup
    and its ancestors, or None when there is no quota (or no cgroups)."""
    mounts = list(_cgroup_mounts(mountinfo))
    limits = []
    for line in (_read(proc_cgroup) or "").splitlines():
        hierarchy, controllers, rel = (line.split(":", 2) + ["", ""])[:3]
        if hierarchy == "0" and controllers == "":
            # cgroup v2: one unified hierarchy.
            for mount_root, mount_point, is_v2, _ in mounts:
                if is_v2:
                    for d in _dirs_to_check(rel, mount_root, mount_point):
                        limits.append(_v2_quota(_read(os.path.join(d, "cpu.max"))))
        elif "cpu" in controllers.split(","):
            # cgroup v1: the hierarchy whose mount carries the cpu controller.
            for mount_root, mount_point, is_v2, options in mounts:
                if not is_v2 and "cpu" in options:
                    for d in _dirs_to_check(rel, mount_root, mount_point):
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
