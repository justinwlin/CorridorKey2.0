#!/usr/bin/env python3
"""Run CorridorKey on remote GPU machines.

Uses your own machines over SSH, or rents machines from a cloud provider for
the run. Uploads this code and your clips, runs infer.py on each clip,
downloads the outputs, and shuts down any machine it started. Flags not
listed below are passed through to infer.py unchanged.

    python remote/run.py --provider ssh --ssh_host gpu-box \\
        --input shot.mp4 --output_dir out
    RUNPOD_API_KEY=... python remote/run.py --input clips/ --output_dir out \\
        --machines 3 --outputs alpha fg cutout

A folder of clips is handled like the drag-and-drop launcher: each clip gets
its own out_<clip> folder and companion hint files are picked up
automatically. Its clips are shared out across the machines. A single clip
always runs on one machine.
"""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remote.providers import PROVIDERS, Provider, SSHTarget  # noqa: E402
from run_interactive import (  # noqa: E402
    check_image_sequence_dir,
    find_alpha_hint,
    find_first_frame_hint,
    get_batch_clips,
    resolve_image_seq_dirs,
)

# Everything the machine needs to run infer.py. Weights are downloaded there.
CODE_PATHS = ("corridorkey", "third_party", "scripts", "infer.py",
              "requirements.txt", "remote/setup_machine.sh")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_print_lock = threading.Lock()


def log(label: str, message: str) -> None:
    with _print_lock:
        print(f"[{label}] {message}", flush=True)


@dataclass
class Clip:
    index: int
    name: str
    input: Path
    output_dir: Path
    hint_first_frame: Path | None = None
    hint_video: Path | None = None


def _resolve(path: str | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None


def collect_clips(args) -> list[Clip]:
    inp = _resolve(args.input)
    out = _resolve(args.output_dir)
    hint_first, hint_video = _resolve(args.hint_first_frame), _resolve(args.hint_video)
    for path in (inp, hint_first, hint_video):
        if path and not path.exists():
            raise SystemExit(f"not found: {path}")
    if inp.is_dir():
        seq_input, seq_hint = check_image_sequence_dir(inp)
        if seq_input:
            return [Clip(0, inp.name, seq_input, out, hint_first, hint_video or seq_hint)]
        batch = get_batch_clips(inp)
        if batch:
            if hint_first or hint_video:
                raise SystemExit("--hint_first_frame and --hint_video apply to a single "
                                 "clip. Batch folders use companion hint files.")
            return [_batch_clip(i, clip, out / f"out_{clip.stem}")
                    for i, clip in enumerate(batch)]
    return [Clip(0, inp.stem, inp, out, hint_first, hint_video)]


def _batch_clip(index: int, clip: Path, out: Path) -> Clip:
    """Same companion-hint rules as the drag-and-drop launcher."""
    if clip.is_dir():
        seq_input, seq_hint = resolve_image_seq_dirs(clip)
        return Clip(index, clip.name, seq_input, out, hint_video=seq_hint)
    hint = find_alpha_hint(clip)
    if hint and hint.suffix.lower() in VIDEO_EXTENSIONS:
        return Clip(index, clip.stem, clip, out, hint_video=hint)
    return Clip(index, clip.stem, clip, out, hint_first_frame=find_first_frame_hint(clip))


class Remote:
    """Shell commands and tar-over-SSH file transfer to one machine."""

    def __init__(self, target: SSHTarget, run_key: Path, label: str, *,
                 installs_run_key: bool = True):
        self.label = label
        key = target.identity_file or (run_key if installs_run_key else None)
        self.ssh = ["ssh"]
        if key:
            self.ssh += ["-i", str(key)]
        if target.port:
            self.ssh += ["-p", str(target.port)]
        if installs_run_key:
            # A fresh machine has a new host key every time.
            self.ssh += ["-o", "StrictHostKeyChecking=no",
                         "-o", f"UserKnownHostsFile={os.devnull}"]
        else:
            self.ssh += ["-o", "StrictHostKeyChecking=accept-new"]
        self.ssh += [
            "-o", "LogLevel=ERROR",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            f"{target.user}@{target.host}" if target.user else target.host,
        ]

    def alive(self) -> bool:
        return subprocess.run(self.ssh + ["true"], stdin=subprocess.DEVNULL,
                              capture_output=True).returncode == 0

    def capture(self, command: str) -> str:
        proc = subprocess.run(self.ssh + [command], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"remote command failed ({proc.returncode}): "
                               f"{command}: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self.alive():
            if time.monotonic() > deadline:
                raise TimeoutError(f"SSH not reachable after {timeout:.0f}s")
            time.sleep(5)

    def run(self, command: str) -> None:
        """Run a shell command remotely, streaming its output with our label."""
        proc = subprocess.Popen(
            self.ssh + ["export PYTHONUNBUFFERED=1; " + command], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            log(self.label, line.rstrip())
        if proc.wait() != 0:
            raise RuntimeError(f"remote command failed ({proc.returncode}): {command}")

    def upload(self, files: list[tuple[Path, str]], dest: str) -> None:
        """Copy (local path, relative name) pairs into dest on the machine."""
        proc = subprocess.Popen(
            self.ssh + [f"mkdir -p {shlex.quote(dest)} && tar xf - --no-same-owner -C {shlex.quote(dest)}"],
            stdin=subprocess.PIPE)
        skip = lambda info: None if "__pycache__" in info.name else info  # noqa: E731
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                for local, name in files:
                    tar.add(local, arcname=name, filter=skip)
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"upload to {dest} failed ({proc.returncode})")

    def download(self, src: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(self.ssh + [f"tar cf - -C {shlex.quote(src)} ."],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            tar.extractall(local_dir, **safe)
        if proc.wait() != 0:
            raise RuntimeError(f"download of {src} failed ({proc.returncode})")


class Machines:
    """Tracks rented machines so every exit path releases them."""

    def __init__(self, provider: Provider, keep: bool):
        self.provider = provider
        self.keep = keep
        self.live: set[str] = set()
        self.stopping = False
        self.lock = threading.Lock()

    def add(self, instance_id: str) -> None:
        with self.lock:
            if not self.stopping:
                self.live.add(instance_id)
                return
        self.provider.destroy(instance_id)
        raise RuntimeError("run stopped while the machine was starting")

    def release(self, instance_id: str) -> None:
        with self.lock:
            if self.keep or instance_id not in self.live:
                return
            self.live.discard(instance_id)
        self.provider.destroy(instance_id)

    def release_all(self) -> None:
        with self.lock:
            self.stopping = True
            ids = [] if self.keep else sorted(self.live)
        for instance_id in ids:
            try:
                self.release(instance_id)
            except Exception as e:
                print(f"[error] could not release {instance_id}, delete it by hand: {e}")


def run_clip(remote: Remote, clip: Clip, code_dir: str, job_dir: str,
             python: str, infer_args: list[str]) -> None:
    files = [(clip.input, f"in/{clip.input.name}")]
    cmd = [python, "infer.py",
           "--input", f"{job_dir}/in/{clip.input.name}",
           "--output_dir", f"{job_dir}/out"]
    if clip.hint_first_frame:
        files.append((clip.hint_first_frame, f"hint_first/{clip.hint_first_frame.name}"))
        cmd += ["--hint_first_frame", f"{job_dir}/hint_first/{clip.hint_first_frame.name}"]
    if clip.hint_video:
        files.append((clip.hint_video, f"hint_video/{clip.hint_video.name}"))
        cmd += ["--hint_video", f"{job_dir}/hint_video/{clip.hint_video.name}"]
    cmd += infer_args

    remote.upload(files, job_dir)
    remote.run(f"cd {shlex.quote(code_dir)} && {shlex.join(cmd)}")
    remote.download(f"{job_dir}/out", clip.output_dir)
    remote.run(f"rm -rf {shlex.quote(job_dir)}")


def machine_worker(number: int, args, provider: Provider, machines: Machines,
                   key: Path, public_key: str, jobs: queue.Queue,
                   results: dict[int, str], run_id: str, infer_args: list[str]) -> None:
    label = f"machine {number}"
    instance_id = None
    try:
        instance_id = provider.create(f"corridorkey-{run_id}-{number}", public_key)
        machines.add(instance_id)
        target = provider.wait_for_ssh(instance_id, args.ready_timeout)
        remote = Remote(target, key, label, installs_run_key=provider.installs_run_key)
        remote.wait_ready(args.ready_timeout)
        log(label, f"connected: {shlex.join(remote.ssh)}")

        # Absolute, because infer.py runs from code_dir and gets job paths.
        workdir = remote.capture(f"mkdir -p {shlex.quote(provider.workdir)} && "
                                 f"cd {shlex.quote(provider.workdir)} && pwd")
        code_dir = f"{workdir}/corridorkey"
        remote.upload([(ROOT / p, p) for p in CODE_PATHS], code_dir)
        remote.run(f"bash {shlex.quote(code_dir)}/remote/setup_machine.sh "
                   f"{shlex.quote(code_dir)} {shlex.quote(provider.python)}")

        while True:
            try:
                clip = jobs.get_nowait()
            except queue.Empty:
                return
            log(label, f"processing {clip.name}")
            started = time.monotonic()
            job_dir = f"{workdir}/corridorkey_jobs/{run_id}/{clip.index:04d}"
            try:
                run_clip(remote, clip, code_dir, job_dir, provider.python, infer_args)
            except Exception as e:
                if not remote.alive():
                    # The machine is gone, not the clip. Leave it for another machine.
                    jobs.put(clip)
                    raise RuntimeError(f"lost connection while processing {clip.name}") from e
                results[clip.index] = f"failed: {e}"
                log(label, f"{clip.name} failed: {e}")
            else:
                results[clip.index] = f"done in {time.monotonic() - started:.0f}s -> {clip.output_dir}"
                log(label, f"{clip.name} done -> {clip.output_dir}")
    except Exception as e:
        log(label, f"machine failed: {e}")
    finally:
        if instance_id:
            machines.release(instance_id)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="runpod")
    ap.add_argument("--input", required=True,
                    help="Video file, frame directory, or folder of clips.")
    ap.add_argument("--output_dir", required=True, help="Local output directory.")
    ap.add_argument("--hint_first_frame", default=None,
                    help="Local first-frame matte, uploaded with the clip.")
    ap.add_argument("--hint_video", default=None,
                    help="Local hint video or frame directory, uploaded with the clip.")
    ap.add_argument("--machines", type=int, default=None,
                    help="Machines to use in parallel for a folder of clips. "
                         "Default: 1 for rented machines, every --ssh_host for ssh.")
    ap.add_argument("--ready_timeout", type=float, default=900,
                    help="Seconds to wait for a machine to accept SSH.")
    ap.add_argument("--keep_machines", action="store_true",
                    help="Leave machines running afterwards for debugging. They "
                         "bill until you delete them.")
    for cls in PROVIDERS.values():
        cls.add_arguments(ap.add_argument_group(f"{cls.name} options"))
    args, infer_args = ap.parse_known_args()

    clips = collect_clips(args)
    provider = PROVIDERS[args.provider](args)
    limit = provider.max_machines
    count = min(args.machines or limit or 1, limit or len(clips), len(clips))
    run_id = uuid.uuid4().hex[:8]
    print(f"{len(clips)} clip(s) on {count} {provider.name} machine(s), run {run_id}")

    # A throwaway SSH key per run, so nothing has to be set up on the account.
    key_dir = Path(tempfile.mkdtemp(prefix="corridorkey-remote-"))
    key = key_dir / "id_ed25519"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
                    f"corridorkey-{run_id}", "-f", str(key)], check=True)
    public_key = key.with_suffix(".pub").read_text().strip()

    jobs: queue.Queue[Clip] = queue.Queue()
    for clip in clips:
        jobs.put(clip)
    results: dict[int, str] = {}
    machines = Machines(provider, args.keep_machines)
    workers = [
        threading.Thread(
            target=machine_worker, daemon=True,
            args=(n, args, provider, machines, key, public_key, jobs, results,
                  run_id, infer_args))
        for n in range(1, count + 1)
    ]
    try:
        for worker in workers:
            worker.start()
        while any(worker.is_alive() for worker in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted, releasing machines.")
    finally:
        machines.release_all()
        if not args.keep_machines:
            for path in key_dir.iterdir():
                path.unlink()
            key_dir.rmdir()

    print("\nSummary:")
    for clip in clips:
        print(f"  {clip.name}: {results.get(clip.index, 'not processed')}")
    if args.keep_machines:
        print(f"Machines were kept running. SSH key: {key}")
    if any(not results.get(c.index, "").startswith("done") for c in clips):
        sys.exit(1)


if __name__ == "__main__":
    main()
