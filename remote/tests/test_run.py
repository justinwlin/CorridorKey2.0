"""Offline tests for remote/run.py. No cloud account, SSH server, or GPU needed.

    python remote/tests/test_run.py

FakeProvider stands in for a cloud API and FakeRemote for the SSH layer, so
these tests cover the runner's scheduling and cleanup, not a real provider.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from remote import run  # noqa: E402
from remote.providers import Provider, SSHTarget  # noqa: E402
from remote.providers.ssh import parse_host  # noqa: E402


class FakeProvider(Provider):
    name = "fake"
    fail_wait = False

    instance: FakeProvider | None = None

    def __init__(self, args):
        super().__init__(args)
        FakeProvider.instance = self
        self.lock = threading.Lock()
        self.created: list[str] = []
        self.destroyed: list[str] = []

    def create(self, name, public_key):
        assert public_key.startswith("ssh-ed25519 ")
        with self.lock:
            instance_id = f"m{len(self.created) + 1}"
            self.created.append(instance_id)
        return instance_id

    def wait_for_ssh(self, instance_id, timeout):
        if self.fail_wait:
            raise TimeoutError("no SSH endpoint")
        return SSHTarget(host=f"{instance_id}.example", port=22)

    def destroy(self, instance_id):
        with self.lock:
            self.destroyed.append(instance_id)


class FakeRemote:
    """Records commands. Inference fails for clips named in `fail_clips`, and
    the first machine to run a clip named in `kill_on` loses its connection."""

    fail_clips: set[str] = set()
    kill_on: set[str] = set()
    commands: list[tuple[str, str]] = []
    uploads: list[str] = []
    # Set by a test to block setup until a clip upload has started.
    setup_waits_for_clip_upload = False

    def __init__(self, target, key, label, *, installs_run_key=True):
        self.host = target.host
        self.ssh = ["ssh", target.host]
        self.dead = False

    def capture(self, command):
        FakeRemote.commands.append((self.host, command))
        return "/home/artist/work"

    def alive(self):
        return not self.dead

    def wait_ready(self, timeout):
        pass

    def run(self, command):
        FakeRemote.commands.append((self.host, command))
        if "setup_machine.sh" in command and FakeRemote.setup_waits_for_clip_upload:
            deadline = time.monotonic() + 5
            while not any("corridorkey_jobs" in d for d in FakeRemote.uploads):
                if time.monotonic() > deadline:
                    raise RuntimeError("no clip upload started during setup")
                time.sleep(0.01)
        if "infer.py" in command:
            for name in FakeRemote.kill_on:
                if name in command:
                    FakeRemote.kill_on = FakeRemote.kill_on - {name}
                    self.dead = True
                    raise RuntimeError("connection reset")
            if any(name in command for name in FakeRemote.fail_clips):
                raise RuntimeError("remote command failed (1)")

    def upload(self, files, dest):
        for local, _ in files:
            assert Path(local).exists(), local
        FakeRemote.uploads.append(dest)

    def download(self, src, local_dir):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "alpha.mp4").write_bytes(b"")


class Fixtures:
    """Three clips with companion hints, a fake provider, and a fake SSH layer."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.clips = self.tmp / "clips"
        self.clips.mkdir()
        for name in ("a", "b", "c"):
            (self.clips / f"shot_{name}.mp4").write_bytes(b"")
        (self.clips / "shot_aFrameHint.png").write_bytes(b"")
        (self.clips / "shot_b_AlphaHint.mp4").write_bytes(b"")
        FakeRemote.fail_clips, FakeRemote.kill_on, FakeRemote.commands = set(), set(), []
        FakeRemote.uploads, FakeRemote.setup_waits_for_clip_upload = [], False
        FakeProvider.fail_wait = False
        for patch in (mock.patch.dict(run.PROVIDERS, {"fake": FakeProvider}),
                      mock.patch.object(run, "Remote", FakeRemote)):
            patch.start()
            self.addCleanup(patch.stop)

    def main(self, *argv):
        with mock.patch.object(sys, "argv", ["run.py", "--provider", "fake", *argv]):
            try:
                run.main()
            except SystemExit as e:
                return e.code or 0
            finally:
                self.provider = FakeProvider.instance
        return 0

    def infer_commands(self):
        return [cmd for _, cmd in FakeRemote.commands if "infer.py" in cmd]


class RunnerTest(Fixtures, unittest.TestCase):
    def test_batch_spreads_clips_and_releases_every_machine(self):
        code = self.main("--input", str(self.clips), "--output_dir", str(self.tmp / "out"),
                         "--machines", "2", "--outputs", "alpha", "fg")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(self.provider.created), ["m1", "m2"])
        self.assertEqual(sorted(self.provider.destroyed), ["m1", "m2"])
        for name in ("a", "b", "c"):
            self.assertTrue((self.tmp / "out" / f"out_shot_{name}" / "alpha.mp4").exists())
        commands = self.infer_commands()
        self.assertEqual(len(commands), 3)
        self.assertTrue(all("--outputs alpha fg" in cmd for cmd in commands))
        by_clip = {name: next(c for c in commands if f"shot_{name}.mp4" in c) for name in "abc"}
        self.assertIn("--hint_first_frame", by_clip["a"])
        self.assertIn("--hint_video", by_clip["b"])
        self.assertNotIn("--hint", by_clip["c"])

    def test_first_clip_uploads_while_setup_runs(self):
        FakeRemote.setup_waits_for_clip_upload = True
        code = self.main("--input", str(self.clips), "--output_dir", str(self.tmp / "out"))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.infer_commands()), 3)

    def test_never_rents_more_machines_than_clips(self):
        code = self.main("--input", str(self.clips / "shot_c.mp4"),
                         "--output_dir", str(self.tmp / "out"), "--machines", "4")
        self.assertEqual(code, 0)
        self.assertEqual(self.provider.created, ["m1"])
        self.assertTrue((self.tmp / "out" / "alpha.mp4").exists())

    def test_failed_clip_is_reported_and_the_rest_still_run(self):
        FakeRemote.fail_clips = {"shot_b"}
        code = self.main("--input", str(self.clips), "--output_dir", str(self.tmp / "out"))
        self.assertEqual(code, 1)
        self.assertEqual(len(self.infer_commands()), 3)
        self.assertEqual(self.provider.destroyed, ["m1"])

    def test_lost_machine_stops_instead_of_failing_every_clip(self):
        FakeRemote.kill_on = {"shot_a"}
        code = self.main("--input", str(self.clips), "--output_dir", str(self.tmp / "out"))
        self.assertEqual(code, 1)
        # The dead machine gave shot_a back and did not try shot_b or shot_c.
        self.assertEqual(len(self.infer_commands()), 1)
        self.assertEqual(self.provider.destroyed, ["m1"])

    def test_machine_that_never_comes_up_is_still_released(self):
        FakeProvider.fail_wait = True
        code = self.main("--input", str(self.clips), "--output_dir", str(self.tmp / "out"))
        self.assertEqual(code, 1)
        self.assertEqual(self.provider.destroyed, ["m1"])

    def test_keep_machines_skips_release(self):
        code = self.main("--input", str(self.clips / "shot_c.mp4"),
                         "--output_dir", str(self.tmp / "out"), "--keep_machines")
        self.assertEqual(code, 0)
        self.assertEqual(self.provider.destroyed, [])

    def test_explicit_hints_rejected_for_a_folder(self):
        with self.assertRaises(SystemExit):
            run.collect_clips(mock.Mock(
                input=str(self.clips), output_dir=str(self.tmp / "out"),
                hint_first_frame=str(self.clips / "shot_aFrameHint.png"), hint_video=None))


class SSHProviderTest(Fixtures, unittest.TestCase):
    """The ssh provider against FakeRemote."""

    def run_ssh(self, *argv):
        with mock.patch.object(sys, "argv", ["run.py", "--provider", "ssh",
                                             "--input", str(self.clips),
                                             "--output_dir", str(self.tmp / "out"), *argv]):
            try:
                run.main()
            except SystemExit as e:
                return e.code or 0
        return 0

    def test_uses_every_host_by_default_and_runs_in_its_workdir(self):
        code = self.run_ssh("--ssh_host", "artist@box1", "--ssh_host", "box2:2222",
                            "--ssh_python", "/opt/ck/venv/bin/python")
        self.assertEqual(code, 0)
        self.assertEqual({host for host, _ in FakeRemote.commands}, {"box1", "box2"})
        commands = self.infer_commands()
        self.assertEqual(len(commands), 3)
        self.assertTrue(all("cd /home/artist/work/corridorkey && /opt/ck/venv/bin/python "
                            "infer.py --input /home/artist/work/corridorkey_jobs/" in c
                            for c in commands))
        setup = [c for _, c in FakeRemote.commands if "setup_machine.sh" in c]
        self.assertTrue(all(c.endswith(" /opt/ck/venv/bin/python") for c in setup))

    def test_machines_flag_can_use_fewer_hosts(self):
        self.assertEqual(self.run_ssh("--ssh_host", "box1", "--ssh_host", "box2",
                                      "--machines", "1"), 0)
        self.assertEqual({host for host, _ in FakeRemote.commands}, {"box1"})

    def test_hosts_flag_is_required(self):
        with self.assertRaises(RuntimeError):
            self.run_ssh()


class SSHCommandTest(unittest.TestCase):
    def test_parse_host(self):
        self.assertEqual(parse_host("gpu-box"), SSHTarget("gpu-box"))
        self.assertEqual(parse_host("me@10.0.0.5:2222"), SSHTarget("10.0.0.5", 2222, "me"))
        self.assertEqual(parse_host("me@gpu-box"), SSHTarget("gpu-box", None, "me"))

    def test_own_machine_uses_ssh_defaults_and_checks_host_keys(self):
        ssh = run.Remote(SSHTarget("gpu-box"), Path("/tmp/run_key"), "m",
                         installs_run_key=False).ssh
        self.assertNotIn("-i", ssh)
        self.assertNotIn("-p", ssh)
        self.assertIn("StrictHostKeyChecking=accept-new", ssh)
        self.assertEqual(ssh[-1], "gpu-box")

    def test_own_machine_with_explicit_key(self):
        ssh = run.Remote(SSHTarget("gpu-box", 2222, "me", "~/.ssh/studio"),
                         Path("/tmp/run_key"), "m", installs_run_key=False).ssh
        self.assertEqual(ssh[1:5], ["-i", "~/.ssh/studio", "-p", "2222"])
        self.assertEqual(ssh[-1], "me@gpu-box")

    def test_rented_machine_uses_run_key_without_host_key_checks(self):
        ssh = run.Remote(SSHTarget("1.2.3.4", 22022, "root"), Path("/tmp/run_key"), "m").ssh
        self.assertEqual(ssh[1:5], ["-i", "/tmp/run_key", "-p", "22022"])
        self.assertIn("StrictHostKeyChecking=no", ssh)


class RunpodProviderTest(unittest.TestCase):
    """The Runpod v2 provider with the HTTP layer replaced by canned responses."""

    def provider(self, responses, **flags):
        from remote.providers.runpod import provider as rp
        args = mock.Mock(runpod_gpu=flags.get("gpus"), runpod_image="img",
                         runpod_cloud_type="SECURE",
                         runpod_network_volume=flags.get("volume"),
                         runpod_gpu_count=flags.get("gpu_count", 1),
                         runpod_disk_gb=100)
        calls = []

        def fake_request(self, method, path, body=None):
            calls.append((method, path, json.loads(json.dumps(body))))
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "test"}), \
                mock.patch.object(rp.RunpodProvider, "_request", fake_request):
            provider = rp.RunpodProvider(args)
        provider._request = lambda *a, **k: fake_request(provider, *a, **k)
        return provider, calls, rp

    def test_tries_gpu_types_in_order_until_one_is_available(self):
        from remote.providers.runpod.provider import RunpodAPIError
        provider, calls, _ = self.provider([
            RunpodAPIError("POST", "/pods", 400, "no instances available"),
            {"id": "pod1", "cost": 0.59},
        ], gpus=["GPU A", "GPU B", "GPU C"])
        self.assertEqual(provider.create("ck", "ssh-ed25519 AAAA"), "pod1")
        self.assertEqual([c[2]["gpu"]["id"] for c in calls], ["GPU A", "GPU B"])
        body = calls[-1][2]
        self.assertEqual(body["gpu"], {"id": "GPU B", "count": 1, "minCudaVersion": "12.8"})
        self.assertEqual(body["env"], {"PUBLIC_KEY": "ssh-ed25519 AAAA"})
        # No volume by default: everything runs on the local container disk.
        self.assertNotIn("mounts", body)
        self.assertEqual(body["disk"], 100)
        self.assertEqual(provider.workdir, "/corridorkey-remote")
        self.assertNotIn("dataCenterIds", body)

    def test_gpu_count_is_sent_for_multi_gpu_pods(self):
        provider, calls, _ = self.provider([{"id": "pod1"}], gpus=["GPU A"], gpu_count=2)
        provider.create("ck", "key")
        self.assertEqual(calls[-1][2]["gpu"]["count"], 2)

    def test_reports_every_gpu_when_none_can_be_rented(self):
        from remote.providers.runpod.provider import RunpodAPIError
        provider, _, _ = self.provider([
            RunpodAPIError("POST", "/pods", 400, "no instances available"),
            RunpodAPIError("POST", "/pods", 422, "unknown GPU type"),
        ], gpus=["GPU A", "GPU B"])
        with self.assertRaisesRegex(RuntimeError, "GPU A: no instances.*\n.*GPU B: unknown"):
            provider.create("ck", "key")

    def test_auth_errors_are_not_retried_on_other_gpus(self):
        from remote.providers.runpod.provider import RunpodAPIError
        provider, calls, _ = self.provider([
            RunpodAPIError("POST", "/pods", 401, "unauthorized"),
        ], gpus=["GPU A", "GPU B"])
        with self.assertRaises(RunpodAPIError):
            provider.create("ck", "key")
        self.assertEqual(len(calls), 1)

    def test_network_volume_mounts_and_pins_the_data_center(self):
        provider, calls, _ = self.provider([
            {"id": "vol1", "dataCenter": "EU-RO-1"},
            {"id": "pod1"},
        ], gpus=["GPU A"], volume="vol1")
        provider.create("ck", "key")
        self.assertEqual(calls[0][:2], ("GET", "/network-volumes/vol1"))
        body = calls[1][2]
        self.assertEqual(body["mounts"], {"network": [{"volumeId": "vol1", "path": "/workspace"}]})
        self.assertEqual(body["dataCenterIds"], ["EU-RO-1"])
        self.assertEqual(provider.workdir, "/workspace")

    def test_waits_for_the_direct_ssh_endpoint(self):
        provider, _, rp = self.provider([
            {"status": "PROVISIONING", "ssh": {"proxy": None, "direct": None}},
            {"status": "RUNNING", "ssh": {"proxy": {"host": "ssh.runpod.io"},
                                          "direct": {"host": "1.2.3.4", "port": 22022,
                                                     "username": "root"}}},
        ])
        with mock.patch.object(rp.time, "sleep"):
            target = provider.wait_for_ssh("pod1", 60)
        self.assertEqual(target, SSHTarget("1.2.3.4", 22022, "root"))

    def test_stopped_pod_fails_instead_of_waiting(self):
        provider, _, _ = self.provider([{"status": "ERROR", "ssh": {}}])
        with self.assertRaisesRegex(RuntimeError, "ERROR"):
            provider.wait_for_ssh("pod1", 60)

    def test_destroying_a_deleted_pod_is_fine(self):
        from remote.providers.runpod.provider import RunpodAPIError
        provider, _, _ = self.provider([RunpodAPIError("DELETE", "/pods/x", 404, "not found")])
        provider.destroy("x")


class MachinesTest(unittest.TestCase):
    def test_machine_created_after_stop_is_destroyed_at_once(self):
        provider = FakeProvider(mock.Mock())
        machines = run.Machines(provider, keep=False)
        machines.release_all()
        with self.assertRaises(RuntimeError):
            machines.add("late")
        self.assertEqual(provider.destroyed, ["late"])


if __name__ == "__main__":
    unittest.main()
