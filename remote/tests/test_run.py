"""Offline tests for remote/run.py. No cloud account, SSH server, or GPU needed.

    python remote/tests/test_run.py

FakeProvider stands in for a cloud API and FakeRemote for the SSH layer, so
these tests cover the runner's scheduling and cleanup, not a real provider.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from remote import run  # noqa: E402
from remote.providers import Provider, SSHTarget  # noqa: E402


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

    def __init__(self, target, key, label):
        self.host = target.host
        self.dead = False

    def alive(self):
        return not self.dead

    def wait_ready(self, timeout):
        pass

    def run(self, command):
        FakeRemote.commands.append((self.host, command))
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

    def download(self, src, local_dir):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "alpha.mp4").write_bytes(b"")


class RunnerTest(unittest.TestCase):
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
