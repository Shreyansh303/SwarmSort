"""Tests for src/watchdog_run.py with tiny Python child scripts (a few seconds each, no training)."""
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import watchdog_run  # noqa: E402
from watchdog_run import StallError, run_watched  # noqa: E402

STALL = 3 / 60  # 3-second stall limit, in minutes


def pid_alive(pid):
    try:
        import psutil
    except ImportError:  # Linux fallback
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class TestWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def script(self, body):
        """A child script; every start appends a line to attempts.txt."""
        path = self.dir / f"child_{len(list(self.dir.glob('child_*.py')))}.py"
        path.write_text(textwrap.dedent(f"""
            import os, sys, time, subprocess
            HERE = {str(self.dir)!r}
            open(os.path.join(HERE, "attempts.txt"), "a").write("x\\n")
        """) + textwrap.dedent(body))
        return [sys.executable, str(path)]

    def attempts(self):
        f = self.dir / "attempts.txt"
        return len(f.read_text().splitlines()) if f.exists() else 0

    def test_a_normal_script_streams_output(self):
        cmd = self.script("""
            for i in range(3):
                print("line", i, flush=True)
            sys.stderr.write("to stderr\\n")
        """)
        result = run_watched(cmd, stall_minutes=STALL, out=self.out)
        self.assertEqual(result.returncode, 0)
        text = self.out.getvalue()
        for s in ("line 0", "line 1", "line 2", "to stderr"):
            self.assertIn(s, text)
        self.assertNotIn("STALL", text)
        self.assertEqual(self.attempts(), 1)

    def test_b_stalled_script_is_killed_and_retried(self):
        cmd = self.script("""
            marker = os.path.join(HERE, "marker")
            if not os.path.exists(marker):
                open(marker, "w").close()
                print("first attempt, now freezing", flush=True)
                time.sleep(120)
                print("never printed", flush=True)
            else:
                print("second attempt finished", flush=True)
        """)
        start = time.monotonic()
        run_watched(cmd, stall_minutes=STALL, max_retries=2, out=self.out)
        text = self.out.getvalue()
        self.assertLess(time.monotonic() - start, 60)
        self.assertIn("first attempt, now freezing", text)
        self.assertIn("STALL", text)
        self.assertIn("retry 1 of 2", text)
        self.assertIn("second attempt finished", text)
        self.assertNotIn("never printed", text)
        self.assertEqual(self.attempts(), 2)

    def test_b2_retry_command_is_used_for_retries(self):
        cmd = self.script("""
            if "--resume" not in sys.argv:
                print("fresh start, freezing", flush=True)
                time.sleep(120)
            print("resumed:", sys.argv[1:], flush=True)
        """)
        run_watched(cmd, retry_cmd=cmd + ["--resume"], stall_minutes=STALL, max_retries=1, out=self.out)
        self.assertIn("resumed: ['--resume']", self.out.getvalue())
        self.assertEqual(self.attempts(), 2)

    def test_c_always_stalling_script_fails_after_max_retries(self):
        cmd = self.script("""
            print("starting", flush=True)
            time.sleep(120)
        """)
        with self.assertRaises(StallError) as ctx:
            run_watched(cmd, stall_minutes=STALL, max_retries=2, out=self.out)
        self.assertIn("stalled 3 times", str(ctx.exception))
        self.assertIn("gave up after 2 retries", str(ctx.exception))
        self.assertEqual(self.attempts(), 3)
        self.assertEqual(self.out.getvalue().count("STALL"), 3)

    def test_d_nonzero_exit_raises_without_retry(self):
        cmd = self.script("""
            print("about to fail", flush=True)
            sys.exit(3)
        """)
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            run_watched(cmd, stall_minutes=STALL, max_retries=3, out=self.out)
        self.assertEqual(ctx.exception.returncode, 3)
        self.assertEqual(self.attempts(), 1)
        self.assertNotIn("STALL", self.out.getvalue())

    def test_e_child_processes_are_killed_too(self):
        cmd = self.script("""
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            open(os.path.join(HERE, "grandchild.pid"), "w").write(str(child.pid))
            print("started a worker, now freezing", flush=True)
            time.sleep(120)
        """)
        with self.assertRaises(StallError):
            run_watched(cmd, stall_minutes=STALL, max_retries=0, out=self.out)
        pid = int((self.dir / "grandchild.pid").read_text())
        deadline = time.monotonic() + 10
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(pid_alive(pid), f"the grandchild {pid} survived the kill")

    def test_f_file_changes_count_as_progress(self):
        """Silent, but writing into the watched folder: not a stall."""
        run_dir = self.dir / "run"
        run_dir.mkdir()
        cmd = self.script(f"""
            print("silent from now on", flush=True)
            for i in range(14):
                time.sleep(0.5)
                open(os.path.join({str(run_dir)!r}, "results.csv"), "a").write(f"{{i}}\\n")
            print("done", flush=True)
        """)
        run_watched(cmd, watch=[run_dir], stall_minutes=2 / 60, max_retries=0, out=self.out)
        self.assertNotIn("STALL", self.out.getvalue())
        self.assertIn("done", self.out.getvalue())

    def make_tree(self):
        src = self.dir / "src"
        for i in range(30):
            (src / "images").mkdir(parents=True, exist_ok=True)
            (src / "labels").mkdir(exist_ok=True)
            (src / "images" / f"{i}.jpg").write_bytes(os.urandom(100 + i))
            (src / "labels" / f"{i}.txt").write_text(f"0 0.5 0.5 0.1 0.1\n{i}\n")
        (src / "data.yaml").write_text("names: [a]\n")
        (src / "labels.cache").write_text("stale")
        return src

    def test_copy_tree_counts_and_sizes(self):
        src = self.make_tree()
        files = watchdog_run.list_tree(src, exclude=["*.cache"])
        self.assertEqual(len(files), 61)
        self.assertNotIn("labels.cache", [f for f, _ in files])
        info = watchdog_run.copy_files(src, self.dir / "dst", files, workers=4, out=self.out)
        self.assertEqual(info["files"], 61)
        self.assertEqual(info["bytes"], sum(s for _, s in files))
        self.assertEqual(watchdog_run.list_tree(self.dir / "dst"), files)
        self.assertIn("copied 61/61 files", self.out.getvalue())

    def test_copy_stall_fails_loudly(self):
        import shutil
        from unittest import mock

        src = self.make_tree()
        files = watchdog_run.list_tree(src)
        real = shutil.copyfile

        def slow(a, b):
            if a.endswith("7.jpg"):
                time.sleep(30)  # a read that hangs on the input mount
            return real(a, b)

        start = time.monotonic()
        with mock.patch.object(shutil, "copyfile", slow), self.assertRaises(TimeoutError) as ctx:
            watchdog_run.copy_files(src, self.dir / "dst", files, workers=1, stall_seconds=1, out=self.out)
        self.assertLess(time.monotonic() - start, 10)
        self.assertIn("made no progress", str(ctx.exception))

    def test_newest_mtime(self):
        self.assertIsNone(watchdog_run.newest_mtime([self.dir / "missing"]))
        (self.dir / "a").mkdir()
        f = self.dir / "a" / "f.txt"
        f.write_text("x")
        os.utime(f, (2e9, 2e9))
        self.assertEqual(watchdog_run.newest_mtime([self.dir, self.dir / "missing"]), 2e9)


if __name__ == "__main__":
    unittest.main()
