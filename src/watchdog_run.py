"""Run a script under a stall watchdog: stream its output, and kill and retry it if it stops making progress.

A Kaggle training can freeze without an error (e.g. dataloader workers blocked on a slow network file system): the
process stays alive, prints nothing and writes nothing until Kaggle's 12-hour limit kills the session. run_watched()
counts two signs of progress: a new piece of output, and a change of the newest modification time under the watched
paths (the run folder, a search log). If neither happens for `stall_minutes`, the whole process group (the script and
its dataloader workers) is killed and the command is started again, up to `max_retries` times. The retry command
should continue the work (search.py replays its log; train_final.py --resume continues last.pt).

A non-zero exit that is not a stall raises subprocess.CalledProcessError at once, like subprocess.run(check=True).

list_tree() and copy_files() copy the attached dataset to local disk before training, with progress prints and a
stall limit, so a stalling input mount fails loudly at the start instead of freezing a training later.

    from watchdog_run import run_watched
    run_watched([sys.executable, "src/train_final.py", ..., "--resume"], watch=["results/gen2/runs/arm_c"])
"""
import codecs
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime


class StallError(RuntimeError):
    """The command stalled on every attempt."""


def newest_mtime(paths):
    """The newest modification time of any file or folder under `paths` (missing paths are skipped), or None."""
    newest = None
    for path in paths:
        path = os.fspath(path)
        try:
            stamps = [os.stat(path).st_mtime]
        except OSError:
            continue
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for name in dirs + files:
                    try:
                        stamps.append(os.stat(os.path.join(root, name)).st_mtime)
                    except OSError:
                        pass  # removed while walking
        newest = max(stamps) if newest is None else max(newest, *stamps)
    return newest


def _start(cmd, env, cwd):
    """Start cmd in its own process group, so it can be killed together with every process it starts."""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # new session and process group, id = the child's pid
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, cwd=cwd, **kwargs)


def kill_tree(proc):
    """Kill proc and every process it started (on Linux the whole process group, on Windows the process tree)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # already gone
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


class _Reader(threading.Thread):
    """Copies the child's output to `out` as it arrives and records when the last piece came."""

    def __init__(self, stream, out):
        super().__init__(daemon=True)
        self.stream, self.out = stream, out
        self.last = time.monotonic()
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def run(self):
        read = getattr(self.stream, "read1", self.stream.read)
        while True:
            try:
                chunk = read(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            self.last = time.monotonic()
            try:
                self.out.write(self.decoder.decode(chunk))
                self.out.flush()
            except Exception:  # a closed or broken output must not stop the reading (or the child blocks)
                pass


def _watch_threads(threads, progress, stall_seconds, what, report=None, report_every=15.0):
    """Wait for daemon threads; raise TimeoutError if progress() (a number) stops growing for stall_seconds.
    A hung read cannot be interrupted, but the caller gets a loud error instead of a silent hang."""
    last_value, last_change, last_report = progress(), time.monotonic(), time.monotonic()
    while True:
        alive = [t for t in threads if t.is_alive()]
        if not alive:
            return
        alive[0].join(timeout=min(1.0, stall_seconds / 10))
        value = progress()
        if value != last_value:
            last_value, last_change = value, time.monotonic()
        if time.monotonic() - last_change > stall_seconds:
            raise TimeoutError(f"{what} made no progress for {stall_seconds:.0f} s (stuck at {value}): the input "
                               f"file system is not responding. Run the notebook again (Save Version).")
        if report and time.monotonic() - last_report >= report_every:
            report()
            last_report = time.monotonic()


def list_tree(src, exclude=(), stall_seconds=120.0):
    """[(relative posix path, size)] of every file under src (sorted), skipping names matching `exclude` globs.
    Raises TimeoutError if the listing stalls."""
    from fnmatch import fnmatch

    src = os.fspath(src)
    files, errors = [], []

    def walk():
        try:
            stack = [""]
            while stack:
                rel = stack.pop()
                with os.scandir(os.path.join(src, rel)) as it:
                    for e in it:
                        r = f"{rel}/{e.name}" if rel else e.name
                        if e.is_dir(follow_symlinks=True):
                            stack.append(r)
                        elif not any(fnmatch(e.name, p) for p in exclude):
                            files.append((r, e.stat().st_size))
        except BaseException as e:  # reported by the caller
            errors.append(e)

    t = threading.Thread(target=walk, daemon=True)
    t.start()
    _watch_threads([t], lambda: len(files), stall_seconds, f"Listing {src}")
    if errors:
        raise errors[0]
    return sorted(files)


def copy_files(src, dst, files, workers=16, stall_seconds=120.0, out=None):
    """Copy `files` (from list_tree) from src to dst with `workers` threads, printing progress. Every copy is
    checked against its listed size. Raises TimeoutError if no file finishes for stall_seconds."""
    import queue
    import shutil

    out = out or sys.stdout
    src, dst = os.fspath(src), os.fspath(dst)
    todo = queue.Queue()
    for item in files:
        todo.put(item)
    state = {"files": 0, "bytes": 0}
    errors, lock, start = [], threading.Lock(), time.monotonic()
    total = sum(size for _, size in files)

    def worker():
        while not errors:
            try:
                rel, size = todo.get_nowait()
            except queue.Empty:
                return
            try:
                target = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(src, rel), target)
                got = os.path.getsize(target)
                if got != size:
                    raise OSError(f"{rel}: copied {got} bytes, the source has {size}")
            except BaseException as e:
                errors.append(e)
                return
            with lock:
                state["files"] += 1
                state["bytes"] += size

    def report():
        s = time.monotonic() - start
        out.write(f"  copied {state['files']}/{len(files)} files, {state['bytes'] / 1e9:.2f}/{total / 1e9:.2f} GB "
                  f"({state['bytes'] / 1e6 / max(s, 1e-9):.0f} MB/s)\n")
        out.flush()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    _watch_threads(threads, lambda: state["files"], stall_seconds, f"Copying {src} -> {dst}", report=report)
    if errors:
        raise errors[0]
    report()
    return {"files": state["files"], "bytes": state["bytes"], "seconds": round(time.monotonic() - start, 1)}


def run_watched(cmd, retry_cmd=None, watch=(), stall_minutes=20.0, max_retries=3, env=None, cwd=None,
                poll_seconds=None, out=None):
    """Run cmd, streaming its stdout and stderr to `out` (default sys.stdout). If there is no new output and no
    change under `watch` for stall_minutes, kill its process group and run retry_cmd (default: cmd again), at most
    max_retries times. Returns subprocess.CompletedProcess of the attempt that finished. Raises
    subprocess.CalledProcessError on a non-zero exit (no retry) and StallError if every attempt stalled."""
    out = out or sys.stdout
    stall_seconds = float(stall_minutes) * 60
    poll = poll_seconds if poll_seconds is not None else max(0.1, min(30.0, stall_seconds / 20))
    watch = [os.fspath(p) for p in watch]

    def log(msg):
        out.write(f"[watchdog {datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        out.flush()

    for attempt in range(max_retries + 1):
        args = [str(a) for a in (cmd if attempt == 0 or retry_cmd is None else retry_cmd)]
        if attempt:
            log(f"retry {attempt} of {max_retries}: " + " ".join(args))
        proc = _start(args, env, cwd)
        reader = _Reader(proc.stdout, out)
        reader.start()
        last_change, seen = time.monotonic(), newest_mtime(watch)
        stalled = False
        try:
            while proc.poll() is None:
                time.sleep(poll)
                stamp = newest_mtime(watch)
                if stamp != seen:
                    seen, last_change = stamp, time.monotonic()
                idle = time.monotonic() - max(reader.last, last_change)
                if idle > stall_seconds:
                    stalled = True
                    log(f"STALL: no output and no file change under {watch or '(nothing watched)'} for "
                        f"{idle / 60:.1f} min (limit {stall_minutes} min); killing process group {proc.pid}")
                    kill_tree(proc)
                    break
        except BaseException:  # e.g. the cell was interrupted: never leave the script running in the background
            kill_tree(proc)
            raise
        reader.join(timeout=10)  # an escaped grandchild may hold the pipe open: do not wait for it forever
        if not reader.is_alive():
            proc.stdout.close()
        if not stalled:
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, args)
            if attempt:
                log(f"finished after {attempt} retr{'y' if attempt == 1 else 'ies'}")
            return subprocess.CompletedProcess(args, 0)
        if attempt < max_retries:
            log(f"stall {attempt + 1}: the command will be started again")
    raise StallError(f"{' '.join(str(a) for a in cmd)} stalled {max_retries + 1} times (no output and no file "
                     f"change for {stall_minutes} min each time); gave up after {max_retries} retries")
