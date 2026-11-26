"""A camera reading's pure part in other processes: the sensor's noise and conversion (hw.sensor.expose) and the ISP
(hw.isp.process), most of a reading's time, off the caller's GIL.

    pool = offload.shared()          # None when switched off (OPTICALNAV_ROBOT_READ_PROCESSES=0)
    reading = cam.spec.read(world, cam, request, outs, develop=pool.develop)

The viewer reads its robot's cameras on threads beside its editor, its HTTP handlers and each other: an HDR reading of
~80 ms alone took 130-145 ms there. Most of a reading is numpy, which lets the GIL go during an operation, but each
operation must take it back, and a thread running Python keeps it for up to the switch interval (5 ms). What stays
with the caller is SimWorld.meter: the lens's image of the render (its sampling table is cached per lens in this
process), the light on the wells, and the auto exposure, which has state and sees the frames one by one, in order
(~15 ms of the HDR reading). The rest is a pure function of the light, the camera's spec, the setting and the seed:
the same raw and picture in whichever process (``develop`` here is that function). What travels is the light (H x W
float64, 2.7 MB for the HDR camera's 720 x 464) one way, the raw and the picture (2.3 MB) back; pickling them takes
0.2 ms, the pipe a few ms.

Measured offline (a synthetic render, the ranger_mini's hdr_12bit_mast, 2 workers): alone, an HDR reading
86 ms here, 79 ms with the pool (its workers' heaps, below); 12-bit 26 and 27 ms. Both cameras read at once beside
three threads of Python and small numpy (an editor's kind of work): HDR 127 ms here, 95 ms with the pool; 12-bit 53
and 36 ms; the busy threads lost 35 % of their speed to the readings, 18-25 % with the pool. Beside one thread of pure
Python: HDR 1250 ms here, 300 ms with the pool.

Each worker is a process with a pipe of its own, which the calling thread writes and reads itself: a
concurrent.futures.ProcessPoolExecutor passes a job through two threads of its own on the way out and one on the way
back, each of which waits for the GIL as the reading's own operations do (with it, the same workers: the HDR reading
alone 91 ms, not 79; beside the thread of pure Python 455 ms, not 300). The workers are started by "spawn" (the viewer
has many threads: a fork would copy locks some of them hold), once, and the readings run here until they are up
(~0.3 s). Spawn imports the caller's main module again in each worker: the viewer's (``-m opticalnav_sim.gui``) only
defines things; a script that uses the pool needs its ``if __name__ == "__main__":`` guard, as with any spawn
(without it, its body runs again in each worker). A pool that fails to start, a worker that dies, an answer that does
not come: the reading runs here, the pool is switched off for good, and that is said once on stderr. An error of the
reading itself is raised as it would be without the pool.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import sys
import threading
from time import monotonic, sleep
from types import SimpleNamespace

from . import isp, sensor as sensor_model

# worker processes the robot cameras share (0: every reading in the calling process). Two: the rigs have one or two
# lit cameras, each with at most one reading under way (robot_view._measure)
PROCESSES = max(0, int(os.environ.get("OPTICALNAV_ROBOT_READ_PROCESSES", "2")))
START_S = 60.0     # a worker not up by then is not coming
TIMEOUT_S = 20.0   # an answer later than this is a worker that hangs


def develop(spec, light, exposure_s: float, gain_db: float, seed, pixels_per_sample: float, intrinsics, pose, time: float,
            noise: bool = True) -> tuple:
    """A camera's raw of the ``light`` on its wells (hw.sensor.irradiance) at this setting, and its ISP's picture of it:
    ``(Raw, H x W x 3 uint8)``. ``intrinsics``, ``pose``, ``time``: the frame's, carried into the raw."""
    frame = SimpleNamespace(intrinsics=intrinsics, pose=pose, time=time, s0=None, s1=None, s2=None)
    raw = sensor_model.expose(frame, spec, exposure_s, gain_db, noise, seed=seed, pixels_per_sample=pixels_per_sample,
                              light=light)
    return raw, isp.process(raw, spec.processing())


def _worker_init() -> None:
    """A worker's start. Its heap keeps the memory a reading frees: glibc gives a block of more than 128 kB back to the
    system when it is freed and faults it in again, page by page, at the next reading. An HDR reading makes ~20 arrays
    of 1-3 MB: 12 000 page faults, 77 ms where the viewer's process (its heap grown long ago) took 67; with the
    thresholds raised, none, 64 ms."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C is the viewer's: a worker ends when its pipe closes
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 256 << 20)   # M_MMAP_THRESHOLD: blocks up to 256 MB from the heap, not mmap
        libc.mallopt(-1, 512 << 20)   # M_TRIM_THRESHOLD: the heap's top is kept up to 512 MB
    except (OSError, AttributeError):
        pass   # not glibc: as it is
    try:
        import scipy.ndimage  # noqa: F401 - the ISP's filters, before the first reading
    except ImportError:
        pass


def _serve(conn) -> None:
    """A worker: jobs from its pipe, answers back, until the pipe closes (the caller's process ended or stopped it)."""
    _worker_init()
    conn.send(("ready", os.getpid()))
    while True:
        try:
            args, kwargs = conn.recv()
        except (EOFError, OSError):
            return
        try:
            answer = ("ok", develop(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - the caller runs it again itself and sees the error there
            answer = ("error", f"{type(exc).__name__}: {exc}")
        try:
            conn.send(answer)
        except (EOFError, OSError):
            return


class Pool:
    """Worker processes that ``develop`` camera readings; see the module."""

    def __init__(self, processes: int = PROCESSES):
        self.processes = int(processes)
        self._lock = threading.Lock()
        self._workers: list = []          # (process, its end of the pipe), started
        self._idle: queue.Queue = queue.Queue()   # the started ones not reading now
        self._up = 0                      # workers that said they are ready
        self.off = None if self.processes > 0 else "switched off"   # why the readings run in the caller, or None

    def start(self, timeout: float | None = None) -> bool:
        """Start the workers now (else at the first reading), and wait up to ``timeout`` s for them. True: they read."""
        with self._lock:
            if self.off is None and not self._workers:
                try:
                    ctx = multiprocessing.get_context("spawn")
                    for i in range(self.processes):
                        mine, theirs = ctx.Pipe()
                        # daemon: ended with the viewer (and a worker whose pipe closes ends by itself)
                        proc = ctx.Process(target=_serve, args=(theirs,), name=f"opticalnav-read-{i}", daemon=True)
                        proc.start()
                        theirs.close()
                        self._workers.append((proc, mine))
                except Exception as exc:  # noqa: BLE001 - no processes here (limits, a sandbox): read in the caller
                    self._stop(exc)
                else:
                    for w in self._workers:
                        threading.Thread(target=self._await, args=(w,), name="opticalnav-read-start", daemon=True).start()
        if timeout is not None:
            end = monotonic() + timeout
            while self.off is None and self._up < len(self._workers) and monotonic() < end:
                sleep(0.02)
        return self.off is None and self._up > 0

    def _await(self, worker) -> None:
        proc, conn = worker
        try:
            if not conn.poll(START_S):
                raise TimeoutError(f"a worker did not start in {START_S:.0f} s")
            conn.recv()
        except Exception as exc:  # noqa: BLE001 - it died while starting (its traceback is on stderr)
            with self._lock:
                self._stop(exc)
            return
        with self._lock:
            if self.off is None:
                self._up += 1
                self._idle.put(worker)

    def _stop(self, exc: BaseException) -> None:   # (under the lock)
        if self.off is None:
            self.off = f"{type(exc).__name__}: {exc}"
            print(f"[offload] camera readings in the viewer's process from now on ({self.off})", file=sys.stderr, flush=True)
        for proc, conn in self._workers:
            for end in (conn.close, proc.terminate):
                try:
                    end()
                except Exception:  # noqa: BLE001 - gone already
                    pass
        self._workers, self._up = [], 0
        for _ in range(self.processes):
            self._idle.put(None)   # a reading waiting for a worker runs here

    def develop(self, *args, **kwargs) -> tuple:
        """``develop`` in a worker when the pool runs (waiting for a free one), else here."""
        if self.off is None and not self._workers:
            self.start()
        worker = None
        if self.off is None and self._up > 0:
            try:
                worker = self._idle.get(timeout=TIMEOUT_S)
            except queue.Empty:
                pass   # all of them busy that long: the ones holding them say what went wrong
        if worker is None or self.off is not None:
            return develop(*args, **kwargs)
        proc, conn = worker
        try:
            conn.send((args, kwargs))
            if not conn.poll(TIMEOUT_S):
                raise TimeoutError(f"no answer from a worker in {TIMEOUT_S:.0f} s")
            status, value = conn.recv()
        except Exception as exc:  # noqa: BLE001 - a dead worker, a pickling failure, a timeout
            out = develop(*args, **kwargs)   # an error of the reading itself is raised here, as without the pool
            with self._lock:
                self._stop(exc)              # it ran here: the pool's failure
            return out
        self._idle.put(worker)               # it answered: free again
        if status == "ok":
            return value
        out = develop(*args, **kwargs)       # the error it sent back: the reading's own is raised here
        with self._lock:
            self._stop(RuntimeError(f"in a worker: {value}"))   # it ran here: the worker's failure (memory, a module)
        return out

    def pids(self) -> list:
        with self._lock:
            return [proc.pid for proc, _ in self._workers]

    def shutdown(self) -> None:
        with self._lock:
            workers, self._workers, self._up, self.off = self._workers, [], 0, self.off or "shut down"
            for _ in range(self.processes):
                self._idle.put(None)
        for proc, conn in workers:
            conn.close()   # the worker's recv ends: it returns
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()


_shared: Pool | None = None
_shared_lock = threading.Lock()


def shared() -> Pool | None:
    """The process's pool (PROCESSES workers, started at its first reading and kept: a robot spawned again, or another
    rig, reads with the same workers), or None when switched off."""
    global _shared
    if PROCESSES <= 0:
        return None
    with _shared_lock:
        if _shared is None:
            _shared = Pool(PROCESSES)
        return _shared
