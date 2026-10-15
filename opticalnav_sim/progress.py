"""Progress of the render server's two long waits: loading a scene and recording a frozen render.

Neither Mitsuba nor Dr.Jit reports a fraction done, but both log their steps:

* Dr.Jit (``jit_set_log_level_callback`` in libdrjit-core, hooked through ctypes at Info level) logs
  ``jit_eval(): launching N kernels`` when tracing ends and code generation starts, then one
  ``-> launching <hash> (... jit=..)`` line as each kernel's code is generated, and ``cache hit/miss``
  for the compile. A recording is tracing (no messages; about 20 s on a Device 1 apartment), code
  generation (about 55 s) and compile (0 s on a kernel-cache hit). ``FreezeProgress`` turns that into a
  stage, kernel k of N and, from the last recording of the same scene and settings (kept in
  ``~/.cache/opticalnav_sim/freeze_timings.json``), a percentage and the time left.
* Mitsuba logs ``Loading mesh from "..."`` and ``Loading PNG file "..."`` (Debug) for each file and
  ``Building scene in OptiX`` at the end. ``LoadProgress`` counts the distinct files against those the
  scene file names.
"""
from __future__ import annotations

import ctypes
import json
import re
import sys
import threading
import time
from pathlib import Path

HISTORY = Path.home() / ".cache" / "opticalnav_sim" / "freeze_timings.json"
# Before any history, estimates scale the scene file: a recording call walks every BSDF, texture and shape
# (tracing) and generates code for their calls (codegen), a load reads every file. Measured on Device 1 with
# infinigen_apartment_natural_v1_20268504 (7685 such elements, 3527 files): 91.5 s per recording call,
# 131-141 s per load.
SECONDS_PER_ELEMENT = 91.5 / 7685
SECONDS_PER_FILE = 136.0 / 3527


def scene_score(text: str) -> int:
    """BSDFs, textures and shapes in a scene file: what a freeze recording traces and generates code for."""
    return sum(len(re.findall(rf"<{tag}\b", text)) for tag in ("bsdf", "texture", "shape"))


def _rate(entries, num: str, den: str, default: float) -> tuple[float, str]:
    """Median seconds per unit over past entries, or the default."""
    rates = sorted(e[num] / e[den] for e in entries if isinstance(e, dict) and e.get(den))
    return (rates[len(rates) // 2], "past") if rates else (default, "default")
_LAUNCHING = re.compile(r"jit_eval\(\): launching (\d+) kernels?")
_FILE = re.compile(r'Loading (?:mesh from|[A-Za-z]+ file) "([^"]+)"')


class FreezeProgress:
    """Stage and estimate of the recording in progress, fed by Dr.Jit's log (see install)."""

    def __init__(self, history: Path = HISTORY):
        self.lock = threading.Lock()
        self.history_path = history
        try:
            self.history: dict[str, float] = json.loads(history.read_text())
        except (OSError, ValueError):
            self.history = {}
        self.key = None
        self.hooked = False
        self._reset()

    def _reset(self) -> None:
        self.started = self.stage_since = 0.0
        self.stage = ""
        self.call = self.calls = 0
        self.kernels = [0, 0]
        self.misses = 0
        self.recorded = False  # a replay that recorded after all (its input layout changed)
        self.remember = True
        self.score = 0
        self.tail: list[str] = []  # the recording's last log lines

    # --- driven by the server ---
    def begin(self, key: str, calls: int = 2, remember: bool = True, score: int = 0) -> None:
        """A recording of ``calls`` calls starts (the server's first recording makes two). ``remember``:
        keep its duration for later estimates; a replay, which records only if its inputs changed
        layout, passes False and calls=1."""
        with self.lock:
            self._reset()
            self.key, self.calls, self.remember, self.score = key, calls, remember, int(score)
            self.started = self.stage_since = time.time()

    def next_call(self) -> None:
        with self.lock:
            self.call += 1
            self.stage, self.stage_since, self.kernels = "tracing", time.time(), [0, 0]

    def save(self) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(self.history, indent=1))
        except OSError:
            pass

    def end(self, ok: bool = True) -> None:
        with self.lock:
            if ok and self.key is not None and self.remember:  # seconds per call, and the scene's score
                self.history[self.key] = {"per_call": round((time.time() - self.started) / max(1, self.calls), 1),
                                          "score": self.score}
                self.save()
            self.key = None
            self._reset()

    # --- driven by Dr.Jit's log ---
    def on_log(self, text: str) -> None:
        if "changed from" in text or "re-recording" in text:
            # why a frozen function records again (Dr.Jit's diff of the new inputs against the recorded ones)
            print(f"[freeze] {text.strip()}", flush=True)
        if self.key is None:
            return
        if "Replaying" in text:
            return
        with self.lock:
            self.recorded = True
            line = text.strip()
            if line and not line.startswith("jit_eval(): done"):
                self.tail = (self.tail + [f"{time.time() - self.started:6.1f}s  {line[:160]}"])[-8:]
            found = _LAUNCHING.search(text)
            if found:
                self.stage, self.stage_since = "codegen", time.time()
                self.kernels = [0, int(found.group(1))]
            elif "-> launching" in text:
                self.kernels[0] += 1
            elif "cache miss" in text:
                self.misses += 1
            elif "jit_eval(): done" in text:
                self.stage, self.stage_since = "tracing", time.time()

    def snapshot(self) -> dict | None:
        with self.lock:
            if self.key is None or (not self.remember and not self.recorded):
                return None  # a plain replay: nothing to report
            now = time.time()
            out = {"stage": self.stage or "tracing", "call": self.call, "calls": self.calls,
                   "kernels": list(self.kernels), "elapsed_s": round(now - self.started, 1),
                   "stage_s": round(now - self.stage_since, 1), "log": self.hooked}
            out["log_tail"] = list(self.tail)
            past = self.history.get(self.key)
            per_call = past.get("per_call") if isinstance(past, dict) else past
            if per_call:
                out["eta_basis"] = "same settings"
            elif self.score:
                rate, basis = _rate(self.history.values(), "per_call", "score", SECONDS_PER_ELEMENT)
                per_call = rate * self.score
                out["eta_basis"] = f"scene size ({self.score} elements, {basis} rate)"
            expected = per_call * max(1, self.calls) if per_call else None
            if expected:
                done = (now - self.started) / expected
                out["percent"] = int(min(99, 100 * done))
                out["eta_s"] = round(max(0.0, expected - (now - self.started)), 1)
            return out


class LoadProgress:
    """Files read so far while a scene loads, from Mitsuba's Debug log (see watch_load)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.total, self.seen, self.stage, self.started = 0, set(), "", 0.0

    def begin(self, scene_text: str) -> None:
        with self.lock:
            self.total = len(set(re.findall(r'name="filename" value="([^"]+)"', scene_text)))
            self.seen, self.stage, self.started, self.active = set(), "files", time.time(), True
            loads = [v for k, v in FREEZE.history.items() if k.startswith("load|")]
            self.rate, self.basis = _rate(loads, "seconds", "files", SECONDS_PER_FILE)

    def on_log(self, text: str) -> None:
        if not self.active:
            return
        found = _FILE.search(text)
        with self.lock:
            if found:
                self.seen.add(Path(found.group(1)).name)
            elif "Building scene in OptiX" in text:
                self.stage = "optix"

    def end(self, name: str | None = None) -> None:
        """``name``: remember this load's time per file (the scene loaded)."""
        with self.lock:
            self.active = False
            seconds = time.time() - self.started
        if name and self.total:
            with FREEZE.lock:
                FREEZE.history[f"load|{name}"] = {"seconds": round(seconds, 1), "files": self.total}
                FREEZE.save()

    def snapshot(self) -> dict | None:
        with self.lock:
            if not self.active:
                return None
            done = min(len(self.seen), self.total)
            elapsed = time.time() - self.started
            expected = self.rate * self.total
            return {"stage": self.stage, "files": [done, self.total], "elapsed_s": round(elapsed, 1),
                    "percent": int(100 * done / self.total) if self.total and self.stage == "files" else None,
                    "eta_s": round(max(0.0, expected - elapsed), 1) if self.total else None,
                    "eta_basis": f"{self.total} files, {self.basis} rate"}


FREEZE = FreezeProgress()
LOAD = LoadProgress()
_KEEP: list = []  # the ctypes callback must outlive the hook


def install() -> bool:
    """Hook Dr.Jit's log (Info level) into FREEZE, its warnings and errors still printed. False when
    libdrjit-core cannot be found; recordings then report elapsed time only."""
    try:
        core = ctypes.CDLL("libdrjit-core.so")
        hook = core._Z26jit_set_log_level_callback8LogLevelPFvS_PKcE  # jit_set_log_level_callback(LogLevel, cb)
    except (OSError, AttributeError):
        return False
    def on_log(level, msg):
        if not msg:
            return
        text = msg.decode(errors="replace")
        if level <= 2:  # Error, Warn: the callback takes Dr.Jit's log over, so these are printed here (a failure's
            sys.stderr.write(f"[drjit] {text}\n")  # reason comes before its abort, else it is lost)
            sys.stderr.flush()
        FREEZE.on_log(text)

    callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)(on_log)
    hook(3, callback)  # LogLevel::Info
    _KEEP.append(callback)
    FREEZE.hooked = True
    return True


class watch_load:
    """While loading ``scene_text``: Mitsuba's logger at Debug feeding LOAD, its usual Info lines still
    printed, the per-file Debug lines not."""

    def __init__(self, mi, scene_text: str, name: str | None = None):
        self.mi, self.text, self.name = mi, scene_text, name

    def __enter__(self):
        mi = self.mi

        class Appender(mi.Appender):
            def append(self, level, text):
                LOAD.on_log(text)
                if int(level) >= int(mi.LogLevel.Info):
                    print(text, flush=True)

            def log_progress(self, progress, name, formatted, eta, ptr=None):
                pass

        self.logger = mi.logger() if hasattr(mi, "logger") else mi.Thread.thread().logger()
        self.level = self.logger.log_level()
        self.saved = [self.logger.appender(i) for i in range(self.logger.appender_count())]
        self.logger.clear_appenders()
        self.appender = Appender()
        self.logger.add_appender(self.appender)
        self.logger.set_log_level(mi.LogLevel.Debug)
        LOAD.begin(self.text)
        return self

    def __exit__(self, exc_type, *exc):
        LOAD.end(None if exc_type else self.name)
        self.logger.set_log_level(self.level)
        self.logger.remove_appender(self.appender)
        for appender in self.saved:
            self.logger.add_appender(appender)
        return False
