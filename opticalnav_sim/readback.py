"""Read a render back without waiting for the renders queued after it.

Dr.Jit runs a thread's work on one CUDA stream, and its readbacks (np.array, dr.sync_thread) wait for the whole
stream. The render server queues frame N+1 before it reads frame N back (Executor.submit_pipelined), so reading
N also waited for N+1, and the CPU half of the next frame (camera writes, the frozen replay: ~10 ms) then ran with
the GPU idle: a frame took GPU + CPU (512x384 16 spp: 38 + 10 ms, the GPU busy 81% of the time; 256x192: 58%).

Here an image's copy to page-locked host memory goes on the stream right behind its own kernels, followed by a
CUDA event, and finishing it waits for that event alone. Dr.Jit's Python API has neither, so this calls Dr.Jit's
core library (jit_cuda_stream, jit_var_data, jit_memcpy_async: exported C++ symbols of Dr.Jit 1.x) and the CUDA
driver through ctypes. ``AsyncReadback.create`` returns None where that is not possible (another build, no
CUDA); callers then read back as before.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

CU_EVENT_DISABLE_TIMING = 0x2
JIT_BACKEND_CUDA = 1  # drjit-core's JitBackend::CUDA


def _loaded(name: str) -> str | None:
    """The path of a shared library this process has mapped."""
    for line in Path("/proc/self/maps").read_text().splitlines():
        if line.endswith(name):
            return line.split(None, 5)[-1]
    return None


class AsyncReadback:
    """Queue device -> host copies of Dr.Jit arrays on the calling thread's stream, each with an event."""

    def __init__(self, core: ctypes.CDLL, cuda: ctypes.CDLL):
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        self._stream = core._Z15jit_cuda_streamv
        self._stream.restype, self._stream.argtypes = vp, []
        self._context = core._Z16jit_cuda_contextv
        self._context.restype, self._context.argtypes = vp, []
        self._var_data = core._Z12jit_var_datajPPv
        self._var_data.restype, self._var_data.argtypes = u32, [u32, ctypes.POINTER(vp)]
        self._dec_ref = core._Z20jit_var_dec_ref_implj
        self._dec_ref.restype, self._dec_ref.argtypes = None, [u32]
        self._memcpy_async = core._Z16jit_memcpy_async10JitBackendPvPKvm
        self._memcpy_async.restype, self._memcpy_async.argtypes = None, [u32, vp, vp, ctypes.c_size_t]
        self.cuda = cuda
        for fn, args in (("cuCtxPushCurrent_v2", [vp]), ("cuCtxPopCurrent_v2", [ctypes.POINTER(vp)]),
                         ("cuEventCreate", [ctypes.POINTER(vp), ctypes.c_uint]), ("cuEventRecord", [vp, vp]),
                         ("cuEventSynchronize", [vp]), ("cuEventDestroy_v2", [vp]),
                         ("cuMemHostAlloc", [ctypes.POINTER(vp), ctypes.c_size_t, ctypes.c_uint]),
                         ("cuMemFreeHost", [vp])):
            f = getattr(cuda, fn)
            f.restype, f.argtypes = ctypes.c_int, args
        self.pool: dict[int, list[int]] = {}  # page-locked buffers by size, reused

    @classmethod
    def create(cls) -> "AsyncReadback | None":
        try:
            path = _loaded("libdrjit-core.so")
            if path is None:
                return None
            return cls(ctypes.CDLL(path), ctypes.CDLL("libcuda.so.1"))
        except (OSError, AttributeError):
            return None

    def _check(self, err: int, what: str) -> None:
        if err:
            raise RuntimeError(f"CUDA driver error {err} in {what}")

    def _push(self) -> None:
        self._check(self.cuda.cuCtxPushCurrent_v2(self._context()), "cuCtxPushCurrent")

    def _pop(self) -> None:
        ctx = ctypes.c_void_p()
        self.cuda.cuCtxPopCurrent_v2(ctypes.byref(ctx))

    def start(self, array, shape: tuple | None = None) -> "Pending":
        """Queue the copy of ``array`` (a float32 Dr.Jit array or tensor: its kernels are queued or done) behind
        the work on this thread's stream. ``shape``: the numpy shape (a tensor's own by default)."""
        flat = getattr(array, "array", array)
        shape = tuple(array.shape) if shape is None and hasattr(array, "array") else tuple(shape or (len(flat),))
        ptr = ctypes.c_void_p()
        index = self._var_data(flat.index, ctypes.byref(ptr))  # evaluates if needed; a new reference
        nbytes = 4 * math.prod(shape)
        self._push()
        try:
            free = self.pool.get(nbytes)
            host = free.pop() if free else None
            if host is None:
                p = ctypes.c_void_p()
                self._check(self.cuda.cuMemHostAlloc(ctypes.byref(p), nbytes, 0), "cuMemHostAlloc")
                host = p.value
            self._memcpy_async(JIT_BACKEND_CUDA, host, ptr.value, nbytes)
            event = ctypes.c_void_p()
            self._check(self.cuda.cuEventCreate(ctypes.byref(event), CU_EVENT_DISABLE_TIMING), "cuEventCreate")
            self._check(self.cuda.cuEventRecord(event, self._stream()), "cuEventRecord")
        finally:
            self._pop()
        return Pending(self, array, index, host, nbytes, shape, event)


class Pending:
    """A queued copy: ``result()`` waits for its event alone and returns a numpy copy."""

    def __init__(self, owner: AsyncReadback, array, index: int, host: int, nbytes: int, shape: tuple, event):
        self.owner, self.array, self.index, self.host = owner, array, index, host
        self.nbytes, self.shape, self.event, self.value = nbytes, shape, event, None

    def result(self) -> np.ndarray:
        if self.value is not None:
            return self.value
        o = self.owner
        o._push()
        try:
            o._check(o.cuda.cuEventSynchronize(self.event), "cuEventSynchronize")
            o.cuda.cuEventDestroy_v2(self.event)
        finally:
            o._pop()
        buf = (ctypes.c_char * self.nbytes).from_address(self.host)
        self.value = np.frombuffer(buf, dtype=np.float32).reshape(self.shape).copy()
        o.pool.setdefault(self.nbytes, []).append(self.host)
        o._dec_ref(self.index)  # the device array may go now
        self.array = None
        return self.value
