"""HTTP client for the render server (stdlib only)."""
from __future__ import annotations

import http.client
import io
import json
import os
from urllib.parse import urlsplit

import numpy as np

DEFAULT_URL = "http://127.0.0.1:18770"


class RenderClient:
    def __init__(self, url: str | None = None, timeout: float = 900.0, token: str | None = None):
        self.url = (url or os.environ.get("OPTICALNAV_SIM_URL") or DEFAULT_URL).rstrip("/")
        self.token = token or os.environ.get("OPTICALNAV_SIM_TOKEN")
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"render server URL must be http(s)://host[:port], got {self.url!r}")
        self._host, self._port, self._https = parts.hostname, parts.port, parts.scheme == "https"
        self._prefix = parts.path.rstrip("/")
        self.timeout = timeout
        self._conn = None
        self.last_render_seconds = 0.0
        self.last_timing: dict[str, float] = {}

    def _connection(self):
        if self._conn is None:
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            self._conn = cls(self._host, self._port, timeout=self.timeout)
        return self._conn

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str = "application/json") -> tuple[bytes, dict]:
        headers = {"Content-Type": content_type} if body is not None else {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in (0, 1):  # one reconnect for a keep-alive socket the server closed
            conn = self._connection()
            try:
                conn.request(method, self._prefix + path, body=body, headers=headers)
                resp = conn.getresponse()
                length = resp.getheader("Content-Length")
                if length is not None:  # read into a buffer of our own: arrays viewing it stay writable
                    data = bytearray(int(length))
                    view, got = memoryview(data), 0
                    while got < len(data):
                        n = resp.readinto(view[got:])
                        if not n:
                            raise ConnectionError("render server closed the connection mid-reply")
                        got += n
                else:
                    data = resp.read()
                break
            except (http.client.HTTPException, ConnectionError, OSError):
                conn.close()
                self._conn = None
                if attempt:
                    raise
        if resp.status != 200:
            try:
                message = json.loads(data).get("error", data[:200])
            except ValueError:
                message = data[:200]
            raise RuntimeError(f"render server {resp.status} on {path}: {message}")
        return data, dict(resp.getheaders())

    def _json(self, path: str):
        return json.loads(self._request("GET", path)[0])

    def info(self) -> dict:
        return self._json("/v1/info")

    def status(self) -> dict:
        """What the render server is doing now (loading a scene, recording a freeze, rendering)."""
        return self._json("/v1/status")

    def scans(self) -> list[str]:
        return self._json("/v1/scans")

    def meta(self, scan: str) -> dict:
        return self._json(f"/v1/scans/{scan}/meta")

    def stats(self, scan: str) -> dict:
        """What the render server's resident scene for ``scan`` holds (shapes, BSDFs, lights, GPU memory)."""
        return self._json(f"/v1/scans/{scan}/stats")

    def connectivity(self, scan: str) -> list[dict]:
        return self._json(f"/v1/scans/{scan}/connectivity")

    def render(self, views: list[dict], stokes_dtype: str = "float16") -> list[dict[str, np.ndarray]]:
        """Render dataset-convention cameras. Each view: scan, variant, camera_to_world (4x4),
        width, height, hfov_deg, spp, seed, mode ("polar" | "rgb"), denoise. Returns per view
        {rgb, s0, s1, s2, s3} in polar mode or {rgb, radiance} in rgb mode."""
        payload = {"views": [dict(v, camera_to_world=np.asarray(v["camera_to_world"]).tolist()) for v in views],
                   "stokes_dtype": stokes_dtype, "format": "raw"}
        data, headers = self._request("POST", "/v1/render", json.dumps(payload).encode())
        self.last_render_seconds = float(headers.get("X-Render-Seconds", 0.0))
        self.last_timing = json.loads(headers.get("X-Render-Timing", "{}"))
        if headers.get("Content-Type") == "application/x-opticalnav-arrays":  # header + raw bytes: views, no copy
            size = int.from_bytes(data[:8], "little")
            arrays = {name: np.frombuffer(data, dtype=np.dtype(dt), count=int(np.prod(shape)) if shape else 1,
                                          offset=8 + size + off).reshape(shape)
                      for name, dt, shape, off, _ in json.loads(data[8:8 + size])}
        else:  # an older server: npz
            npz = np.load(io.BytesIO(data))
            arrays = {key: npz[key] for key in npz.files}
        out = [{} for _ in views]
        for key, value in arrays.items():
            name, _, index = key.rpartition("_")
            out[int(index)][name] = value
        return out

    def denoise(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """OptiX-denoise finished images on the render server: ``s0``, ``s1``, ``s2`` (polar; S3 is left
        alone) or ``radiance``, each HxWx3, and/or the near-infrared ``nir_s0``, ``nir_s1``, ``nir_s2``
        (HxW), all in one call. Denoise an accumulated mean, not each pass before averaging. Sent and
        returned as float16; returned as float32."""
        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v, dtype=np.float16) for k, v in images.items()})
        data, _ = self._request("POST", "/v1/denoise", buf.getvalue(), "application/octet-stream")
        npz = np.load(io.BytesIO(data))
        return {k: npz[k].astype(np.float32) for k in npz.files}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
