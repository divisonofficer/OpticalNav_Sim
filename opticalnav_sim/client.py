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

    def _request(self, method: str, path: str, body: bytes | None = None) -> tuple[bytes, dict]:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in (0, 1):  # one reconnect for a keep-alive socket the server closed
            conn = self._connection()
            try:
                conn.request(method, self._prefix + path, body=body, headers=headers)
                resp = conn.getresponse()
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

    def scans(self) -> list[str]:
        return self._json("/v1/scans")

    def meta(self, scan: str) -> dict:
        return self._json(f"/v1/scans/{scan}/meta")

    def connectivity(self, scan: str) -> list[dict]:
        return self._json(f"/v1/scans/{scan}/connectivity")

    def render(self, views: list[dict], stokes_dtype: str = "float16") -> list[dict[str, np.ndarray]]:
        """Render dataset-convention cameras. Each view: scan, variant, camera_to_world (4x4),
        width, height, hfov_deg, spp, seed, mode ("polar" | "rgb"), denoise. Returns per view
        {rgb, s0, s1, s2, s3} in polar mode or {rgb, radiance} in rgb mode."""
        payload = {"views": [dict(v, camera_to_world=np.asarray(v["camera_to_world"]).tolist()) for v in views],
                   "stokes_dtype": stokes_dtype}
        data, headers = self._request("POST", "/v1/render", json.dumps(payload).encode())
        self.last_render_seconds = float(headers.get("X-Render-Seconds", 0.0))
        self.last_timing = json.loads(headers.get("X-Render-Timing", "{}"))
        npz = np.load(io.BytesIO(data))
        out = [{} for _ in views]
        for key in npz.files:
            name, _, index = key.rpartition("_")
            out[int(index)][name] = npz[key]
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
