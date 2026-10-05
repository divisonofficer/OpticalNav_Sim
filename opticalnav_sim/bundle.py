"""Write simulator renders in the layout of robomituba's OpticalNav export bundle.

    <out>/
      index.jsonl                      one row per (frame, camera, modality, variant), same fields as the export
      dataset_meta.json                scenes, variants, counts, tonemap, render settings, schemas
      images/<variant>/<frame_id>__polar_cam__<modality>.jpg|png
      polarization_raw/<variant>/<frame_id>__polar_cam__stokes.npz   s0..s3 float16 HxWx3 (+ .source.json)
      hdr/<variant>/<frame_id>__polar_cam__s0.exr                     optional linear S0
      graph/<scene>__navigation_support_graph.json                    when given
      episodes/<split>/<episode_id>.json

``frame_id`` is ``<scene>_<node>_<heading>`` and ``vp_id`` / ``heading_id`` join
frames to episode steps, as in the export. Variants use the export's names
(``perturbed_active_polar`` for the simulator's ``active_polar``). Each row adds a
``render`` block (renderer, spp, seed, mode, denoiser) that the export does not have.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import stokes
from .episodes import flat, frame_id
from .tonemap import Tone

STOKES_SCHEMA = "minimal_rgb_stokes_f16_v2"
CAMERA_ID = "polar_cam"
MODALITIES = ("polar_rgb_preview", "dop", "aolp", "s1_over_s0", "s2_over_s0")
EXPORT_VARIANT = {"base": "base", "perturbed": "perturbed", "active_polar": "perturbed_active_polar"}


def export_variant(variant: str) -> str:
    return EXPORT_VARIANT.get(variant, variant)


class BundleWriter:
    def __init__(self, out: Path, *, modalities=("polar_rgb_preview", "dop", "aolp"), image_format: str = "jpeg",
                 jpeg_quality: int = 95, hdr=("npz",), tone: Tone | None = None, polar_scale: float = 0.2):
        unknown = set(modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)} (have {MODALITIES})")
        if image_format not in ("jpeg", "png", "none"):
            raise ValueError("image_format must be jpeg, png or none")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.modalities = tuple(modalities) if image_format != "none" else ()
        self.image_format, self.jpeg_quality = image_format, int(jpeg_quality)
        self.hdr, self.tone, self.polar_scale = tuple(hdr), tone, float(polar_scale)
        self.ext = {"jpeg": "jpg", "png": "png", "none": ""}[image_format]
        self.written: set[tuple[str, str]] = set()
        self.scenes: set[str] = set()
        self.variants: set[str] = set()
        self.raw_files: list[dict] = []
        self.episode_count = 0
        self.rows = 0
        self.index = (self.out / "index.jsonl").open("a", encoding="utf-8")

    def has(self, variant: str, fid: str) -> bool:
        return (export_variant(variant), fid) in self.written

    def add_frame(self, *, scene_id: str, variant: str, node_id: str, heading_id: str, camera_to_world: np.ndarray,
                  base_pose: np.ndarray | None, fov_deg: float, resolution: tuple[int, int], stokes_images: dict,
                  render: dict) -> str:
        v = export_variant(variant)
        fid = frame_id(scene_id, node_id, heading_id)
        if (v, fid) in self.written:
            return fid
        s0, s1, s2, s3 = (np.asarray(stokes_images[k], dtype=np.float32) for k in ("s0", "s1", "s2", "s3"))
        common = {"frame_id": fid, "variant": v, "scene_id": scene_id, "camera_id": CAMERA_ID,
                  "camera_to_world": flat(camera_to_world),
                  "base_pose": flat(base_pose) if base_pose is not None else None,
                  "fov_deg": float(fov_deg), "resolution": [int(resolution[0]), int(resolution[1])],
                  "vp_id": node_id, "heading_id": heading_id, "yaw_deg": float(heading_id.split("_", 1)[1]),
                  "render_mode": "simulator_replay", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "render": render}
        if self.modalities:
            d = stokes.derived(s0, s1, s2)
            for modality in self.modalities:
                rel = f"images/{v}/{fid}__{CAMERA_ID}__{modality}.{self.ext}"
                self._save_image(self._image(modality, s0, d), self.out / rel)
                self.index.write(json.dumps({**common, "image": rel, "modality": modality}) + "\n")
                self.rows += 1
        if "npz" in self.hdr:
            rel = f"polarization_raw/{v}/{fid}__{CAMERA_ID}__stokes.npz"
            dst = self.out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, s0=s0.astype(np.float16), s1=s1.astype(np.float16),
                                s2=s2.astype(np.float16), s3=s3.astype(np.float16))
            info = {"schema": STOKES_SCHEMA, "dtype": "float16", "shape": list(s0.shape),
                    "axis_order": ["height", "width", "rgb"], "component_order": ["S0", "S1", "S2", "S3"],
                    "source": "opticalnav_sim", "render": render, "lossy": True}
            dst.with_suffix(dst.suffix + ".source.json").write_text(json.dumps(info, sort_keys=True))
            self.raw_files.append({"path": rel, **info, "bytes": dst.stat().st_size})
        if "exr" in self.hdr:
            self._save_exr(s0, self.out / f"hdr/{v}/{fid}__{CAMERA_ID}__s0.exr")
        self.written.add((v, fid))
        self.scenes.add(scene_id)
        self.variants.add(v)
        return fid

    def add_episode(self, raw: dict) -> None:
        dst = self.out / "episodes" / (raw.get("split") or "unsplit") / f"{raw['episode_id']}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(raw, ensure_ascii=False, indent=1))
        self.episode_count += 1

    def add_graph(self, scene_id: str, data: bytes) -> None:
        dst = self.out / "graph" / f"{scene_id}__navigation_support_graph.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    def close(self, extra: dict | None = None) -> dict:
        self.index.close()
        if self.raw_files:
            (self.out / "polarization_raw_manifest.json").write_text(json.dumps(
                {"schema": STOKES_SCHEMA, "format": "minimal-f16", "files": self.raw_files, "failures": []}, indent=1))
        meta = {
            "source": "opticalnav_sim",
            "scenes": sorted(self.scenes),
            "cameras": [CAMERA_ID],
            "variants": sorted(self.variants),
            "frame_count": len(self.written),
            "index_rows": self.rows,
            "episode_count": self.episode_count,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality if self.image_format == "jpeg" else None,
            "modalities": list(self.modalities),
            "polarization_raw": {"included": "npz" in self.hdr, "format": "minimal-f16" if "npz" in self.hdr else None,
                                 "schema": STOKES_SCHEMA if "npz" in self.hdr else None, "count": len(self.raw_files)},
            "hdr_exr": "exr" in self.hdr,
            "tonemap": {**(self.tone.describe() if self.tone else {"operator": "none"}),
                        "applies_to": "polar_rgb_preview (linear S0)"},
            "colormaps": {"dop": f"black to red, full red at DoLP {self.polar_scale}",
                          "aolp": f"hue around the colour circle for AoLP 0-180 deg, brightness DoLP/{self.polar_scale}",
                          "s1_over_s0": f"blue-white-red over +-{self.polar_scale}",
                          "s2_over_s0": f"blue-white-red over +-{self.polar_scale}"},
            "index_schema": {
                "image": "relative path to the encoded image",
                "camera_to_world": "row-major flattened 4x4 (Mitsuba/USD convention, column-translation last)",
                "base_pose": "robot base 4x4 flattened",
                "fov_deg": "horizontal field of view",
                "resolution": "[width, height]",
                "vp_id": "support graph node id; join key into episodes path_nodes",
                "heading_id": "discrete heading; join key into episodes path_headings",
                "yaw_deg": "heading yaw in degrees",
                "render": "simulator render settings (renderer, mitsuba variant, spp, seed, mode, denoiser)",
            },
            "navigation_schema": {"join": "index.jsonl (scene_id, vp_id, heading_id) <- episodes path_nodes[i], path_headings[i]"},
            **(extra or {}),
        }
        (self.out / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    # --- encoding ---
    def _image(self, modality: str, s0: np.ndarray, d: dict) -> np.ndarray:
        if modality == "polar_rgb_preview":
            return self.tone(s0) if self.tone else stokes.quick_preview(s0)
        if modality == "dop":
            return stokes.dolp_rgb(d["dolp"], self.polar_scale)
        if modality == "aolp":
            return stokes.aolp_rgb(d["aolp_deg"], d["dolp"], self.polar_scale)
        return stokes.signed_rgb(d[modality] / self.polar_scale)

    def _save_image(self, img: np.ndarray, path: Path) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        if self.image_format == "jpeg":
            Image.fromarray(img).save(path, quality=self.jpeg_quality)
        else:
            Image.fromarray(img).save(path)

    @staticmethod
    def _save_exr(img: np.ndarray, path: Path) -> None:
        import os

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), np.ascontiguousarray(img[:, :, ::-1], dtype=np.float32)):
            raise RuntimeError(f"could not write {path} (OpenCV built without OpenEXR?)")
