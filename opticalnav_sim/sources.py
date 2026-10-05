"""Where episodes and the dataset's own Stokes frames come from.

``open_source(path)`` accepts
* an export bundle: an unzipped ``bundle/`` directory, ``bundle.zip``, or the
  wizard's split upload (``bundle.zip.part000`` ... or the directory holding the
  parts). Split parts are read in place as one file, so a 36 GB upload need not
  be joined or unzipped to replay a few episodes;
* a robomituba OpticalNav project (``out/opticalnav/<version>``) or one scene of it;
* a simulator scene pack (episodes copied by build_pack, and its reference frames);
* a single episode JSON file.

Every source answers ``episodes(scene, splits)`` and ``stokes(scene, variant, node, heading)``
(None when that view's arrays are not stored, e.g. a project whose raw renders were pruned).
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .episodes import Episode, frame_id, parse

EXPORT_VARIANT = {"base": "base", "perturbed": "perturbed", "active_polar": "perturbed_active_polar"}
OBSERVATION_DIRS = {"base": "observations", "perturbed": "observations_perturbed",
                    "active_polar": "observations_perturbed_active_polar"}


class PartsFile(io.RawIOBase):
    """Read-only, seekable view of byte-split parts (``cat part* > whole``) as one file."""

    def __init__(self, parts: list[Path]):
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.size, self.pos = sum(self.sizes), 0
        self.handles: dict[int, object] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self.pos = {os.SEEK_SET: 0, os.SEEK_CUR: self.pos, os.SEEK_END: self.size}[whence] + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, buffer) -> int:
        view, done = memoryview(buffer), 0
        while done < len(view) and self.pos < self.size:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            handle = self.handles.get(i) or self.handles.setdefault(i, self.parts[i].open("rb"))
            handle.seek(self.pos - self.starts[i])
            n = handle.readinto(view[done:done + min(len(view) - done, self.starts[i] + self.sizes[i] - self.pos)])
            if not n:
                break
            done, self.pos = done + n, self.pos + n
        return done

    def close(self) -> None:
        for h in self.handles.values():
            h.close()
        super().close()


def _parts_in(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("bundle.zip.part*[0-9]"))
    if ".part" in path.name:
        return sorted(path.parent.glob(path.name.split(".part")[0] + ".part*[0-9]"))
    return []


class _Tree:
    """Relative-path reads over a directory or a zip (entries may sit under a ``bundle/`` prefix)."""

    def __init__(self, path: Path):
        self.zip = None
        parts = _parts_in(path)
        if parts:
            self.zip = zipfile.ZipFile(io.BufferedReader(PartsFile(parts), buffer_size=1 << 20))
        elif path.is_file() and zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
        if self.zip is not None:
            names = self.zip.namelist()
            self.prefix = "bundle/" if any(n.startswith("bundle/") for n in names) else ""
            self.names = {n[len(self.prefix):] for n in names if n.startswith(self.prefix)}
        else:
            self.root = path / "bundle" if (path / "bundle" / "index.jsonl").is_file() else path

    def exists(self, rel: str) -> bool:
        return rel in self.names if self.zip is not None else (self.root / rel).is_file()

    def read(self, rel: str) -> bytes:
        return self.zip.read(self.prefix + rel) if self.zip is not None else (self.root / rel).read_bytes()

    def glob_episodes(self) -> list[str]:
        if self.zip is not None:
            return sorted(n for n in self.names if n.startswith("episodes/") and n.endswith(".json"))
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.glob("episodes/*/*.json"))


class Source:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tree = None if self.path.suffix == ".json" and self.path.is_file() else _Tree(self.path)

    @property
    def kind(self) -> str:
        if self.tree is None:
            return "episode file"
        if self.tree.exists("index.jsonl"):
            return "export bundle"
        if self.tree.zip is None and (self.tree.root / "pack.json").is_file():
            return "scene pack"
        if self.tree.zip is None and (self.tree.root / "scenes").is_dir():
            return "robomituba project"
        return "episode folder"

    def episodes(self, scene: str | None = None, splits: Iterable[str] | None = None) -> list[Episode]:
        if self.tree is None:
            return [parse(json.loads(self.path.read_text()), self.path)]
        wanted = set(splits) if splits else None
        rels = self.tree.glob_episodes()
        if self.tree.zip is None and scene:  # project or pack: prefer the scene's own episode set
            scene_dir = self.tree.root / "scenes" / scene / "episodes"
            if scene_dir.is_dir() and any(scene_dir.glob("*/*.json")):
                rels = sorted(p.relative_to(self.tree.root).as_posix() for p in scene_dir.glob("*/*.json"))
        out = []
        for rel in rels:
            split = rel.split("/")[-2]
            if wanted is not None and split not in wanted:
                continue
            if scene and not rel.split("/")[-1].startswith(scene + "_"):
                raw = json.loads(self.tree.read(rel))
                if raw.get("scene_id") != scene:
                    continue
            else:
                raw = json.loads(self.tree.read(rel))
            out.append(parse(raw, Path(rel)))
        return out

    def stokes(self, scene: str, variant: str, node: str, heading: str) -> dict | None:
        if self.tree is None:
            return None
        fid = frame_id(scene, node, heading)
        rel = f"polarization_raw/{EXPORT_VARIANT.get(variant, variant)}/{fid}__polar_cam__stokes.npz"
        if self.tree.exists(rel):
            return dict(np.load(io.BytesIO(self.tree.read(rel))))
        if self.tree.zip is not None:
            return None
        root = self.tree.root
        ref = root / "scenes" / scene / "reference" / variant / f"{node}__{heading}" / "stokes_data.npz"
        if ref.is_file():  # scene pack
            return dict(np.load(ref))
        pointer = root / "scenes" / scene / OBSERVATION_DIRS.get(variant, "") / node / heading / "current.json"
        if not pointer.is_file():
            return None
        manifest = json.loads((root / json.loads(pointer.read_text())["bundle_ref"] / "manifest.json").read_text())
        # a render version may reuse an older one's arrays; artifact paths are relative to the robomituba repo
        art = next(((a.get("artifact_paths") or {}).get("stokes_npz") for a in manifest.get("artifacts", [])
                    if a.get("camera_id") == "polar_cam" and (a.get("artifact_paths") or {}).get("stokes_npz")), None)
        for base in (root, *root.resolve().parents):
            if art and (base / art).is_file():
                return dict(np.load(base / art))
        return None

    def graph(self, scene: str) -> bytes | None:
        """The scene's navigation support graph, if the source holds one."""
        if self.tree is None:
            return None
        for rel in (f"graph/{scene}__navigation_support_graph.json", f"scenes/{scene}/navigation_support_graph.json"):
            if self.tree.exists(rel):
                return self.tree.read(rel)
        return None


def open_source(path: str | Path) -> Source:
    return Source(path)
