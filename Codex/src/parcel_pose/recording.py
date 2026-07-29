"""Raw RGB-D session persistence and deterministic replay."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .output import to_jsonable
from .session import RecordedFrame, SessionMetadata, SessionValidationError


MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SessionWriter:
    def __init__(
        self,
        root: str | Path,
        metadata: SessionMetadata,
        *,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"recording destination already exists: {self.root}")
            if (
                self.root.is_symlink()
                or not self.root.is_dir()
                or not (self.root / MANIFEST_NAME).is_file()
            ):
                raise FileExistsError(
                    "refusing to overwrite a path that is not an existing "
                    f"parcel-pose recording session: {self.root}"
                )
            shutil.rmtree(self.root)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True)
        self.metadata = metadata
        self._entries: list[dict[str, Any]] = []
        self._closed = False
        self._write_manifest(complete=False)

    @property
    def frame_count(self) -> int:
        return len(self._entries)

    def _write_manifest(self, *, complete: bool) -> None:
        payload = {
            "metadata": self.metadata.to_dict(),
            "complete": bool(complete),
            "frame_count": len(self._entries),
            "frames": self._entries,
        }
        manifest = self.root / MANIFEST_NAME
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(manifest)

    def add_frame(self, frame: RecordedFrame) -> int:
        if self._closed:
            raise RuntimeError("cannot add a frame after the session is closed")
        if frame.raw_depth_z16.shape != (
            self.metadata.depth_profile.intrinsics.height,
            self.metadata.depth_profile.intrinsics.width,
        ):
            raise SessionValidationError("raw depth shape does not match the active depth profile")
        if frame.raw_color_bgr.shape[:2] != (
            self.metadata.color_profile.intrinsics.height,
            self.metadata.color_profile.intrinsics.width,
        ):
            raise SessionValidationError("raw color shape does not match the active color profile")
        index = len(self._entries)
        relative_path = Path("frames") / f"{index:06d}.npz"
        target = self.root / relative_path
        arrays: dict[str, np.ndarray] = {
            "raw_depth_z16": frame.raw_depth_z16,
            "raw_color_bgr": frame.raw_color_bgr,
        }
        if frame.color_on_depth_bgr is not None:
            arrays["color_on_depth_bgr"] = frame.color_on_depth_bgr
        np.savez_compressed(target, **arrays)
        entry = {
            "index": index,
            "file": relative_path.as_posix(),
            "sha256": _sha256(target),
            "depth_timestamp_ms": frame.depth_timestamp_ms,
            "color_timestamp_ms": frame.color_timestamp_ms,
            "depth_frame_number": frame.depth_frame_number,
            "color_frame_number": frame.color_frame_number,
            "hardware_timestamp_ms": frame.hardware_timestamp_ms,
            "system_timestamp_ns": frame.system_timestamp_ns,
            "frame_metadata": dict(frame.frame_metadata),
            "has_color_on_depth": frame.color_on_depth_bgr is not None,
        }
        self._entries.append(to_jsonable(entry))
        self._write_manifest(complete=False)
        return index

    def close(self) -> None:
        if not self._closed:
            self._write_manifest(complete=True)
            self._closed = True

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()


class SessionReader:
    def __init__(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise SessionValidationError(f"recording manifest is missing: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionValidationError(f"cannot read recording manifest: {exc}") from exc
        if not isinstance(payload, dict) or "metadata" not in payload or "frames" not in payload:
            raise SessionValidationError("manifest requires metadata and frames")
        self.metadata = SessionMetadata.from_dict(payload["metadata"])
        self.complete = bool(payload.get("complete", False))
        entries = payload["frames"]
        if not isinstance(entries, list):
            raise SessionValidationError("manifest frames must be an array")
        if int(payload.get("frame_count", -1)) != len(entries):
            raise SessionValidationError("manifest frame_count does not match frame entries")
        self._entries = entries
        self._verify_hashes = verify_hashes

    def __len__(self) -> int:
        return len(self._entries)

    def _load_entry(self, entry: dict[str, Any]) -> RecordedFrame:
        relative = Path(str(entry.get("file", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise SessionValidationError("recorded frame path must stay inside the session")
        path = self.root / relative
        root_resolved = self.root.resolve()
        if not path.resolve().is_relative_to(root_resolved):
            raise SessionValidationError("recorded frame path escapes the session")
        if not path.is_file():
            raise SessionValidationError(f"recorded frame file is missing: {path}")
        if self._verify_hashes and _sha256(path) != entry.get("sha256"):
            raise SessionValidationError(f"recorded frame checksum mismatch: {path}")
        try:
            with np.load(path, allow_pickle=False) as arrays:
                names = set(arrays.files)
                if not {"raw_depth_z16", "raw_color_bgr"}.issubset(names):
                    raise SessionValidationError(f"recorded frame arrays are incomplete: {path}")
                aligned = (
                    np.array(arrays["color_on_depth_bgr"], copy=True)
                    if "color_on_depth_bgr" in names
                    else None
                )
                return RecordedFrame(
                    raw_depth_z16=np.array(arrays["raw_depth_z16"], copy=True),
                    raw_color_bgr=np.array(arrays["raw_color_bgr"], copy=True),
                    color_on_depth_bgr=aligned,
                    depth_timestamp_ms=float(entry["depth_timestamp_ms"]),
                    color_timestamp_ms=float(entry["color_timestamp_ms"]),
                    depth_frame_number=int(entry["depth_frame_number"]),
                    color_frame_number=int(entry["color_frame_number"]),
                    hardware_timestamp_ms=entry.get("hardware_timestamp_ms"),
                    system_timestamp_ns=entry.get("system_timestamp_ns"),
                    frame_metadata=dict(entry.get("frame_metadata", {})),
                )
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, SessionValidationError):
                raise
            raise SessionValidationError(f"cannot load recorded frame {path}: {exc}") from exc

    def __iter__(self) -> Iterator[RecordedFrame]:
        expected_index = 0
        for entry in self._entries:
            if not isinstance(entry, dict) or int(entry.get("index", -1)) != expected_index:
                raise SessionValidationError("frame indices must be contiguous and deterministic")
            yield self._load_entry(entry)
            expected_index += 1


def replay_session(
    root: str | Path,
    processor: Callable[[RecordedFrame, SessionMetadata], Any] | None = None,
) -> list[Any]:
    """Replay frames in manifest order and return strict JSON-compatible results."""

    reader = SessionReader(root)
    results: list[Any] = []
    for frame in reader:
        if processor is None:
            value: Any = {
                "depth_frame_number": frame.depth_frame_number,
                "color_frame_number": frame.color_frame_number,
                "depth_timestamp_ms": frame.depth_timestamp_ms,
                "valid_depth_pixels": int(np.count_nonzero(frame.raw_depth_z16)),
            }
        else:
            value = processor(frame, reader.metadata)
        results.append(to_jsonable(value))
    return results


def recording_summary(root: str | Path) -> dict[str, Any]:
    reader = SessionReader(root)
    return {
        "schema_version": reader.metadata.schema_version,
        "complete": reader.complete,
        "frame_count": len(reader),
        "camera_serial": reader.metadata.camera_serial,
        "depth_profile": reader.metadata.depth_profile.to_dict(),
        "color_profile": reader.metadata.color_profile.to_dict(),
    }


__all__ = [
    "MANIFEST_NAME",
    "SessionReader",
    "SessionWriter",
    "recording_summary",
    "replay_session",
]
