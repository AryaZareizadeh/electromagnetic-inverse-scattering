"""Verify that reorganizing the project preserved the supplied files."""

import hashlib
import json
from pathlib import Path


def verify_assets(root: str | Path) -> list[dict]:
    root = Path(root).absolute()
    manifest = json.loads((root / "asset_manifest.json").read_text(encoding="utf-8"))
    results = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest path must stay within the project: {relative}")
        path = root / relative
        expected = entry.get("exported_source_sha256", entry["sha256"])
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
        else:
            actual = None
        results.append({"path": entry["path"], "ok": actual == expected,
                        "expected_sha256": expected, "actual_sha256": actual})
    return results
