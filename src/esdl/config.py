"""Locations for local data, pretrained parameters, and new experiment outputs."""

from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    """Find the supplied assets from a checkout, including editable installs."""
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents, Path(__file__).absolute().parents[2]):
        if (candidate / "asset_manifest.json").is_file():
            return candidate
    return cwd


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        return cls(Path(root).absolute() if root is not None else find_project_root())

    @property
    def data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def pretrained(self) -> Path:
        return self.root / "artifacts" / "pretrained"

    @property
    def checkpoint(self) -> Path:
        return self.pretrained / "my_checkpoint.pth.tar"

    @property
    def matrices(self) -> Path:
        return self.pretrained / "Ks.pt"

    @property
    def initial(self) -> Path:
        return self.pretrained / "Xk.pt"
