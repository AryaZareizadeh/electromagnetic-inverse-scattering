import hashlib
import json

import numpy as np

from esdl.cli import main


def test_generate_writes_encoded_arrays_and_provenance(tmp_path):
    output = tmp_path / "generated"
    assert main(["--root", str(tmp_path), "generate", "--count", "1", "--seed", "7",
                 "--output", str(output)]) == 0
    domains = np.load(output / "Domains_cy.npy")
    fields = np.load(output / "Field_Data_cy.npy")
    assert domains.shape == (1, 2, 28, 28)
    assert fields.shape == (1, 2, 20, 20)
    assert np.isfinite(fields).all()
    np.testing.assert_array_equal(domains[:, 0], 0)
    assert json.loads((output / "metadata.json").read_text())["forward_config"]["formulation"] == "legacy"


def test_existing_generation_directory_is_not_overwritten(tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep")
    assert main(["generate", "--count", "1", "--output", str(tmp_path)]) == 1
    assert sentinel.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sentinel.txt"]


def test_verify_assets_detects_modified_file(tmp_path):
    asset = tmp_path / "original.npy"
    asset.write_bytes(b"original asset")
    manifest = {"files": [{"path": "original.npy",
                 "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}]}
    (tmp_path / "asset_manifest.json").write_text(json.dumps(manifest))
    args = ["--root", str(tmp_path), "verify-assets"]
    assert main(args) == 0
    asset.write_bytes(b"changed")
    assert main(args) == 1


def test_observed_fields_are_validated_before_checkpoint_loading(tmp_path):
    fields = tmp_path / "bad.npy"
    np.save(fields, np.ones((2, 20)))
    assert main(["--root", str(tmp_path), "reconstruct", "--fields", str(fields)]) == 1
