# Research project cleanup implementation plan

This is the historical plan for the initial local cleanup. The subsequent request to create a public GitHub repository supersedes its no-publication and no-Git constraints; source, tests, documentation, and historical exports are published, while the original data and trained artifacts remain local.

> **For agentic workers:** use the safe-refactor workflow and verify each numerical boundary against the historical source. Independent subsystem implementation is delegated in this session.

**Goal:** Recover a runnable Python project, preserve supplied data and trained parameters, and remove every notebook file.

**Architecture:** A `src/esdl` package separates array conventions, the forward solver, the exact CNN, training, checkpoint handling, and supervised descent. A CLI connects the stages. Historical source and outputs survive as Python, Markdown, and PNG files.

**Tech stack:** Python 3.10+, NumPy, SciPy, PyTorch; torchvision and matplotlib optional.

**Spec:** the user's cleanup request and `docs/methodology.md`, whose compatibility findings come from the original source and saved artifacts.

## Global constraints

- Preserve every original non-notebook file byte-for-byte, recording SHA-256 before and after relocation.
- No `.ipynb` files or notebook-based execution route remain in the project.
- Preserve CNN parameter names, sizes, nonlinearities, tensor channel order, descent residual sign, update sign, regularization, and stopping rule.
- Preserve the legacy forward equations and the supplied two-channel initializer; scientific changes require an explicit separate mode.
- Do not run full retraining, overwrite supplied assets, invent a license, or publish the project.
- This folder is not a Git checkout; implementation occurs in the user-authorized folder without creating a worktree or commits.

## Review focus

- CPU loading must accept tensors originally saved on CUDA.
- Mini-batches must keep remainder samples and support datasets smaller than a batch.
- Original artifact storage contains more matrices than its logical tensor shape; use 342 active matrices.
- Generated arrays must use domain channels `[zero, permittivity]` and field channels `[real, imaginary]`.
- Existing output directories and supplied assets must not be overwritten by a CLI run.

## Tasks

- [x] Export all source cells and historical outputs; hash assets; relocate arrays into `data`, parameters into `artifacts`, and stale reviews into `archive`.
- [x] Implement `data.encode_domains`, `data.encode_fields`, and `data.load_split`; validate using `python -m pytest tests/test_data.py`.
- [x] Implement the exact `DConvNet`, safe checkpoint loading, batched training/evaluation, and honest field metrics; validate meta architecture and tiny train/save/resume cases.
- [x] Implement vectorized legacy MoM and seeded domain generation; compare literal equations and saved first samples numerically.
- [x] Implement regularized descent fitting and reconstruction; compare the original SVD expression, plus update, and stopping matrix retention.
- [x] Connect `inspect`, `verify-assets`, `generate`, `train-forward`, `evaluate`, `train-descent`, and `reconstruct` CLI commands; add optional plots and documentation.
- [x] Run the full tests, wheel build, data/hash audit, actual weighted inference and representative reconstruction; obtain an independent review and resolve material findings.
- [x] Remove verified original notebook files and generated caches; verify recursively that no notebooks remain.

## Progress and decisions

- Audit complete: all four notebooks read; actual saved artifact metadata inspected using memory mapping.
- Legacy solver reproduces train/test/cylinder sample 0 to roughly 1e-15 relative error.
- Ruling: direct implementation proceeds under the user's cleanup authorization; no extra design approval is required for preserving the recovered workflow.
- Ruling: use `Path.absolute()` for workspace paths. This host's Python `realpath` unexpectedly duplicates the long directory basename.
- Completion evidence: 73 tests passed; original pretrained CNN predicts two test examples; all 342 descent matrices reconstruct one example; a two-iteration offline fit saves compact directions; all 20 manifest entries pass; zero notebook files remain. See `docs/validation.md` for commands, metrics, and limits.
- Independent review: no critical/important findings; additional warm Adam continuation and dataset-free complex-observation reconstruction probes passed.
