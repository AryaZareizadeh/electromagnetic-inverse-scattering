# Cleanup validation — 14 October 2026

This record distinguishes numerical compatibility, actual pretrained execution, and work that was not repeated.

## Asset and source preservation

All **20 manifest entries** passed `python scripts/esdl.py verify-assets`: 16 original non-notebook files matched their original SHA-256 hashes, and four Python source exports matched their recorded export hashes. Data and parameters were moved within the same volume without altering their contents.

Before deletion, every original code cell was found verbatim in its ordered Python export; all four exports parsed as Python, original notebook hashes matched the baseline, and every embedded PNG output had a corresponding extracted file. Historical text outputs and logs are available in the accompanying Markdown files. A recursive file scan after removal found **zero `.ipynb` files**.

## Numerical compatibility

The vectorized legacy solver reproduced the first supplied example from each split:

| Split | Relative L2 discrepancy |
|---|---:|
| training | 6.9 × 10⁻¹⁶ |
| test | 6.34 × 10⁻¹⁶ |
| cylinders | 6.27 × 10⁻¹⁶ |

Unit checks compare the solver to literal historical equations, verify channel/flatten order, confirm the exact 399,880,950-parameter CNN on the meta device, compare the regularized SVD fit to the historical expression, and test descent update signs and stopping behavior. Tiny trainable models exercise tail batches, checkpoint save/load, optimizer continuation, and full-batch settings without allocating the full training model.

## Actual pretrained execution

These commands used the original trained files, CPU execution, and four PyTorch threads:

```powershell
python scripts/esdl.py --threads 4 evaluate --samples 2 --batch-size 1 --device cpu --plot --output runs/validation-forward
python scripts/esdl.py --threads 4 reconstruct --samples 1 --batch-size 1 --device cpu --plot --output runs/validation-reconstruction
python scripts/esdl.py --threads 4 train-descent --samples 3 --batch-size 2 --iterations 2 --device cpu --output runs/validation-descent
```

| Check | Result |
|---|---|
| CNN prediction on test samples 0–1 | Field MSE 2.35516845 × 10⁻⁷; maximum sample MSE 3.16211270 × 10⁻⁷ |
| Full 342-matrix reconstruction of test sample 0 | Domain MSE 0.0083632385; field MSE 2.16190706 × 10⁻⁷ |
| Two offline fitting iterations on three training samples | Finite stopping scores 0.0446010418 and 0.0362860002; compact `(2,800,1568)` matrix tensor saved |

Forward fields for reconstruction metrics were recomputed after the final update. The generated arrays, metrics, run configurations, and comparison plots remain under the listed `runs/validation-*` directories in the local working copy; those directories are excluded from GitHub. These small checks establish that the recovered execution path works with the supplied parameters; they do not measure full-dataset performance.

## Packaging, tests, and independent review

The complete test suite passes: **73 tests**. A wheel was built with `python -m build --wheel --no-isolation`, installed into an isolated verification directory without network access, and its CLI help entry point ran successfully. The source-checkout runner also works without installation.

An independent reviewer found no critical or important issues. Additional review probes established that warm Adam-state continuation matches uninterrupted tiny-model training and does not alter its input checkpoint, and that external complex observations can pass through the reconstruction CLI without a dataset directory.

PyTorch reports an expected warning for the preserved even-sized convolution kernel with `padding="same"`; changing that layer would break historical model compatibility. This host's sandbox restricts its default temporary directory, so verification used a writable `--basetemp` for pytest and an isolated build temporary directory. These host-specific settings do not affect numerical source behavior.

Verified environment: Python 3.13.0, NumPy 2.1.2, SciPy 1.16.3, PyTorch 2.8.0+cu129, on Windows. Full CNN retraining, all 800 offline fitting iterations, a complete test-set evaluation, and independent physical validation of the historical solver were not performed.
