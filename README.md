# Electromagnetic inverse scattering with CNNs and supervised descent

This research project reconstructs the spatial relative permittivity of dielectric objects from complex scattered electromagnetic fields. It combines a method-of-moments data generator, a convolutional neural network that approximates the forward scattering map, and regularized supervised descent for inverse reconstruction.

The workflow starts with 28×28 permittivity maps of digit-shaped or cylindrical scatterers and computes their fields for 20 receivers and 20 incident directions. The CNN learns to predict those fields from a map. During offline training, supervised descent learns a sequence of matrices that turn field prediction errors into domain updates; online reconstruction applies those updates to estimate an unknown object's permittivity from its observed fields.

The historical notebooks have been replaced by a Python package and command-line workflow. The recovered equations and CNN architecture are preserved, with numerical compatibility tests and documented scientific limitations. Historical code, logs, and figures are in `archive/`; no notebooks are required to run the project.

**Asset availability:** this GitHub repository contains source code, tests, documentation, and historical exports. The original datasets and pretrained parameters are preserved unchanged in the author's local working copy and excluded from Git; no public download location is currently provided. Commands that use them require the paths documented in [data and artifacts](docs/data.md). Data generation and the core test suite work without the original assets.

## What it does

- Generates digit-shaped and two-cylinder permittivity maps and their scattered fields.
- Trains, saves, resumes, and evaluates a CNN forward surrogate.
- Fits regularized supervised descent matrices and reconstructs domains from observed fields, including observations supplied without ground truth.
- Saves predictions, reconstructions, regression metrics, and optional comparison plots.
- Checks array conventions, historical numerical behavior, and the integrity of locally supplied assets.

## Layout

```text
src/esdl/                Data, forward solver, CNN, training, descent, CLI
tests/                   Numerical fidelity and workflow tests
scripts/esdl.py           Run directly from the source checkout
data/processed/          Canonical train, test, and cylinder arrays
data/legacy/             Older pickle datasets and the distinct Domains01 array
artifacts/pretrained/    Original CNN checkpoint, descent matrices, initial guesses
archive/legacy/          Historical Python source, Markdown outputs, PNG figures
archive/reviews/         Earlier assessments, retained as historical documents
docs/                    Methodology, data conventions, validation
asset_manifest.json      Original hashes and relocation paths
runs/                    New experiment outputs (ignored by Git)
```

## Setup

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,plots,generation]"
esdl --help
```

PyTorch must match your intended CPU/CUDA environment. Core execution requires NumPy, SciPy, and PyTorch; torchvision is needed only for generating MNIST data, and matplotlib only for plots. The cleanup was verified with Python 3.13.0, NumPy 2.1.2, SciPy 1.16.3, and PyTorch 2.8.0+cu129.

With dependencies already installed, installation is optional:

```powershell
python scripts/esdl.py --help
python scripts/esdl.py inspect
python scripts/esdl.py verify-assets
```

Installed commands work from this checkout or with `esdl --root "path/to/project" ...`. `--root` and `--threads` precede the subcommand. Input paths are explicit; Colab, Google Drive mounts, helper-script downloads, and Jupyter are unnecessary.

## Use the existing trained parameters

This section requires the original assets; they are not downloaded by cloning the repository. The CNN alone contains **399,880,950 parameters**, about **1.49 GiB** of float32 weights. The original checkpoint is **4.47 GiB**, including Adam state. The descent file is **3.74 GiB**, although its 342 active matrices occupy about **1.60 GiB**. Loading uses CPU memory mapping; reconstruction transfers matrices one at a time. These are substantial research models: use a small sample count for the first run.

Predict fields for two test objects:

```powershell
esdl --threads 4 evaluate --samples 2 --batch-size 1 --device cpu --plot --output runs/forward-check
```

Reconstruct one test object using all 342 trained descent matrices:

```powershell
esdl --threads 4 reconstruct --samples 1 --batch-size 1 --device cpu --plot --output runs/reconstruction-check
```

For a quick execution check, add `--iterations 2`; this is a partial reconstruction, not the full trained result. Use `--split cylinders` to evaluate the cylindrical dataset. To reconstruct your own observations without ground truth:

```powershell
esdl reconstruct --fields path/to/observed_fields.npy --samples 1 --output runs/observed-reconstruction
```

Observed fields must be real `(N,2,20,20)` with channels `[real, imaginary]`, or complex `(N,20,20)`. Every output directory must be new. Evaluation writes predicted fields and metrics; reconstruction also writes the recovered domain arrays and run configuration.

## Train and generate new data

Mini-batch training preserves the historical defaults: 400 epochs, ordered batches of 100, Adam learning rate `1e-6`, MSE loss. It now includes remainder samples, saves a usable checkpoint, and supports CPU execution.

```powershell
esdl train-forward --device cuda --epochs 400 --batch-size 100 --output runs/forward-training
esdl train-forward --resume artifacts/pretrained/my_checkpoint.pth.tar --epochs 10 --device cuda --output runs/continued-training
```

Resume restores Adam state and its saved learning rate (`1e-7` in the supplied checkpoint); use `--resume-learning-rate` for an explicit override. `--full-batch` preserves the alternative experiment's one full-data update per epoch and default `5e-5` learning rate. Full training needs much more memory than inference: weights, gradients, and Adam moments alone require roughly 6 GiB, before activations.

Learn new descent matrices using the trained surrogate and saved initial guesses:

```powershell
esdl train-descent --checkpoint runs/forward-training/forward_checkpoint.pt --device cuda --iterations 800 --output runs/descent-training
```

Descent fitting uses `alpha=0.5` and the historical improvement threshold `0.02`. `--samples` limits any training command for an execution experiment; its output is a different experiment from training on all 5,000 samples.

Generate deterministic new cylindrical data, or download/sample the MNIST test split:

```powershell
esdl generate --kind cylinders --count 10 --seed 42 --output data/generated/cylinders-example
esdl generate --kind mnist --split test --count 10 --seed 42 --output data/generated/mnist-example
esdl evaluate --data-dir data/generated/mnist-example --samples 2 --output runs/generated-evaluation
```

Generation defaults to the recovered **legacy** equations that reproduce the supplied data. An explicit `--formulation free-space` changes the questionable impedance argument; those fields differ from the data used to train the supplied parameters. See [methodology](docs/methodology.md) before using that mode in an experiment.

## Verification and project history

```powershell
python -m pytest
python -m build --wheel
esdl verify-assets
```

Tests use tiny numerical systems and meta-device CNN construction rather than retraining the large network. [Validation](docs/validation.md) records checks actually run, including numerical reproduction and real pretrained execution. [Dataset documentation](docs/data.md) explains the channel convention and preserved legacy files. [Methodology](docs/methodology.md) describes the recovered algorithm and scientific limitations.

Large local datasets, model files, and experiment outputs are ignored by Git; `asset_manifest.json` records the original assets' paths and hashes. `verify-assets` checks a working copy containing those assets and will report missing files in a fresh clone. This repository currently has no declared license and no published dataset or pretrained-model downloads.
