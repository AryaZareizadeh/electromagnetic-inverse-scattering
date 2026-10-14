# Project review — 13 October 2026

**Scope update:** The intended GitHub release includes training/testing data and labels, but excludes learned model parameters. Three notebooks recovered from `G:/Master study/2nd Semester/Computational Electromagnetics/CE project/Codes` now supply several previously missing source components. See the recovered-notebook assessment below. The original-directory findings describe the initial snapshot; a pretrained checkpoint is not required in the intended release if the workflow trains and produces the needed parameters locally.

The project contains a recognizable research workflow: train a convolutional surrogate for electromagnetic scattering, learn regularized supervised descent matrices offline, and apply those matrices to inverse reconstruction online. It needs several reproducibility and correctness repairs before someone else can reliably run it from GitHub.

This review covers every source cell in all three notebooks and every supplied data file. The original notebooks and datasets were left unchanged. Notebook cell numbers below start at **1**, counting both code and markdown cells.

## What is present

| File | Role |
| --- | --- |
| `DConvNet.ipynb` | Mini-batch training of the forward surrogate, with saved training logs and field plots. |
| `DConvNet_FullData.ipynb` | Full-batch alternative using the same architecture; saved execution ends with a KeyboardInterrupt. |
| `SDM.ipynb` | Offline descent-matrix learning and online reconstruction code. |
| `Domains.npy`, `Field_Data.npy` | 5,000 paired training samples. |
| `Domains_test.npy`, `Field_Data_test.npy` | 1,000 paired testing samples. |
| `Domains_cy.npy`, `Field_Data_cy.npy` | 1,000 cylindrical-domain testing samples. |
| `Labels.npy`, `Labels_test.npy` | Digit labels, 0–9; not used in the active training code. |
| `Domains01.npy` | Another 5,000-domain dataset; its purpose and matching field data are undocumented. |
| `dataset_train.pkl`, `dataset_test.pkl` | Older encodings of the main training/testing pairs. |

The original folder contains 14 files totaling **272,044,183 bytes** (272 MB, approximately 259 MiB). It has no `.git` directory, README, environment specification, license, tests, or standalone Python modules.

## Priority 1: restore a reproducible run

1. **The trained forward-model checkpoint is missing.** `SDM.ipynb`, cell 10, loads `my_checkpoint.pth.tar`. Neither training notebook actively saves this file: the checkpoint code in cell 13 is commented out. Recover the original checkpoint, or add and verify a documented training-and-save step. Without it, the supplied descent notebook cannot reproduce its recorded run.

2. **The initial reconstruction file is missing.** `SDM.ipynb`, cells 12 and 16, require `Xk.pt`. Cell 18 contains two commented initialization alternatives, after the cells that need the file. Specify the initialization actually used, make its creation an explicit earlier step, and distinguish training and testing sample counts. The test notebook currently reads the same filename used for training.

3. **All active dataset paths point to a personal Colab Drive folder.** Both CNN notebooks use `/content/drive/MyDrive/Colab Notebooks/` without mounting Drive themselves; the SDM notebook explicitly imports `google.colab`. Introduce one configurable data/artifact directory and document either a local execution route or a complete Colab setup.

4. **The advertised CPU fallback does not work for data loading.** Selecting `device = "cuda" if ... else "cpu"` is followed or preceded by `.type(torch.cuda.FloatTensor)` in active data-loading cells. Load data as float32 and transfer it to the chosen device. Checkpoint loading also needs an explicit device mapping for portability.

5. **The helper module is fetched from a moving external branch at runtime.** `helper_functions.py` is absent, and all notebooks download it from the `main` branch of another repository. Only `print_train_time` is actually used; the imported `accuracy_fn` is unused. A local timing helper or a pinned, attributed dependency would make execution reproducible and avoid an unnecessary download.

6. **There is no tested environment specification or run order.** Record Python/PyTorch/CUDA and library versions, installation steps, hardware, expected runtime, and the sequence: train/load surrogate → construct initial guesses → learn/save descent matrices → evaluate reconstruction. `Ks.pt` is also absent, but the offline step can generate it; it is only required as a supplied artifact if offering inference without retraining.

## Priority 2: confirmed correctness problems

| Location | Finding and effect |
| --- | --- |
| `DConvNet.ipynb`, cell 12 | `bestloss` is read before it has been initialized. Its only assignment is inside the conditional that already reads it. A fresh run reaches a NameError after the first epoch's calculations. |
| Both CNN notebooks, cell 11 | `accuracy_func(y_true, y_pred)` ignores both arguments and uses global `y_test` and `test_pred`. Consequently, the per-example plotting cells report the whole test-set error rather than the selected example's error. |
| Both CNN notebooks, cells 11–12 | The quantity labeled “Accuracy” is a sum of absolute errors, not an accuracy percentage or mean error. It scales with dataset size. Define and label regression metrics explicitly. |
| `SDM.ipynb`, cell 9 | `RMS_X` and `RMS_E` compute the mean of `(sum(error²) / sum(truth²))²`. For a single sample this is the fourth power of relative L2 error, not a root-mean-square or relative L2 error. A probe with error norm 1 and truth norm 2 returned 0.0625; relative L2 error is 0.5. Zero-energy truth produces infinity. Confirm the intended paper definition before changing it. |
| `SDM.ipynb`, cells 9, 13 and 16 | `trainloss` sums across samples and averages across coordinates. For an offline batch it is batch size times elementwise MSE; for the online singleton it becomes elementwise MSE. The logged “MMSE” and the 0.02 stopping threshold therefore depend on the number of training samples. |
| `DConvNet_FullData.ipynb`, cell 17 | The scratch expression `torch.abs()` raises TypeError and prevents a clean Run All even after training succeeds. |

The argument bug was reproduced with two radically different input pairs: both returned the same error from the global tensors. The undefined-variable and scratch-cell errors were also reproduced in isolated probes.

Additional behavior to review: mini-batch slicing discards a final incomplete batch when the sample count is not divisible by 100; the supplied 5,000 samples divide evenly, so this does not currently discard data. The SDM stopping rule includes the latest update even when loss worsens, and its displayed residuals are calculated before that update. The final online field plot likewise uses `Ek` from before the last reconstruction update. These choices should be made explicit or adjusted when cleaning up evaluation.

## Priority 3: explain the science and preserve the evidence

- **Dataset generation and provenance are missing.** The notebooks declare wavelength, transmitter/receiver counts, receiver radius, and other constants, but no electromagnetic solver or data-generation code is supplied. Document the governing equations, polarization, material representation, transmitter/receiver geometry and ordering, boundary conditions, discretization, field normalization, and noise assumptions. Recover the generator or explain how the supplied arrays were produced.
- **Channel meanings need documentation.** The `.npy` object arrays have shape `(N, 2, 28, 28)`; channel 0 is entirely zero and channel 1 ranges from 1 to 2. Fields have shape `(N, 2, 20, 20)`. The pickle fields confirm real/imaginary channel ordering for the main train/test `.npy` files. Explain whether object values represent permittivity or contrast, and the ordering of its two channels. The code's “contrast” comments alone do not settle the definition.
- **There is no stated reference or scope of the enhancement.** Add the original report/paper, methodological citations, the baseline method, and which part this project improves. The class name `AlexNet` does not describe the implemented architecture accurately enough for a reader; document its actual layers.
- **Saved outputs do not match the current source completely.** `DConvNet.ipynb` now requests 400 epochs but its saved log ends at epoch 199, and its current source contains the uninitialized `bestloss`. The full-batch run was interrupted. Preserve these as historical outputs if useful, but regenerate a clearly identified verified run before presenting them as reproducible results.
- **Inverse reconstruction results are not recorded comprehensively.** The SDM notebook has saved offline logs but no saved output from its online testing cell. That cell evaluates only the first 20 examples; cylindrical testing is commented out. Add aggregate reconstruction and field errors across the documented evaluation sets, representative reconstructions, runtime, and a baseline comparison if claiming improved performance.
- **There is no validation split, seed setup, or saved experiment configuration.** Both CNN notebooks report test-set loss every epoch, and “best” tracking uses training loss. Introduce validation for model selection and reserve a final test evaluation; record the chosen epoch, parameters, seed, and checkpoint. Inspecting test loss every epoch alone does not prove that test data influenced model selection, but the current files do not document that decision.
- **The surrogate is also the online forward evaluator.** To support physical fidelity, validate reconstructed objects against the original electromagnetic solver where available. A small residual against the learned surrogate alone does not establish a small physical forward-model residual.

## Data checks and storage decisions

All nine `.npy` arrays are float64 and contain finite values. Main training and testing sample counts and shapes agree. SHA-256 hashes of complete samples found **no exact train/test overlap** in either domains or fields. This does not establish source-level independence, alignment with the true physical solver, or the absence of transformed/near-duplicate samples.

Training domains have 4,784 unique samples out of 5,000; test domains have 958 out of 1,000. Fields have the same unique-sample counts. Document whether these repeated examples were intentional.

The pickle files store a custom `__main__.NumpyDataset` with real float64 domain tensors shaped `(N, 28, 28)` and complex128 fields shaped `(N, 20, 20)`. Their stored tensors exactly match `Domains[:, 1]` and `Field_Data[:, 0] + 1j * Field_Data[:, 1]`, including the test equivalents. They contribute roughly 76 MB and are unnecessary for the active notebook paths. Their contents were inspected and compared without executing their serialized objects. Keep one canonical representation when preparing the release.

`Domains01.npy` is **not** a duplicate of `Domains.npy`. It shares 357 unique maps with it, but differs overall. Preserve it until its provenance and purpose are understood.

Three current files exceed 50 MiB: `Domains.npy`, `Domains01.npy`, and `dataset_train.pkl`. No current file exceeds 100 MiB. GitHub warns for files above 50 MiB and blocks regular Git files above 100 MiB; plan data and artifact storage before the first commit. Git LFS or separate downloadable release assets are supported options. A full float32 model checkpoint would itself be about 1.6 GB before optimizer state. See [GitHub's large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Hardware and architecture

All three notebooks define the same **399,880,950-parameter** model. Its `19600 → 19600` dense layer accounts for about 384 million parameters. Float32 weights alone occupy approximately 1.49 GiB; weights, gradients, and two Adam moment buffers amount to roughly 5.96 GiB before activations and other allocations. Full-batch training creates additional large activations for 5,000 examples. Document actual peak memory and make mini-batch training the clear default unless the full-batch variant is needed for the research.

The two fully connected layers have no intervening activation. Their composition is affine, so this is an expensive factorization of a single affine map after the convolutional feature extractor. This is a design observation, not proof of a bug. Preserve the historical architecture for reproduction; evaluate architecture changes separately.

The SDM notebook preallocates `Ks` with shape `(800, 800, 1568)` in float32: **4,014,080,000 bytes**, about 3.74 GiB of CPU memory, even when stopping earlier. Document this and consider storing only learned iterations. The recorded run stops after 342 matrices.

## Smallest useful GitHub cleanup

1. Recover the original checkpoint, initial-guess definition, generator, and project report if available.
2. Fix the confirmed runtime and metric problems, paths, device handling, and checkpoint saving.
3. Keep one main training notebook and a clearly documented SDM notebook; label the full-batch notebook as an alternative experiment and remove scratch cells from the runnable route.
4. Add `README.md`, a tested environment/dependency file, `.gitignore`, a chosen license, dataset documentation, and attribution for borrowed code. Include model/data download instructions and a small verified example.
5. Verify a fresh-kernel run with a small subset, checkpoint save/load, metric calculations, and descent learning/application. Then record one full experiment and initialize/commit the repository.

A notebook-based release is sufficient; a large package refactor is not required to make this work understandable and reproducible.

## Verification limits

All three notebooks pass notebook-schema validation, and all source cells parse as Python. Dataset integrity and overlap checks, storage comparisons, isolated metric/error probes, and architecture construction/output-shape checks were performed. The model was instantiated on PyTorch's meta device for the architecture checks, so these establish parameter count and output shape `(1, 800)`, not numerical model accuracy. Probe environment: Python 3.13.0, NumPy 2.1.2, PyTorch 2.8.0+cu129; these are review-tool versions, not a proposed reproduction environment.

No full training, actual weighted forward pass, offline SVD training, or end-to-end reconstruction was run. Missing artifacts prevent reproduction of the original experiment, and the checked source/data alone cannot establish the scientific validity of the claimed enhancement.

## Recovered-notebook assessment

All source cells in the three recovered notebooks were read. They pass notebook-schema validation and Python syntax parsing. They were not copied into the project, and their original files were not modified. Full data generation and training were not run.

### `Data_Creation.ipynb`

This supplies the missing data-generation source: downloading MNIST, sampling 5,000 training images, mapping pixel intensities to real permittivity values `1 + image`, retaining digit labels, and computing complex scattered fields with a method-of-moments implementation. It also generates 1,000 two-cylinder domains and their fields. The field array is indexed as `[sample, receiver, incident-wave direction]`.

Remaining gaps:

- It downloads the MNIST test split but does not generate `Domains_test.npy`, `Field_Data_test.npy`, or `Labels_test.npy`.
- It produces domains shaped `(N, 28, 28)` and complex fields shaped `(N, 20, 20)`, while the training notebooks expect two-channel arrays. The recovered training notebook contains a commented conversion example for cylindrical data only. A consistent conversion step for every split is still needed. Use distinct raw/processed paths to avoid overwriting incompatible formats under the same filenames.
- On a fresh generation run, cell 4 creates lowercase `labels`, but cell 5 reads uppercase `Labels`; only the load-existing-files branch defines uppercase `Labels`.
- Both field-generation cells call `Z_mn_func(..., k0, eps_r[m], s)`, passing material permittivity into the function argument named `Z0`. Other solver terms use the free-space value `Z0 = 376.7303`. This is an explicit inconsistency requiring comparison with the intended equations before regenerated fields can be trusted; this review did not validate the electromagnetic formulation.
- Sampling uses random indices with replacement and no recorded seed or source indices. This can explain repeated domains, although the recovered code alone does not prove it produced the exact supplied datasets.

### `DConvNet_batched.ipynb`

This is a stronger starting point than the project's current mini-batch notebook. It initializes `bestloss = 5`, uses the supplied arguments in `accuracy_func`, mounts Colab Drive, and adds digit/cylindrical example evaluations and sample-error histograms.

It defines checkpoint save/load functions, but cell 13 unconditionally loads `my_checkpoint.pth.tar` **before training**; the save invocation remains commented. Make checkpoint loading an optional resume operation, and save or pass the newly trained model for SDM. Such generated weights can remain excluded from Git. The error function still returns a sum of absolute errors labeled “Accuracy.” Personal Drive paths, CUDA-only casts, the external helper download, and the earlier validation/seed/documentation gaps remain.

### Recovered `SDM.ipynb`

This differs from the current project version. Cell 16 actively constructs and saves an initial guess consisting of a `(5000, 2, 28, 28)` tensor filled with 2. That recovers an initialization implementation, though it does not establish which initialization produced the earlier saved results.

The initializer occurs **after** offline training has already tried to load `Xk.pt` in cell 12, so it must move earlier. Size initial guesses from the relevant dataset and explicitly document both channels. Cell 14 now loads `Ks.pt` instead of saving the newly learned `Ks`, which introduces an unnecessary external-file dependency and overwrites the matrices just learned in memory. The metric formulas and unconditional model-checkpoint load remain unchanged. Online evaluation runs only the first 10 digit samples, with cylindrical evaluation commented out.

### Revised release assessment

The recovered notebooks cover much of the missing implementation: a forward solver, dataset-generation logic, improved mini-batch training/evaluation, checkpoint helper definitions, and an initial-guess definition. A clean release can use these three notebooks as its basis while keeping the supplied data and labels. It still needs test-set generation and format conversion, a fresh-run sequence that generates model weights/descent matrices locally, the noted bug fixes, and README/dependencies/license/ignore rules. Model weights and descent matrices need not be committed. Preserve the current notebooks' historical outputs separately if they are useful evidence.
