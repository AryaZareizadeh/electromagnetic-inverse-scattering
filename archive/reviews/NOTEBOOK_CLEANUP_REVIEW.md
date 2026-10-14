# Saved artifacts and notebook consolidation

Reviewed 13 October 2026. This assessment compares the three notebooks in the project directory with the three recovered notebooks in `G:/Master study/2nd Semester/Computational Electromagnetics/CE project/Codes`. It covers active code, commented alternatives, and saved outputs. Cell numbers start at 1 and include markdown cells. No notebooks or datasets were changed or deleted.

## Filenames to look for

| Exact filename | Contents expected by the code | Where it is referenced |
| --- | --- | --- |
| `my_checkpoint.pth.tar` | CNN checkpoint dictionary with `state_dict`; the training save helper also includes `optimizer` for resuming training. | Both SDM versions, cell 10; recovered `DConvNet_batched.ipynb`, cell 13; commented blocks in both current CNN notebooks, cell 13. |
| `Ks.pt` | The learned supervised descent matrices, shaped `(number_of_iterations, 800, 1568)`. | Both SDM versions, cell 14. The current project version saves them; the recovered version loads them. |
| `Xk.pt` | Initial reconstruction guesses, normally an unflattened `(N, 2, 28, 28)` tensor in the creation blocks. This is initialization data, not learned weights. | Both SDM versions' offline and online loading cells; current SDM cell 18 and recovered SDM cell 16 contain creation alternatives. |

All three are referenced at `/content/drive/MyDrive/Colab Notebooks/` in the original workflow: the `Colab Notebooks` folder under Google Drive's `MyDrive`. None is present in either inspected local directory. These names identify what to locate; the actual checkpoint contents cannot be validated until supplied. The code has no separate named hyperparameter file: learning rate, epochs, batch size, regularization, and iteration limit are notebook variables.

`helper_functions.py` is a separate downloaded helper dependency, not a model parameter file.

## Which versions to use

### CNN notebooks

**Use the recovered `DConvNet_batched.ipynb` as the main training source.** It is the most complete of the inspected versions, although timestamps/version lineage have not been established.

| Difference | Current `DConvNet.ipynb` | Current `DConvNet_FullData.ipynb` | Recovered `DConvNet_batched.ipynb` |
| --- | --- | --- | --- |
| Architecture | Same architecture | Same architecture | Same architecture |
| Batch strategy | 100 samples | All 5,000 samples | 100 samples |
| Learning rate | `1e-6` | `5e-5` | `1e-6` |
| Epoch variable | 400 | 400 | 200 |
| `bestloss` initialization | Absent | Present | Present |
| Error function arguments | Ignored | Ignored | Used correctly |
| Checkpoint helper | Commented, including a missing colon in the loader definition | Same commented block | Active valid helper definitions, active load, commented save invocation |
| Cylindrical evaluation | Absent | Absent | Example plots and error distribution |
| Error distributions | Absent | Absent | Digit and cylindrical datasets |
| Historical outputs | Completed 200-epoch log and field plots | Interrupted training and plots | Only Drive-mount text retained |

The class definitions in **all five model-bearing notebooks**, including both SDM copies, are structurally identical after comments are excluded. Their common model has 399,880,950 parameters. These are alternative workflows around the same architecture, not separate model architectures.

Preserve the current mini-batch notebook's historical results. The full-batch variant's distinct scientific settings can be retained as an explicitly labeled experiment/configuration if useful; it does not need a duplicate architecture and duplicated plotting code. Do not silently treat full-batch and mini-batch training as the same experiment: batch size, learning rate, and epoch count differ.

### SDM notebooks

**Combine the two versions rather than replacing one wholesale.** The common offline algorithm and metric functions are the same.

- Keep the current project version's `torch.save(Ks, ...)` operation after learning.
- Keep the recovered version's load operation as an explicit option for using previously learned matrices; do not automatically overwrite the matrices just learned.
- Keep the recovered version's initial-guess creation block and labeled ground-truth/reconstruction/field plots.
- Retain the commented centered-square initialization as an optional experiment, with its assumptions documented.
- Put initialization before offline training, and size it for the relevant dataset.
- Digit online evaluation currently selects the first 20 cases in the project copy and the first 10 in the recovered copy. Make sample selection explicit.
- Preserve the project's saved offline log, which records 342 learned matrices. The recovered copy contains only checkpoint-loading text, not a comparable completed reconstruction record.

### Data creation

**Keep the recovered `Data_Creation.ipynb` as the provenance/generation notebook.** Its main and cylindrical field-solver blocks are identical after dataset-variable names are substituted. They can share one solver function, preserving their actual computations until the earlier `Z0`/`eps_r` argument inconsistency is resolved.

For the intended release, supplied datasets can be used directly. Re-running generation is optional; missing test-generation code does not prevent publishing or training on the supplied test data.

## Useful commented code to preserve or turn into options

| Commented block | Location | Recommended treatment |
| --- | --- | --- |
| Checkpoint dictionary creation and saving | Current CNN notebooks, cell 13; recovered batched notebook, cell 13 | Keep one working implementation. Distinguish loading for inference from loading model plus optimizer for resuming training. |
| Raw-to-two-channel array conversion | Recovered batched notebook, cell 6 | Preserve as a conversion utility. It explains the stored domain channel 0 being zero and channel 1 carrying real permittivity, and field channels being real/imaginary parts. Apply only to raw arrays; supplied project arrays are already converted. |
| Lower learning rate after training | Recovered batched notebook, cell 15 | Retain as an explicitly optional continuation/fine-tuning setting, not a hidden action in Run All. |
| Centered-square initial guess | Current SDM, cell 18; recovered SDM, cell 16 | Keep as an alternative to the recovered active uniform guess. Select and document the intended initialization. |
| Saved descent-matrix loading/saving | Both SDM versions, cell 14 | Keep both modes, with clear train-versus-load selection. |
| Cylindrical reconstruction dataset selection | Current SDM, cell 16; recovered SDM, cell 17 | Keep as a dataset option or separate evaluation cell instead of swapping lines manually. |
| Plots during offline training and every fifth online iteration | Current SDM, cells 13 and 16; recovered SDM, cells 13 and 17 | Keep as optional progress visualization. Avoid repeated copies of plotting code. |
| Forward-field comparisons for reconstructed objects | Current SDM, cell 17; recovered SDM, cell 19 | Preserve the diagnostic idea, but rewrite variable selection. The old block references `Domains_test` even though those variables are not defined in SDM, hardcodes 5,000 samples while `Xt` is later the 1,000-sample test set, and includes a comparison against `yvis[0]` inside a loop over other samples. |

Two commented experiments need care:

- The centered-square initializer starts from zeros, including a zero background in the material channel, whereas supplied domains use background 1. The active recovered uniform initializer fills **both** channels with 2, whereas supplied domain channel 0 is always zero. Preserve them as evidence of past experiments; they are not interchangeable, and neither should become a default silently.
- The commented `torch.mul((Xk_temp >= 1), Xk_temp) + (Xk_temp < 1)` is a lower-bound projection. As written it acts on **both** channels, including the normally zero first channel. Its channel scope needs an explicit decision before enabling it.

## Redundant or obsolete blocks

- The commented `NumpyDataset`/pickle loaders and their `DataLoader` declarations in the current CNN notebooks belong to an older data representation. The pickle tensors are single-channel real domains and complex fields; simply uncommenting them will not match the two-channel CNN inputs/real-valued targets. If adopting DataLoader, build it over the supplied two-channel arrays instead of reviving the incompatible path unchanged.
- Commented layer-shape prints, batch-index prints, and scratch `a`/`b` array calculations are development diagnostics, not separate features.
- The current full-batch notebook's `torch.abs()` cell is a scratch expression that raises TypeError.
- Per-epoch field plots in the current CNN copies largely duplicate the explicit evaluation plots in the recovered batched notebook. Keep one plotting implementation and invoke it where useful.
- Unused imports such as `pandas`, training-notebook `scipy`, `accuracy_fn`, and many torchvision helpers can be removed from the runnable route after final consolidation. Torchvision remains needed by the data-generation notebook.
- The downloaded helper module is only used for timing in these notebooks. It can be replaced by a small local timing helper.

The recovered error histograms are useful functionality, but their horizontal coordinates currently use `minimum_error * bin_number`, while the counts are constructed from bins beginning at `minimum_error` with a fixed `errorLength`. Preserve the feature and correct the coordinate construction during cleanup.

## Data files

Keep the main training/testing arrays and labels requested for the release, plus the cylindrical arrays if retaining those evaluations. Earlier checks showed the two pickle datasets store exactly the same main train/test data in an older representation; they can be archived outside the release if keeping the `.npy` route. `Domains01.npy` is a distinct dataset, not an exact duplicate. Its intended role is still unestablished; do not delete it based only on its name.

## Suggested final notebook set

1. `Data_Creation.ipynb`: optional generation/provenance and raw-array conversion.
2. `DConvNet_batched.ipynb`: one canonical CNN architecture, training/resume/load, checkpoint save, and digit/cylindrical evaluation.
3. `SDM.ipynb`: one canonical offline/online workflow with explicit initialization, matrix train/load/save, dataset selection, and plots.

Archive alternative source versions and historical outputs before removing duplicate notebooks from the eventual release. The core implementation is already available; cleanup should preserve useful options and consolidate repeated code. No full generation, training, or checkpoint execution was attempted for this comparison.
