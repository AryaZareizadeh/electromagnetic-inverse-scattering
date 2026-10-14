# Supplied data and trained artifacts

All original non-notebook files remain available in the author's local working copy. Datasets and trained artifacts are excluded from GitHub and currently have no public download location. Their SHA-256 values and original/new locations are recorded in `asset_manifest.json`; `esdl verify-assets` recomputes the hashes when those assets are present. Original arrays retain float64 storage; model inputs are converted to float32 per batch.

## Canonical arrays

| Split | Domains | Fields | Labels | Samples |
|---|---|---|---|---:|
| train | `data/processed/Domains.npy` | `data/processed/Field_Data.npy` | `data/processed/Labels.npy` | 5,000 |
| test | `data/processed/Domains_test.npy` | `data/processed/Field_Data_test.npy` | `data/processed/Labels_test.npy` | 1,000 |
| cylinders | `data/processed/Domains_cy.npy` | `data/processed/Field_Data_cy.npy` | none | 1,000 |

Domains have shape `(N,2,28,28)`: **channel 0 is zero; channel 1 is relative permittivity**. Background is 1 and scatterers range up to 2. The domain channels are not the ordinary real/imaginary order. Fields have shape `(N,2,20,20)`, with channels `[real, imaginary]` and spatial axes `[receiver, incident direction]`. C-order flattening places 400 real field values before 400 imaginary field values, and 784 zero-domain values before 784 permittivity values.

The recovered generator initially made raw real maps `(N,28,28)` and complex fields `(N,20,20)`. `encode_domains` and `encode_fields` now supply the previously missing conversion step. Generation writes into a new directory, using the canonical filenames and a JSON provenance record.

MNIST sampling uses replacement, so repeated domains may occur. New runs record a seed and source indices. The original notebook does not record its seed or source indices; exact re-sampling of the historical collection is unavailable. New MNIST intensity conversion preserves float32 `pixel/255 + 1` values before storage, matching the historical torchvision path.

## Preserved legacy arrays

`data/legacy/dataset_train.pkl` and `dataset_test.pkl` retain the older serialized representation. The runnable route uses NumPy arrays and does not need to unpickle these objects. `data/legacy/Domains01.npy` is a distinct collection with uncertain provenance; it was preserved without assigning it to training or deleting it.

## Trained files

| File under `artifacts/pretrained/` | Contents | Bytes |
|---|---|---:|
| `my_checkpoint.pth.tar` | CNN state plus Adam optimizer | 4,798,581,225 |
| `Ks.pt` | 342 descent matrices, logical shape `(342,800,1568)` | 4,014,080,668 |
| `Xk.pt` | Initial guesses `(5000,2,28,28)` | 31,360,668 |

The descent tensor is a slice of an originally allocated 800-matrix tensor. Its serialized backing storage contains all 800 slots; only the 342 logical matrices are applied. The original is unchanged. Newly fitted matrices save a compact tensor with only learned iterations.

Every value in `Xk.pt` is 2, including both domain channels. This differs from the domain training convention, but reproduces the supplied descent initialization. Loading uses the requested sample count; a uniform saved initializer can also be broadcast to a larger collection. No projection or channel correction is silently added.

The checkpoint's Adam learning rate is `1e-7` and all parameter steps are 14,259. Neither current training notebook specifies that learning rate, and the file has no epoch or configuration metadata. Its exact training history cannot be inferred solely from its filename. New training saves epoch, effective configuration, optimizer state, and metrics.
