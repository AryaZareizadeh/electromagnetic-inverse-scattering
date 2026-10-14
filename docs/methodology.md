# Recovered method and compatibility decisions

## Workflow

1. Generate a 28×28 relative permittivity map from MNIST intensities or two rasterized cylinders.
2. Compute complex scattered fields for 20 receiver positions and 20 incident plane-wave directions using the recovered method-of-moments equations.
3. Train a CNN to approximate the forward mapping from maps to 800 real field components.
4. Offline, fit a sequence of regularized descent matrices from the surrogate's field residuals and the known domain errors.
5. Online, repeatedly apply those matrices to measured-field residuals, starting from the supplied initial guess.

The CNN predicts fields; the descent stage performs inverse reconstruction.

## Exact CNN

The notebook class was named `AlexNet`, but its architecture is a custom forward surrogate:

```text
(N,2,28,28)
  Conv2d(2,50,kernel=2,stride=2) → ReLU
  Conv2d(50,100,kernel=2,stride=1,padding="same") → ReLU
  flatten to 19600
  Linear(19600,19600)
  Linear(19600,800)
```

There is no activation between the dense layers. The package preserves the eight state-dictionary parameter keys exactly (`conv1.0`, `conv2.0`, `fc1`, `fc2`, weights and biases). Its parameter count is 399,880,950; the first dense layer contributes 384,160,000 weights. `DConvNet` is the public class name, with an `AlexNet` compatibility alias.

The mini-batch source uses Adam/MSE, batch 100, learning rate `1e-6`, and 400 epochs. The full-batch source uses `5e-5` and one update over all data per epoch. They are distinct experiments. Cleanup fixes CPU loading, dropped tail batches, undefined `bestloss`, metrics that referenced global variables, missing checkpoint saving, and plotting after stale predictions. It does not change the architecture.

## Forward equations

Defaults are wavelength 1 m, object side 1 m, receiver radius 20 m, field amplitude 1, free-space impedance 376.7303, and `k0=2π`. Active pixels satisfy `eps_r>1`. For image row j, column k, spacing Δ=L/H, and cell area s=Δ²:

```text
x = -L/2 + (k+1/2)Δ
y = +L/2 - (j+1/2)Δ
V[n,t] = E0 exp(i k0 (x[n] cos(phi[t]) + y[n] sin(phi[t])))
Z[m,n] = k0 q[m] s/4 H0^(2)(k0 distance(m,n))       (m != n)
Z[m,m] = k0 q[m] s/4 (1 - 2i/π log(k0*1.7180724*s/(4e)))
Zh[n,n] = -i Z0 / (k0 (eps_r[n]-1))
J = solve(Z + Zh, V)
G[r,n] = -k0 Z0 s/4 H0^(2)(k0 distance(receiver[r],cell[n]))
Fields = G J
```

The original calls `Z_mn_func(..., eps_r[m], ...)` where the argument is named `Z0`. Consequently **the legacy formulation has `q[m]=eps_r[m]`**. This is inconsistent with the free-space impedance used in other terms. Repeating these exact equations reproduces the first supplied train, test, and cylinder fields at roughly `1e-15` relative error; replacing q with 376.7303 changes the first training field by approximately 46% in relative L2 norm. The supplied trained parameters therefore require the legacy formulation for comparable data.

`--formulation free-space` explicitly changes q to the free-space impedance. It addresses that argument inconsistency only; neither the historical diagonal area approximation nor the constant 1.7180724 has been independently established here as physically correct. This cleanup establishes computational fidelity, not a validation of the scientific model.

The refactored solver builds the domain-dependent system once and solves all incident directions together. It uses a linear solve instead of explicitly forming an inverse. Background-only maps produce zero scattered fields. Cylinder generation preserves the exact 69-cell raster shape at independently sampled centers in `[4,24)`; overlapping shapes form a union.

## Supervised descent

With flattened domain estimates X, true training domains Xt, observed fields Et, and forward surrogate F:

```text
R = F(X) - Et
D = Xt - X
R = U diag(S) Vᵀ
K = V diag(S / (S² + alpha²)) Uᵀ D
X_next = X + R K
```

The residual is predicted minus measured and the update is a **plus** operation. Defaults are `alpha=0.5`, maximum 800 offline iterations. The historical stopping score is `mean(sum((X-Xt)², dim=0))`, which scales with the number of training samples. After an update, training stops if improvement over the best score is ≤0.02 and **retains the stopping matrix**. The recorded original run stopped at iteration 341, giving 342 matrices and final score about 30.3627853.

Online reconstruction applies the learned matrices in order without needing ground truth. The active historical code has no clipping, lower-bound projection, or replacement of channel 0. Those changes would define a different reconstruction procedure. Metrics/plots now recompute forward fields after the final update.

## Metrics and preserved evidence

The notebook's “Accuracy” was a sum of absolute field errors; it was not classification accuracy. Outputs use `mse`, `mae`, `l1_sum`, and `max_sample_mse` with explicit meanings. Historical “RMS” functions computed squared relative energy ratios, not standard root mean square errors. Their logged values survive in archived outputs; the fitting stopping score remains unchanged.

All original code cells, comments, historical logs, and embedded PNG figures were exported under `archive/legacy`. Archived source contains historical bugs and external Colab paths and is reference material. The canonical executable implementation lives in `src/esdl`.
