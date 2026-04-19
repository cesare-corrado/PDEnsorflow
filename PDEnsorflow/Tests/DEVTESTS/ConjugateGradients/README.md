# Tests/DEVTESTS/ConjugateGradients

## Description
This test measures the temporal performance of the conjugate gradient
solver (`gpuSolve.linearsolvers.conjgrad.ConjGrad`) with the Jacobi
preconditioner (`gpuSolve.linearsolvers.jacobi_precond.JacobiPrecond`).

For each input mesh, the global mass `M` and stiffness `K` matrices are
assembled via the finite element method, and the linear system is
built as:

```
A = alpha * M + beta * K
```

with defaults `alpha = 0.01` and `beta = 1.0`.

A gold-truth solution `x*` is prescribed (pseudo-random, seed = 0).
The right-hand side is obtained as `b = A x*`. The initial guess is `0`.
The solver is run several times and mean and standard deviation of the
elapsed time are reported, together with the number of CG iterations
and the relative residual `||A x - b|| / ||b||`.

The test uses two input meshes of different sizes:
- **triangulated_square.pkl**: coarse triangular mesh (63,001 nodes, 125,000 elements)
- **triangulated_square_fine_mm.pkl**: fine triangular mesh (1,002,001 nodes, 2,000,000 elements)

## Usage
From the `ConjugateGradients` directory, run:
```bash
conda activate PDEnsorflow
source ~/TestCase/init.sh
python ConjugateGradients.py               # both meshes, 5 reps
python ConjugateGradients.py --mesh coarse --nrep 3
python ConjugateGradients.py --alpha 0.01 --beta 1.0 --toll 1e-7 --maxiter 30000
```

Command-line arguments:
- `--alpha`     (float, default `0.01`)
- `--beta`      (float, default `1.0`)
- `--nrep`      (int,   default `5`)
- `--maxiter`   (int,   default `5000`)
- `--toll`      (float, default `1e-7`)
- `--mesh`      (`coarse` / `fine` / `both`, default `both`)

Environment-variable switches:
- `GPUSOLVE_EAGER=1`        run with `tf.config.run_functions_eagerly(True)`
  (required for the *unmodified* baseline code, which is only graph-mode
  compatible after the optimisation).
- `GPUSOLVE_USE_SYSPATH=1`  do not prepend the in-tree `gpuSolve` package
  to `sys.path`; use whatever is installed in the active environment.
  This is the switch used to compare against the original code.

## Output
For each mesh the script prints:
- Elapsed solve time at each repetition
- Number of CG iterations
- Relative residual `||A x - b|| / ||b||`
- Relative `L2` error `||x - x*|| / ||x*||`
- Mean and standard deviation of elapsed times (full and steady-state)

## Profiling Results

Run on NVIDIA GPU, TensorFlow 2.12.0, CUDA 11.2, single-precision
(`float32`), `alpha = 0.01`, `beta = 1.0`, `toll = 1e-7`.

### Baseline (before optimization)
Original `ConjGrad` code path (run with `GPUSOLVE_EAGER=1`). The Python
`for` loop in `solve()` forces a GPU→CPU synchronisation at every
iteration (`if (tf.sqrt(self._residual) < self._toll)`). In the
preconditioned path, `||r||²` is recomputed every iteration even
though only `r·z` is needed for `beta`. `JacobiPrecond.build_preconditioner`
iterates in Python over all non-zeros.

| Mesh   | Nodes     | nnz        | Iters | Mean (s) | Steady-state mean (s) |
|--------|-----------|------------|-------|----------|-----------------------|
| Coarse |  63,001   |    439,001 |  1119 |   2.028  |   2.024               |
| Fine   | 1,002,001 |  7,006,001 |  2899 |   8.396  |   8.404               |

### Optimized — default path (`_use_graph_loop = False`)
- `ConjGrad.solve()` keeps a python-driven loop (so that each
  `_iterate` call dispatches to a single cached `@tf.function`
  graph), but reads the residual back to the host only every
  `self._check_every` iterations (default 5). This removes 4 out
  of 5 per-iteration CPU stalls without introducing a pathological
  number of extra iterations once CG has converged. The small
  block size is chosen so that warm-started CG (e.g. time-stepping
  a parabolic PDE that converges in <10 iterations) still exits
  quickly and does not drift into denormals. The convergence check
  also guards against NaN/Inf: once CG has fully converged in
  `float32`, `rzold` can underflow and the next
  `alpha = rzold/(p·Ap)` becomes `0/0 → NaN`; the guard breaks out
  of the loop in that case.
- `JacobiPrecond.build_preconditioner` vectorises the diagonal
  extraction with numpy boolean masking (no Python loop over nnz).

| Mesh   | Nodes     | nnz        | Iters | Mean (s) | Steady-state mean (s) | Speedup |
|--------|-----------|------------|-------|----------|-----------------------|---------|
| Coarse |  63,001   |    439,001 |  1120 |   1.839  |   1.858               |  ~1.09x |
| Fine   | 1,002,001 |  7,006,001 |  2900 |   7.858  |   7.867               |  ~1.07x |

### Optimized — full GPU-resident path (`_use_graph_loop = True`)
`ConjGrad._solve_graph` wraps the entire CG loop in one
`@tf.function` containing a `tf.while_loop`. The condition, the
`break`, and the NaN/Inf guard are all tensor ops; `X`, `r`, `p`,
`rzold` are carried as loop variables. The CPU dispatches exactly
one graph launch per `solve()` call and never reads the residual
back until the loop finishes. Enabled by passing
`use_graph_loop=True` in the config dict (or `--graph-loop` on the
command line).

| Mesh   | Nodes     | nnz        | Iters | Mean (s) | Steady-state mean (s) | vs default    |
|--------|-----------|------------|-------|----------|-----------------------|---------------|
| Coarse |  63,001   |    439,001 |  1119 |   2.725  |   2.651               |  ~1.4x slower |
| Fine   | 1,002,001 |  7,006,001 |  2901 |  83.5    |  83.5                 |  ~10x slower  |

The GPU-resident path is slower on this problem, and upgrading the
CUDA toolchain does not change that:

- **Upgrading `ptxas` 10.1 → 11.8** (via `conda install -c nvidia
  cuda-nvcc=11.8`) silences the miscompile-warning spam and would
  let XLA run without complaint, but it does **not** improve the
  solver's runtime on its own, because TF only reaches XLA by
  explicit opt-in.
- **Enabling XLA auto-clustering** (`TF_XLA_FLAGS=--tf_xla_auto_jit=1`)
  makes XLA crash at compile time with a cuDNN
  `RET_CHECK failure ... dnn != nullptr` inside the while-loop
  cluster — XLA tries to compile ops it cannot, for this shape mix.
- **Forcing `@tf.function(jit_compile=True)` on `_solve_graph`**
  fails immediately with:
  `InvalidArgumentError: Detected unsupported operations ... No
  registered 'SparseTensorDenseMatMul' OpKernel for XLA_GPU_JIT`.

The last error is the fundamental one. **In TF 2.12 there is no
XLA GPU kernel for `tf.sparse.sparse_dense_matmul`.** Since every
iteration of CG is built around a sparse SpMV, XLA cannot fuse the
body on GPU regardless of the ptxas/cuDNN versions. Without fusion,
`tf.while_loop`'s per-iteration control-flow + loop-carried
[N × 1] state costs more than the CPU-side stall it removes — so
the GPU-resident path remains ~1.4× / ~10× slower on these two
meshes.

Conclusion for this hardware and this TF version: the Python
for-loop dispatching a cached `@tf.function` body (the default
path) is already CPU-efficient enough that the remaining CPU stall
is below the threshold `tf.while_loop` can beat. The GPU-resident
path is still kept behind the flag because (a) it is the correct
path if and when a future TF version adds an XLA GPU kernel for
sparse matmul, and (b) it is useful as a structured baseline for
experimenting with alternative sparse representations (e.g. CSR via
`tf.raw_ops.CSRSparseMatrixMatMul`, or an indices/values-based
manual SpMV that XLA *can* fuse).

Runs with `GPUSOLVE_EAGER=1` (see "Usage"): the unmodified baseline
only works in that mode. The optimised code runs with the same
eager-mode setting for a like-for-like comparison.

Correctness (all configurations):
- Coarse: `||A x - b|| / ||b|| ≈ 9.3e-7` (below the 1e-6 target).
- Fine:   `||A x - b|| / ||b|| ≈ 3.4e-6` (float32 round-off floor for
  a matrix with ~10⁶ rows — the same residual is obtained by the
  baseline; the tolerance `toll = 1e-7` is simply unreachable in
  single precision on this problem size).
- `||x - x*|| / ||x*||` in the `3e-5` range in both cases.
