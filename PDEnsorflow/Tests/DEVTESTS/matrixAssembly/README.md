# Tests/DEVTESTS/matrixAssembly

## Description
This test measures the temporal performance of the global matrix assembly
(mass and stiffness matrices) using the finite element method.

The test uses two input meshes of different sizes:
- **triangulated_square.pkl**: a coarse triangular mesh (63,001 nodes, 125,000 elements)
- **triangulated_square_fine_mm.pkl**: a fine triangular mesh (1,002,001 nodes, 2,000,000 elements)

For each mesh, the assembly is repeated several times and the mean and standard
deviation of the elapsed time are reported.

## Usage
From the `matrixAssembly` directory, run:
```bash
conda activate PDEnsorflow
source ~/TestCase/init.sh
python matrixAssembly.py
```

## Output
The script prints, for each mesh:
- Assembly time for each repetition
- Mean and standard deviation of assembly times
- Matrix dimensions and number of nonzeros

## Profiling Results

### Baseline (before optimization)
Serial per-element Python loop with `np.where` index lookups.
No GPU involvement in assembly.

| Mesh   | Nodes     | Elements  | Mean (s) | Std (s) |
|--------|-----------|-----------|----------|---------|
| Coarse | 63,001    | 125,000   | 35.696   | 7.034   |

### Optimized (after optimization, GPU)
Vectorized contravariant basis, Sigma, and local matrix computation.
Direct COO construction with TensorFlow `unsorted_segment_sum`: every element contribution is scattered into its global (row, column) entry in one call, instead of a per-element loop.
Running with GPU detected (`/physical_device:GPU:0`), TensorFlow 2.12.0.

| Mesh   | Nodes     | Elements  | Mean (s) | Std (s) | Speedup |
|--------|-----------|-----------|----------|---------|---------|
| Coarse | 63,001    | 125,000   | 0.780    | 0.584   | ~46x    |
| Fine   | 1,002,001 | 2,000,000 | 7.858    | 0.041   | ~73x    |

Note: first repetition includes TF/GPU warmup; steady-state performance

**Change Log**: 19 April 2026: matrices now are CSR
is ~0.45s (coarse) and ~7.8s (fine).

**Change Log**: 23 September 2026: the element entries are summed on the host.
Each one is located in the sparsity pattern the solver already computes
(binary search on the sorted pattern keys), then summed with `np.add.at`; only
the finished CSR matrices go to the device. The device path (`tf.unique` +
`unsorted_segment_sum`) needed about 12 bytes of scratch per element entry and
ran out of memory on a 12 GB card for an 18.2 M-tetrahedron mesh; it was also
not reproducible to the bit (the device sum adds with atomics). The host sum
is deterministic and agrees with the device one to about 2 float32 ulps.
Measured on the coarse mesh: 1.0 s (host) vs 2.4 s (device); fine mesh:
15.2 s vs 8.6 s, with the per-element callback of the benchmark script.
