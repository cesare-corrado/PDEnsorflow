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
Direct COO construction with TensorFlow `unsorted_segment_sum` (torchcor-style).
Running with GPU detected (`/physical_device:GPU:0`), TensorFlow 2.12.0.

| Mesh   | Nodes     | Elements  | Mean (s) | Std (s) | Speedup |
|--------|-----------|-----------|----------|---------|---------|
| Coarse | 63,001    | 125,000   | 0.780    | 0.584   | ~46x    |
| Fine   | 1,002,001 | 2,000,000 | 7.858    | 0.041   | ~73x    |

Note: first repetition includes TF/GPU warmup; steady-state performance
is ~0.45s (coarse) and ~7.8s (fine).
