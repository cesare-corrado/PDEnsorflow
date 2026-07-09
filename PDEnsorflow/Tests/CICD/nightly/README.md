# Tests/CICD/nightly

Tier-2 continuous-integration tests: **heavier, reduced-config** solver
regressions with numeric sanity checks (e.g. CG residual below tolerance,
finite solution fields, stable matrix `nnz`). These are intended to run on a
**scheduled** basis on a **self-hosted GPU runner**, not on every push.

These tests are added incrementally, typically by taking the profiling scripts
in `Tests/DEVTESTS` and the examples in `Tests/FD` / `Tests/FEM`, shrinking the
problem size / number of steps, and adding assertions.

`pytest` does **not** collect this folder by default; the nightly workflow runs
it explicitly:

```bash
pytest PDEnsorflow/Tests/CICD/nightly
```

*(Placeholder &mdash; no nightly tests committed yet.)*
