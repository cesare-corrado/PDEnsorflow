# Tests/CICD

Continuous-integration test-suite for **PDEnsorflow**. Unlike `Tests/DEVTESTS`
(developer / profiling scripts, run by hand) and `Tests/FD` / `Tests/FEM`
(illustrative examples), everything here is an **automated pass/fail test**
executed by GitHub Actions.

## Layout

| Folder     | Tier   | Runs on                         | Purpose                                                     |
|------------|--------|---------------------------------|-------------------------------------------------------------|
| `unit/`    | Tier 1 | every push / pull request (CPU) | fast, CPU-only checks of individual components              |
| `nightly/` | Tier 2 | scheduled, self-hosted GPU      | heavier reduced-config solver regressions (numeric sanity) |

The GitHub-hosted Tier-1 job (`.github/workflows/ci.yml`) installs the **CPU**
TensorFlow build and runs `pytest`, which by default collects only `unit/` (a
`gpuSolve` import smoke plus the component tests; see `pytest.ini` `testpaths`).

**Tier-2 (nightly GPU) is not yet implemented.** A full implementation spec —
the scheduled self-hosted-GPU workflow plus a 2-D mMS regression modelled on
`unit/test_mms_1d.py` — is kept as a maintainer planning prompt in
`Tier2_nightly_GPU_prompt.md` (outside the repository tree).

## Running locally

```bash
conda activate PDEnsorflow          # or Claude_testing
python -m pip install pytest        # once
# from the repository root (the folder that contains setup.py):
pytest                              # Tier-1 unit tests only
pytest PDEnsorflow/Tests/CICD/nightly   # heavier suite, when present
```

Shared fixtures live in `conftest.py`; it also prepends the in-tree package to
`sys.path`, so the suite always exercises the edited `gpuSolve` in this
repository even without `pip install -e .`.
