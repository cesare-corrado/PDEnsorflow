# DEVTESTS

This folder contains tests for development and testing. They are useful to debug/improve performances.

## matrixAssembly
This test is used to profile the performances of matrix assembly

## ionic
Single-cell 100-beat pacing test for the ionic models (Fenton4v, ModifiedMS2v, CourtemancheRamirezNattel, TenTusscherPanfilov). Saves the last beat trace as numpy arrays. `tomek.py` paces the Tomek model (ENDO, EPI, MCELL) and can compare the last beat with the reference single-cell tool. `singlecell_vs_bench.py` compares the `singlecell` executable with the reference single-cell tool on every shared model, and swaps state files between the two.

## LAT
`lat_detector_scaling.py` measures what one step of `LatDetector` costs at
63k, 250k, 1M and 4M nodes, for both execution knobs
(`set_use_graph_check`, `set_use_deferred_count`) and both recording modes. It
needs an **idle** GPU: an earlier run on a card another process held at 99%
reversed the conclusion for `all = 0`. The numbers it produced on an RTX 5090
are in its docstring.

## betaJump
`beta_jump_cable.py` shows what the place of beta changes. On a 20 mm cable
whose second half has four times the membrane surface per unit volume, it runs
the same problem twice, once with beta dividing the conductivity (the
formulation before this change) and once with beta on the mass integral and the
conductivity divided by the single reference value, and prints the arrival time
of the front at both. Inside each half the two agree, because a constant beta
divides out exactly there; the whole difference is at the jump, where dividing
by a function of space is not an identity.

## IO
The automated `BaseWriter` unit test (`test_basewriter.py`) is now part of the
continuous-integration suite at `Tests/CICD/unit/`, where it runs on every push
and pull request. See `Tests/CICD/README.md`.
