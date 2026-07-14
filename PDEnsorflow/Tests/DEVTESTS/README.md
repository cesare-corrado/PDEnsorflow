# DEVTESTS

This folder contains tests for development and testing. They are useful to debug/improve performances.

## matrixAssembly
This test is used to profile the performances of matrix assembly

## ionic
Single-cell 100-beat pacing test for all ionic models (Fenton4v, ModifiedMS2v, CourtemancheRamirezNattel, TenTusscherPanfilov). Saves the last beat trace as numpy arrays.

## IO
The automated `BaseWriter` unit test (`test_basewriter.py`) is now part of the
continuous-integration suite at `Tests/CICD/unit/`, where it runs on every push
and pull request. See `Tests/CICD/README.md`.
