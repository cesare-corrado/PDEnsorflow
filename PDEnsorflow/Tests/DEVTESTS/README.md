# DEVTESTS

This folder contains tests for development and testing. They are useful to debug/improve performances.

## matrixAssembly
This test is used to profile the performances of matrix assembly

## ionic
Single-cell 100-beat pacing test for all ionic models (Fenton4v, ModifiedMS2v, CourtemancheRamirezNattel, TenTusscherPanfilov). Saves the last beat trace as numpy arrays.

## IO
Tests for the `gpuSolve.IO.writers` submodule: `test_basewriter.py` checks the
`BaseWriter` GPU container chunking (iteration-based `every_N` and memory-based
`max_chunk_mb` triggers) and the final aggregation to a NumPy array.
