# DEVTESTS

This folder contains tests for development and testing. They are useful to debug/improve performances.

## ConjugateGradients

This test builds the matrix of the *FEM/mMS.py* example (it is possible to change `dt` increase the wheights of the stiffness over the mass ) and solves the linear system `nb_of_tests` times. The inital value and the RHS are both randomly chosen.

## matrixAssembly
This test is used to profile the performances of matrix assembly

## ionic
Single-cell 100-beat pacing test for all ionic models (Fenton4v, ModifiedMS2v, CourtemancheRamirezNattel, TenTusscherPanfilov). Saves the last beat trace as numpy arrays.
