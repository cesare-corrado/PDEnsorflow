# DEVTESTS

## Description
This folder contains tests for development and testing. They are useful to debug/improve performances.

### mMS0d

The *FEM/mMS.py* example with no CG solution

### ConjugateGradients

This test builds the matrix of the *FEM/mMS.py* example (it is possible to change `dt` increase the wheights of the stiffness over the mass ) and solves the linear system `nb_of_tests` times. The inital value and the RHS are both randomly chosen.
