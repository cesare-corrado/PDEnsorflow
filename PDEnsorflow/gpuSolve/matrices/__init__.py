"""
matrices submodule of gpuSolve.
This module contains definitions of local Finite Element matrices
and functions to assemble global sparse matrices. 
Contains:
  * localMass:      a collection of functions to assemble local mass matrices in cartesian coordinates
  * localStiffness: a collection of functions to assemble local stiffness matrices in cartesian coordinates
  * globalMatrices: a collection of functions to assemble global sparse matrices
"""
import tensorflow as tf

__version__=['1','1','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)

from gpuSolve.matrices.localMass import localMass 
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.globalMatrices import assemble_mass_matrix
from gpuSolve.matrices.globalMatrices import assemble_stiffness_matrix
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
