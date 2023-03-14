"""
diffop32 submodule of gpuSolve.
This module contains 2D partial differential operators
Contains:
  * laplace_homogeneous_isotropic_diffusion: 2D laplace solver for homogeneous isotropic diffusion  
  * laplace_heterogeneous_isotropic_diffusion: 2D laplace solver for heterogeneous isotropic diffusion  
"""

__version__=['1','1','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


from gpuSolve.diffop2D.laplace_heterogeneous_isotropic_diffusion import laplace_heterogeneous_isotropic_diffusion as laplace_heterog
from gpuSolve.diffop2D.laplace_homogeneous_isotropic_diffusion import laplace_homogeneous_isotropic_diffusion as laplace_homog
