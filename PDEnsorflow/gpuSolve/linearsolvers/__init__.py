"""
linearsolvers submodule of gpuSolve.
This module contains definitions of linear solvers
to solve linear problems (matrix inversion)
Contains:
  * conjGrad: conjugate gradient

"""

__version__=['1','2','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)

from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.linearsolvers.jacobi_precond import JacobiPrecond
