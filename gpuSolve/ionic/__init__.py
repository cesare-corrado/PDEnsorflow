"""
ionic submodule of gpuSolve.
This module contains the 0D ionic models
Contains:
    * fenton4v:  the 4 variables fenton model

"""


__version__=['1','0','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)

