"""
ionic submodule of gpuSolve.
This module contains the 0D ionic models
Contains:
    * IonicModel: the base class for the ionic models that implements
    * fenton4v:   the 4 variables Fenton model
    * mms2v.py:   the 2 variables modified Mitchell Schaeffer model
    * ms2v.py:    the 2 variables Mitchell Schaeffer model
"""


__version__=['1','1','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


