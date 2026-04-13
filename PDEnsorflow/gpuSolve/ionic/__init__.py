"""
ionic submodule of gpuSolve.
This module contains the 0D ionic models
Contains:
    * IonicModel:                  the base class for the ionic models
    * fenton4v:                    the 4 variables Fenton model
    * mms2v.py:                    the 2 variables modified Mitchell Schaeffer model
    * ms2v.py:                     the 2 variables Mitchell Schaeffer model
    * courtemanche_ramirez_nattel: the Courtemanche-Ramirez-Nattel (1998) human atrial model
    * ten_tusscher_panfilov:       the ten Tusscher-Panfilov (2006) human ventricular model
"""


__version__=['1','3','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


