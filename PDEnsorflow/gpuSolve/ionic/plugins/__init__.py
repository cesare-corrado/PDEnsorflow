"""
plugins submodule of gpuSolve.ionic.
This module contains the ionic plugins: currents with their own state
variables that are added to the current of a cell model (e.g. membrane
electroporation, fibroblast coupling). A plugin never runs on its own: it is
attached to a cell model through gpuSolve.ionic.ionicmodelwithplugins.
Contains:
    * IonicPlugin:                         the base class for the plugins
    * electroporation_debruin_krassowska98: the DeBruin-Krassowska (1998) membrane electroporation current
    * defib_ashihara_trayanova:             the Ashihara-Trayanova (2004) outward current activated by strong shocks
"""


from gpuSolve._version import __version__


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)
