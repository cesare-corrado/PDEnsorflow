"""
carp_compatibility submodule of gpuSolve.
This module implements the parameter-file / command-line front end, so a
simulation can be described by a `.par` file and a list of flags instead of a
hand-written Python script.
Contains:
    * ParFileReader:    lexer for the `.par` parameter-file format
    * OptionReader:     assembles files and flags into one ordered store
    * ParameterMapper:  turns that store into gpuSolve objects and settings
    * SimulationRunner: builds the solver, owns the time loop and the output
    * main:             the entry point of the `PDEnsorflow` executable
"""


from gpuSolve._version import __version__


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)
