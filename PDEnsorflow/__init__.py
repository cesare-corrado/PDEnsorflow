import os
import pathlib
import sys


"""
This is the PDEonsorflow package that configures the environment correctly

"""

if not 'LD_LIBRARY_PATH' in os.environ.keys():
    os.environ['LD_LIBRARY_PATH']=''


if 'CONDA_PREFIX' in os.environ:
    conda_prefix = pathlib.Path(os.environ['CONDA_PREFIX'])
    conda_lib = (conda_prefix / 'lib').as_posix()
    if not conda_lib in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':'+conda_prefix
        try:
            #os.execv(sys.argv[0], sys.argv)
            os.execv(sys.executable, ['python'] + [sys.argv[0]])
        except Exception as e:
            sys.exit('EXCEPTION: Failed to Execute after adding CONDA_PREFIX to LD_LIBRARY_PATH, '+e)


__version__=['1','0','1']

def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)

