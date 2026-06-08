import os
import glob
import pathlib
import sysconfig
import sys

"""
This is a gpuSolve package that implements functions to solve PDEs using tensorflow.

"""

if not 'LD_LIBRARY_PATH' in os.environ.keys():
    os.environ['LD_LIBRARY_PATH']='/lib64'


# Directories the dynamic linker must see so TensorFlow can dlopen the CUDA
# runtime with no manual setup: the active conda environment lib dir (for a
# conda-installed CUDA) plus the pip-installed CUDA wheels (nvidia-*-cu12) that
# ship with tensorflow[and-cuda]. LD_LIBRARY_PATH is read by the linker at
# process start, so when an entry is missing we add it and re-exec once.
lib_dirs = []
if 'CONDA_PREFIX' in os.environ:
    conda_prefix = pathlib.Path(os.environ['CONDA_PREFIX'])
    lib_dirs.append((conda_prefix / 'lib').as_posix())
for site_dir in {sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')}:
    if site_dir:
        lib_dirs += sorted(glob.glob(os.path.join(site_dir, 'nvidia', '*', 'lib')))

missing = [d for d in lib_dirs if d and d not in os.environ['LD_LIBRARY_PATH'].split(':')]
if missing:
    if os.environ['LD_LIBRARY_PATH']=='':
        os.environ['LD_LIBRARY_PATH'] = ':'.join(missing)
    else:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(missing)+':'+os.environ['LD_LIBRARY_PATH']
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        sys.exit('EXCEPTION: Failed to Execute after adding the CUDA libs to LD_LIBRARY_PATH, '+str(e))


__version__=['1','3','1']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)
    

#print(' This is gpuSolve Version {}'.format(__version__),flush=True)




