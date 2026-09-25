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


# Where the pip CUDA wheels (nvidia-*-cu12, pulled in by tensorflow[and-cuda])
# are installed. Both the linker path below and the ptxas pin further down look
# for them here, so the lookup is done once.
site_dirs = [d for d in {sysconfig.get_paths().get('purelib'),
                         sysconfig.get_paths().get('platlib')} if d]

# Directories the dynamic linker must see so TensorFlow can dlopen the CUDA
# runtime with no manual setup: the active conda environment lib dir (for a
# conda-installed CUDA) plus the CUDA wheels. LD_LIBRARY_PATH is read by the
# linker at process start, so when an entry is missing we add it and re-exec once.
lib_dirs = []
if 'CONDA_PREFIX' in os.environ:
    conda_prefix = pathlib.Path(os.environ['CONDA_PREFIX'])
    lib_dirs.append((conda_prefix / 'lib').as_posix())
for site_dir in site_dirs:
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



def cuda_data_dir(search_dirs: list) -> str:
    """ cuda_data_dir(search_dirs) returns the first directory under search_dirs that
        is a usable CUDA root for XLA, i.e. the nvidia-cuda-nvcc-cu12 wheel, which
        holds both bin/ptxas and nvvm/libdevice. Returns '' when there is none.
    """
    for search_dir in search_dirs:
        candidate = os.path.join(search_dir, 'nvidia', 'cuda_nvcc')
        # both are needed: ptxas assembles the PTX, libdevice is linked into it.
        if os.access(os.path.join(candidate, 'bin', 'ptxas'), os.X_OK) and \
           os.path.isdir(os.path.join(candidate, 'nvvm', 'libdevice')):
            return(candidate)
    return('')


def pin_cuda_data_dir(environment: dict, search_dirs: list) -> str:
    """ pin_cuda_data_dir(environment, search_dirs) adds --xla_gpu_cuda_data_dir to
        XLA_FLAGS in environment, pointing at the CUDA root shipped with the pip
        CUDA wheels, and returns the new XLA_FLAGS value.
        Every ionic model compiles differentiate() with XLA, and XLA shells out to
        ptxas to assemble the PTX. It probes a list of candidate CUDA roots and takes
        the first one it accepts, so a CUDA toolkit installed system-wide can win
        over the wheel: releases 12.0 to 12.6.2 are rejected outright for a clamping
        miscompile, and then no GPU kernel compiles at all. The directory named by
        this flag is the first candidate probed, so pinning it means no system path
        is ever consulted.
        Unlike LD_LIBRARY_PATH this needs no re-exec: XLA_FLAGS is read when the
        first kernel is compiled, long after process start.
        Inert when the caller already named a data dir (an explicit choice wins) or
        when no wheel is installed.
    """
    xla_flags = environment.get('XLA_FLAGS', '')
    if '--xla_gpu_cuda_data_dir' in xla_flags:
        return(xla_flags)
    data_dir = cuda_data_dir(search_dirs)
    if not data_dir:
        return(xla_flags)
    xla_flags = '{}--xla_gpu_cuda_data_dir={}'.format(
        xla_flags + ' ' if xla_flags else '', data_dir)
    environment['XLA_FLAGS'] = xla_flags
    return(xla_flags)


# Done after the re-exec above, not before: this one only has to be in place by
# the time the first kernel is compiled.
pin_cuda_data_dir(os.environ, site_dirs)

from gpuSolve._version import __version__


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)
    

#print(' This is gpuSolve Version {}'.format(__version__),flush=True)




