#!/usr/bin/env python
"""
    Checks that a run resumed from a saved state reproduces the uninterrupted run.

    Run the parameter file once saving the state at 100 ms, then resume from it
    into another output folder:

        PDEnsorflow +F parameters.par -num_tsav 1 -tsav[0] 100
        PDEnsorflow +F parameters.par -simID OUT_mMS_restart -start_statef OUT_mMS/state.100.0
        python check_restart.py

    The state is saved as the reference simulator saves it (state_format =
    roe1, the default): OUT_mMS/state.100.0.roe, a binary state file the
    reference can restart from too.

    The restarted output starts with the saved potential and then records the
    same frames as the uninterrupted run from 100 ms on, so the two are compared
    frame by frame over the overlap.

    They cannot agree to round-off. The CG warm-start history U^{n-1} is not part
    of a checkpoint, so the first step after the restart starts CG from a
    different initial guess, and the two runs agree to within the CG tolerance.
    On a GPU there is a second contribution of the same size: re-running the same
    parameter file twice already differs by about 2e-2 mV (see README.md), so
    that is the floor to compare the restart difference against.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

from gpuSolve.IO.readers import IGBReader
from gpuSolve.IO.readers import RoeReader

FULL_IGB : str    = os.path.join('OUT_mMS', 'vm.igb')
RESTART_IGB : str = os.path.join('OUT_mMS_restart', 'vm.igb')
STATE_FILE : str  = os.path.join('OUT_mMS', 'state.100.0.roe')
FRAME_DT : float  = 1.0          # ms, the spacedt of the parameter file


def read_igb(path: str) -> np.ndarray:
    """ read_igb(path) returns the solution as a (nframes, nnodes) array """
    reader = IGBReader()
    reader.read(path)
    header = reader.header()
    return(np.array(reader.data()).reshape(header['t'], header['x']))


if __name__ == '__main__':
    state   = RoeReader()
    content = state.read(STATE_FILE)
    print('state file : {} (format {}) at t = {:.4f} ms, {} nodes, regions {}'.format(
        STATE_FILE, content['format'], state.time(), state.num_nodes(),
        ', '.join('{} ({} nodes)'.format(region['model'], state.region_nodes(index).size)
                  for index, region in enumerate(content['regions']))))
    saved_Vm = content['globals']['Vm']

    V_full    = read_igb(FULL_IGB)
    V_restart = read_igb(RESTART_IGB)
    print('frames     : uninterrupted {}, restarted {}'.format(V_full.shape[0], V_restart.shape[0]))

    # the first restarted frame is the saved state itself
    print('restart frame 0 vs saved Vm: max |dV| = {:.3e} mV'.format(
        np.abs(V_restart[0, :] - saved_Vm).max()))

    # the remaining frames are the last ones of the uninterrupted run
    overlap = V_restart.shape[0] - 1
    diff    = np.abs(V_restart[1:, :] - V_full[-overlap:, :])
    print('restarted vs uninterrupted over the last {} frames ({:.0f} ms):'.format(
        overlap, overlap * FRAME_DT))
    print('            max |dV| = {:.3e} mV, mean |dV| = {:.3e} mV, all finite: {}'.format(
        diff.max(), diff.mean(), bool(np.all(np.isfinite(V_restart)))))
    per_frame = diff.max(axis=1)
    print('            max |dV| on the first / last compared frame: {:.3e} / {:.3e} mV'.format(
        per_frame[0], per_frame[-1]))
