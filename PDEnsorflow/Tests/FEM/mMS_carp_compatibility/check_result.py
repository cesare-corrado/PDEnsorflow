#!/usr/bin/env python
"""
    Checks the result of this example against the number that can be derived
    analytically, and against the run produced by the Python API.

    Ahead of the activation front the fast gate is still closed, so the
    modified Mitchell-Schaeffer reaction reduces to the Nagumo bistable form
    and the planar front travels at

        CV = 0.5 (1 - 2 u_crit) sqrt(2 D / tau_in)

    with D the diffusion coefficient the solver actually used, which for this
    parameter file is sigma / beta = 1000 um^2/ms. The measured speed comes
    from a straight-line fit of the local activation time against x over the
    interior of the sheet, away from the paced strip and the far edge.

    When equivalent_script.py has also been run, the two results are compared
    node by node: they must agree to round-off, because they are the same
    solver reached two different ways.

        python check_result.py

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

from gpuSolve.IO.readers import IGBReader

FRONTEND_IGB : str = os.path.join('OUT_mMS', 'vm.igb')
API_IGB : str      = os.path.join('OUT_mMS_api', 'vm.igb')
MESH_PTS : str     = 'square.pts'

D_COEFF : float  = 1000.0        # um^2/ms, sigma / beta as the parameter file sets them
TAU_IN : float   = 0.15
U_CRIT : float   = 0.1
FRAME_DT : float = 1.0           # ms, the spacedt of the parameter file
V_THRESHOLD : float = -30.0      # mid-upstroke of the [-80, 20] band


def read_igb(path: str) -> np.ndarray:
    """ read_igb(path) returns the solution as a (nframes, nnodes) array """
    reader = IGBReader()
    reader.read(path)
    header = reader.header()
    return(np.array(reader.data()).reshape(header['t'], header['x']))


def read_x(path: str) -> np.ndarray:
    """ read_x(path) returns the x coordinate of every node, in micrometres """
    with open(path, 'r') as fpts:
        npt = int(fpts.readline().strip())
        return(np.array([float(fpts.readline().split()[0]) for _ipt in range(npt)]))


def conduction_velocity(V: np.ndarray, xcoord: np.ndarray) -> float:
    """ conduction_velocity(V, xcoord) measures the planar front speed in um/ms
        from the local activation times of the sheet interior
    """
    lat = np.full(V.shape[1], np.nan)
    for jnode in range(V.shape[1]):
        above = np.where(V[:, jnode] >= V_THRESHOLD)[0]
        if above.size > 0:
            lat[jnode] = above[0] * FRAME_DT
    # average the activation time over each column of nodes sharing an x, then
    # fit the interior: the paced strip and the far edge are not on the front
    xmin, xmax = xcoord.min(), xcoord.max()
    lo, hi     = xmin + 0.25 * (xmax - xmin), xmin + 0.85 * (xmax - xmin)
    inside     = np.logical_and(np.logical_and(xcoord >= lo, xcoord <= hi), np.isfinite(lat))
    slope      = np.polyfit(xcoord[inside], lat[inside], 1)[0]
    return(1.0 / slope)


if __name__ == '__main__':
    xcoord = read_x(MESH_PTS)
    V      = read_igb(FRONTEND_IGB)
    print('front end : {} frames x {} nodes'.format(V.shape[0], V.shape[1]))
    print('            V in [{:.3f}, {:.3f}] mV, all finite: {}'.format(
        V.min(), V.max(), bool(np.all(np.isfinite(V)))))

    measured = conduction_velocity(V, xcoord)
    analytic = 0.5 * (1.0 - 2.0 * U_CRIT) * np.sqrt(2.0 * D_COEFF / TAU_IN)
    print('CV measured: {:7.3f} um/ms ({:.2f} cm/s)'.format(measured, measured / 10.0))
    print('CV analytic: {:7.3f} um/ms ({:.2f} cm/s)'.format(analytic, analytic / 10.0))
    print('rel. error : {:.2%}'.format(abs(measured - analytic) / analytic))

    if os.path.isfile(API_IGB):
        W = read_igb(API_IGB)
        if W.shape != V.shape:
            print('API run has a different shape ({} vs {}): not comparable'.format(W.shape, V.shape))
        else:
            diff = np.abs(V - W)
            print('vs Python API: max |dV| = {:.3e} mV, mean |dV| = {:.3e} mV'.format(
                diff.max(), diff.mean()))
    else:
        print('({} not found: run equivalent_script.py to compare the two)'.format(API_IGB))
