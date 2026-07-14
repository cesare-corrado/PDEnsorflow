#!/usr/bin/env python
"""
    Tier-2 nightly GPU regression for the 2D monodomain (modified
    Mitchell-Schaeffer) solver (gpuSolve.physics.MonodomainSolver driving
    ModifiedMS2v) on the real triangulated_square data mesh (10x10 mm sheet,
    63001 nodes). Heavier than the 1D unit test (test_mms_1d.py, the template):
    it exercises the actual mesh loader and reverse Cuthill-McKee renumbering.

    A depolarised block over the first 5% of the sheet in x (V = +20 mV over the
    resting V = -80 mV -- the model's dimensional [vmin, vmax] = [-80, 20] range)
    launches a PLANAR activation front that travels in +x. The diffusion tensor
    is forced isotropic (sigma*I, ignoring the mesh fibres/regions) so the front
    speed is the same analytic Nagumo value as the 1D cable, independent of
    direction:

        CV = 0.5 (1 - 2 u_crit) sqrt(2 D / tau_in),   with D = sigma

    (the solver's equation (M + dt K) U^{n+1} = M (U^n + dt (dU + I0)) carries no
    Cm / surface-to-volume factor, so the diffusion coefficient is exactly the
    stiffness conductivity sigma).

    The simulation runs ONCE (module-scoped fixture); two tests share that run:
      * propagation (qualitative): the planar front is unidirectional -- the mean
        local activation time per x-column increases monotonically along the
        sheet interior, the potential stays in the physical band, and nothing
        goes non-finite. (A per-node monotonicity check as in the 1D test does
        not apply: on the 2D grid many nodes share each x, so the front is
        checked column by column.)
      * conduction velocity (quantitative): CV measured from a linear fit of LAT
        vs x over the sheet interior matches the analytic value within 10%
        (~4.2% at these parameters; the margin absorbs the linear-FEM /
        forward-Euler discretisation bias).

    Requires eager execution (ConjGrad stores r/p between traced calls; see the
    module fixture). Marked nightly + gpu: intended for the scheduled self-hosted
    GPU runner, where it also runs the whole Tier-1 unit suite (un-skipping the
    device-gated csr_axpby native-path cases).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.physics import MonodomainSolver

# Sheet / solver parameters, tuned on the GPU box so the planar front is well
# resolved (~7.9 elements across it) and the run is a few seconds; the measured
# CV error is ~4.2% and reproducible bit-for-bit run to run.
_MESH_FILE = 'triangulated_square.pkl'
_SIGMA  = 0.5                                     # isotropic conductivity == diffusion coefficient D
_DT     = 0.05
_TEND   = 10.0
_VMIN   = -80.0                                   # resting potential (dimensional mV)
_VMAX   = 20.0                                    # plateau potential (dimensional mV)
_VTH    = -30.0                                   # activation threshold (mid-upstroke), rising edge
_XFRAC  = 0.05                                    # depolarised block over the first 5% of the sheet in x
_INT_LO = 0.4                                     # sheet interior for the LAT fit: [0.4, 0.8] * Lx
_INT_HI = 0.8
_CV_TOL = 0.10                                    # relative tolerance on conduction velocity


@pytest.fixture(scope='module', autouse=True)
def _eager_mode():
    """ConjGrad stores r/p between traced calls, so it must run eagerly. Set the
    flag in fixture *setup* (not at import time) so it holds regardless of the
    eager state another test module may have left behind, then restore it."""
    prev = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(bool(prev))


def _sigma_iso(elemtype: str, iElem: int, domain, matprop) -> np.ndarray:
    """Isotropic diffusion tensor sigma*I (fibres/regions deliberately ignored so
    the planar front carries the analytic Nagumo speed)."""
    return(_SIGMA * np.eye(3))


def _dfmass(elemtype: str, iElem: int, domain, matprop):
    """Empty mass-property callback."""
    return(None)


def _local_activation_times(Vrec: np.ndarray, times: np.ndarray, vth: float) -> np.ndarray:
    """LAT per node = interpolated time of the first rising crossing of vth
    (NaN where the node never crosses, e.g. the initially-depolarised block)."""
    npt = Vrec.shape[1]
    lat = np.full(npt, np.nan)
    for j in range(npt):
        v = Vrec[:, j]
        k = int(np.argmax((v[:-1] < vth) & (v[1:] >= vth)))
        if (v[k] < vth) and (v[k + 1] >= vth):
            lat[j] = times[k] + (vth - v[k]) / (v[k + 1] - v[k]) * (times[k + 1] - times[k])
    return(lat)


@pytest.fixture(scope='module')
def mms2d_run(_eager_mode, data_dir):
    """Run the 2D mMS sheet ONCE and return the recorded fields plus the measured
    and analytic conduction velocities, shared by the tests below. Depends on
    _eager_mode so eager execution is guaranteed active before the solve."""
    mesh = os.path.join(data_dir, _MESH_FILE)

    ionic = ModifiedMS2v(dt=_DT)
    cfg   = {'mesh_file_name': mesh, 'use_renumbering': True,
             'dt': _DT, 'dt_per_plot': 1, 'Tend': _TEND}
    model = MonodomainSolver(ionic, cfg)
    model.add_element_material_property('sigma_l', 'region', {1: _SIGMA, 2: _SIGMA, 3: _SIGMA, 4: _SIGMA})
    model.add_element_material_property('sigma_t', 'region', {1: _SIGMA, 2: _SIGMA, 3: _SIGMA, 4: _SIGMA})
    model.add_material_function('mass', _dfmass)
    model.add_material_function('stiffness', _sigma_iso)
    model.assemble_matrices()

    x  = model.domain().Pts()[:, 0].copy()
    Lx = float(x.max())
    U0 = np.where(x < _XFRAC * Lx, _VMAX, _VMIN).astype(np.float32)
    model.set_initial_condition(U0)
    model.solver().set_maxiter(model.domain().Pts().shape[0] // 2)
    model.finalize_for_run()

    nt    = model.nt()
    Vrec  = np.empty(shape=(nt + 1, x.shape[0]), dtype=np.float32)
    times = np.empty(shape=(nt + 1,), dtype=np.float64)
    Vrec[0]  = model.U().numpy().ravel()
    times[0] = 0.0
    ctime = 0.0
    for i in range(nt):
        ctime += _DT
        model.step(ctime)
        Vrec[i + 1]  = model.U().numpy().ravel()
        times[i + 1] = ctime

    lat  = _local_activation_times(Vrec, times, _VTH)
    mask = (x >= _INT_LO * Lx) & (x <= _INT_HI * Lx) & np.isfinite(lat)
    slope   = float(np.polyfit(x[mask], lat[mask], 1)[0])
    cv_meas = 1.0 / slope
    u_crit  = float(ionic.get_parameter('u_crit'))
    tau_in  = float(ionic.get_parameter('tau_in'))
    cv_an   = 0.5 * (1.0 - 2.0 * u_crit) * np.sqrt(2.0 * _SIGMA / tau_in)
    return({'x': x, 'Lx': Lx, 'lat': lat, 'Vrec': Vrec,
            'cv_meas': cv_meas, 'cv_analytic': cv_an})


@pytest.mark.nightly
@pytest.mark.gpu
def test_mms_2d_front_propagates(mms2d_run):
    """A single unidirectional planar front: the mean activation time per
    x-column increases monotonically along the sheet interior and the potential
    stays bounded."""
    x    = mms2d_run['x']
    Lx   = mms2d_run['Lx']
    lat  = mms2d_run['lat']
    Vrec = mms2d_run['Vrec']

    assert np.all(np.isfinite(Vrec)), 'non-finite potential in the simulation'

    # planar front: collapse to one activation time per x-column (many nodes
    # share each x on the 2D grid) and require it to increase across the interior
    xr      = np.round(x, 6)
    xu      = np.sort(np.unique(xr))
    xu_int  = xu[(xu >= _INT_LO * Lx) & (xu <= _INT_HI * Lx)]
    col_lat = np.array([np.nanmean(lat[xr == xv]) for xv in xu_int])
    assert np.all(np.isfinite(col_lat)), 'front did not cross the sheet interior'
    assert np.all(np.diff(col_lat) > 0.0), 'activation times not monotone (front not unidirectional)'
    # potential stays within the model's physical band [vmin, vmax]
    assert Vrec.min() > _VMIN - 1.0 and Vrec.max() < _VMAX + 1.0, \
        'potential left the physical band [{0}, {1}]'.format(_VMIN, _VMAX)


@pytest.mark.nightly
@pytest.mark.gpu
def test_mms_2d_conduction_velocity(mms2d_run):
    """Measured conduction velocity matches the analytic Nagumo front speed."""
    cv_meas = mms2d_run['cv_meas']
    cv_an   = mms2d_run['cv_analytic']
    relerr  = abs(cv_meas - cv_an) / cv_an
    assert relerr < _CV_TOL, \
        'CV {0:.4f} vs analytic {1:.4f} (rel err {2:.1%}) exceeds {3:.0%}'.format(
            cv_meas, cv_an, relerr, _CV_TOL)
