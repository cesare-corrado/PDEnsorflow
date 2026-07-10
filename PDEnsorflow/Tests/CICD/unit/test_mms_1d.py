#!/usr/bin/env python
"""
    Tier-1 unit test for the 1D monodomain (modified Mitchell-Schaeffer) solver
    (gpuSolve.physics.MonodomainSolver driving ModifiedMS2v) on a uniform cable.

    A depolarised block at one end (V = +20 mV over the resting V = -80 mV -- the
    model's dimensional [vmin, vmax] = [-80, 20] range) launches a travelling
    activation front. Ahead of the front the fast gate H ~= 1, so the modified
    Mitchell-Schaeffer reaction reduces to the Nagumo bistable form
    f(u) = (1/tau_in) u (u - u_crit)(1 - u), whose front speed is analytic:

        CV = 0.5 (1 - 2 u_crit) sqrt(2 D / tau_in),   with D = sigma

    (the solver's equation (M + dt K) U^{n+1} = M (U^n + dt (dU + I0)) carries no
    Cm / surface-to-volume factor, so the diffusion coefficient is exactly the
    stiffness conductivity sigma).

    The simulation runs ONCE (module-scoped fixture); two tests share that run:
      * propagation (qualitative): a single unidirectional front -- local
        activation times increase monotonically along the cable interior, the
        potential stays in the physical band, and nothing goes non-finite.
      * conduction velocity (quantitative): CV measured from a linear fit of LAT
        vs x over the cable interior matches the analytic value within 10%
        (~3.6% at these parameters; the margin absorbs the linear-FEM /
        forward-Euler discretisation bias).

    CPU-only; ConjGrad requires eager execution (see the module fixture).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle
import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.physics import MonodomainSolver

# Cable / solver parameters, tuned so the front is well resolved and the run is
# fast; the measured CV error is ~3.6% at these values.
_SIGMA  = 0.1                                     # isotropic conductivity == diffusion coefficient D
_NELEM  = 150
_LENGTH = 2.0
_DT     = 0.05
_TEND   = 7.0
_VMIN   = -80.0                                   # resting potential (dimensional mV)
_VMAX   = 20.0                                    # plateau potential (dimensional mV)
_VTH    = -30.0                                   # activation threshold (mid-upstroke), rising edge
_XSTIM  = 0.05 * _LENGTH                          # depolarised block over the first 5% of the cable
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


def _write_1d_mesh(fname: str, nelem: int, length: float):
    """Write a uniform 1D edge mesh over [0, length] as a Triangulation .pkl."""
    npt         = nelem + 1
    Pts         = np.zeros(shape=(npt, 3), dtype=np.float64)
    Pts[:, 0]   = np.linspace(0.0, length, npt)
    Edges       = np.zeros(shape=(nelem, 3), dtype=np.int32)
    Edges[:, 0] = np.arange(0, nelem)
    Edges[:, 1] = np.arange(1, nelem + 1)
    Edges[:, 2] = 1
    with open(fname, 'wb') as fout:
        pickle.dump({'Pts': Pts, 'Elems': {'Edges': Edges}, 'Fibres': None},
                    fout, protocol=pickle.HIGHEST_PROTOCOL)


def _sigma_iso(elemtype: str, iElem: int, domain, matprop) -> np.ndarray:
    """Isotropic diffusion tensor sigma*I (the 1D cable carries no fibres)."""
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
def mms_run(_eager_mode, tmp_path_factory):
    """Run the 1D mMS cable ONCE and return the recorded fields plus the measured
    and analytic conduction velocities, shared by the tests below. Depends on
    _eager_mode so eager execution is guaranteed active before the solve."""
    mesh = str(tmp_path_factory.mktemp('mms') / 'cable.pkl')
    _write_1d_mesh(mesh, _NELEM, _LENGTH)

    ionic = ModifiedMS2v(dt=_DT)
    cfg   = {'mesh_file_name': mesh, 'use_renumbering': False,
             'dt': _DT, 'dt_per_plot': 1, 'Tend': _TEND}
    model = MonodomainSolver(ionic, cfg)
    model.add_element_material_property('sigma_l', 'region', {1: _SIGMA})
    model.add_element_material_property('sigma_t', 'region', {1: _SIGMA})
    model.add_material_function('mass', _dfmass)
    model.add_material_function('stiffness', _sigma_iso)
    model.assemble_matrices()

    x  = model.domain().Pts()[:, 0].copy()
    U0 = np.where(x < _XSTIM, _VMAX, _VMIN).astype(np.float32)
    model.set_initial_condition(U0)
    model.solver().set_maxiter(_NELEM)
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
    mask = (x >= 0.4 * _LENGTH) & (x <= 0.8 * _LENGTH) & np.isfinite(lat)
    slope   = float(np.polyfit(x[mask], lat[mask], 1)[0])
    cv_meas = 1.0 / slope
    u_crit  = float(ionic.get_parameter('u_crit'))
    tau_in  = float(ionic.get_parameter('tau_in'))
    cv_an   = 0.5 * (1.0 - 2.0 * u_crit) * np.sqrt(2.0 * _SIGMA / tau_in)
    return({'x': x, 'lat': lat, 'Vrec': Vrec, 'cv_meas': cv_meas, 'cv_analytic': cv_an})


def test_mms_1d_front_propagates(mms_run):
    """A single unidirectional travelling front: activation times increase
    monotonically along the cable interior and the potential stays bounded."""
    x    = mms_run['x']
    lat  = mms_run['lat']
    Vrec = mms_run['Vrec']

    assert np.all(np.isfinite(Vrec)), 'non-finite potential in the simulation'

    interior = (x >= 0.4 * _LENGTH) & (x <= 0.8 * _LENGTH)
    lat_int  = lat[interior]
    assert np.all(np.isfinite(lat_int)), 'front did not cross the cable interior'
    # unidirectional front: LAT strictly increases with x through the interior
    assert np.all(np.diff(lat_int) > 0.0), 'activation times not monotone (front not unidirectional)'
    # potential stays within the model's physical band [vmin, vmax]
    assert Vrec.min() > _VMIN - 1.0 and Vrec.max() < _VMAX + 1.0, \
        'potential left the physical band [{0}, {1}]'.format(_VMIN, _VMAX)


def test_mms_1d_conduction_velocity(mms_run):
    """Measured conduction velocity matches the analytic Nagumo front speed."""
    cv_meas = mms_run['cv_meas']
    cv_an   = mms_run['cv_analytic']
    relerr  = abs(cv_meas - cv_an) / cv_an
    assert relerr < _CV_TOL, \
        'CV {0:.4f} vs analytic {1:.4f} (rel err {2:.1%}) exceeds {3:.0%}'.format(
            cv_meas, cv_an, relerr, _CV_TOL)
