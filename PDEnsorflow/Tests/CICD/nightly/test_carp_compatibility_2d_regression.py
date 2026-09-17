#!/usr/bin/env python
"""
    Tier-2 nightly GPU regression for the parameter-file front end
    (gpuSolve.carp_compatibility.main) on the real data sheet.

    test_mms_2d_regression.py drives the same physics through the Python API.
    This one drives it through a `.par` file and the `PDEnsorflow` entry point,
    on the same 10 x 10 mm sheet (63001 nodes, 125000 triangles), and checks the
    planar front against the same analytic speed.

    It exists because the front end has whole layers the unit tests cannot
    reach on a small cable: the vectorised assembly path (which builds the
    diffusion tensor from the region maps WITHOUT calling the material
    function, and so has to apply beta itself), reverse Cuthill-McKee
    renumbering of a real mesh, and the IGB writer's chunked flush. Two defects
    of exactly that kind were found by running the real example rather than the
    unit suite, which is the argument for this test.

    The mesh is written in the external three-file format with coordinates in
    MICROMETRES, because that is the unit that format specifies and the unit
    the front end converts from. Tests/data stores the sheet in millimetres, so
    the fixture scales it on the way out, into a temporary directory pytest
    removes afterwards.

    Ahead of the front the fast gate is still closed, so the reaction reduces to
    the Nagumo bistable form and the front speed is analytic:

        CV = 0.5 (1 - 2 u_crit) sqrt(2 D / tau_in),   with D = sigma / beta

    The parameter file asks for g_il = g_it = 0.7 S/m with
    cellSurfVolRatio = 0.14, which the mapping turns into
    D = 1e5 * 0.7 / 0.14 = 5.0e5 um^2/ms, the same coefficient the API
    regression uses (0.5 mm^2/ms). u_crit and tau_in are left at the cell
    model's own defaults, so a parameter file and a script agree.

    Runs the traced solver kernels and the XLA-compiled cell model, as the
    front end does (see the module fixture). Marked nightly + gpu.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle

import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.carp_compatibility.main import main
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.IO.readers import IGBReader

_MESH_FILE = 'triangulated_square.pkl'
_MM_TO_UM  = 1.0e3
_G_IL      = 0.7                                  # S/m
_BETA      = 0.14                                 # um^-1
_SIGMA     = 1.0e5 * _G_IL / _BETA                # 5.0e5 um^2/ms, as the API regression
_DT_US     = 50.0                                 # microseconds
_SPACEDT   = 0.1                                  # ms between recorded frames
_TEND      = 10.0                                 # ms
_VMIN      = -80.0
_VMAX      = 20.0
_VTH       = -30.0                                # activation threshold, rising edge
_XFRAC     = 0.05                                 # paced strip: the first 5% of the sheet in x
_INT_LO    = 0.4                                  # sheet interior for the LAT fit: [0.4, 0.8] * Lx
_INT_HI    = 0.8
_CV_TOL    = 0.10                                 # relative tolerance on conduction velocity
_ELEMENT_CODES = {'Edges': 'Ln', 'Trias': 'Tr', 'Quads': 'Qd',
                  'Tetras': 'Tt', 'Hexas': 'Hx', 'Pyras': 'Py', 'Prisms': 'Pr'}


@pytest.fixture(scope='module', autouse=True)
def _graph_mode():
    """Run with traced kernels and the XLA-compiled cell model, as the solver
    and the front end do. Set in fixture *setup* (not at import time) so it
    holds whatever eager state another module left behind, then restore it."""
    prev = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(False)
    yield
    tf.config.run_functions_eagerly(bool(prev))


def _write_mesh_in_um(pkl_path: str, folder: str, basename: str) -> float:
    """Write the pickled sheet as .pts/.elem/.lon with coordinates in
    micrometres, and return the extent in x."""
    with open(pkl_path, 'rb') as fmesh:
        mesh = pickle.load(fmesh)
    points = np.asarray(mesh['Pts'], dtype=float) * _MM_TO_UM
    with open(os.path.join(folder, '{}.pts'.format(basename)), 'w') as fout:
        fout.write('{}\n'.format(points.shape[0]))
        fout.write('\n'.join('{:.6f} {:.6f} {:.6f}'.format(*row) for row in points))
        fout.write('\n')
    rows = []
    for elemtype, elements in mesh['Elems'].items():
        if elements is None or len(elements) == 0:
            continue
        code = _ELEMENT_CODES[elemtype]
        for elem in np.asarray(elements, dtype=int):
            rows.append('{} {} {}'.format(code, ' '.join(str(n) for n in elem[:-1]), elem[-1]))
    with open(os.path.join(folder, '{}.elem'.format(basename)), 'w') as fout:
        fout.write('{}\n'.format(len(rows)))
        fout.write('\n'.join(rows))
        fout.write('\n')
    with open(os.path.join(folder, '{}.lon'.format(basename)), 'w') as fout:
        fout.write('1\n')
        fout.write('\n'.join('{:.6f} {:.6f} {:.6f}'.format(*row)
                             for row in np.asarray(mesh['Fibres'], dtype=float)))
        fout.write('\n')
    return(float(points[:, 0].max()))


def _write_par(folder: str, basename: str, xstim: float):
    """The simulation, as a parameter file. Everything the front end must
    convert is expressed in the units that format uses: S/m, micrometres,
    microseconds."""
    with open(os.path.join(folder, 'sheet.par'), 'w') as fout:
        fout.write('meshname        = {}\n'
                   'simID           = OUT\n'
                   'tend            = {}\n'
                   'dt              = {}\n'
                   'spacedt         = {}\n'
                   'timedt          = 100.0\n'
                   'renumbering     = 1\n'
                   'bidm_eqv_mono   = 0\n'
                   'cg_maxit_parab  = 5000\n'
                   'num_imp_regions = 1\n'
                   'imp_region[0].im               = "mMS"\n'
                   'imp_region[0].cellSurfVolRatio = {}\n'
                   'num_gregions    = 1\n'
                   'gregion[0].g_il = {}\n'
                   'gregion[0].g_it = {}\n'
                   'num_stim = 1\n'
                   'stim[0].name           = "S1"\n'
                   'stim[0].crct.type      = 0\n'
                   'stim[0].pulse.strength = 60.0\n'
                   'stim[0].ptcl.start     = 0.0\n'
                   'stim[0].ptcl.duration  = 1.0\n'
                   'stim[0].ptcl.npls      = 1\n'
                   'stim[0].elec.p0[0]     = 0.0\n'
                   'stim[0].elec.p1[0]     = {}\n'.format(
                       basename, _TEND, _DT_US, _SPACEDT, _BETA, _G_IL, _G_IL, xstim))


def _local_activation_times(V: np.ndarray, times: np.ndarray, vth: float) -> np.ndarray:
    """LAT per node = interpolated time of the first rising crossing of vth
    (NaN where the node never crosses)."""
    lat = np.full(V.shape[1], np.nan)
    for jnode in range(V.shape[1]):
        trace = V[:, jnode]
        kcross = int(np.argmax((trace[:-1] < vth) & (trace[1:] >= vth)))
        if (trace[kcross] < vth) and (trace[kcross + 1] >= vth):
            lat[jnode] = times[kcross] + ((vth - trace[kcross])
                                          / (trace[kcross + 1] - trace[kcross])
                                          * (times[kcross + 1] - times[kcross]))
    return(lat)


@pytest.fixture(scope='module')
def sheet_run(_graph_mode, data_dir, tmp_path_factory) -> dict:
    """Drive the front end ONCE through a parameter file and return the fields
    plus the measured and analytic conduction velocities."""
    folder   = str(tmp_path_factory.mktemp('carp_nightly'))
    basename = 'square'
    Lx       = _write_mesh_in_um(os.path.join(data_dir, _MESH_FILE), folder, basename)
    _write_par(folder, basename, _XFRAC * Lx)

    cwd = os.getcwd()
    try:
        os.chdir(folder)
        status = main(['+F', 'sheet.par'])
    finally:
        os.chdir(cwd)

    reader = IGBReader()
    reader.read(os.path.join(folder, 'OUT', 'vm.igb'))
    header = reader.header()
    V      = np.array(reader.data()).reshape(header['t'], header['x'])
    times  = np.arange(header['t'], dtype=np.float64) * _SPACEDT

    xcoord = np.array([float(line.split()[0])
                       for line in open(os.path.join(folder, '{}.pts'.format(basename)))
                       .readlines()[1:]])

    lat    = _local_activation_times(V, times, _VTH)
    inside = ((xcoord >= _INT_LO * Lx) & (xcoord <= _INT_HI * Lx) & np.isfinite(lat))
    slope  = float(np.polyfit(xcoord[inside], lat[inside], 1)[0])

    reference = ModifiedMS2v(dt=_DT_US * 1.0e-3)
    u_crit    = float(reference.get_parameter('u_crit'))
    tau_in    = float(reference.get_parameter('tau_in'))
    return({'status': status, 'V': V, 'x': xcoord, 'Lx': Lx, 'lat': lat,
            'cv_meas': 1.0 / slope,
            'cv_analytic': 0.5 * (1.0 - 2.0 * u_crit) * np.sqrt(2.0 * _SIGMA / tau_in)})


@pytest.mark.nightly
@pytest.mark.gpu
def test_carp_front_end_front_propagates(sheet_run):
    """The run completes, the potential stays physical and the planar front is
    unidirectional: the mean activation time per x-column increases along the
    sheet (a per-node check does not apply, since many nodes share each x)."""
    assert sheet_run['status'] == 0
    V = sheet_run['V']
    assert np.all(np.isfinite(V))
    assert V.min() >= _VMIN - 1.0
    assert V.max() <= _VMAX + 6.0
    assert V.max() > 0.0

    xcoord, lat = sheet_run['x'], sheet_run['lat']
    columns = np.unique(np.round(xcoord, 3))
    means   = []
    for xval in columns:
        here = np.isclose(xcoord, xval) & np.isfinite(lat)
        if np.any(here):
            means.append(float(np.mean(lat[here])))
    means = np.array(means)
    assert means.size > columns.size // 2
    assert np.all(np.diff(means) >= -1.0e-9)


@pytest.mark.nightly
@pytest.mark.gpu
def test_carp_front_end_conduction_velocity(sheet_run):
    """The speed measured from a parameter-file run matches the analytic Nagumo
    value, so every conversion between the two vocabularies is right: S/m and
    beta to a diffusion coefficient, microseconds to milliseconds, and a
    micrometre mesh used as it stands."""
    measured = sheet_run['cv_meas']
    analytic = sheet_run['cv_analytic']
    error    = abs(measured - analytic) / analytic
    print('CV measured {:.3f} um/ms, analytic {:.3f} um/ms, error {:.2%}'.format(
        measured, analytic, error))
    assert error < _CV_TOL
