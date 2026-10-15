#!/usr/bin/env python
"""
    Tier-1 end-to-end test of the parameter-file front end
    (gpuSolve.carp_compatibility.main).

    One short run drives the whole chain on a small cable: the `.par` file is
    read, the defaults and the unit conversions are applied, the mesh is loaded
    from the three-file format, the materials and the stimulus are assigned,
    the time loop runs and the result is written as IGB.

    The mesh is written with `Ln` line elements, the type specifier an external
    `.elem` file carries, so the run also covers reading that spelling.

    The run is deliberately small (a 5 mm cable, 6 ms, 240 steps) and CPU-only.
    The quantitative conduction-velocity check lives with the example under
    Tests/FEM, where the mesh is fine enough for the front to be resolved; here
    the assertions are that the chain completes, the output header describes
    what was written, the potential stays physical, and the front travels.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.carp_compatibility.main import main
from gpuSolve.IO.readers import IGBReader

_NELEM  = 50
_DX     = 100.0                                   # micrometres
_NPT    = _NELEM + 1
_TEND   = 6.0                                     # ms
_DT_US  = 25.0                                    # microseconds
_SPACE  = 1.0                                     # ms between recorded frames
_VMIN   = -80.0
_VMAX   = 20.0
_LAT_THRESHOLD = -30.0                            # mid-upstroke of the band


def _write_cable(folder: str):
    """Write a uniform 1D cable as the three files of the external mesh format,
    with coordinates in micrometres and a single element tag."""
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * _DX))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} 1\n'.format(iel, iel + 1))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


def _write_par(folder: str):
    """A parameter file that leaves most keys at their defaults on purpose, so
    the defaults are exercised too."""
    with open(os.path.join(folder, 'cable.par'), 'w') as fout:
        fout.write('# a cable paced at one end\n'
                   'meshname      = cable\n'
                   'simID         = OUT\n'
                   'tend          = {}\n'
                   'dt            = {}          # microseconds\n'
                   'spacedt       = {}\n'
                   'timedt        = 100.0\n'
                   'bidm_eqv_mono = 0\n'
                   'num_imp_regions = 1\n'
                   'imp_region[0].im       = "mMS"\n'
                   'imp_region[0].im_param = "V_gate=0.1,a_crit=0.1"\n'
                   'imp_region[0].cellSurfVolRatio = 0.14\n'
                   'num_gregions  = 1\n'
                   'gregion[0].g_il = 0.174\n'
                   'gregion[0].g_it = 0.174\n'
                   'num_stim = 1\n'
                   'stim[0].name           = "S1"\n'
                   'stim[0].pulse.strength = 60.0\n'
                   'stim[0].ptcl.start     = 0.0\n'
                   'stim[0].ptcl.duration  = 2.0\n'
                   'stim[0].ptcl.npls      = 1\n'
                   'stim[0].elec.p0[0]     = 0.0\n'
                   'stim[0].elec.p1[0]     = 250.0\n'
                   # two detectors on Vm and one on Phie, which has to be
                   # dropped: there is no extracellular solve here
                   'num_LATs = 3\n'
                   'lats[0].all       = 0\n'
                   'lats[0].threshold = {}\n'
                   'lats[1].ID        = "every_act"\n'
                   'lats[1].all       = 1\n'
                   'lats[1].threshold = {}\n'
                   'lats[2].ID        = "phie_act"\n'
                   'lats[2].measurand = 1\n'.format(_TEND, _DT_US, _SPACE,
                                                     _LAT_THRESHOLD, _LAT_THRESHOLD))


@pytest.fixture(scope='module')
def cable_run(tmp_path_factory) -> dict:
    """Run the front end once; every test below reads the same result."""
    folder = str(tmp_path_factory.mktemp('carp_cli'))
    _write_cable(folder)
    _write_par(folder)
    cwd = os.getcwd()
    try:
        # simID is a relative directory, as it is in the format being matched
        os.chdir(folder)
        status = main(['+F', 'cable.par', '+Save', 'resolved.par'])
    finally:
        os.chdir(cwd)
    return({'folder': folder, 'status': status})


def test_the_run_completes_and_writes_its_output(cable_run):
    """Exit status 0, the IGB file is there and +Save wrote the resolved set."""
    assert cable_run['status'] == 0
    assert os.path.isfile(os.path.join(cable_run['folder'], 'OUT', 'vm.igb'))
    assert os.path.isfile(os.path.join(cable_run['folder'], 'resolved.par'))


def test_the_output_header_describes_what_was_written(cable_run):
    """nx is the node count and nt the number of frames actually recorded:
    one before the loop, then one at every multiple of dt_per_plot steps
    (t = 0, spacedt, ..., tend)."""
    reader = IGBReader()
    reader.read(os.path.join(cable_run['folder'], 'OUT', 'vm.igb'))
    header      = reader.header()
    nsteps      = int(round(_TEND / (_DT_US * 1.0e-3)))
    dt_per_plot = int(round(_SPACE / (_DT_US * 1.0e-3)))
    assert header['x'] == _NPT
    assert header['t'] == 1 + nsteps // dt_per_plot


def test_the_potential_stays_physical_and_the_front_travels(cable_run):
    """Nothing goes non-finite, the potential stays inside the model's band,
    and the activation times increase along the cable."""
    reader = IGBReader()
    reader.read(os.path.join(cable_run['folder'], 'OUT', 'vm.igb'))
    V = np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x'])
    assert np.all(np.isfinite(V))
    assert V.min() >= _VMIN - 1.0
    assert V.max() <= _VMAX + 6.0                      # the mMS upstroke overshoots slightly
    assert V.max() > 0.0                               # something actually depolarised

    # local activation time per node, on the frames that were recorded
    threshold = 0.5 * (_VMIN + _VMAX)
    lat       = np.full(V.shape[1], np.nan)
    for jnode in range(V.shape[1]):
        above = np.where(V[:, jnode] >= threshold)[0]
        if above.size > 0:
            lat[jnode] = above[0]
    activated = np.where(np.isfinite(lat))[0]
    assert activated.size > V.shape[1] // 3            # the front covered a good part of the cable
    assert np.all(np.diff(lat[activated]) >= 0.0)      # and it travelled in one direction


def test_the_lat_detectors_write_the_activation_times_they_were_asked_for(cable_run):
    """num_LATs / lats[] produce the reference's two file layouts, and only for
    the entries that can be measured: `all = 0` writes one first-activation
    time per node as init_acts_<ID>.dat (-1 where a node never activated),
    `all = 1` writes a node/time table as <ID>.dat, and the Phie entry writes
    nothing at all, because this front end has no extracellular potential.

    The activation times are checked against the recorded frames: the detector
    sees every step, so its time for a node must fall inside the frame interval
    in which that node crossed the threshold.
    """
    outdir = os.path.join(cable_run['folder'], 'OUT')
    first  = os.path.join(outdir, 'init_acts_vm_act.dat')      # the LAT_ID[0] default
    every  = os.path.join(outdir, 'every_act.dat')
    assert os.path.isfile(first)
    assert os.path.isfile(every)
    assert not os.path.isfile(os.path.join(outdir, 'phie_act.dat'))
    assert not os.path.isfile(os.path.join(outdir, 'init_acts_phie_act.dat'))

    lat = np.loadtxt(first)
    assert lat.size == _NPT                                    # one line per node
    activated = np.where(lat >= 0.0)[0]
    assert activated.size > 0
    assert np.all(np.diff(lat[activated]) > 0.0)               # the front travels one way
    assert np.all(lat[activated] <= _TEND)
    # the nodes that never activated are the ones beyond the front, and they
    # are marked, not omitted
    assert np.all(lat[activated.max() + 1:] == -1.0)

    # the table of every activation holds the same nodes: on this cable each
    # one crosses the threshold once
    table = np.loadtxt(every)
    assert np.array_equal(np.sort(table[:, 0].astype(int)), activated)
    assert np.allclose(np.sort(table[:, 1]), lat[activated])

    # against the frames: node j crossed between frame k-1 and frame k
    reader = IGBReader()
    reader.read(os.path.join(outdir, 'vm.igb'))
    V = np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x'])
    for jnode in activated:
        above = np.where(V[:, jnode] >= _LAT_THRESHOLD)[0]
        assert above.size > 0
        assert (above[0] - 1) * _SPACE <= lat[jnode] <= above[0] * _SPACE


def test_a_vertex_file_electrode_stimulates_exactly_those_nodes(tmp_path):
    """An elec.vtx_file names the stimulated nodes outright instead of
    describing a box. A short run (2 ms, the stimulus on throughout) checks that
    the named nodes depolarise and that a distant one does not: over 2 ms the
    diffusion length is about 700 um, so node 50 at 5000 um cannot be reached.
    """
    folder = str(tmp_path)
    _write_cable(folder)
    stimulated = [0, 1, 2, 3, 4]
    with open(os.path.join(folder, 'electrode.vtx'), 'w') as fout:
        fout.write('{}\nintra\n'.format(len(stimulated)))
        fout.write('\n'.join(str(node) for node in stimulated))
        fout.write('\n')
    with open(os.path.join(folder, 'vtx.par'), 'w') as fout:
        fout.write('meshname = cable\n'
                   'simID    = OUT_VTX\n'
                   'tend     = 2.0\n'
                   'dt       = {}\n'
                   'spacedt  = 1.0\n'
                   'timedt   = 100.0\n'
                   'bidm_eqv_mono = 0\n'
                   'imp_region[0].im       = "mMS"\n'
                   'imp_region[0].im_param = "V_gate=0.1,a_crit=0.1"\n'
                   'imp_region[0].cellSurfVolRatio = 0.14\n'
                   'gregion[0].g_il = 0.174\n'
                   'gregion[0].g_it = 0.174\n'
                   'num_stim = 1\n'
                   'stim[0].name           = "S1"\n'
                   'stim[0].pulse.strength = 60.0\n'
                   'stim[0].ptcl.start     = 0.0\n'
                   'stim[0].ptcl.duration  = 2.0\n'
                   'stim[0].ptcl.npls      = 1\n'
                   'stim[0].elec.vtx_file  = "electrode.vtx"\n'.format(_DT_US))
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        status = main(['+F', 'vtx.par'])
    finally:
        os.chdir(cwd)
    assert status == 0

    reader = IGBReader()
    reader.read(os.path.join(folder, 'OUT_VTX', 'vm.igb'))
    V = np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x'])
    final = V[-1, :]
    assert np.all(np.isfinite(final))
    for node in stimulated:
        assert final[node] > _VMIN + 20.0, 'node {} was not stimulated'.format(node)
    assert final[_NPT - 1] == pytest.approx(_VMIN, abs=1.0)


def test_a_vertex_file_naming_a_node_outside_the_mesh_is_rejected(tmp_path, capsys):
    """An out-of-range index would otherwise index the mask out of bounds or,
    worse, wrap round and stimulate the wrong end of the mesh."""
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'bad.vtx'), 'w') as fout:
        fout.write('2\nintra\n0\n999999\n')
    with open(os.path.join(folder, 'bad.par'), 'w') as fout:
        fout.write('meshname = cable\ntend = 1.0\ndt = 100\nspacedt = 1.0\n'
                   'imp_region[0].im = "mMS"\n'
                   'num_stim = 1\n'
                   'stim[0].pulse.strength = 60.0\n'
                   'stim[0].elec.vtx_file  = "bad.vtx"\n')
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        status = main(['+F', 'bad.par'])
    finally:
        os.chdir(cwd)
    assert status == 1
    assert '999999' in capsys.readouterr().err


def test_notes_collected_during_setup_reach_the_banner(tmp_path, capsys):
    """Some notes are only produced while the simulation is being built: a
    region tag that is not in the mesh, an electrode given both ways. They are
    printed after the setup for that reason, and a regression that moved them
    back before it would drop exactly the diagnostics worth reading."""
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'note.par'), 'w') as fout:
        fout.write('meshname = cable\nsimID = OUT_NOTE\ntend = 1.0\ndt = {}\n'
                   'spacedt = 1.0\ntimedt = 100.0\n'
                   'imp_region[0].im = "mMS"\n'
                   'gregion[0].ID = 1,77\n'.format(_DT_US))
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        status = main(['+F', 'note.par'])
    finally:
        os.chdir(cwd)
    assert status == 0
    printed = capsys.readouterr().out
    assert 'region tag 77' in printed


def test_a_pure_diffusion_run_stays_finite(tmp_path):
    """No cell model makes it a heat-equation run, which starts from U = 0. With
    no stimulus the first linear system is exactly zero, and the solve used to
    turn it into NaN (0/0 in the CG step length), leaving every later frame
    non-finite. The output must stay finite, and stay at 0 with nothing to
    drive it."""
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'heat.par'), 'w') as fout:
        fout.write('meshname = cable\nsimID = OUT_HEAT\ntend = 1.0\ndt = {}\n'
                   'spacedt = 0.25\ntimedt = 100.0\n'.format(_DT_US))
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        status = main(['+F', 'heat.par'])
    finally:
        os.chdir(cwd)
    assert status == 0
    reader = IGBReader()
    reader.read(os.path.join(folder, 'OUT_HEAT', 'vm.igb'))
    V = np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x'])
    assert V.shape[0] > 1
    assert np.all(np.isfinite(V))
    np.testing.assert_array_equal(V, 0.0)


def test_a_missing_mesh_is_reported_by_name(tmp_path, capsys):
    """meshname has a default, so a mesh that is not there is the commonest
    first mistake. It must name the files it looked for rather than raise a
    bare FileNotFoundError from inside the reader."""
    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        status = main(['-meshname', 'nosuch', '-tend', '1.0', '-dt', '100'])
    finally:
        os.chdir(cwd)
    assert status == 1
    assert 'nosuch.pts' in capsys.readouterr().err


def test_help_and_a_bad_key_are_reported_without_running(capsys):
    """+Help prints the usage and stops; an unknown key exits non-zero."""
    assert main(['+Help']) == 0
    assert 'PDEnsorflow' in capsys.readouterr().out
    assert main(['-no_such_parameter', '1']) == 1


@pytest.mark.parametrize('meshname', ['cable', 'cable.pkl'])
def test_the_mesh_export_is_named_without_the_binary_extension(tmp_path, meshname):
    """gridout_i exports the mesh as the three text files under simID. A mesh
    read from a binary file (meshname = cable.pkl) must be exported as
    cable.pts/.elem/.lon, the same names as from the text files, not as
    cable.pkl.pts."""
    from gpuSolve.entities.triangulation import Triangulation
    folder = str(tmp_path)
    _write_cable(folder)
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        domain = Triangulation()
        domain.readMesh('cable')
        domain.saveMesh('cable.pkl')
        with open('grid.par', 'w') as fout:
            fout.write('meshname = {}\nsimID = OUT_GRID\ntend = 0.5\ndt = {}\n'
                       'spacedt = 0.25\ntimedt = 100.0\ngridout_i = 1\n'.format(meshname, _DT_US))
        status = main(['+F', 'grid.par'])
    finally:
        os.chdir(cwd)
    assert status == 0
    written = sorted(os.listdir(os.path.join(folder, 'OUT_GRID')))
    for suffix in ('.pts', '.elem', '.lon'):
        assert 'cable{}'.format(suffix) in written, written
    assert not any(name.startswith('cable.pkl') for name in written), written


def _run_cable_for(folder: str, tend: float, simid: str) -> tuple:
    """Run the paced cable to tend and return (frames, final state in mesh order)."""
    from gpuSolve.carp_compatibility.optionreader import OptionReader
    from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
    from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        mapper = ParameterMapper()
        mapper.resolve(OptionReader().read(['+F', 'cable.par', '-tend', str(tend), '-simID', simid]))
        runner = SimulationRunner({'verbose': False})
        runner.set_mapper(mapper)
        runner.build()
        runner.run()
        # U() returns the mesh's node order whether or not the run renumbered
        final = np.reshape(runner.model().U().numpy(), (-1,))
    finally:
        os.chdir(cwd)
    reader = IGBReader()
    reader.read(os.path.join(folder, simid, 'vm.igb'))
    frames = np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x'])
    return((frames, final))


def test_frames_fall_on_multiples_of_spacedt(tmp_path):
    """Frame k holds the solution at t = k*spacedt, as the reference writes
    it: the last frame is the state at tend, and frame 1 of a longer run is
    the last frame of a run that stops at spacedt. Recording on the step
    INDEX instead put every frame one step late and dropped the one at tend."""
    folder = str(tmp_path)
    _write_cable(folder)
    _write_par(folder)
    short, _ = _run_cable_for(folder, _SPACE, 'OUT_SHORT')
    full, final = _run_cable_for(folder, 3.0 * _SPACE, 'OUT_FULL')
    assert short.shape[0] == 2 and full.shape[0] == 4
    np.testing.assert_array_equal(full[-1, :], final.astype(full.dtype))
    np.testing.assert_array_equal(full[1, :], short[-1, :])


@pytest.mark.parametrize('tend,dt,nsteps', [(100.0, 0.02, 5000), (1.0, 0.025, 40),
                                            (250.0, 0.1, 2500), (1.01, 0.02, 50)])
def test_the_run_reaches_tend(tend, dt, nsteps):
    """The step count reaches tend when dt divides it, although dt is not exact
    in binary (100 // 0.02 is 4999), and never steps past tend when it does
    not divide it (1.01 ms at 0.02 ms is 50 steps, to 1.00 ms)."""
    from gpuSolve.physics import HeatSolver
    assert HeatSolver({'Tend': tend, 'dt': dt}).nt() == nsteps


def _write_two_tag_cable(folder: str):
    """The same cable, with the left half tagged 1 and the right half tagged 2."""
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * _DX))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(iel, iel + 1, 1 if iel < _NELEM // 2 else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


def test_a_node_takes_the_highest_claimed_region(tmp_path):
    """imp_region[1] claims tag 1 only, so tag 2 is claimed by nobody.

    The nodal assignment must then be the reference's: a node is governed by
    the highest tag an imp_region claims among the elements it belongs to, and
    a node no claimed tag reaches falls to imp_region[0]. Node 25 belongs to
    one element of each tag, so it goes to region 1, not to the default.

    The assertion is on the per-node value of a cell parameter, which is where
    the assignment is actually felt, and it also covers the lookup of the
    nodes that carry the no-region sentinel.
    """
    from gpuSolve.carp_compatibility.optionreader import OptionReader
    from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
    from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner
    folder = str(tmp_path)
    _write_two_tag_cable(folder)
    with open(os.path.join(folder, 'two.par'), 'w') as fout:
        fout.write('meshname = cable\nsimID = OUT_TWO\ntend = 0.5\n'
                   'dt = {}\nspacedt = 0.25\ntimedt = 100.0\n'
                   'num_imp_regions = 2\n'
                   'imp_region[0].im = "mMS"\n'
                   'imp_region[0].im_param = "tau_in=0.3"\n'
                   'imp_region[1].im = "mMS"\n'
                   'imp_region[1].ID = 1\n'
                   'imp_region[1].im_param = "tau_in=0.9"\n'
                   'num_gregions = 1\ngregion[0].g_il = 0.174\n'
                   'gregion[0].g_it = 0.174\n'.format(_DT_US))
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        mapper = ParameterMapper()
        mapper.resolve(OptionReader().read(['+F', 'two.par']))
        runner = SimulationRunner({'verbose': False})
        runner.set_mapper(mapper)
        runner.build()
        regions = np.reshape(np.asarray(runner.model().domain().point_region_ids()), (-1,))
        tau_in  = np.reshape(np.asarray(runner.model().ionic_model().get_parameter('tau_in')), (-1,))
        # the cell parameters live in the solver's node order once the run has
        # been finalized, the region IDs in the mesh's: bring the first back
        renumbering = runner.model().renumbering()
        if renumbering is not None:
            tau_in = np.take(tau_in, renumbering['iperm'])
    finally:
        os.chdir(cwd)
    # nodes 0..25 meet an element of tag 1, which is the only claimed tag
    assert np.array_equal(regions[:1 + _NELEM // 2], np.ones(1 + _NELEM // 2))
    # nodes 26.. meet tag 2 only, which no region claims
    assert np.array_equal(regions[1 + _NELEM // 2:], np.zeros(_NPT - 1 - _NELEM // 2) - 1.0)
    assert np.allclose(tau_in[:1 + _NELEM // 2], 0.9)
    assert np.allclose(tau_in[1 + _NELEM // 2:], 0.3)
