#!/usr/bin/env python
"""
    Tier-1 tests of single-cell prepacing (gpuSolve.physics.Prepacer), of the
    activation-time reader it is guided by, and of the parameter-file keys that
    drive it.

    What is checked, and why each check is the one that matters:

      * the save-time arithmetic. It is the whole idea of prepacing: a node that
        activates early must take a *later* state of the paced cell than one
        that activates late, and an unactivated node must take the earliest
        state of all. A sign error here would still produce a plausible looking
        conditioned state, so it is tested on numbers rather than through a run;
      * the state a node receives equals a single cell integrated independently
        for that node's own number of steps, with the same protocol. This is the
        test that says the distribution puts the right state on the right node;
      * strategies A (one cell per region) and B (one cell per node) agree
        exactly on a mesh whose parameters are uniform. They are the same ODEs,
        so anything other than bitwise equality means one of the two paths does
        not integrate what it claims to;
      * the automatic choice between them follows the way the cell parameters
        were registered, not the number of regions;
      * the front end reads the keys, and says so when prepacing is asked for
        but cannot run.

    The runs are deliberately tiny (a 16-node cable, 2 beats of 20 ms at
    dt = 0.05 ms) and CPU-only: prepacing costs one ionic step per cell per step
    and nothing else, so a small case exercises exactly the same code path as a
    large one.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import Prepacer
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import no_mass_property
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.IO.readers import LatReader
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper


_NELEM   = 15
_NPT     = _NELEM + 1
_DX      = 100.0                                  # micrometres
_DT      = 0.05                                   # ms
_BCL     = 20.0                                   # ms
_BEATS   = 2
_STIMDUR = 1.0                                    # ms
_STIMSTR = 60.0                                   # uA/uF
_TEND    = 5.0                                    # ms


def _write_cable(folder: str):
    """A uniform 1D cable in the three-file external mesh format."""
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


def _build_solver(folder: str) -> MonodomainSolver:
    """A monodomain solver on the cable, assembled and started from rest but
    not finalized: prepacing runs at exactly this point of the setup."""
    config = {'mesh_file_name': os.path.join(folder, 'cable'),
              'dt': _DT, 'Tend': _TEND, 'dt_per_plot': 2}
    solver = MonodomainSolver(ModifiedMS2v(dt=_DT), config)
    solver.add_element_material_property('sigma_l', 'uniform', 0.174)
    solver.add_element_material_property('sigma_t', 'uniform', 0.174)
    solver.add_element_material_property('beta', 'uniform', 1400.0)
    solver.add_material_function('mass', no_mass_property)
    solver.add_material_function('stiffness', conductivity_tensor)
    solver.assemble_matrices()
    solver.set_initial_condition()
    return(solver)


def _prepacer(**overrides) -> Prepacer:
    config = {'bcl': _BCL, 'beats': _BEATS, 'stimdur': _STIMDUR, 'stimstr': _STIMSTR,
              'dt': _DT, 'tend': _TEND, 'verbose': False}
    config.update(overrides)
    prepacer = Prepacer(config)
    prepacer.set_model_factory(lambda: ModifiedMS2v(dt=_DT))
    return(prepacer)


@pytest.fixture(scope='module')
def cable(tmp_path_factory) -> str:
    folder = str(tmp_path_factory.mktemp('prepacing'))
    _write_cable(folder)
    return(folder)


# ---- the save-time arithmetic ------------------------------------------------
def test_an_early_node_takes_a_later_state_than_a_late_one(cable):
    """save = bcl*beats - (LAT - offset): the earlier a node activates, the
    further into the prepacing train the state it is handed comes from."""
    solver   = _build_solver(cable)
    lats     = np.linspace(0.0, 5.0, _NPT)
    prepacer = _prepacer()
    prepacer.set_lats(lats)
    prepacer.prepace(solver)
    save = prepacer.save_times()
    assert np.all(np.diff(save) < 0.0)
    # the earliest node is handed the cell at the very end of the train
    assert save[0] == pytest.approx(_BCL * _BEATS)
    # and every other node exactly its own activation time earlier
    assert save == pytest.approx(_BCL * _BEATS - lats)


def test_the_offset_is_a_whole_number_of_cycles(cable):
    """The train is kept in phase with the activation sequence: shifting every
    LAT by one cycle length must leave the save times unchanged."""
    solver   = _build_solver(cable)
    lats     = np.linspace(0.0, 5.0, _NPT)
    first    = _prepacer()
    first.set_lats(lats)
    first.prepace(solver)
    second   = _prepacer()
    second.set_lats(lats + _BCL)
    second.prepace(_build_solver(cable))
    assert second.save_times() == pytest.approx(first.save_times())


def test_a_node_that_never_activates_takes_the_least_prepaced_state(cable):
    """A negative LAT means the wavefront never arrives; the reference pushes
    such a node past the end of the run, so it takes the smallest save time."""
    solver   = _build_solver(cable)
    lats     = np.linspace(0.0, 5.0, _NPT)
    lats[3]  = -1.0
    prepacer = _prepacer()
    prepacer.set_lats(lats)
    prepacer.prepace(solver)
    save = prepacer.save_times()
    assert save[3] == np.min(save)
    assert save[3] < np.min(np.delete(save, 3))


def test_activation_times_that_are_all_negative_are_refused(cable):
    """Nothing activates anywhere: the file does not describe this run, and
    prepacing every node from the same meaningless offset would hide that."""
    prepacer = _prepacer()
    prepacer.set_lats(np.full(_NPT, -1.0))
    with pytest.raises(ValueError, match='no node ever activates'):
        prepacer.prepace(_build_solver(cable))


# ---- what each node actually receives ---------------------------------------
def test_each_node_gets_the_paced_cell_at_its_own_save_time(cable):
    """The reference implementation for this test is a single cell integrated
    here, step by step, with the same protocol: the state node i receives must
    be the one that cell holds after round(save[i]/dt) steps."""
    solver   = _build_solver(cable)
    resting  = float(np.reshape(solver.checkpoint()['Vm'], (-1,))[0])
    lats     = np.linspace(0.0, 5.0, _NPT)
    prepacer = _prepacer()
    prepacer.set_lats(lats)
    prepacer.prepace(solver)

    snap     = np.rint(np.maximum(prepacer.save_times(), 0.0) / _DT).astype(np.int64)
    wanted   = set(int(step) for step in np.unique(snap))
    cell     = ModifiedMS2v(dt=_DT)
    U        = tf.Variable(np.full((1, 1), resting, dtype=np.float32))
    cell.initialize_state_variables(U)
    names    = tuple(cell.state_variable_names())
    recorded : dict = {}
    for step in range(int(snap.max()) + 1):
        if step in wanted:
            recorded[step] = (float(U.numpy()[0, 0]),
                              {name: float(np.reshape(cell.state_variable(name).numpy(),
                                                      (-1,))[0]) for name in names})
        time = step * _DT
        if np.mod(time, _BCL) < _STIMDUR and time < _BCL * _BEATS - 1.0:
            U.assign_add(tf.fill(tf.shape(U), tf.constant(_STIMSTR * _DT, dtype=U.dtype)))
        U.assign_add(_DT * cell.differentiate(U))

    checkpoint = solver.checkpoint()
    Vm         = np.reshape(checkpoint['Vm'], (-1,))
    for node in range(_NPT):
        expected_Vm, expected_states = recorded[int(snap[node])]
        assert Vm[node] == pytest.approx(expected_Vm, abs=1.0e-6)
        for name in names:
            assert checkpoint['state_variables'][name][node] == pytest.approx(
                expected_states[name], abs=1.0e-6)


def test_prepacing_moves_the_state_away_from_rest(cable):
    """A guard against a prepacing that silently does nothing: the conditioned
    state must differ from the resting one it started at."""
    solver  = _build_solver(cable)
    resting = np.reshape(solver.checkpoint()['Vm'], (-1,)).copy()
    prepacer = _prepacer()
    prepacer.set_lats(np.linspace(0.0, 5.0, _NPT))
    prepacer.prepace(solver)
    assert np.max(np.abs(np.reshape(solver.checkpoint()['Vm'], (-1,)) - resting)) > 1.0e-3


# ---- the two strategies ------------------------------------------------------
def test_one_cell_per_region_and_one_per_node_agree(cable):
    """A and B integrate the same ODEs; on a mesh whose cell parameters are
    uniform they must agree to the bit, or one of the two is not doing what it
    says."""
    lats = np.linspace(0.0, 5.0, _NPT)
    results : list = []
    for group in (True, False):
        solver   = _build_solver(cable)
        prepacer = _prepacer(group_by_region=group)
        prepacer.set_lats(lats)
        prepacer.prepace(solver)
        results.append((solver.checkpoint(), prepacer))
    (grouped, by_region), (nodal, by_node) = results
    assert by_region.ncells() == 1
    assert by_node.ncells() == _NPT
    assert not by_region.per_node()
    assert by_node.per_node()
    assert np.array_equal(np.asarray(grouped['Vm']), np.asarray(nodal['Vm']))
    for name, values in grouped['state_variables'].items():
        assert np.array_equal(np.asarray(values),
                              np.asarray(nodal['state_variables'][name]))


def test_the_default_paces_one_cell_per_region(cable):
    """Nothing registers a per-node cell parameter in this run, so the
    automatic choice must be A, whatever the mesh looks like."""
    solver = _build_solver(cable)
    assert not solver.has_nodal_cell_parameters()
    prepacer = _prepacer()
    prepacer.set_lats(np.linspace(0.0, 5.0, _NPT))
    prepacer.prepace(solver)
    assert not prepacer.per_node()


def test_a_per_node_cell_parameter_selects_one_cell_per_node(cable):
    """A parameter registered as 'nodal' may vary between two nodes of the same
    region, so a representative cell no longer stands for its neighbours and
    the automatic choice must switch to B."""
    config = {'mesh_file_name': os.path.join(cable, 'cable'),
              'dt': _DT, 'Tend': _TEND, 'dt_per_plot': 2}
    solver = MonodomainSolver(ModifiedMS2v(dt=_DT), config)
    solver.add_element_material_property('sigma_l', 'uniform', 0.174)
    solver.add_element_material_property('sigma_t', 'uniform', 0.174)
    solver.add_element_material_property('beta', 'uniform', 1400.0)
    solver.add_material_function('mass', no_mass_property)
    solver.add_material_function('stiffness', conductivity_tensor)
    solver.assemble_matrices()
    # tau_in varies smoothly from node to node, which no region map can express
    solver.add_nodal_material_property('tau_in', 'nodal',
                                       np.linspace(0.2, 0.4, _NPT).tolist())
    solver.assign_nodal_properties()
    solver.set_initial_condition()
    assert solver.has_nodal_cell_parameters()
    assert 'tau_in' in solver.nodal_cell_parameters()
    prepacer = _prepacer()
    prepacer.set_lats(np.linspace(0.0, 5.0, _NPT))
    prepacer.prepace(solver)
    assert prepacer.per_node()
    assert prepacer.ncells() == _NPT
    # the per-node parameter reached the paced cells: equal save times would
    # otherwise give equal states, and here the two ends differ
    states = solver.checkpoint()['state_variables']
    name   = next(iter(states))
    assert states[name][0] != pytest.approx(states[name][-1], abs=1.0e-9)


def test_prepacing_after_finalize_is_refused(cable):
    """The prepacer works in the user's node order; after finalize_for_run()
    the solver is in its own, and writing one into the other would scramble the
    state silently."""
    solver = _build_solver(cable)
    solver.finalize_for_run()
    prepacer = _prepacer()
    prepacer.set_lats(np.linspace(0.0, 5.0, _NPT))
    with pytest.raises(ValueError, match='finalize_for_run'):
        prepacer.prepace(solver)


def test_a_disabled_prepacer_reports_it(cable):
    """The reference's switch is the cycle length."""
    assert not _prepacer(bcl=-1.0).enabled()
    assert not _prepacer(beats=0).enabled()
    assert _prepacer().enabled()


# ---- the activation-time reader ---------------------------------------------
def test_the_reader_takes_the_layout_the_detector_writes(tmp_path):
    """One value per line, as LatDetector.write() emits with all = 0."""
    path = str(tmp_path / 'init_acts_lat.dat')
    with open(path, 'w') as fout:
        fout.write('0.000000\n1.500000\n-1.000000\n3.250000\n')
    times = LatReader().read(path, 4)
    assert times == pytest.approx([0.0, 1.5, -1.0, 3.25])


def test_the_reader_refuses_a_file_written_for_another_mesh(tmp_path):
    """A short file is almost always the wrong file, and the nodes it does not
    reach would be prepaced from an uninitialised activation time."""
    path = str(tmp_path / 'short.dat')
    with open(path, 'w') as fout:
        fout.write('0.0\n1.0\n')
    with pytest.raises(ValueError, match='but the mesh has'):
        LatReader().read(path, 8)


# ---- the parameter-file keys -------------------------------------------------
def _mapper(argv: list) -> ParameterMapper:
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-meshname', 'cable', '-tend', '5.0',
                                        '-dt', '25.0'] + argv))
    return(mapper)


def test_the_front_end_reads_the_prepacing_keys():
    mapper   = _mapper(['-prepacing_lats', 'acts.dat',
                        '-prepacing_beats', '20',
                        '-prepacing_bcl', '500.0',
                        '-prepacing_stimdur', '2.0',
                        '-prepacing_stimstr', '40.0'])
    settings = mapper.prepacing_settings()
    assert settings['lats_file'] == 'acts.dat'
    assert settings['beats'] == 20
    assert settings['bcl'] == pytest.approx(500.0)
    assert settings['stimdur'] == pytest.approx(2.0)
    assert settings['stimstr'] == pytest.approx(40.0)
    # dt is converted from the microseconds the format carries
    assert settings['dt'] == pytest.approx(0.025)


def test_the_defaults_leave_prepacing_off():
    settings = _mapper([]).prepacing_settings()
    assert settings['bcl'] == pytest.approx(-1.0)
    assert settings['beats'] == 0
    assert settings['stimdur'] == pytest.approx(1.0)
    assert settings['stimstr'] == pytest.approx(60.0)
    assert not Prepacer({key: value for key, value in settings.items()
                         if key != 'lats_file'}).enabled()


def test_prepacing_asked_for_without_activation_times_is_noted():
    """Switched on, but with nothing to say where each cell sits: the run would
    otherwise start from rest without a word."""
    mapper = _mapper(['-prepacing_bcl', '500.0', '-prepacing_beats', '20'])
    assert any('prepacing_lats' in note for note in mapper.notes())


def test_prepacing_beats_without_a_cycle_length_is_noted():
    """The cycle length is the switch, so beats alone does nothing."""
    mapper = _mapper(['-prepacing_beats', '20'])
    assert any('prepacing_beats' in note and 'prepacing_bcl' in note
               for note in mapper.notes())
