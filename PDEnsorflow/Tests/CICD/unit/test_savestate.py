#!/usr/bin/env python
"""
    Tier-1 unit tests for saving and restoring the simulation state:
    StateWriter / StateReader, the state-variable contract of the cell models,
    HeatSolver / MonodomainSolver.checkpoint() and restore_checkpoint(), and the
    savestate keys of the parameter-file front end.

    The central check is that a run saved half way and resumed matches the
    uninterrupted run. It cannot match bit for bit: the warm-start history of the
    CG solver (U^{n-1}) is not part of a checkpoint, so the first step after the
    restart starts CG from a different initial guess. The two runs therefore
    agree to within the CG tolerance, not to round-off. The restart is also run
    with renumbering switched on while the saving run had it off, which checks
    that a checkpoint is stored in the user's node order.

    The second check is that every variable a cell model advances in time is
    declared by it. A variable left out would restart from its initial value,
    and would not be renumbered with the rest of the state.

    CPU-only and small (a 51-node cable, 6 ms).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import pickle
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.carp_compatibility.main import main
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner
from gpuSolve.IO.readers import IGBReader
from gpuSolve.IO.readers import StateReader
from gpuSolve.IO.writers import StateWriter
from gpuSolve.ionic.ms2v import MitchellSchaeffer2v
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.ionic.fenton4v import Fenton4v
from gpuSolve.ionic.courtemanche_ramirez_nattel import CourtemancheRamirezNattel
from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov

ALL_MODELS = [MitchellSchaeffer2v, ModifiedMS2v, Fenton4v,
              CourtemancheRamirezNattel, TenTusscherPanfilov]

_NELEM  = 50
_DX     = 100.0                                   # micrometres
_NPT    = _NELEM + 1
_TEND   = 6.0                                     # ms
_DT_US  = 25.0                                    # microseconds
_DT_MS  = _DT_US * 1.0e-3
_SPACE  = 1.0                                     # ms between recorded frames
_TSAV   = 3.0                                     # ms, half way
# largest |dV| accepted between the restarted and the uninterrupted run, in mV.
# The measured difference is set out in the report of the test below; the bound
# leaves an order of magnitude of margin over it.
_RESTART_TOL = 1.0e-2


def _ids(models):
    return([m.__name__ for m in models])


def _write_cable(folder: str, shuffle: bool = False):
    """Write a uniform 1D cable as .pts/.elem/.lon. With shuffle the node
    numbers are scrambled along the cable, so the renumbering is far from the
    identity and a mix-up between the two node orders cannot go unnoticed."""
    order = np.arange(_NPT)
    if shuffle:
        order = np.random.default_rng(7).permutation(_NPT)
    xcoord = np.zeros(_NPT)
    xcoord[order] = np.arange(_NPT) * _DX
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(xcoord[ipt]))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} 1\n'.format(order[iel], order[iel + 1]))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


def _par(simid: str, extra: str = '', im: str = 'mMS',
         im_param: str = 'V_gate=0.1,a_crit=0.1') -> str:
    """The paced cable of test_carp_compatibility_run, plus extra lines."""
    return('meshname = cable\n'
           'simID    = {}\n'
           'tend     = {}\n'
           'dt       = {}\n'
           'spacedt  = {}\n'
           'timedt   = 100.0\n'
           'bidm_eqv_mono = 0\n'
           'imp_region[0].im       = "{}"\n'
           'imp_region[0].im_param = "{}"\n'
           'imp_region[0].cellSurfVolRatio = 0.14\n'
           'gregion[0].g_il = 0.174\n'
           'gregion[0].g_it = 0.174\n'
           'num_stim = 1\n'
           'stim[0].name           = "S1"\n'
           'stim[0].pulse.strength = 60.0\n'
           'stim[0].ptcl.start     = 0.0\n'
           'stim[0].ptcl.duration  = 2.0\n'
           'stim[0].ptcl.npls      = 1\n'
           'stim[0].elec.p0[0]     = 0.0\n'
           'stim[0].elec.p1[0]     = 250.0\n'.format(simid, _TEND, _DT_US, _SPACE,
                                                   im, im_param) + extra)


def _run_in(folder: str, argv: list) -> int:
    """Run main() with folder as the working directory, as the format expects."""
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        return(main(argv))
    finally:
        os.chdir(cwd)


def _read_igb(path: str) -> np.ndarray:
    reader = IGBReader()
    reader.read(path)
    return(np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x']))


# ---- the file ----------------------------------------------------------------
def _checkpoint() -> dict:
    return({'ionic_model': 'ModifiedMS2v',
            'time': 12.5,
            'num_nodes': 4,
            'Vm': np.array([-80.0, -20.0, 10.0, -79.0], dtype=np.float32),
            'state_variables': {'H_state': np.array([1.0, 0.5, 0.1, 0.9], dtype=np.float32)}})


def test_a_checkpoint_round_trips_through_the_file(tmp_path):
    """What is written is what is read, arrays included; the folder is created."""
    fname = str(tmp_path / 'sub' / 'state.12.5.pkl')
    StateWriter().write(_checkpoint(), fname)
    assert not os.path.exists('{}.partial'.format(fname))
    reader = StateReader()
    back   = reader.read(fname)
    assert reader.ionic_model() == 'ModifiedMS2v'
    assert reader.time() == 12.5
    assert reader.num_nodes() == 4
    np.testing.assert_array_equal(back['Vm'], _checkpoint()['Vm'])
    np.testing.assert_array_equal(back['state_variables']['H_state'],
                                  _checkpoint()['state_variables']['H_state'])
    # plain values only: loading must not need TensorFlow objects
    with open(fname, 'rb') as fstate:
        raw = pickle.load(fstate)
    assert isinstance(raw['Vm'], np.ndarray)


@pytest.mark.parametrize('damage,message', [
    (lambda c: c.pop('Vm'), 'missing Vm'),
    (lambda c: c.update({'Vm': np.zeros(3)}), 'Vm has 3 values'),
    (lambda c: c['state_variables'].update({'H_state': np.array([1.0, np.nan, 0.0, 0.0])}),
     'non-finite'),
    (lambda c: c.update({'time': -1.0}), 'invalid time'),
])
def test_a_damaged_checkpoint_is_refused_by_name(tmp_path, damage, message):
    """A truncated or inconsistent file stops at the reader, naming the file,
    instead of failing later inside the solver."""
    content = _checkpoint()
    damage(content)
    fname = str(tmp_path / 'bad.pkl')
    with open(fname, 'wb') as fstate:
        pickle.dump(content, fstate)
    with pytest.raises(ValueError, match=message):
        StateReader().read(fname)


# ---- the cell models ---------------------------------------------------------
def _depolarised(Model):
    """A model whose nodes sit 40 mV above rest, so the gates move at once."""
    model = Model(dt=0.01, n_nodes=4)
    rest  = model.get_parameter('V_init')
    if rest is None:
        rest = model.get_parameter('vmin')
    U = tf.Variable((float(rest) + 40.0) * tf.ones(shape=(4, 1), dtype=tf.float32))
    model.initialize_state_variables(U)
    return(model, U)


@pytest.mark.parametrize('Model', ALL_MODELS, ids=_ids(ALL_MODELS))
def test_state_variable_names_cover_what_differentiate_advances(Model):
    """Every tf.Variable that differentiate() changes must be declared as a state
    variable. A variable left out would silently restart from its initial value."""
    model, U = _depolarised(Model)
    before = {attr: value.numpy().copy() for attr, value in vars(model).items()
              if isinstance(value, tf.Variable)}
    model.differentiate(U)
    changed = {attr[1:] for attr, value in before.items()
               if not np.array_equal(value, getattr(model, attr).numpy())}
    declared = set(model.state_variable_names())
    assert len(changed) > 0
    assert changed <= declared, '{}: undeclared state {}'.format(
        Model.__name__, sorted(changed - declared))
    assert set(model.get_state_variables().keys()) == declared


@pytest.mark.parametrize('Model', ALL_MODELS, ids=_ids(ALL_MODELS))
def test_state_variables_round_trip_through_the_model(Model):
    """set_state_variables() followed by get_state_variables() returns the values."""
    model, _U = _depolarised(Model)
    states = {name: (0.5 * values + np.arange(values.size, dtype=values.dtype))
              for name, values in model.get_state_variables().items()}
    model.set_state_variables(states)
    for name, values in model.get_state_variables().items():
        np.testing.assert_allclose(values, states[name], rtol=1.0e-6)


def test_a_foreign_set_of_state_variables_is_refused():
    """A Fenton state pushed into the mMS model must not load half of itself."""
    model, _U = _depolarised(ModifiedMS2v)
    with pytest.raises(ValueError, match='not in the model: V_state'):
        model.set_state_variables({'H_state': np.ones(4), 'V_state': np.ones(4)})


# ---- the front end -----------------------------------------------------------
@pytest.fixture(scope='module')
def restart_runs(tmp_path_factory) -> dict:
    """Run the cable once uninterrupted, saving at tsav and every 2 ms, then
    resume from the tsav state with renumbering switched on."""
    folder = str(tmp_path_factory.mktemp('savestate'))
    _write_cable(folder)
    with open(os.path.join(folder, 'full.par'), 'w') as fout:
        fout.write(_par('OUT_A', 'num_tsav = 1\ntsav[0] = {}\nchkpt_intv = 2.0\n'.format(_TSAV)))
    with open(os.path.join(folder, 'restart.par'), 'w') as fout:
        # the extension is left out on purpose: it is accepted either way
        fout.write(_par('OUT_B', 'start_statef = OUT_A/state.3\nrenumbering = 1\n'))
    status_a = _run_in(folder, ['+F', 'full.par'])
    status_b = _run_in(folder, ['+F', 'restart.par'])
    return({'folder': folder, 'status_a': status_a, 'status_b': status_b})


def test_the_save_times_write_their_files(restart_runs):
    """tsav writes <write_statef>.<time>.pkl; checkpointing writes one file every
    chkpt_intv from chkpt_start, and none at tend, which the last step (t = tend - dt)
    does not reach."""
    assert restart_runs['status_a'] == 0
    outdir = os.path.join(restart_runs['folder'], 'OUT_A')
    for name in ('state.3.pkl', 'checkpoint.0.pkl', 'checkpoint.2.pkl', 'checkpoint.4.pkl'):
        assert os.path.isfile(os.path.join(outdir, name)), name
    assert not os.path.isfile(os.path.join(outdir, 'checkpoint.6.pkl'))

    reader = StateReader()
    reader.read(os.path.join(outdir, 'state.3.pkl'))
    assert reader.ionic_model() == 'ModifiedMS2v'
    assert reader.num_nodes() == _NPT
    assert reader.time() == pytest.approx(_TSAV, abs=0.5 * _DT_MS)
    assert set(reader.state_variables().keys()) == {'H_state'}


def test_the_restarted_run_matches_the_uninterrupted_one(restart_runs):
    """The restarted output starts with the saved potential, is recorded on the
    same steps as the uninterrupted run, and agrees with it within tolerance."""
    assert restart_runs['status_b'] == 0
    folder = restart_runs['folder']
    V_full    = _read_igb(os.path.join(folder, 'OUT_A', 'vm.igb'))
    V_restart = _read_igb(os.path.join(folder, 'OUT_B', 'vm.igb'))

    nsteps      = int(_TEND // _DT_MS)
    dt_per_plot = int(round(_SPACE / _DT_MS))
    first       = int(round(_TSAV / _DT_MS))
    # frames of the uninterrupted run: the initial one, then step indices 0, 40, ...
    full_steps    = list(range(0, nsteps, dt_per_plot))
    restart_steps = [istep for istep in full_steps if istep >= first]
    assert V_full.shape[0] == 1 + len(full_steps)
    assert V_restart.shape[0] == 1 + len(restart_steps)

    saved = StateReader()
    saved.read(os.path.join(folder, 'OUT_A', 'state.3.pkl'))
    np.testing.assert_allclose(V_restart[0, :], saved.Vm(), atol=1.0e-5)

    V_matching = V_full[1 + full_steps.index(restart_steps[0]):, :]
    diff = np.abs(V_restart[1:, :] - V_matching)
    print('\n  restart vs uninterrupted: max |dV| = {:.3e} mV, mean |dV| = {:.3e} mV'.format(
        diff.max(), diff.mean()))
    assert np.all(np.isfinite(V_restart))
    assert V_full.max() > 0.0                          # the front did travel
    assert diff.max() < _RESTART_TOL


def test_a_restored_state_survives_renumbering_for_every_variable(tmp_path, monkeypatch):
    """With a scrambled mesh and renumbering on, every ten Tusscher-Panfilov
    variable (most of them are not named *_state) is renumbered on the way in
    and undone on the way out."""
    _write_cable(str(tmp_path), shuffle=True)
    monkeypatch.chdir(str(tmp_path))
    with open('ttp.par', 'w') as fout:
        fout.write(_par('OUT_T', 'renumbering = 1\n', im='tenTusscherPanfilov', im_param=''))

    def build(argv: list) -> SimulationRunner:
        mapper = ParameterMapper()
        mapper.resolve(OptionReader().read(argv))
        runner = SimulationRunner({'verbose': False})
        runner.set_mapper(mapper)
        runner.build()
        return(runner)

    model = build(['+F', 'ttp.par']).model()
    perm  = model.renumbering()['perm']
    assert not np.array_equal(perm, np.arange(_NPT))

    # node-dependent values, so a permutation error changes them
    written = model.checkpoint()
    ramp    = np.arange(_NPT, dtype=np.float32)
    written['time'] = 40 * _DT_MS
    written['Vm']   = -85.0 + 0.5 * ramp
    for iname, name in enumerate(sorted(written['state_variables'].keys())):
        base = written['state_variables'][name]
        written['state_variables'][name] = base * (1.0 + 1.0e-2 * ramp) + 1.0e-3 * iname
    StateWriter().write(written, 'saved.pkl')

    restored = build(['+F', 'ttp.par', '-start_statef', 'saved.pkl']).model()
    assert restored.ctime() == pytest.approx(40 * _DT_MS)
    back = restored.checkpoint()
    np.testing.assert_allclose(back['Vm'], written['Vm'], rtol=1.0e-6)
    solver_order = restored.ionic_model().get_state_variables()
    for name, values in written['state_variables'].items():
        np.testing.assert_allclose(back['state_variables'][name], values, rtol=1.0e-6)
        np.testing.assert_allclose(solver_order[name], values[perm], rtol=1.0e-6)


def test_a_pure_diffusion_run_saves_and_resumes(tmp_path):
    """Without a cell model the checkpoint holds the potential only."""
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'heat.par'), 'w') as fout:
        fout.write('meshname = cable\nsimID = OUT_H\ntend = 1.0\ndt = {}\nspacedt = 0.25\n'
                   'timedt = 100.0\nnum_tsav = 1\ntsav[0] = 0.5\n'.format(_DT_US))
    assert _run_in(folder, ['+F', 'heat.par']) == 0
    reader = StateReader()
    reader.read(os.path.join(folder, 'OUT_H', 'state.0.5.pkl'))
    assert reader.ionic_model() == ''
    assert reader.state_variables() == {}
    # the restart reads its own file: overriding num_tsav to 0 while the file
    # still assigns tsav[0] is refused by the counter check, as it should be
    with open(os.path.join(folder, 'heat_restart.par'), 'w') as fout:
        fout.write('meshname = cable\nsimID = OUT_H2\ntend = 1.0\ndt = {}\nspacedt = 0.25\n'
                   'timedt = 100.0\nstart_statef = OUT_H/state.0.5.pkl\n'.format(_DT_US))
    assert _run_in(folder, ['+F', 'heat_restart.par']) == 0
    V = _read_igb(os.path.join(folder, 'OUT_H2', 'vm.igb'))
    assert np.all(np.isfinite(V))
    np.testing.assert_array_equal(V, 0.0)


@pytest.mark.parametrize('argv,message', [
    (['-imp_region[0].im', 'Fenton', '-imp_region[0].im_param', ''], 'cell model'),
    (['-tend', '2.0'], 'no step before tend'),
    (['-start_statef', 'nosuch'], 'cannot find nosuch'),
])
def test_an_unusable_checkpoint_stops_the_run(restart_runs, capsys, argv, message):
    """A checkpoint from another cell model, one saved after tend, and a missing
    file are reported and exit non-zero before any step is taken."""
    folder = restart_runs['folder']
    status = _run_in(folder, ['+F', 'restart.par', '-simID', 'OUT_ERR'] + argv)
    assert status == 1
    assert message in capsys.readouterr().err


def test_a_checkpoint_from_another_mesh_stops_the_run(restart_runs, capsys):
    folder  = restart_runs['folder']
    content = _checkpoint()
    StateWriter().write(content, os.path.join(folder, 'small.pkl'))
    status = _run_in(folder, ['+F', 'restart.par', '-simID', 'OUT_ERR', '-start_statef', 'small.pkl'])
    assert status == 1
    assert 'holds 4 nodes but the mesh has {}'.format(_NPT) in capsys.readouterr().err


# ---- the parameters ----------------------------------------------------------
def _mapper(argv: list) -> ParameterMapper:
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


def test_savestate_defaults_follow_the_format():
    """A tsav left unset is one step before tend, named after its time; interval
    checkpointing is off and stops at tend."""
    settings = _mapper(['-tend', '100', '-dt', '25', '-num_tsav', '1']).savestate_settings()
    assert settings['save_times'] == [(pytest.approx(99.975), 'state.99.975')]
    assert settings['start_statef'] == ''
    assert settings['chkpt_intv'] == 0.0
    assert settings['chkpt_stop'] == 100.0


def test_save_times_take_their_names_from_tsav_ext_and_write_statef():
    settings = _mapper(['-num_tsav', '2', '-tsav[0]', '10', '-tsav[1]', '12.5',
                        '-tsav_ext[1]', 'late', '-write_statef', 'sv']).savestate_settings()
    assert settings['save_times'] == [(10.0, 'sv.10'), (12.5, 'sv.late')]


def test_savestate_ranges_are_enforced():
    with pytest.raises(ValueError, match='at most 50'):
        _mapper(['-num_tsav', '51']).savestate_settings()
    with pytest.raises(ValueError, match='chkpt_intv'):
        _mapper(['-tend', '10', '-chkpt_intv', '20']).savestate_settings()
    notes = _mapper(['-tend', '10', '-tsav[0]', '20']).notes()
    assert any('after tend' in note for note in notes)
