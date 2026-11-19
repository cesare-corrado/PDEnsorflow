#!/usr/bin/env python
"""
    Tier-1 tests of the binary state files (`.roe`) in the parameter-file front
    end: state_format, the reference's file names and save steps, the restart
    from a `.roe` file, the checks a file must pass, and the stored cell
    parameters a restart takes over (with a warning).

    The files are those of gpuSolve itself here; the exchange with the
    reference (a restart of either simulator from the other's file) needs the
    reference installed and is Tests/CICD/nightly/test_roe_reference_exchange.py.

    CPU-only and small: a 51-node cable in two ionic regions, 6 ms.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.carp_compatibility.main import main
from gpuSolve.IO.readers import IGBReader
from gpuSolve.IO.readers.roereader import RoeReader

_NELEM  = 50
_DX     = 100.0                                   # micrometres
_NPT    = _NELEM + 1
_TEND   = 6.0                                     # ms
_DT_US  = 25.0
_DT_MS  = _DT_US * 1.0e-3
# the restarted run starts CG from U^n instead of the warm start U^{n-1}, so it
# agrees with the uninterrupted one to the CG tolerance (see test_savestate)
_RESTART_TOL = 1.0e-2


def _write_cable(folder: str):
    """ a uniform cable; the second half of its elements carries tag 2 """
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


def _par(extra: str = '', im: str = 'mMS', im_param: str = 'V_gate=0.1,a_crit=0.1',
         plugins: str = '') -> str:
    """ the paced cable of test_savestate, in two ionic regions """
    return('meshname = cable\n'
           'tend     = {}\n'
           'dt       = {}\n'
           'spacedt  = 1.0\n'
           'timedt   = 100.0\n'
           'bidm_eqv_mono = 0\n'
           'num_imp_regions = 2\n'
           'imp_region[0].im       = "{im}"\n'
           'imp_region[0].im_param = "{p}"\n'
           'imp_region[1].im       = "{im}"\n'
           'imp_region[1].im_param = "{p}"\n'
           'imp_region[1].num_IDs  = 1\n'
           'imp_region[1].ID[0]    = 2\n'
           'imp_region[1].plugins  = "{plugins}"\n'
           'gregion[0].g_il = 0.174\n'
           'gregion[0].g_it = 0.174\n'
           'num_stim = 1\n'
           'stim[0].pulse.strength = 60.0\n'
           'stim[0].ptcl.start     = 0.0\n'
           'stim[0].ptcl.duration  = 2.0\n'
           'stim[0].ptcl.npls      = 1\n'
           'stim[0].elec.p0[0]     = 0.0\n'
           'stim[0].elec.p1[0]     = 250.0\n'.format(_TEND, _DT_US, im=im, p=im_param,
                                                    plugins=plugins) + extra)


def _run_in(folder: str, argv: list) -> int:
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


# the saves of the uninterrupted runs, given on their command line only, so
# that the restarts read the same parameter file without them
_SAVES = ['-num_tsav', '2', '-tsav[0]', '3.0', '-tsav[1]', '4.31', '-tsav_ext[1]', 'late',
          '-chkpt_intv', '2.0']


@pytest.fixture(scope='module')
def runs(tmp_path_factory) -> dict:
    """ an uninterrupted run saving with the reference's defaults, and a
        restart from its tsav file given without the extension
    """
    folder = str(tmp_path_factory.mktemp('roe_state'))
    _write_cable(folder)
    with open(os.path.join(folder, 'run.par'), 'w') as fout:
        fout.write(_par())
    status_a = _run_in(folder, ['+F', 'run.par', '-simID', 'OUT_A'] + _SAVES)
    status_b = _run_in(folder, ['+F', 'run.par', '-simID', 'OUT_B',
                                '-start_statef', 'OUT_A/state.3.0'])
    return({'folder': folder, 'status_a': status_a, 'status_b': status_b})


def test_the_files_have_the_reference_names_and_content(runs):
    """ roe1 is the default; tsav_ext defaults to the time with one decimal,
        checkpoints are named after their step time, and the file describes
        the two ionic regions node by node
    """
    assert runs['status_a'] == 0
    outdir = os.path.join(runs['folder'], 'OUT_A')
    names = sorted(name for name in os.listdir(outdir) if name.endswith('.roe'))
    assert names == ['checkpoint.0.0.roe', 'checkpoint.2.0.roe', 'checkpoint.4.0.roe',
                     'checkpoint.6.0.roe', 'state.3.0.roe', 'state.late.roe']
    content = RoeReader().read(os.path.join(outdir, 'state.3.0.roe'))
    assert content['format'] == 1 and content['revision'] == 0
    assert content['time'] == pytest.approx(3.0, abs=1.0e-6)
    assert content['num_nodes'] == _NPT
    assert list(content['globals'].keys()) == ['Vm', 'Iion']
    assert [region['model'] for region in content['regions']] == ['mMS', 'mMS']
    assert [region['size'] for region in content['regions']] == [72, 72]
    # the nodes of tag-2 elements only, the shared node 25 included, are
    # region 1 (a node takes the highest tag a region claims)
    expected = np.zeros(_NPT, dtype=np.int8)
    expected[_NELEM // 2:] = 1
    np.testing.assert_array_equal(content['mask'], expected)
    # tsav = 4.31 falls on step lround(4.31 / 0.025) = 172, t = 4.3
    late = RoeReader().read(os.path.join(outdir, 'state.late.roe'))
    assert late['time'] == pytest.approx(4.3, abs=1.0e-6)


def test_the_restart_from_a_roe_file_matches_the_uninterrupted_run(runs):
    assert runs['status_b'] == 0
    folder = runs['folder']
    full    = _read_igb(os.path.join(folder, 'OUT_A', 'vm.igb'))
    restart = _read_igb(os.path.join(folder, 'OUT_B', 'vm.igb'))
    # frames every 1 ms: the restart records t = 3, 4, 5, 6
    np.testing.assert_allclose(restart, full[3:], atol=_RESTART_TOL)
    assert full.max() > 0.0


def test_stored_parameters_win_with_a_warning(runs, capsys):
    """ the reference restores the parameter fields of its records, so a
        parameter changed in the parameter file does not take effect on a
        restart; it is reported, and the run is the uninterrupted one
    """
    folder = runs['folder']
    status = _run_in(folder, ['+F', 'run.par', '-simID', 'OUT_W', '-start_statef', 'OUT_A/state.3.0.roe',
                              '-imp_region[1].im_param', 'V_gate=0.1,a_crit=0.1,tau_out=6.0'])
    assert status == 0
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'tau_out = 9 where the parameters give 6' in out
    restart = _read_igb(os.path.join(folder, 'OUT_W', 'vm.igb'))
    full    = _read_igb(os.path.join(folder, 'OUT_A', 'vm.igb'))
    np.testing.assert_allclose(restart, full[3:], atol=_RESTART_TOL)


@pytest.mark.parametrize('argv,message', [
    (['-imp_region[0].im', 'Fenton', '-imp_region[1].im', 'Fenton',
      '-imp_region[0].im_param', '', '-imp_region[1].im_param', ''], 'holds cell model mMS'),
    (['-num_imp_regions', '3', '-imp_region[2].im', 'mMS'], 'with 2 ionic region(s), this run has 3'),
    (['-imp_region[1].ID[0]', '1'], 'different ionic region than this run does at'),
    (['-imp_region[1].plugins', 'Electroporation_DeBruinKrassowska98'], 'holds 0 plugin(s)'),
])
def test_a_state_of_another_configuration_is_refused(runs, capsys, argv, message):
    """ the checks of the reference's restore_state, each with its message """
    status = _run_in(runs['folder'], ['+F', 'run.par', '-simID', 'OUT_ERR', '-start_statef', 'OUT_A/state.3.0']
                     + argv)
    assert status == 1
    assert message in capsys.readouterr().err


def test_a_misnamed_state_file_is_refused(runs, capsys):
    """ the first bytes tell the two kinds apart, and the name must agree """
    folder = runs['folder']
    with open(os.path.join(folder, 'OUT_A', 'state.3.0.roe'), 'rb') as fin:
        data = fin.read()
    with open(os.path.join(folder, 'wrong.pkl'), 'wb') as fout:
        fout.write(data)
    status = _run_in(folder, ['+F', 'run.par', '-simID', 'OUT_ERR', '-start_statef', 'wrong.pkl'])
    assert status == 1
    assert 'is a binary state file (.roe)' in capsys.readouterr().err


def test_a_plugin_state_restarts(tmp_path):
    """ a plugin on one region: its records cover that region's nodes only """
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'plug.par'), 'w') as fout:
        fout.write(_par(plugins='Electroporation_DeBruinKrassowska98'))
    assert _run_in(folder, ['+F', 'plug.par', '-simID', 'OUT_P', '-num_tsav', '1',
                            '-tsav[0]', '3.0']) == 0
    content = RoeReader().read(os.path.join(folder, 'OUT_P', 'state.3.0.roe'))
    assert content['regions'][0]['plugins'] == []
    plugin = content['regions'][1]['plugins'][0]
    assert plugin['name'] == 'Electroporation_DeBruinKrassowska98'
    # the true size: the reference restores a plugin only from such a table
    assert plugin['size'] == 8 and plugin['stored_size'] == 8
    assert _run_in(folder, ['+F', 'plug.par', '-simID', 'OUT_Q',
                            '-start_statef', 'OUT_P/state.3.0']) == 0
    full    = _read_igb(os.path.join(folder, 'OUT_P', 'vm.igb'))
    restart = _read_igb(os.path.join(folder, 'OUT_Q', 'vm.igb'))
    np.testing.assert_allclose(restart, full[3:], atol=_RESTART_TOL)


@pytest.mark.parametrize('state_format,name,fmt', [('roe3', 'state.3.0.roe', 3),
                                                   ('pkl', 'state.3.0.pkl', None)])
def test_state_format_selects_the_file(tmp_path, state_format, name, fmt):
    """ roe3 writes the packed format with fingerprints; pkl gpuSolve's own
        checkpoint, under the same name with its extension; both restart
    """
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'f.par'), 'w') as fout:
        fout.write(_par('state_format = {}\n'.format(state_format)))
    assert _run_in(folder, ['+F', 'f.par', '-simID', 'OUT_F', '-num_tsav', '1',
                            '-tsav[0]', '3.0']) == 0
    assert sorted(os.listdir(os.path.join(folder, 'OUT_F'))) == sorted([name, 'vm.igb'])
    if fmt is not None:
        content = RoeReader().read(os.path.join(folder, 'OUT_F', name))
        assert content['format'] == fmt and content['regions'][0]['fingerprint'] is not None
    assert _run_in(folder, ['+F', 'f.par', '-simID', 'OUT_G',
                            '-start_statef', 'OUT_F/state.3.0']) == 0
    full    = _read_igb(os.path.join(folder, 'OUT_F', 'vm.igb'))
    restart = _read_igb(os.path.join(folder, 'OUT_G', 'vm.igb'))
    np.testing.assert_allclose(restart, full[3:], atol=_RESTART_TOL)


def test_mitchell_schaeffer_with_a_threshold_keeps_its_name(tmp_path):
    """ MitchellSchaeffer with a_crit runs ModifiedMS2v, but the run is the
        reference's MitchellSchaeffer, and so is the file
    """
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'ms.par'), 'w') as fout:
        fout.write(_par('num_tsav = 1\ntsav[0] = 1.0\n', im='MitchellSchaeffer'))
    assert _run_in(folder, ['+F', 'ms.par', '-simID', 'OUT_M', '-tend', '1.0']) == 0
    content = RoeReader().read(os.path.join(folder, 'OUT_M', 'state.1.0.roe'))
    assert [region['model'] for region in content['regions']] == ['MitchellSchaeffer'] * 2
