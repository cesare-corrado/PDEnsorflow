#!/usr/bin/env python
"""
    Tier-1 end-to-end tests of the `singlecell` executable
    (gpuSolve.carp_compatibility.singlecell.main).

    The numbers come from the reference single-cell tool (bench, openCARP
    v16), run with the same command lines on the MitchellSchaeffer model with
    V_min=0, V_max=1, tau_out=5 (the reference's defaults; gpuSolve's differ),
    which is fast and exercises the timing rules without a slow model build:
      * a regular train (2 stimuli, BCL 400 ms, output every 50 ms);
      * irregular stimuli given as diastolic intervals, with a delayed output
        start and no --duration (it follows from the last stimulus);
      * a state file saved at 150 ms.
    gpuSolve's MitchellSchaeffer2v is float32 and the reference float64: the
    potentials agree to about 1e-6.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.carp_compatibility.singlecell import main
from gpuSolve.IO.readers.svfilereader import SvFileReader

MS = ['--imp', 'MitchellSchaeffer', '--imp-par', 'V_min=0,V_max=1,tau_out=5',
      '--stim-curr', '0.5', '--no-trace']

# bench --imp MitchellSchaeffer --imp-par V_min=0,V_max=1,tau_out=5 --stim-curr 0.5
#       --numstim 2 --bcl 400 --duration 800 --dt-out 50
REFERENCE_TRAIN = np.array([
    [0.0, 0.00000000e+00], [50.0, 9.08978949e-01], [100.0, 8.67145282e-01],
    [150.0, 8.00011916e-01], [200.0, 6.74584074e-01], [250.0, 1.98829805e-01],
    [300.0, 2.19441413e-05], [350.0, 9.86505667e-10], [400.0, 4.43410554e-14],
    [450.0, 8.76305167e-01], [500.0, 8.15203937e-01], [550.0, 7.05889979e-01],
    [600.0, 3.69628660e-01], [650.0, 1.38898025e-04], [700.0, 6.24930622e-09],
    [750.0, 2.80891289e-13], [800.0, 1.26253873e-17]])

# ... --stim-times 10,390 --DIA --past-stim 100 --dt-out 25 --start-out 380
REFERENCE_DIA = np.array([
    [380.0, 1.47311378e-11], [405.0, 9.07733341e-01], [430.0, 8.89567472e-01],
    [455.0, 8.66199345e-01], [480.0, 8.36673309e-01]])


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """ an empty working directory, and an environment restored afterwards
        (main() sets the thread count and the visible devices) """
    monkeypatch.chdir(tmp_path)
    for name in ('CUDA_VISIBLE_DEVICES', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    return(tmp_path)


def _table(fname: str) -> np.ndarray:
    return(np.loadtxt(fname))


def test_regular_train_matches_the_reference(workdir):
    assert main(MS + ['--numstim', '2', '--bcl', '400', '--duration', '800', '--dt-out', '50',
                      '--fout=train']) == 0
    table = _table('train.txt')
    np.testing.assert_array_equal(table[:, 0], REFERENCE_TRAIN[:, 0])
    np.testing.assert_allclose(table[:, 1], REFERENCE_TRAIN[:, 1], atol=1e-5)


def test_diastolic_intervals_and_output_start(workdir):
    assert main(MS + ['--stim-times', '10,390', '--DIA', '--past-stim', '100', '--dt-out', '25',
                      '--start-out', '380', '--fout=dia']) == 0
    table = _table('dia.txt')
    np.testing.assert_array_equal(table[:, 0], REFERENCE_DIA[:, 0])
    np.testing.assert_allclose(table[:, 1], REFERENCE_DIA[:, 1], atol=1e-5)


def test_validate_writes_the_reference_binaries(workdir):
    assert main(MS + ['--duration', '20', '-v']) == 0
    lines = open('BENCH_REG_header.txt').read().splitlines()
    assert lines[0] == '0  # is bigendian'
    listed = [line.split() for line in lines[1:]]
    assert [entry[0] for entry in listed[:3]] == ['BENCH_REG.Vm.bin', 'BENCH_REG.Iion.bin',
                                                  'BENCH_REG.t.bin']
    assert ['BENCH_REG_MitchellSchaeffer.h.bin', 'GlobalData_t', '8', '21'] in listed
    t = np.fromfile('BENCH_REG.t.bin')
    np.testing.assert_allclose(t, np.arange(21.0))
    V = np.fromfile('BENCH_REG.Vm.bin')
    assert V.shape == (21,) and V[0] == 0.0 and V[2] > 0.5
    # a parameter the reference keeps in the state is dumped with its value
    np.testing.assert_allclose(np.fromfile('BENCH_REG_MitchellSchaeffer.tau_out.bin'), 5.0)


def test_state_file_matches_the_reference_and_restarts(workdir):
    assert main(MS + ['--duration', '200', '--dt-out', '100', '-F', 'ms150.sv', '-S', '150']) == 0
    reader = SvFileReader()
    reader.read('ms150.sv')
    values = dict(reader.global_values())
    [(section, entries)] = reader.sections()
    assert section == 'MitchellSchaeffer'
    # bench: Vm 0.800012, h 0.370946 (6 significant digits)
    assert values['Vm'] == pytest.approx(0.800012, abs=2e-6)
    assert dict(entries)['h'] == pytest.approx(0.370946, abs=2e-6)
    assert dict(entries)['tau_out'] == 5.0
    # restarting from it continues the paced run: the potential 50 ms later is
    # the one of the uninterrupted run at 200 ms
    assert main(MS + ['--read-ini-file', 'ms150.sv', '--numstim', '0', '--duration', '50',
                      '--dt-out', '50', '--fout=restart']) == 0
    restart = _table('restart.txt')
    assert restart[-1, 1] == pytest.approx(REFERENCE_TRAIN[4, 1], abs=1e-5)


def test_state_file_parameters_are_reported_not_applied(workdir, capsys):
    assert main(MS + ['--duration', '10', '-F', 'ms.sv', '-S', '10']) == 0
    capsys.readouterr()
    # the file says tau_out = 5; this run keeps gpuSolve's default (6)
    assert main(['--imp', 'MitchellSchaeffer', '--imp-par', 'V_min=0,V_max=1',
                 '--read-ini-file', 'ms.sv', '--duration', '5', '--no-trace']) == 0
    err = capsys.readouterr().err
    warning = 'WARNING: state file ms.sv sets tau_out = 5 (MitchellSchaeffer); ignored, this run uses 6'
    assert err.count(warning) == 2      # at the start and at the end
    # parameters that agree give no warning
    assert main(MS + ['--read-ini-file', 'ms.sv', '--duration', '5']) == 0
    assert 'WARNING: state file' not in capsys.readouterr().err


def test_state_file_of_another_model_is_refused(workdir, capsys):
    assert main(MS + ['--duration', '10', '-F', 'ms.sv', '-S', '10']) == 0
    assert main(['--imp', 'Fenton', '--read-ini-file', 'ms.sv', '--duration', '5']) == 1
    assert 'the state file holds MitchellSchaeffer, but this run has Fenton' in capsys.readouterr().err


def test_sv_dump_list(workdir, capsys):
    assert main(MS + ['--duration', '4', '-u', 'h']) == 0
    assert os.path.exists('BENCH_REG_MitchellSchaeffer.h.bin')
    assert np.fromfile('BENCH_REG_MitchellSchaeffer.h.bin').shape == (5,)
    # bench's manual writes the list with colons, but bench splits it on
    # commas and skips what it does not know; singlecell stops instead
    assert main(MS + ['--duration', '4', '-u', 'h:V_gate']) == 1
    assert 'has no entry "h:V_gate"' in capsys.readouterr().err


def test_stdout_table_and_modifier_banner(workdir, capsys):
    assert main(MS + ['--duration', '2']) == 0
    out = capsys.readouterr().out
    assert 'Ionic model: MitchellSchaeffer' in out
    assert 'tau_out              modifier: =5              value: 5' in out
    rows = [line for line in out.splitlines() if line.startswith('     ')]
    assert rows[0] == '     0.000 +0.00000000e+00 +0.00000000e+00 '
    assert len(rows) == 3


def test_refusals(workdir, capsys):
    assert main(['--restitute', 'S1S2']) == 1
    assert 'is a bench option that singlecell does not support yet' in capsys.readouterr().err
    assert main(['--num', '4']) == 1
    assert main(['--target', 'mlir-rocm']) == 1
    assert 'Available targets are: auto, cpu, mlir-cpu, mlir-cuda' in capsys.readouterr().err
    assert main(['--imp', 'DrouhardRoberge', '--duration', '1']) == 1
    assert 'unknown cell model "DrouhardRoberge"' in capsys.readouterr().err
    assert main(['--imp', 'MitchellSchaeffer', '--imp-par', 'tau_outt=5', '--duration', '1']) == 1
    assert 'has no parameter "tau_outt"' in capsys.readouterr().err
    assert main(['--numstim', '0', '--past-stim', '-5']) == 1
    assert 'negative' in capsys.readouterr().err


def test_info_modes(workdir, capsys):
    assert main(['--list-imps']) == 0
    out = capsys.readouterr().out
    assert '\tTomek' in out and '\tElectroporation_DeBruinKrassowska98' in out
    assert main(['--imp', 'MitchellSchaeffer', '--imp-info']) == 0
    out = capsys.readouterr().out
    assert 'tau_out\t6' in out and '\t\t                   h' in out
