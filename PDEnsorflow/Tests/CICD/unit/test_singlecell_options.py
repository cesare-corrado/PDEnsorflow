#!/usr/bin/env python
"""
    Tier-1 tests of SingleCellOptionReader, the command line of the
    `singlecell` executable.

    The expected behaviour is that of the reference single-cell tool (bench),
    checked against the installed binary when the reader was written: the
    abbreviation, conflict and error rules below, and the exact wording of its
    messages. The one deliberate difference (a stray word is an error, where
    bench drops it) is pinned too.

    The reader must not start TensorFlow: singlecell chooses the device and the
    thread count from the options, before TensorFlow starts.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import subprocess
import sys

import pytest

from gpuSolve.carp_compatibility.singlecelloptionreader import SingleCellOptionReader
from gpuSolve.carp_compatibility.singlecelloptionreader import BENCH_OPTIONS
from gpuSolve.carp_compatibility.singlecelloptionreader import SUPPORTED_OPTIONS


def _read(argv: list) -> SingleCellOptionReader:
    reader = SingleCellOptionReader()
    reader.read(argv)
    return(reader)


def _error(argv: list) -> str:
    with pytest.raises(ValueError) as caught:
        _read(argv)
    return(str(caught.value))


def test_long_forms_and_abbreviation():
    reader = _read(['--imp', 'Tomek', '--duration=250', '--imp-par', 'GKr*0.5', '--bc', '800'])
    assert reader.value('imp') == 'Tomek'
    assert reader.value('duration') == 250.0
    assert reader.value('imp-par') == 'GKr*0.5'
    # a unique prefix is accepted, as getopt_long does
    assert reader.value('bcl') == 800.0
    assert reader.given_options() == ['imp', 'duration', 'imp-par', 'bcl']


def test_exact_name_wins_over_a_longer_one():
    # --dt is also a prefix of --dt-out
    assert _read(['--dt', '0.02']).value('dt') == 0.02


def test_short_forms_grouping_and_glued_values():
    reader = _read(['-vB', '-a2', '-D', '0.02', '-IFenton'])
    assert reader.value('validate') and reader.value('bin')
    assert reader.value('duration') == 2.0
    assert reader.value('dt') == 0.02
    assert reader.value('imp') == 'Fenton'


def test_defaults():
    reader = _read([])
    # the default model is singlecell's own (bench's DrouhardRoberge is not in gpuSolve)
    assert reader.value('imp') == 'Courtemanche'
    assert reader.value('dt') == 0.01
    assert reader.value('bcl') == 1000.0
    assert reader.value('stim-curr') == 60.0
    assert reader.value('fout') == 'BENCH_REG'
    assert reader.value('duration') is None
    assert not reader.given('duration')


def test_optional_argument_only_when_glued():
    assert _read(['--fout=run1']).value('fout') == 'run1'
    assert _read(['-Orun2']).value('fout') == 'run2'
    bare = _read(['--fout'])
    assert bare.given('fout') and bare.value('fout') == 'BENCH_REG'
    # bench leaves `run3` behind and ignores it; singlecell refuses it
    assert "unexpected argument 'run3'" in _error(['--fout', 'run3'])


def test_bench_error_messages():
    assert _error(['--dt', '0.01', '--dt', '0.02']) == \
        "singlecell: `--dt' (`-D') option given more than once"
    assert _error(['--numstim', '2', '--stim-times', '1,2']) == \
        'singlecell: option --stim-times conflicts with option --numstim'
    assert _error(['--stim-curr', '20', '--stim-volt', '10']) == \
        'singlecell: 2 options of group stimtype were given. At most one is required.'
    assert _error(['--dt']) == "singlecell: option '--dt' requires an argument"
    assert _error(['--dt', 'abc']) == 'singlecell: invalid numeric value: abc'
    assert _error(['--numstim', '1.5']) == 'singlecell: invalid numeric value: 1.5'
    assert _error(['--foo', '1']) == "singlecell: unrecognized option '--foo'"
    assert _error(['--bin=1']) == "singlecell: option '--bin' doesn't allow an argument"
    assert _error(['-x']) == "singlecell: invalid option -- 'x'"


def test_ambiguous_prefix_lists_candidates_in_bench_order():
    message = _error(['--st', '1'])
    assert message.startswith("singlecell: option '--st' is ambiguous; possibilities: "
                              "'--stim-start' '--stim-times' '--stim-curr'")
    assert message.endswith("'--strain-rate' '--start-out'")


def test_stray_word_is_an_error():
    assert "unexpected argument 'foo'" in _error(['--duration', '2', 'foo'])


def test_unsupported_bench_options_are_recognised():
    reader = _read(['--restitute', 'S1S2', '--clamp', '-20', '--imp', 'Tomek'])
    assert reader.unsupported_options() == ['restitute', 'clamp']
    assert _read(['--imp', 'Tomek']).unsupported_options() == []
    # every bench option is known: none is "unrecognized", and each one that is
    # not implemented is reported as such
    for name, _short, kind, _default, _family, _text in BENCH_OPTIONS:
        if name in SUPPORTED_OPTIONS:
            continue
        if kind in ('flag', 'optstring'):
            argv = ['--{}'.format(name)]
        elif name == 'restitute':
            argv = ['--restitute', 'dyn']
        else:
            argv = ['--{}'.format(name), '1' if kind in ('int', 'double', 'float') else 'x']
        assert _read(argv).unsupported_options() == [name]


def test_closed_value_set():
    assert 'possible values: S1S2, dyn, S1S2f' in _error(['--restitute', 'S1'])


def test_gpusolve_option():
    assert _read(['--reference-scheme']).value('reference-scheme')
    assert 'reference-scheme' in SUPPORTED_OPTIONS


def test_help_text_lists_what_is_implemented():
    text = SingleCellOptionReader().usage_text()
    assert '--imp=STRING' in text and "(default=`Courtemanche')" in text
    assert 'gpuSolve options (not in bench):' in text
    assert '--restitute' in text.split('not supported yet')[1]


def test_reader_does_not_start_tensorflow():
    code = ('import sys\n'
            'from gpuSolve.carp_compatibility.singlecelloptionreader import SingleCellOptionReader\n'
            'SingleCellOptionReader().read(["--imp", "Tomek"])\n'
            'import gpuSolve.carp_compatibility.singlecell\n'
            'print("tensorflow" in sys.modules)\n')
    # the in-tree sources, as conftest.py puts them first on sys.path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    env = dict(os.environ, PYTHONPATH=root)
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env=env, check=True)
    assert result.stdout.strip().splitlines()[-1] == 'False'
