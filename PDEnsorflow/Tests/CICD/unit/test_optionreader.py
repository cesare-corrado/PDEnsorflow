#!/usr/bin/env python
"""
    Tier-1 unit tests for the parameter-file lexer and the command-line reader
    (gpuSolve.carp_compatibility.parfilereader / .optionreader).

    These exercise the SHAPE of the input only, so they touch no mesh and no
    TensorFlow work and run in well under a second.

    The rule that matters most here is the resolution order: files and flags are
    applied strictly left to right and the last assignment of a key wins, so a
    flag placed before a `+F` loses to the file and one placed after it does
    not. That is measurable behaviour of the format being matched, and it is
    the opposite of the "command line always overrides the file" rule one might
    assume, hence a test of its own.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os

import pytest

from gpuSolve.carp_compatibility.parfilereader import ParFileReader
from gpuSolve.carp_compatibility.parfilereader import strip_comments
from gpuSolve.carp_compatibility.parfilereader import decode_quoted
from gpuSolve.carp_compatibility.parfilereader import split_assignment
from gpuSolve.carp_compatibility.optionreader import OptionReader


def _write(tmp_path, name: str, text: str) -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, 'w') as fout:
        fout.write(text)
    return(path)


# ---- the lexer -------------------------------------------------------------
@pytest.mark.parametrize('line,expected', [
    ('dt = 25 # the time step', 'dt = 25 '),
    ('# a whole comment line', ''),
    ('im_param = "a#b" # tail', 'im_param = "a#b" '),
    ('x = "he said \\"hi\\"" # tail', 'x = "he said \\"hi\\"" '),
    ('dt = 25', 'dt = 25'),
])
def test_strip_comments(line, expected):
    """A `#` ends the line, unless it sits inside a quoted string."""
    assert strip_comments(line) == expected


@pytest.mark.parametrize('raw,expected', [
    ('"normal"', 'normal'),
    ('bare', 'bare'),
    ('"a\\"b"', 'a"b'),
    ('"V_gate=0.1,a_crit=0.1"', 'V_gate=0.1,a_crit=0.1'),
])
def test_decode_quoted(raw, expected):
    """A quoted value is unquoted and its escapes resolved; a bare one is kept."""
    assert decode_quoted(raw) == expected


def test_split_assignment_uses_the_first_equals():
    """Cell-parameter lists carry further `=` signs that belong to the value."""
    key, value = split_assignment('imp_region[0].im_param = "V_gate=0.1,a_crit=0.1"')
    assert key == 'imp_region[0].im_param'
    assert value == 'V_gate=0.1,a_crit=0.1'


def test_par_file_reads_comments_continuations_and_order(tmp_path):
    """A file keeps its order, joins continuations with one space and does not
    collapse a repeated key, because the caller resolves it by replaying."""
    path = _write(tmp_path, 'a.par',
                  '# header\n'
                  'dt = 25            # microseconds\n'
                  'meshname = mesh\n'
                  '\n'
                  'gregion[0].ID = 1:3, \\\n'
                  '                7\n'
                  'dt = 50\n')
    pairs = ParFileReader().read(path)
    assert pairs == [('dt', '25'), ('meshname', 'mesh'),
                     ('gregion[0].ID', '1:3, 7'), ('dt', '50')]


def test_par_file_reports_a_bad_line_with_its_number(tmp_path):
    """A statement that is not an assignment names the file and the line."""
    path = _write(tmp_path, 'bad.par', 'dt = 25\nthis is not an assignment\n')
    with pytest.raises(ValueError) as excinfo:
        ParFileReader().read(path)
    assert 'bad.par:2' in str(excinfo.value)


# ---- the command line ------------------------------------------------------
def test_flag_before_file_loses_and_after_file_wins(tmp_path):
    """Strict left-to-right resolution, verified in both directions."""
    path = _write(tmp_path, 'r.par', 'mass_lumping = 0\ndt = 10\n')
    before = OptionReader().read(['-mass_lumping', '1', '+F', path])
    after  = OptionReader().read(['+F', path, '-mass_lumping', '1'])
    assert before['mass_lumping'] == '0'
    assert after['mass_lumping'] == '1'
    # the key the file alone sets is unaffected either way
    assert before['dt'] == '10' and after['dt'] == '10'


def test_two_files_resolve_in_order(tmp_path):
    """The second file overrides the first where they overlap."""
    first  = _write(tmp_path, 'one.par', 'dt = 10\ntend = 100\n')
    second = _write(tmp_path, 'two.par', 'dt = 20\n')
    store  = OptionReader().read(['+F', first, '+F', second])
    assert store['dt'] == '20'
    assert store['tend'] == '100'


@pytest.mark.parametrize('argv', [['-dt', '25'], ['--dt', '25'], ['--dt=25']])
def test_flag_spellings_are_equivalent(argv):
    """Single dash, double dash and the glued form all name the same key."""
    assert OptionReader().read(argv)['dt'] == '25'


def test_negative_values_need_no_escaping():
    """The value is always the next token, so a leading minus is not an option."""
    store = OptionReader().read(['-stim[0].elec.p0[0]', '-1500.0'])
    assert store['stim[0].elec.p0[0]'] == '-1500.0'


def test_long_file_alias_and_save_round_trip(tmp_path):
    """--file is accepted for +F, and +Save writes the resolved set back out in
    a form that parses to the same store."""
    path  = _write(tmp_path, 'r.par', 'dt = 10\nimp_region[0].im_param = "V_gate=0.1,a_crit=0.1"\n')
    saved = os.path.join(str(tmp_path), 'resolved.par')
    reader = OptionReader()
    store  = reader.read(['--file', path, '--save', saved, '-dt', '25'])
    assert store['dt'] == '25'
    reader.write_save_file()
    again = OptionReader().read(['+F', saved])
    assert again == store


@pytest.mark.parametrize('argv,message', [
    (['-dt'], 'expects a value'),
    (['stray'], 'Unexpected argument'),
    (['+Nope', 'x'], 'Unrecognized keyword'),
    (['+F'], 'expects a parameter file'),
])
def test_malformed_command_lines_are_rejected(argv, message):
    """A command line that cannot be read stops before any work is done."""
    with pytest.raises(Exception) as excinfo:
        OptionReader().read(argv)
    assert message in str(excinfo.value)
