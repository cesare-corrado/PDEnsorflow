#!/usr/bin/env python
"""
    Tier-1 tests of the single-cell state files (`.sv`): SvFileReader,
    SvFileWriter and the per-model layouts of carp_compatibility.svlayouts.

    The reference reads a state file by POSITION, so a layout is only right if
    its entries are those of the reference's state structure, in its order. The
    expected orders below are copied from the reference's generated headers
    (limpet/src/imps_src/<Model>.h), and the file text from a file written by
    `bench --save-ini-file` (MitchellSchaeffer, V_min=0, V_max=1, tau_out=5,
    saved at 150 ms).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pytest

from gpuSolve.IO.readers.svfilereader import SvFileReader
from gpuSolve.IO.writers.svfilewriter import SvFileWriter
from gpuSolve.carp_compatibility.parametermapper import IONIC_MODELS
from gpuSolve.carp_compatibility.parametermapper import IONIC_PLUGINS
from gpuSolve.carp_compatibility.svlayouts import SV_LAYOUTS
from gpuSolve.carp_compatibility.svlayouts import IMP_DATA_NAMES
from gpuSolve.carp_compatibility.svlayouts import sv_layout
from gpuSolve.ionic.ms2v import MitchellSchaeffer2v
from gpuSolve.ionic.mms2v import ModifiedMS2v

# A state file written by bench, verbatim.
BENCH_FILE = """0.800012            # Vm
-                   # Lambda
-                   # delLambda
-                   # Tension
-                   # Ke
-                   # Nae
-                   # Cae
0.00173675          # Iion
-                   # tension_component
-                   # illum
MitchellSchaeffer
0.13                # V_gate
1                   # V_max
0                   # V_min
0                   # a_crit
0.370946            # h
150                 # tau_close
0.3                 # tau_in
120                 # tau_open
5                   # tau_out

"""

# The reference's state structures, in order (limpet/src/imps_src/<Model>.h).
REFERENCE_ORDER = {
    'Courtemanche': ('Ca_rel Ca_up Cai Ki d f f_Ca h j m oa oi u ua ui v w xr xs'),
    'tenTusscherPanfilov': ('CaSR CaSS Cai D F F2 FCaSS GCaL GKr GKs Gto H J Ki M Nai R R_ '
                            'S Xr1 Xr2 Xs'),
    'MitchellSchaeffer': 'V_gate V_max V_min a_crit h tau_close tau_in tau_open tau_out',
    'Tomek': ('C1 C2 C3 CaMKt Cai Cajsr Cansr Cass I Jrel_np Jrel_p Ki Kss Nai Nass O a ap d '
              'fCaf fCafp fCas ff ffp fs h hL hLp hp iF iFp iS iSp j jCa jp m mL nCa_i nCa_ss '
              'xs1 xs2'),
    'Electroporation_DeBruinKrassowska98': 'n',
    'Defib_AshiharaTrayanova': 'Ki __sl_i2c_local',
}

# The entries the reference stores as single-precision gates (Gatetype).
REFERENCE_GATES = {
    'Courtemanche': 'd f f_Ca h j m oa oi u ua ui v w xr xs',
    'tenTusscherPanfilov': 'D F F2 FCaSS H J M R S Xr1 Xr2 Xs',
}


def test_read_a_bench_file(tmp_path):
    fname = tmp_path / 'ms.sv'
    fname.write_text(BENCH_FILE)
    reader = SvFileReader()
    reader.read(str(fname))
    assert [name for name, _value in reader.global_values()] == list(IMP_DATA_NAMES)
    values = dict(reader.global_values())
    assert values['Vm'] == 0.800012 and values['Iion'] == 0.00173675
    assert values['Lambda'] is None
    [(section, entries)] = reader.sections()
    assert section == 'MitchellSchaeffer'
    assert dict(entries)['h'] == 0.370946
    assert [name for name, _value in entries] == REFERENCE_ORDER['MitchellSchaeffer'].split()


def test_write_then_read_is_exact(tmp_path):
    fname = str(tmp_path / 'x.sv')
    # values a 6-digit %g would round, and one longer than the 20-character padding
    sections = [('Tomek', [('C1', 0.0006992506583711837), ('Cai', 1.0 / 3.0)]),
                ('Electroporation_DeBruinKrassowska98', [('n', 503726.1234567891)])]
    globals_ = [(name, None) for name in IMP_DATA_NAMES]
    globals_[0] = ('Vm', -88.8490687094118)
    writer = SvFileWriter({'fname': fname})
    writer.write(globals_, sections)
    reader = SvFileReader()
    reader.read(fname)
    assert reader.sections() == sections
    assert reader.global_values() == globals_
    # the layout of each line is the reference's: value, padding, "# name"
    first = open(fname).readline()
    assert first == '-88.8490687094118   # Vm\n'


def test_malformed_files_are_refused(tmp_path):
    reader = SvFileReader()
    bad = tmp_path / 'bad.sv'
    bad.write_text('-80 # Vm\n')
    with pytest.raises(ValueError, match='no model section'):
        reader.read(str(bad))
    bad.write_text('-80 # Vm\nCourtemanche\nabc # Cai\n')
    with pytest.raises(ValueError, match='not a number'):
        reader.read(str(bad))
    bad.write_text('-80 # Vm\nCourtemanche\n- # Cai\n')
    with pytest.raises(ValueError, match='has no value'):
        reader.read(str(bad))


@pytest.mark.parametrize('classname', sorted(SV_LAYOUTS.keys()))
def test_layouts_follow_the_reference_structures(classname):
    section, entries = SV_LAYOUTS[classname]
    assert [entry[0] for entry in entries] == REFERENCE_ORDER[section].split()
    gates = REFERENCE_GATES.get(section, '').split()
    assert [entry[0] for entry in entries if entry[4]] == gates


def _instance(name: str):
    cls = MitchellSchaeffer2v if name == 'MitchellSchaeffer' else IONIC_MODELS[name]
    model = cls(dt=0.01)
    if hasattr(model, 'set_use_rush_larsen'):
        # the default Tomek is slow to build; the layout does not depend on it
        model.set_use_rush_larsen(False)
    return(model)


@pytest.mark.parametrize('name', sorted(set(IONIC_MODELS.keys())))
def test_every_state_and_parameter_of_a_layout_exists(name):
    model = _instance(name)
    section, entries = sv_layout(model, name)
    states = [entry[2] for entry in entries if entry[1] == 'state']
    # every state variable of the model is in the file exactly once, so a
    # saved state restarts the run completely
    assert sorted(states) == sorted(model.state_variable_names())
    for entry in entries:
        if entry[1] == 'parameter':
            assert model.get_parameter(entry[2]) is not None, entry


def test_models_without_reference_counterpart_use_their_own_names():
    # ModifiedMS2v is not the reference's MitchellSchaeffer (its outward
    # current has a (1 - h) factor), so it gets its own section
    section, entries = sv_layout(ModifiedMS2v(dt=0.01), 'mMS')
    assert section == 'mMS'
    assert entries == [('H_state', 'state', 'H_state', 1.0, False)]


@pytest.mark.parametrize('name', sorted(IONIC_PLUGINS.keys()))
def test_plugin_layouts(name):
    plugin = IONIC_PLUGINS[name](dt=0.01)
    section, entries = sv_layout(plugin, name)
    assert section == name
    states = [entry[2] for entry in entries if entry[1] == 'state']
    assert sorted(states) == sorted(plugin.state_variable_names())
