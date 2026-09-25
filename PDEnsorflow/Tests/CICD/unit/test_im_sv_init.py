#!/usr/bin/env python
"""
    Tier-1 tests of `imp_region[].im_sv_init`: the single-cell state file whose
    state is copied onto every node of an ionic region, the equivalent of what
    the reference simulator does when it sets its cell models up ("read in
    single cell state vector and spread it out over the entire region").

    What is checked, and why each check is the one that matters:

      * the state reaches the nodes of the region that names the file, and only
        those. A key that quietly conditioned the whole mesh, or nothing at all,
        would still produce a run that looks plausible;
      * both the potential and the state variables are taken from the file:
        a file that only moved Vm would be undone by the first step;
      * a file written for another cell model is refused. The reference's own
        reader refuses it ("IMPs do not match region") and the tissue solver
        makes that fatal;
      * a parameter stored in the file is NOT applied, and the difference is
        reported. The parameters of a run come from the model defaults and the
        `im_param` modifiers, so a state file must not change them silently;
      * the mapper reads the key, and says so when the file reaches no node.

    The cable is deliberately tiny (a 16-node cable of two ionic regions, one
    step) and CPU-only: the key acts once, while the run is built, so a small
    case exercises exactly the same code path as a large one.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.IO.writers.svfilewriter import SvFileWriter
from gpuSolve.ionic.ms2v import MitchellSchaeffer2v
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner
from gpuSolve.carp_compatibility.svlayouts import IMP_DATA_NAMES
from gpuSolve.carp_compatibility.svstatefile import SvStateFile


_NELEM  = 15
_NPT    = _NELEM + 1
_DX     = 100.0                                   # micrometres
_SPLIT  = 8                                       # first node of the second region
_FILE_H = 0.42                                    # the h the state file holds
_FILE_V = -47.5                                   # the Vm the state file holds, in mV


def _write_cable(folder: str):
    """A uniform 1D cable of two element tags, 1 and 2, in the three-file
    external mesh format."""
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * _DX))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(iel, iel + 1, 1 if iel < _SPLIT else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


def _write_par(folder: str, extra: str = ''):
    """Two ionic regions of the same model, the second one conditioned by a
    state file. One step is enough: the key acts while the run is built."""
    with open(os.path.join(folder, 'cable.par'), 'w') as fout:
        fout.write('meshname        = cable\n'
                   'simID           = OUT\n'
                   'tend            = 0.025\n'
                   'dt              = 25.0\n'
                   'spacedt         = 0.025\n'
                   'timedt          = 100.0\n'
                   'bidm_eqv_mono   = 0\n'
                   'num_gregions    = 1\n'
                   'num_imp_regions = 2\n'
                   'imp_region[0].im       = "MitchellSchaeffer"\n'
                   'imp_region[0].num_IDs  = 1\n'
                   'imp_region[0].ID[0]    = 1\n'
                   'imp_region[1].im       = "MitchellSchaeffer"\n'
                   'imp_region[1].num_IDs  = 1\n'
                   'imp_region[1].ID[0]    = 2\n'
                   'imp_region[1].im_sv_init = "region1.sv"\n'
                   + extra)


def _write_state_file(folder: str, fname: str = 'region1.sv', section: str = None,
                      h: float = _FILE_H, Vm: float = _FILE_V, tau_in: float = None):
    """A state file in the reference's layout for MitchellSchaeffer: the
    parameters the model itself holds, and a chosen h and Vm."""
    svfile = SvStateFile({'model': MitchellSchaeffer2v(dt=0.025),
                          'imp_name': 'MitchellSchaeffer'})
    name, entries, prefix = svfile.sections()[0]
    values : list = []
    for entry, kind, source, scale, _gate in entries:
        if kind == 'state':
            values.append((entry, h))
        elif entry == 'tau_in' and tau_in is not None:
            values.append((entry, tau_in))
        else:
            values.append((entry, svfile.entry_value(kind, prefix, source) * scale))
    known  = {'Vm': Vm}
    writer = SvFileWriter({'fname': os.path.join(folder, fname)})
    writer.write([(gname, known.get(gname)) for gname in IMP_DATA_NAMES],
                 [(section if section is not None else name, values)])


def _build(folder: str) -> SimulationRunner:
    """Builds the run described by cable.par without stepping it: the state
    file is applied while the run is built."""
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        mapper = ParameterMapper()
        mapper.resolve(OptionReader().read(['+F', 'cable.par']))
        runner = SimulationRunner({'verbose': False})
        runner.set_mapper(mapper)
        runner.build()
    finally:
        os.chdir(cwd)
    return(runner)


@pytest.fixture(scope='module')
def conditioned(tmp_path_factory) -> SimulationRunner:
    """One build with a state file on the second region; most tests read it."""
    folder = str(tmp_path_factory.mktemp('im_sv_init'))
    _write_cable(folder)
    _write_par(folder)
    _write_state_file(folder)
    return(_build(folder))


# ---- what the file does to the mesh -----------------------------------------
def test_the_state_reaches_the_region_that_names_the_file(conditioned):
    """Every node of the second region holds the h of the file, and no node of
    the first one does."""
    checkpoint = conditioned.model().checkpoint()
    h = np.reshape(np.asarray(checkpoint['state_variables']['H_state']), (-1,))
    assert np.all(h[_SPLIT + 1:] == pytest.approx(_FILE_H))
    assert np.all(np.abs(h[:_SPLIT] - _FILE_H) > 1.0e-3)


def test_the_potential_comes_from_the_file_too(conditioned):
    """Vm is part of the state the reference spreads: the conditioned nodes
    start from the file's potential, the others from the model's rest."""
    checkpoint = conditioned.model().checkpoint()
    Vm = np.reshape(np.asarray(checkpoint['Vm']), (-1,))
    resting = float(np.reshape(np.asarray(
        conditioned.model().ionic_model().get_parameter('vmin')), (-1,))[0])
    assert np.all(Vm[_SPLIT + 1:] == pytest.approx(_FILE_V, abs=1.0e-4))
    assert np.all(Vm[:_SPLIT] == pytest.approx(resting, abs=1.0e-4))


def test_the_nodes_of_the_first_region_stay_at_rest(conditioned):
    """The nodes the file does not claim keep the state the model starts from
    (h = 1 at rest), so one region's file cannot condition the whole mesh."""
    checkpoint = conditioned.model().checkpoint()
    h = np.reshape(np.asarray(checkpoint['state_variables']['H_state']), (-1,))
    assert np.all(h[:_SPLIT] == pytest.approx(1.0))


# ---- what the file is not allowed to do -------------------------------------
def test_a_file_written_for_another_model_is_refused(tmp_path):
    """The reference logs "IMPs do not match region" and its tissue solver
    exits; a file read into the wrong model would otherwise be nonsense."""
    folder = str(tmp_path)
    _write_cable(folder)
    _write_par(folder)
    _write_state_file(folder, section='Courtemanche')
    with pytest.raises(ValueError, match='the state file holds Courtemanche'):
        _build(folder)


def test_a_parameter_in_the_file_is_reported_and_not_applied(tmp_path):
    """A state file also stores the parameters of the run that wrote it. They
    are not applied, because the parameters come from the defaults and from
    im_param; the difference is reported instead of being silent."""
    folder = str(tmp_path)
    _write_cable(folder)
    _write_par(folder)
    _write_state_file(folder, tau_in=0.123)
    runner = _build(folder)
    notes  = ' | '.join(runner.mapper().notes())
    assert 'tau_in' in notes
    assert 'ignored' in notes
    used = float(np.reshape(np.asarray(
        runner.model().ionic_model().get_parameter('tau_in')), (-1,))[0])
    assert used != pytest.approx(0.123)


# ---- the mapper --------------------------------------------------------------
def test_the_mapper_reads_the_key():
    """The key is registered, and reported with the region's tags and plugins."""
    mapper = ParameterMapper()
    mapper.resolve({'num_imp_regions':          '2',
                    'imp_region[0].im':         'MitchellSchaeffer',
                    'imp_region[0].num_IDs':    '1',
                    'imp_region[0].ID[0]':      '1',
                    'imp_region[1].im':         'MitchellSchaeffer',
                    'imp_region[1].num_IDs':    '1',
                    'imp_region[1].ID[0]':      '2',
                    'imp_region[1].im_sv_init': 'region1.sv'})
    entries = mapper.state_init_files({1, 2})
    assert len(entries) == 1
    assert entries[0]['index'] == 1
    assert entries[0]['file'] == 'region1.sv'
    assert entries[0]['tags'] == [2]
    assert entries[0]['plugins'] == []


def test_a_file_on_a_region_with_no_tag_of_the_mesh_is_noted():
    """A file that reaches no node is almost always a mistake in the tag list,
    so the run says so instead of starting silently from rest."""
    mapper = ParameterMapper()
    mapper.resolve({'num_imp_regions':          '2',
                    'imp_region[0].im':         'MitchellSchaeffer',
                    'imp_region[0].num_IDs':    '1',
                    'imp_region[0].ID[0]':      '1',
                    'imp_region[1].im':         'MitchellSchaeffer',
                    'imp_region[1].num_IDs':    '1',
                    'imp_region[1].ID[0]':      '7',
                    'imp_region[1].im_sv_init': 'region1.sv'})
    assert mapper.state_init_files({1, 2}) == []
    assert any('im_sv_init' in note for note in mapper.notes())


def test_no_key_means_no_work():
    """The default is empty, as in the reference: nothing to read, nothing to
    report."""
    mapper = ParameterMapper()
    mapper.resolve({'num_imp_regions':       '1',
                    'imp_region[0].im':      'MitchellSchaeffer'})
    assert mapper.value('imp_region[0].im_sv_init') == ''
    assert mapper.state_init_files({1}) == []
