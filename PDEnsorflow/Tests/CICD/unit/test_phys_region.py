#!/usr/bin/env python
"""
    Tier-1 tests of the physics regions (num_phys_regions, phys_region[]) of
    the parameter-file front end.

    The reference makes the physics regions optional: without them it builds
    one intracellular region from the elements with a fibre direction
    (simulator/sim_utils.cc). With them, the intracellular mesh is the elements
    whose tags the intracellular regions (ptype 0) list, or, when none lists an
    ID, again the elements with a fibre direction. gpuSolve solves the
    monodomain equation on myocardium only, so a mesh with bath is refused when
    the physics regions say there is one; without physics regions nothing is
    checked, so that an older parameter file runs exactly as before.

    CPU-only: a 21-node cable, 1 ms.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.carp_compatibility.main import main
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.IO.readers import IGBReader

_NELEM = 20
_NPT   = _NELEM + 1


def _write_cable(folder: str, bath: int = 0):
    """ a cable whose second half carries tag 2; the last `bath` elements have
        no fibre direction
    """
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * 100.0))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(iel, iel + 1, 1 if iel < _NELEM // 2 else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for iel in range(_NELEM):
            fout.write('0.0 0.0 0.0\n' if iel >= _NELEM - bath else '1.0 0.0 0.0\n')


_PAR = ('meshname = cable\ntend = 1.0\ndt = 25\nspacedt = 0.25\ntimedt = 100.0\n'
        'imp_region[0].im = "mMS"\n'
        'gregion[0].g_il = 0.174\ngregion[0].g_it = 0.174\n'
        'num_stim = 1\nstim[0].pulse.strength = 60.0\nstim[0].ptcl.duration = 1.0\n'
        'stim[0].elec.p0[0] = 0.0\nstim[0].elec.p1[0] = 250.0\n')

_INTRA = ['-num_phys_regions', '1', '-phys_region[0].ptype', '0',
          '-phys_region[0].num_IDs', '2', '-phys_region[0].ID[0]', '1',
          '-phys_region[0].ID[1]', '2']


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
    return(np.array(reader.data()))


def _mapper(argv: list) -> ParameterMapper:
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


def test_the_intracellular_region_follows_the_reference():
    assert _mapper([]).intracellular_region() == (False, None)
    assert _mapper(_INTRA).intracellular_region() == (True, {1, 2})
    # no ID at all: the elements with a fibre direction
    assert _mapper(['-num_phys_regions', '1']).intracellular_region() == (True, None)
    # every intracellular region adds its IDs to one set; one without IDs adds nothing
    both = ['-num_phys_regions', '2', '-phys_region[0].ID', '1', '-phys_region[1].ptype', '0']
    assert _mapper(both).intracellular_region() == (True, {1})
    # an extracellular region is noted and ignored
    extra = _mapper(_INTRA[:1] + ['2'] + _INTRA[2:] + ['-phys_region[1].ptype', '1'])
    assert extra.intracellular_region() == (True, {1, 2})
    assert any('extracellular region' in note for note in extra.notes())


@pytest.mark.parametrize('argv,message', [
    (['-num_phys_regions', '1', '-phys_region[0].ptype', '2'], '(eikonal) is not provided'),
    (['-num_phys_regions', '1', '-phys_region[0].ptype', '6'], '(EMI) is not provided'),
    (['-num_phys_regions', '1', '-phys_region[0].ptype', '3'], 'is not a physics type'),
    (['-num_phys_regions', '1', '-phys_region[0].ptype', '1'], 'no phys_region is intracellular'),
])
def test_physics_gpusolve_does_not_provide_is_refused(argv, message):
    with pytest.raises(ValueError, match=message.replace('(', r'\(').replace(')', r'\)')):
        _mapper(argv).intracellular_region()


def test_a_myocardium_only_mesh_runs_as_without_physics_regions(tmp_path):
    folder = str(tmp_path)
    _write_cable(folder)
    with open(os.path.join(folder, 'c.par'), 'w') as fout:
        fout.write(_PAR)
    assert _run_in(folder, ['+F', 'c.par', '-simID', 'OUT_0']) == 0
    assert _run_in(folder, ['+F', 'c.par', '-simID', 'OUT_1'] + _INTRA) == 0
    np.testing.assert_array_equal(_read_igb(os.path.join(folder, 'OUT_1', 'vm.igb')),
                                  _read_igb(os.path.join(folder, 'OUT_0', 'vm.igb')))


@pytest.mark.parametrize('bath,argv,message', [
    (0, ['-num_phys_regions', '1', '-phys_region[0].ID', '1'], 'element tag(s) 2 belong to no'),
    (3, ['-num_phys_regions', '1'], '3 element(s) have no fibre direction'),
])
def test_a_mesh_with_bath_is_refused(tmp_path, capsys, bath, argv, message):
    folder = str(tmp_path)
    _write_cable(folder, bath=bath)
    with open(os.path.join(folder, 'c.par'), 'w') as fout:
        fout.write(_PAR)
    assert _run_in(folder, ['+F', 'c.par', '-simID', 'OUT_B'] + argv) == 1
    assert message in capsys.readouterr().err


def test_without_physics_regions_nothing_is_checked(tmp_path):
    """ an older parameter file keeps its behaviour: elements without a fibre
        direction are solved on as tissue, as before the physics regions were
        read (the reference would call them bath)
    """
    folder = str(tmp_path)
    _write_cable(folder, bath=3)
    with open(os.path.join(folder, 'c.par'), 'w') as fout:
        fout.write(_PAR)
    assert _run_in(folder, ['+F', 'c.par', '-simID', 'OUT_OLD']) == 0
