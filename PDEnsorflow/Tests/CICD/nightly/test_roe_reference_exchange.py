#!/usr/bin/env python
"""
    Tier-2 nightly test of the binary state files (`.roe`) exchanged with the
    reference simulator: the same parameter file is run by the reference and by
    the parameter-file front end, each saving its state at 10 ms, and each then
    restarts from the OTHER's file.

    On a 525-node slab with two ionic regions (tag 2 on the half x > 0):
      * the two runs write the same file names, and their files have the same
        format, region table, record sizes and region mask (node by node);
      * the reference restarted from gpuSolve's file continues gpuSolve's
        uninterrupted run, and gpuSolve restarted from the reference's file
        continues the reference's: max |dVm| over 10..20 ms within 0.05 mV.
        The two simulators differ from each other by about 0.2 mV over the same
        interval (their solvers differ), so a restart that took the wrong state
        or the wrong nodes would show here. Measured: 0.008 mV (Tomek) and
        0.005 mV (tenTusscherPanfilov with the electroporation plugin).
    The tenTusscherPanfilov case carries a plugin on one region, whose region
    table release 19.0 of the reference gets wrong (see RoeReader); the mMS
    case runs only where the reference has that user model (bench
    --list-imps), with its parameters given explicitly because the two
    simulators' defaults differ.

    The whole module is skipped, without an error, on a machine where the
    reference (`openCARP`, `mesher`) is not installed. Marked nightly.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import shutil
import subprocess

import numpy as np
import pytest

from gpuSolve.carp_compatibility.main import main
from gpuSolve.IO.readers import IGBReader
from gpuSolve.IO.readers.roereader import RoeReader

pytestmark = pytest.mark.skipif(shutil.which('openCARP') is None or shutil.which('mesher') is None,
                                reason='the reference simulator (openCARP, mesher) is not installed')

_TOL     = 0.05          # mV, a restart against the run that wrote its file
_RESTART = 10.0          # ms
_TEND    = 20.0          # ms
_SPACEDT = 0.1           # ms

# one parameter file for both simulators
_PAR = """meshname  = slab
bidomain  = 0
dt        = 25
timedt    = 1
spacedt   = {spacedt}
num_gregions           = 1
gregion[0].num_IDs     = 2
gregion[0].ID[0]       = 1
gregion[0].ID[1]       = 2
num_imp_regions        = 2
imp_region[0].num_IDs  = 1
imp_region[0].ID[0]    = 1
imp_region[1].num_IDs  = 1
imp_region[1].ID[0]    = 2
num_stim               = 1
stim[0].crct.type      = 0
stim[0].ptcl.start     = 1.0
stim[0].ptcl.duration  = 2.0
stim[0].pulse.strength = 60.0
stim[0].elec.p0[0]     = -1100
stim[0].elec.p0[1]     = -300
stim[0].elec.p0[2]     = -300
stim[0].elec.p1[0]     = -850
stim[0].elec.p1[1]     = 300
stim[0].elec.p1[2]     = 300
num_phys_regions       = 1
phys_region[0].name    = "myocard"
phys_region[0].ptype   = 0
phys_region[0].num_IDs = 2
phys_region[0].ID[0]   = 1
phys_region[0].ID[1]   = 2
"""

_MMS_PARAM = 'V_min=-80,V_max=20,V_gate=0.1,a_crit=0.1,tau_in=0.1,tau_out=9,tau_open=100,tau_close=120'

CASES = {
    'Tomek': ['-imp_region[0].im', 'Tomek', '-imp_region[1].im', 'Tomek'],
    'tenTusscherPanfilov': ['-imp_region[0].im', 'tenTusscherPanfilov',
                            '-imp_region[1].im', 'tenTusscherPanfilov',
                            '-imp_region[1].plugins', 'Electroporation_DeBruinKrassowska98'],
    'mMS': ['-imp_region[0].im', 'mMS', '-imp_region[1].im', 'mMS',
            '-imp_region[0].im_param', _MMS_PARAM, '-imp_region[1].im_param', _MMS_PARAM],
}


def _reference_models() -> str:
    """ the listing of the reference's cell models """
    result = subprocess.run(['bench', '--list-imps'], capture_output=True, text=True)
    return(result.stdout + result.stderr)


@pytest.fixture(scope='module')
def slab(tmp_path_factory) -> str:
    """ a 2 x 0.4 x 0.4 mm slab at 100 um, tag 2 on the half x > 0 """
    folder = str(tmp_path_factory.mktemp('roe_exchange'))
    subprocess.check_call(['mesher', '-size[0]', '0.2', '-size[1]', '0.04', '-size[2]', '0.04',
                           '-resolution[0]', '100', '-resolution[1]', '100',
                           '-resolution[2]', '100', '-mesh', 'slab'],
                          cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pts   = np.loadtxt(os.path.join(folder, 'slab.pts'), skiprows=1)
    lines = open(os.path.join(folder, 'slab.elem')).read().split('\n')
    out   = [lines[0]]
    for line in lines[1:int(lines[0]) + 1]:
        tokens = line.split()
        tokens[5] = '2' if pts[[int(t) for t in tokens[1:5]], 0].mean() > 0.0 else '1'
        out.append(' '.join(tokens))
    with open(os.path.join(folder, 'slab.elem'), 'w') as felem:
        felem.write('\n'.join(out) + '\n')
    with open(os.path.join(folder, 'slab.par'), 'w') as fpar:
        fpar.write(_PAR.format(spacedt=_SPACEDT))
    return(folder)


def _run_reference(folder: str, argv: list):
    with open(os.path.join(folder, 'reference.log'), 'a') as flog:
        code = subprocess.call(['openCARP', '+F', 'slab.par'] + argv,
                               cwd=folder, stdout=flog, stderr=subprocess.STDOUT)
    assert code == 0, 'the reference failed, see {}/reference.log'.format(folder)


def _run_front(folder: str, argv: list):
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        assert main(['+F', 'slab.par'] + argv) == 0
    finally:
        os.chdir(cwd)


def _frames(path: str) -> np.ndarray:
    reader = IGBReader()
    reader.read(path)
    return(np.array(reader.data()).reshape(reader.nt(), -1))


@pytest.mark.nightly
@pytest.mark.parametrize('case', sorted(CASES.keys()))
def test_states_are_exchanged_with_the_reference(slab, case):
    if case == 'mMS' and 'mMS' not in _reference_models().split():
        pytest.skip('the reference has no mMS model on this machine')
    argv  = CASES[case] + ['-tend', str(_TEND)]
    saves = ['-num_tsav', '1', '-tsav[0]', str(_RESTART)]
    ref, front = 'R_' + case, 'F_' + case
    _run_reference(slab, argv + saves + ['-simID', ref])
    _run_front(slab, argv + saves + ['-simID', front])

    names = {sim: sorted(name for name in os.listdir(os.path.join(slab, sim))
                         if name.endswith('.roe')) for sim in (ref, front)}
    assert names[ref] == names[front] == ['state.10.0.roe']
    stored = {sim: RoeReader().read(os.path.join(slab, sim, 'state.10.0.roe')) for sim in (ref, front)}
    assert stored[ref]['format'] == stored[front]['format'] == 1
    assert stored[ref]['time'] == stored[front]['time']
    table = {sim: [(region['model'], region['size'],
                    [(plugin['name'], plugin['size']) for plugin in region['plugins']])
                   for region in stored[sim]['regions']] for sim in (ref, front)}
    assert table[ref] == table[front]
    np.testing.assert_array_equal(stored[ref]['mask'], stored[front]['mask'])

    _run_reference(slab, argv + ['-simID', ref + '_from_front',
                                 '-start_statef', os.path.join(front, 'state.10.0')])
    _run_front(slab, argv + ['-simID', front + '_from_ref',
                             '-start_statef', os.path.join(ref, 'state.10.0')])
    first   = int(round(_RESTART / _SPACEDT))
    full    = {sim: _frames(os.path.join(slab, sim, 'vm.igb'))[first:] for sim in (ref, front)}
    from_front = _frames(os.path.join(slab, ref + '_from_front', 'vm.igb'))
    from_ref   = _frames(os.path.join(slab, front + '_from_ref', 'vm.igb'))
    between = np.max(np.abs(full[ref] - full[front]))
    print('\n  {}: reference vs front end {:.3f} mV; reference from the front end file vs '
          'front end {:.4f} mV; front end from the reference file vs reference {:.4f} mV'.format(
              case, between, np.max(np.abs(from_front - full[front])),
              np.max(np.abs(from_ref - full[ref]))))
    assert np.max(np.abs(from_front - full[front])) < _TOL
    assert np.max(np.abs(from_ref - full[ref])) < _TOL
