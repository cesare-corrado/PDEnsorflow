#!/usr/bin/env python
"""
    imp_region[].im_sv_init: start a tissue simulation from a single-cell state
    file, the equivalent of the reference simulator's key of the same name.

    The example is built so that the answer is known in advance. A single cell
    is paced with `singlecell`, its state is saved 30 ms into the action
    potential, and a cable is then started from that file with no stimulus at
    all. The initial state is the same at every node, so the diffusion term is
    exactly zero (a stiffness matrix annihilates a constant field) and every
    cell of the cable must follow the single-cell trajectory that continues from
    the same file. The comparison is therefore quantitative: the cable's Vm(t)
    against the cell's Vm(t), millivolt by millivolt.

    The same run without the key is the control: it stays at rest, which is what
    a key that silently did nothing would give.

        E=/home/cc14/Libraries/anaconda3/envs/Claude_testing
        cd PDEnsorflow/Tests/FEM/StateInit
        $E/bin/python state_init.py

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

from gpuSolve.carp_compatibility.main import main as tissue_main
from gpuSolve.carp_compatibility.singlecell import main as cell_main
from gpuSolve.IO.readers import IGBReader


NELEM   = 20
NPT     = NELEM + 1
DX      = 100.0                                   # micrometres
DT      = 0.01                                    # ms, both simulations
SAVE_AT = 30.0                                    # ms into the action potential
TAIL    = 170.0                                   # ms simulated from the file
PROBE   = NPT // 2                                # the node the trace is read at


def write_cable():
    """A uniform 1D cable in the three-file external mesh format, in
    micrometres, with a single element tag."""
    with open('cable.pts', 'w') as fout:
        fout.write('{}\n'.format(NPT))
        for ipt in range(NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * DX))
    with open('cable.elem', 'w') as fout:
        fout.write('{}\n'.format(NELEM))
        for iel in range(NELEM):
            fout.write('Ln {} {} 1\n'.format(iel, iel + 1))
    with open('cable.lon', 'w') as fout:
        fout.write('1\n')
        for _iel in range(NELEM):
            fout.write('1.0 0.0 0.0\n')


def write_par(fname: str, outdir: str, state_file: str = ''):
    """The cable, run with no stimulus for TAIL ms. With state_file it starts
    from that state everywhere; without it, from the model's rest state."""
    with open(fname, 'w') as fout:
        fout.write('meshname        = cable\n'
                   'simID           = {}\n'
                   'tend            = {}\n'
                   'dt              = {}            # microseconds\n'
                   'spacedt         = 1.0\n'
                   'timedt          = 50.0\n'
                   'bidm_eqv_mono   = 0\n'
                   'num_stim        = 0\n'
                   'num_gregions    = 1\n'
                   'gregion[0].g_il = 0.01\n'
                   'gregion[0].g_it = 0.01\n'
                   'num_imp_regions = 1\n'
                   'imp_region[0].name = "cable"\n'
                   'imp_region[0].im   = "MitchellSchaeffer"\n'
                   'imp_region[0].cellSurfVolRatio = 1.0\n'.format(outdir, TAIL, DT * 1000.0))
        if len(state_file) > 0:
            fout.write('imp_region[0].im_sv_init = "{}"\n'.format(state_file))


def cell_trace(fname: str) -> np.ndarray:
    """The Vm column of a `singlecell` output table (Time Vm Iion; the file
    the reference writes carries no header line)."""
    return(np.loadtxt(fname, usecols=1))


def run_single_cell():
    """One paced cell: the state at SAVE_AT ms goes to cell.sv, and the
    trajectory that continues from it is the reference for the cable."""
    status = cell_main(['--imp', 'MitchellSchaeffer', '--dt', str(DT),
                        '--numstim', '1', '--stim-start', '1.0', '--stim-dur', '1.0',
                        '--stim-curr', '60.0', '--duration', str(SAVE_AT),
                        '--save-ini-file', 'cell.sv', '--save-ini-time', str(SAVE_AT),
                        '--fout=cell_paced', '--dt-out', '1.0', '--no-trace'])
    if status != 0:
        raise RuntimeError('the paced single cell failed')
    status = cell_main(['--imp', 'MitchellSchaeffer', '--dt', str(DT),
                        '--numstim', '0', '--duration', str(TAIL),
                        '--read-ini-file', 'cell.sv',
                        '--fout=cell_tail', '--dt-out', '1.0', '--no-trace'])
    if status != 0:
        raise RuntimeError('the single cell continued from cell.sv failed')


def run_cable(par: str, outdir: str, state_file: str = '') -> np.ndarray:
    """One cable run; returns its potential as (frame, node)."""
    write_par(par, outdir, state_file)
    if tissue_main(['+F', par]) != 0:
        raise RuntimeError('the cable run of {} failed'.format(par))
    reader = IGBReader()
    reader.read(os.path.join(outdir, 'vm.igb'))
    return(np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x']))


def main():
    write_cable()
    run_single_cell()
    reference = cell_trace('cell_tail.txt')

    conditioned = run_cable('state_init.par', 'OUT_sv_init', 'cell.sv')
    control     = run_cable('rest.par', 'OUT_rest')

    frames = min(reference.shape[0], conditioned.shape[0])
    cable  = conditioned[:frames, PROBE]
    cell   = reference[:frames]
    spread = np.max(np.abs(conditioned[0, :] - conditioned[0, 0]))

    print('')
    print('state file cell.sv written at t = {:g} ms'.format(SAVE_AT))
    print('cable      : {} nodes, {} frames, no stimulus'.format(NPT, frames))
    print('t = 0      : Vm = {:.6f} mV at every node (spread {:.2e} mV)'.format(
        conditioned[0, 0], spread))
    print('             the control run starts at {:.6f} mV (the model rest state)'.format(
        control[0, 0]))
    print('cable vs the single cell continued from the same file, over {:g} ms:'.format(
        (frames - 1) * 1.0))
    print('             max |dV| = {:.3e} mV, mean |dV| = {:.3e} mV'.format(
        float(np.max(np.abs(cable - cell))), float(np.mean(np.abs(cable - cell)))))
    print('cable vs the control run (no im_sv_init), same window:')
    print('             max |dV| = {:.3e} mV'.format(
        float(np.max(np.abs(conditioned[:frames, PROBE] - control[:frames, PROBE])))))
    print('Vm at the probe: {:.3f} -> {:.3f} mV (cell: {:.3f} -> {:.3f} mV)'.format(
        cable[0], cable[-1], cell[0], cell[-1]))


if __name__ == '__main__':
    main()
