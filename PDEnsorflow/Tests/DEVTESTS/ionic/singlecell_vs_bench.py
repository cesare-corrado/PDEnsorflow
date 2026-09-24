#!/usr/bin/env python
"""
    A TensorFlow-based Cardiac Electrophysiology Modeler

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

    Runs the reference single-cell tool (bench, which must be on the PATH) and
    the `singlecell` executable with the same command lines, for every cell
    model the two share, and compares the traces: largest |Vm difference|,
    APD90 and peak of the last beat, and the wall time of each. Then it swaps
    state files between the two (each continues from the other's saved state)
    to check that the files are interchangeable.

    Everything is written to a scratch folder (a new temporary folder unless
    --workdir is given), never to the repository.

        python singlecell_vs_bench.py [--beats 2] [--bcl 1000] [--workdir DIR]
"""

import os
import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

# Command lines shared by both programs, per model. MitchellSchaeffer takes the
# reference's defaults explicitly (gpuSolve's differ); Tomek runs with the
# reference's integration scheme, which bench always uses.
CASES = (
    ('Courtemanche',        ['--imp', 'Courtemanche'], []),
    ('tenTusscherPanfilov', ['--imp', 'tenTusscherPanfilov'], []),
    ('Tomek',               ['--imp', 'Tomek'], ['--reference-scheme']),
    ('MitchellSchaeffer',   ['--imp', 'MitchellSchaeffer', '--imp-par', 'V_min=0,V_max=1,tau_out=5',
                             '--stim-curr', '0.5'], []),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='singlecell against bench, model by model')
    parser.add_argument('--beats', type=int, default=2, help='number of paced beats')
    parser.add_argument('--bcl', type=float, default=1000.0, help='basic cycle length (ms)')
    parser.add_argument('--workdir', default='', help='scratch folder (default: a new temporary one)')
    return(parser.parse_args())


def run(command: list, folder: str) -> float:
    """ runs command in folder, stdout to out.txt; returns the wall time """
    os.makedirs(folder, exist_ok=True)
    start = time.time()
    with open(os.path.join(folder, 'out.txt'), 'w') as fout, \
         open(os.path.join(folder, 'err.txt'), 'w') as ferr:
        status = subprocess.run(command, cwd=folder, stdout=fout, stderr=ferr).returncode
    if status != 0:
        raise RuntimeError('{} failed in {} (see err.txt)'.format(' '.join(command), folder))
    return(time.time() - start)


def table(folder: str) -> np.ndarray:
    """ the `Time Vm Iion` rows of out.txt """
    rows = [line.split() for line in open(os.path.join(folder, 'out.txt'))
            if len(line.split()) == 3 and line.split()[0][:1].isdigit()]
    return(np.array(rows, dtype=np.float64))


def last_beat(trace: np.ndarray, beats: int, bcl: float) -> tuple:
    """ (APD90, peak) of the last beat, on the output sampling """
    start = (beats - 1) * bcl
    mask = (trace[:, 0] >= start) & (trace[:, 0] < start + bcl)
    t, V = trace[mask, 0], trace[mask, 1]
    threshold = V.max() - 0.9 * (V.max() - V[0])
    up   = int(np.argmax(V > threshold))
    down = up + int(np.argmax(V[up:] < threshold))
    return(float(t[down] - t[up]), float(V.max()))


def main():
    args = parse_arguments()
    if shutil.which('bench') is None:
        sys.exit('bench is not on the PATH')
    singlecell = shutil.which('singlecell') or os.path.join(os.path.dirname(sys.executable), 'singlecell')
    workdir = args.workdir if len(args.workdir) > 0 else tempfile.mkdtemp(prefix='singlecell_vs_bench_')
    duration = args.beats * args.bcl
    pacing = ['--numstim', str(args.beats), '--bcl', str(args.bcl), '--duration', str(duration),
              '--no-trace']
    print('scratch folder: {}'.format(workdir))
    print('{:20s} {:>12s} {:>14s} {:>14s} {:>16s} {:>9s} {:>11s}'.format(
        'model', 'max|dVm|', 'APD90 bench', 'APD90 ours', 'peak bench/ours', 'bench s', 'singlecell s'))
    for name, options, extra in CASES:
        folder = os.path.join(workdir, name)
        t_bench = run(['bench'] + options + pacing, os.path.join(folder, 'bench'))
        t_ours  = run([singlecell] + options + extra + pacing, os.path.join(folder, 'singlecell'))
        b, s = table(os.path.join(folder, 'bench')), table(os.path.join(folder, 'singlecell'))
        if not np.array_equal(b[:, 0], s[:, 0]):
            raise RuntimeError('{}: the output times differ'.format(name))
        apd_b, peak_b = last_beat(b, args.beats, args.bcl)
        apd_s, peak_s = last_beat(s, args.beats, args.bcl)
        print('{:20s} {:12.3g} {:14.0f} {:14.0f} {:8.3f}/{:<8.3f} {:9.2f} {:11.2f}'.format(
            name, np.abs(b[:, 1] - s[:, 1]).max(), apd_b, apd_s, peak_b, peak_s, t_bench, t_ours))

    # state files: each program saves at the end of the paced run, and each
    # then continues for one beat from the other's file and from its own
    print('\nstate files (Tomek, reference scheme): one more beat from each saved state')
    folder = os.path.join(workdir, 'svswap')
    save = ['--imp', 'Tomek'] + pacing + ['-F', 'paced.sv', '-S', str(duration)]
    run(['bench'] + save, os.path.join(folder, 'bench_save'))
    run([singlecell, '--reference-scheme'] + save, os.path.join(folder, 'ours_save'))
    resume = ['--imp', 'Tomek', '--numstim', '1', '--bcl', str(args.bcl), '--duration', str(args.bcl),
              '--no-trace']
    traces = {}
    for runner, flags in (('bench', []), ('ours', ['--reference-scheme'])):
        for source in ('bench_save', 'ours_save'):
            key = '{}_from_{}'.format(runner, source)
            program = 'bench' if runner == 'bench' else singlecell
            run([program] + flags + resume + ['--read-ini-file', os.path.join(folder, source, 'paced.sv')],
                os.path.join(folder, key))
            traces[key] = table(os.path.join(folder, key))
    for a, b in (('bench_from_bench_save', 'bench_from_ours_save'),
                 ('ours_from_ours_save', 'ours_from_bench_save'),
                 ('bench_from_bench_save', 'ours_from_ours_save')):
        print('  {:24s} vs {:24s} max|dVm| = {:.3g} mV'.format(
            a, b, np.abs(traces[a][:, 1] - traces[b][:, 1]).max()))


if __name__ == '__main__':
    main()
