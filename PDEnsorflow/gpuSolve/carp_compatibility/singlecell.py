#!/usr/bin/env python
"""
    Entry point of the `singlecell` executable: single-cell experiments with
    the gpuSolve ionic models, driven by the options of the reference
    single-cell tool (bench), with the same spelling:

        singlecell --imp Courtemanche --imp-par "GKr*1.6" --stim-curr 20 \\
                   --numstim 4 --bcl 1000
        singlecell --imp Tomek --numstim 50 --bcl 1000 -F paced.sv -S 49000
        singlecell --imp Tomek --read-ini-file paced.sv --duration 1000 -v

    A bench option singlecell does not implement yet stops the run with a
    message, so a command line copied from a bench script never runs a
    different experiment without saying so. `singlecell --help` lists what is
    implemented.

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
"""
import os
import sys

# Only modules that do not start TensorFlow are imported here: the device and
# the thread count are chosen from the options, and both must be in the
# environment before TensorFlow starts.
from gpuSolve._version import __version__
from gpuSolve.carp_compatibility.singlecelloptionreader import SingleCellOptionReader
from gpuSolve.carp_compatibility.singlecelloptionreader import PROGRAM_NAME


# --target values, as bench spells them, and the device each one runs on.
# They name bench's code generators (plain C++, or MLIR for the CPU, NVIDIA and
# AMD GPUs); gpuSolve always runs TensorFlow, so only the hardware carries
# over. There is no ROCm build of gpuSolve, so mlir-rocm is not listed.
TARGET_DEVICES = {'auto': 'cpu', 'cpu': 'cpu', 'mlir-cpu': 'cpu', 'mlir-cuda': 'gpu'}


def version_text() -> str:
    """ version_text() returns the --version line """
    return('{} (PDEnsorflow) {}'.format(PROGRAM_NAME, '.'.join(__version__)))


def check_options(reader: SingleCellOptionReader):
    """ check_options(reader) raises ValueError for a bench option singlecell
        does not implement, and for a value it cannot honour
    """
    unsupported = reader.unsupported_options()
    if len(unsupported) > 0:
        raise ValueError('{}: {} {} bench option{} that singlecell does not support yet; '
                         'see {} --help'.format(
                             PROGRAM_NAME, ', '.join('--{}'.format(name) for name in unsupported),
                             'is a' if len(unsupported) == 1 else 'are',
                             '' if len(unsupported) == 1 else 's', PROGRAM_NAME))
    if reader.value('num') != 1:
        raise ValueError('{}: --num {}: singlecell runs one cell; several cells are not '
                         'supported yet'.format(PROGRAM_NAME, reader.value('num')))
    if reader.value('target') not in TARGET_DEVICES:
        raise ValueError('Unkown target: {}\nAvailable targets are: {}'.format(
            reader.value('target'), ', '.join(TARGET_DEVICES.keys())))


def prepare_environment(reader: SingleCellOptionReader):
    """ prepare_environment(reader) sets the device and the thread count for
        TensorFlow, which must not have started yet
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # one thread: on arrays of one element the thread pool costs more than it
    # gives (Courtemanche, 1 s of activity: 1.57 s with the default pool,
    # 0.97 s with one thread). This process only; tissue runs are not affected.
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    if TARGET_DEVICES[reader.value('target')] == 'cpu':
        # one cell runs faster on the CPU than on a GPU (see SingleCellRunner)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def print_banner(reader: SingleCellOptionReader):
    """ print_banner(reader) reports the version and the compute device, and
        stops when --target mlir-cuda asks for a GPU that is not there
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if TARGET_DEVICES[reader.value('target')] == 'gpu' and len(gpus) == 0:
        raise ValueError('--target {} asks for a GPU, but TensorFlow sees none'.format(
            reader.value('target')))
    print(version_text(), file=sys.stderr)
    print('{} device'.format('GPU' if len(gpus) > 0 else 'CPU'), file=sys.stderr)
    print('Tensorflow version is: {0}'.format(tf.__version__), file=sys.stderr, flush=True)


def main(argv: list = None) -> int:
    """ main(argv) reads the options, runs the experiment and returns 0, or 1
        when the options or the inputs are wrong. argv excludes the program
        name; it defaults to sys.argv[1:].
    """
    arguments = sys.argv[1:] if argv is None else argv
    reader = SingleCellOptionReader()
    try:
        reader.read(arguments)
        if reader.help_requested():
            print(reader.usage_text())
            return(0)
        if reader.version_requested():
            print(version_text())
            return(0)
        check_options(reader)
    except ValueError as err:
        print(err, file=sys.stderr)
        return(1)
    prepare_environment(reader)
    from gpuSolve.carp_compatibility import singlecellrunner

    try:
        if reader.value('list-imps'):
            print(singlecellrunner.list_models())
            return(0)
        if reader.value('plugin-outputs'):
            print(singlecellrunner.plugin_outputs())
            return(0)
        if reader.value('imp-info'):
            print(singlecellrunner.imp_info(reader.value('imp'), reader.value('plug-in')))
            return(0)
        print_banner(reader)
        if reader.value('buildinfo'):
            print('\n*** --buildinfo is deprecated in bench; the build information is the banner '
                  'above\n', file=sys.stderr)
            return(0)
        runner = singlecellrunner.SingleCellRunner()
        runner.set_options(reader)
        runner.build()
        for note in runner.notes():
            print('NOTE: {}'.format(note), file=sys.stderr)
        for warning in runner.warnings():
            print('WARNING: {}'.format(warning), file=sys.stderr)
        print('Running simulation on target {}'.format(reader.value('target')), file=sys.stderr,
              flush=True)
        runner.run()
    except ValueError as err:
        # a bad input: say what is wrong and stop. Anything else is a defect in
        # the library and keeps its traceback, which is what a bug report needs
        print('\n*** {}\n'.format(err), file=sys.stderr)
        return(1)
    # repeated at the end, where a long run leaves the reader
    for warning in runner.warnings():
        print('WARNING: {}'.format(warning), file=sys.stderr)
    simulated = runner.last_step() * runner.dt()
    print('\n\nAll done!\n', file=sys.stderr)
    print('main loop time      {:.6f} s'.format(runner.elapsed()), file=sys.stderr)
    if runner.elapsed() > 0.0:
        print('real time factor    {:.6f}\n'.format(simulated / (runner.elapsed() * 1.0e3)),
              file=sys.stderr, flush=True)
    return(0)


if __name__ == '__main__':
    sys.exit(main())
