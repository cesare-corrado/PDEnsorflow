#!/usr/bin/env python
"""
    Entry point of the `PDEnsorflow` executable: a parameter-file front end for
    the finite-element monodomain and heat solvers.

        PDEnsorflow +F run.par
        PDEnsorflow +F run.par -meshname atrium -tend 500
        PDEnsorflow -dt 25 +F run.par +Save resolved.par

    Options are read from left to right and the last definition of a key wins,
    so a flag placed BEFORE `+F` is overridden by the file and one placed after
    it is not. `+Save` writes the resolved parameter set back out, which is the
    quickest way to see what a command line actually asked for.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import sys

# Keep TensorFlow quiet before anything imports it. The gpuSolve import below
# pulls it in transitively through the cell models.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gpuSolve
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.parametermapper import REGISTRY
from gpuSolve.carp_compatibility.parametermapper import IONIC_MODELS
from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner


USAGE = """PDEnsorflow [+F file | --file file] [+Save file] [-key value] ...

  +F file, --file file   read a .par parameter file at this position
  +Save file, --save file  write the resolved parameters back out
  +Help, --help          print this message

Options are applied strictly left to right, so the LAST definition of a key
wins whether it came from a file or from the command line."""


def print_usage():
    """ print_usage() prints the command line summary and every key the front
        end understands, so an unknown key can be tracked down without the
        source
    """
    print(USAGE)
    print('\nParameters ([] stands for an array index):\n')
    for pattern in sorted(REGISTRY.keys()):
        vtype, default, actuated = REGISTRY[pattern]
        shown = 'derived' if default is None else default
        print('  {:34s} {:6s} default {:<12} {}'.format(
            pattern, vtype, str(shown), '' if actuated else '(accepted, not actuated)'))
    print('\nCell models for imp_region[].im: {}'.format(', '.join(sorted(IONIC_MODELS.keys()))))


def print_banner(mapper: ParameterMapper):
    """ print_banner(mapper) reports the version, the compute device and every
        place where the resolved parameters ask for something this solver does
        differently
    """
    import tensorflow as tf
    print('PDEnsorflow {}'.format(gpuSolve.version()))
    if tf.config.list_physical_devices('GPU'):
        print('GPU device')
    else:
        print('CPU device')
    print('Tensorflow version is: {0}'.format(tf.__version__))
    for note in mapper.notes():
        print('  NOTE: {}'.format(note), flush=True)


def main(argv: list = None) -> int:
    """ main(argv) reads the parameters, builds the simulation and runs it.
        argv excludes the program name; it defaults to sys.argv[1:].
        Returns 0 on success and 1 when the parameters cannot be read.
    """
    arguments = sys.argv[1:] if argv is None else argv
    reader    = OptionReader()
    mapper    = ParameterMapper()
    try:
        store = reader.read(arguments)
        if reader.help_requested():
            print_usage()
            return(0)
        mapper.resolve(store)
        if reader.save_fname() is not None:
            reader.write_save_file()
    except Exception as err:
        # the reference prints the offending keyword and stops before doing any
        # work, which is the useful behaviour: a typo must not start a run
        print('\n*** {}\n\n*** Error reading parameters'.format(err), file=sys.stderr)
        return(1)

    print_banner(mapper)
    runner = SimulationRunner()
    runner.set_mapper(mapper)
    try:
        runner.build()
        runner.run()
    except (ValueError, OSError) as err:
        # a bad input: say what is wrong and stop. Anything else is a defect in
        # the library and keeps its traceback, which is what a bug report needs
        print('\n*** {}\n\n*** Error running the simulation'.format(err), file=sys.stderr)
        return(1)
    print('results written to {}'.format(runner.outdir()), flush=True)
    return(0)


if __name__ == '__main__':
    sys.exit(main())
