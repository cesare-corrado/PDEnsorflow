#!/usr/bin/env python
"""
    GPU sanity gate for the Tier-2 nightly workflow.

    Imports gpuSolve FIRST (its __init__ fixes LD_LIBRARY_PATH so TensorFlow can
    dlopen the CUDA libraries and actually see the device), then asserts that at
    least one physical GPU is visible. Exits non-zero if not, so the scheduled
    job fails loudly instead of silently falling back to CPU (which would let the
    device-gated csr_axpby native-path cases skip again and defeat the purpose of
    running on the self-hosted GPU runner).

    Run it as a SCRIPT (never `python -c "import gpuSolve"`, which the __init__
    re-exec breaks). Requires an environment where LD_LIBRARY_PATH is already set
    at process start (the conda env's activate.d CUDA shim), so importing gpuSolve
    does not re-exec the interpreter mid-run.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gpuSolve                       # noqa: F401  (import side effect: CUDA LD_LIBRARY_PATH)
import tensorflow as tf


def main() -> int:
    gpus = tf.config.list_physical_devices('GPU')
    print('TensorFlow {0}; visible GPUs: {1}'.format(tf.__version__, gpus))
    if not gpus:
        print('ERROR: no physical GPU visible -- refusing to run the nightly suite on CPU.',
              file=sys.stderr)
        return(1)
    return(0)


if __name__ == '__main__':
    sys.exit(main())
