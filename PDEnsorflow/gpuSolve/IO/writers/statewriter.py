#!/usr/bin/env python
"""
    StateWriter: writes a checkpoint, the state a simulation needs to resume
    from the time it was saved, as a pickled Python dict.

    The dict is the one returned by HeatSolver.checkpoint() /
    MonodomainSolver.checkpoint(); its entries are described in
    gpuSolve.IO.readers.statereader, which reads it back.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import pickle

import numpy as np

from gpuSolve.IO.readers.statereader import CHECKPOINT_KEYS


# Protocol 4 handles objects larger than 4 GB (a large mesh with a detailed
# cell model gets there) and is readable by every Python 3 from 3.4 on.
_PICKLE_PROTOCOL : int = 4


class StateWriter:
    """
    class StateWriter: writes a checkpoint dict to a file.
    """

    def __init__(self, config: dict = None):
        self._fname : str = 'state.pkl'

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def set_fname(self, fname: str):
        """ set_fname(fname) sets the output file name """
        self._fname = fname

    def fname(self) -> str:
        """ fname() returns the output file name """
        return(self._fname)

    def write(self, checkpoint: dict, fname: str = None):
        """ write(checkpoint, fname) writes checkpoint to fname (default: fname()).
            Only plain values are stored (str, float, int and numpy arrays), never
            TensorFlow objects, so the file does not depend on the TensorFlow
            version that wrote it.
        """
        outname = fname if fname is not None else self._fname
        try:
            missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
            if len(missing) > 0:
                raise ValueError('cannot write {}: the checkpoint misses {}'.format(
                    outname, ', '.join(missing)))
            # arrays keep the dtype they were held in, so a float64 build of a
            # cell model restores exactly
            content = {'ionic_model': str(checkpoint['ionic_model']),
                       'time': float(checkpoint['time']),
                       'num_nodes': int(checkpoint['num_nodes']),
                       'Vm': np.reshape(np.array(checkpoint['Vm']), (-1,)),
                       'state_variables': {str(name): np.reshape(np.array(values), (-1,))
                                           for name, values in checkpoint['state_variables'].items()}}
            folder = os.path.dirname(outname)
            if len(folder) > 0:
                os.makedirs(folder, exist_ok=True)
            # write next to the target, then rename: a run killed while writing
            # (the case checkpoints exist for) leaves the previous file intact
            # instead of a truncated one, because the rename is atomic
            partial = '{}.partial'.format(outname)
            with open(partial, 'wb') as fstate:
                pickle.dump(content, fstate, protocol=_PICKLE_PROTOCOL)
            os.replace(partial, outname)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
