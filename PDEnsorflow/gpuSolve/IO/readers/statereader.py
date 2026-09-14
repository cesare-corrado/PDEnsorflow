#!/usr/bin/env python
"""
    StateReader: reads a checkpoint file written by StateWriter, so a simulation
    can resume from the time the state was saved.

    The file is a pickled Python dict with five entries:

        'ionic_model'      name of the cell-model class ('' for pure diffusion)
        'time'             simulation time of the state, in ms
        'num_nodes'        number of mesh nodes
        'Vm'               nodal transmembrane potential, flat, num_nodes values
        'state_variables'  {name: flat array of num_nodes values}, one entry per
                           variable the cell model advances in time

    Every nodal array is in the user's node order (never the renumbered one).

    Loading a pickle can run code, so read only checkpoint files you produced.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import pickle

import numpy as np


# The entries every checkpoint carries. StateWriter imports this tuple, so the
# writer and the reader cannot drift apart.
CHECKPOINT_KEYS = ('ionic_model', 'time', 'num_nodes', 'Vm', 'state_variables')


class StateReader:
    """
    class StateReader: utility to read a checkpoint file.
    Provided a file name, this class returns the checkpoint dict, validated.
    """

    def __init__(self):
        self.__fname : str       = None
        self.__checkpoint : dict = None

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self.__fname)

    def checkpoint(self) -> dict:
        """ checkpoint() returns the checkpoint dict of the last read; None before """
        return(self.__checkpoint)

    def ionic_model(self) -> str:
        """ ionic_model() returns the cell-model name stored in the checkpoint """
        return(self.__checkpoint['ionic_model'])

    def time(self) -> float:
        """ time() returns the simulation time of the checkpoint, in ms """
        return(self.__checkpoint['time'])

    def num_nodes(self) -> int:
        """ num_nodes() returns the number of mesh nodes of the checkpoint """
        return(self.__checkpoint['num_nodes'])

    def Vm(self) -> np.ndarray:
        """ Vm() returns the nodal transmembrane potential of the checkpoint """
        return(self.__checkpoint['Vm'])

    def state_variables(self) -> dict:
        """ state_variables() returns the {name: nodal values} dict of the checkpoint """
        return(self.__checkpoint['state_variables'])

    def read(self, fname: str) -> dict:
        """ read(fname) reads the checkpoint file fname, checks that it is
            complete and consistent, and returns the checkpoint dict
        """
        try:
            with open(fname, 'rb') as fstate:
                content = pickle.load(fstate)
            self.__checkpoint = self.__validate(content, fname)
            self.__fname      = fname
            return(self.__checkpoint)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __validate(self, content, fname: str) -> dict:
        """ checks the structure of the file before anything uses it. A
            truncated or foreign file must be refused here, with its name, rather
            than fail later with a shape error deep inside the solver. Entries
            beyond the five required ones are ignored, so a file from a later
            version that stores more still loads.
        """
        if not isinstance(content, dict):
            raise ValueError('{}: not a checkpoint (expected a dict, found {})'.format(
                fname, type(content).__name__))
        missing = [key for key in CHECKPOINT_KEYS if key not in content]
        if len(missing) > 0:
            raise ValueError('{}: not a checkpoint, missing {}'.format(fname, ', '.join(missing)))
        if not isinstance(content['ionic_model'], str):
            raise ValueError('{}: ionic_model must be a string'.format(fname))
        ctime = float(content['time'])
        if not np.isfinite(ctime) or ctime < 0.0:
            raise ValueError('{}: invalid time {}'.format(fname, content['time']))
        num_nodes = int(content['num_nodes'])
        if num_nodes <= 0:
            raise ValueError('{}: invalid num_nodes {}'.format(fname, content['num_nodes']))
        Vm = self.__nodal_array(content['Vm'], 'Vm', num_nodes, fname)
        if not isinstance(content['state_variables'], dict):
            raise ValueError('{}: state_variables must be a dict'.format(fname))
        states : dict = {}
        for name, values in content['state_variables'].items():
            states[str(name)] = self.__nodal_array(values, 'state variable {}'.format(name),
                                                   num_nodes, fname)
        return({'ionic_model': content['ionic_model'],
                'time': ctime,
                'num_nodes': num_nodes,
                'Vm': Vm,
                'state_variables': states})

    def __nodal_array(self, values, what: str, num_nodes: int, fname: str) -> np.ndarray:
        """ checks that one nodal array has a value per node and no NaN or Inf:
            resuming from a state that has already blown up only wastes a run
        """
        array = np.asarray(values)
        if array.size != num_nodes:
            raise ValueError('{}: {} has {} values but num_nodes is {}'.format(
                fname, what, array.size, num_nodes))
        if not np.all(np.isfinite(array)):
            raise ValueError('{}: {} carries non-finite values'.format(fname, what))
        return(np.reshape(array, (-1,)))
