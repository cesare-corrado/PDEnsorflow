#!/usr/bin/env python
"""
    LatReader: reads a nodal vector of local activation times, one value per
    node, as written by LatDetector.write() with all = 0 (init_acts_<ID>.dat)
    and as read by the reference to guide the distribution of prepacing states.

    The format carries no header and no node indices: it is simply the node
    values in mesh order, whitespace separated. The reference parser scans
    exactly as many numbers as the mesh has nodes and stops (`root_read_ascii`
    in the reference's parallel I/O layer scans with "%lf" in a loop over the
    local chunk sizes), so line breaks carry no meaning and a value may sit on
    a line of its own or share one with its neighbours.

    An activation time is negative (the reference writes -1) where a node never
    activated. That is data, not an error: the caller decides what an
    unactivated node means. This class passes the value through unchanged.

    One deliberate difference from the reference: a file that holds fewer
    values than the mesh has nodes is refused here, where the reference leaves
    the rest of the vector at whatever it held. A short file is almost always
    a file written for a different mesh, and prepacing every remaining node
    from an uninitialised activation time is silently wrong in a way that is
    very hard to see in the result.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import numpy as np


class LatReader:
    """
    class LatReader: utility to read a nodal activation-time vector.
    Provided a file name and the number of nodes, this class returns the
    activation time of each node as a numpy array of float.
    """

    def __init__(self):
        self.__fname : str          = None
        self.__times : np.ndarray   = None

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self.__fname)

    def times(self) -> np.ndarray:
        """ times() returns the activation times of the last read as a numpy
            array of float64; None if nothing was read yet
        """
        return(self.__times)

    def read(self, fname: str, npt: int = 0) -> np.ndarray:
        """ read(fname,npt) reads the activation-time file fname and returns
            the times as a numpy array of float64. When npt is greater than 0,
            the file must hold exactly npt values; the count is otherwise
            whatever the file carries.
        """
        try:
            values : list = []
            with open(fname, 'r') as flat:
                for line in flat:
                    body = line.strip()
                    if len(body) == 0 or body[0] == '#':
                        continue
                    for field in body.split():
                        values.append(float(field))
            if len(values) == 0:
                raise ValueError('{}: holds no activation time'.format(fname))
            if npt > 0 and len(values) != npt:
                raise ValueError('{}: holds {} activation times but the mesh has {} nodes. The '
                                 'file carries one value per node, in mesh order'.format(
                                     fname, len(values), npt))
            self.__fname = fname
            self.__times = np.array(values, dtype=np.float64)
            return(self.__times)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
