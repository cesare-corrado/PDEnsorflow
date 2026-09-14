#!/usr/bin/env python
"""
    VtxReader: reads a `.vtx` vertex-specification file, a list of node indices
    used to name a set of nodes explicitly (a stimulus electrode, a set of
    recording sites) instead of describing it geometrically.

    The file is:

        n                 the number of indices that follow
        intra | extra     which domain the indices refer to (optional)
        node_0
        ...
        node_(n-1)

    with 0-based indices into the `.pts` node list.

    Two details of the format are worth stating, because they are what make a
    file written by one tool readable by another:

      * the `intra` / `extra` line is NOT counted and NOT required. The reference
        parser reads lines and keeps the ones that scan as an integer, so a
        keyword line is simply skipped. Anything else that does not scan as an
        integer is skipped the same way;
      * reading stops once n indices have been collected, so trailing content is
        ignored rather than being an error.

    A file carrying a per-vertex scaling value next to each index is a different
    thing (it is what an `elec.vtx_fcn` electrode uses) and is not read here:
    the second column would be silently dropped, so it is refused instead.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import numpy as np


class VtxReader:
    """
    class VtxReader: utility to read a `.vtx` vertex specification file.
    Provided a file name, this class returns the node indices it names.
    """

    def __init__(self):
        self.__fname : str          = None
        self.__indices : np.ndarray = None
        self.__domain : str         = None

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self.__fname)

    def indices(self) -> np.ndarray:
        """ indices() returns the node indices of the last read as a numpy
            array of int; None if nothing was read yet
        """
        return(self.__indices)

    def domain(self) -> str:
        """ domain() returns the 'intra' / 'extra' keyword the file carried,
            or None when it named none
        """
        return(self.__domain)

    def read(self, fname: str) -> np.ndarray:
        """ read(fname) reads the vertex file fname and returns its node
            indices as a numpy array of int
        """
        try:
            with open(fname, 'r') as fvtx:
                lines = fvtx.readlines()
            self.__fname = fname
            self.__parse(lines, fname)
            return(self.__indices)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __parse(self, lines: list, fname: str):
        """ collects the declared number of indices, skipping the lines that do
            not scan as one
        """
        self.__domain  = None
        expected : int = None
        indices : list = []
        for line in lines:
            body = line.strip()
            if len(body) == 0 or body[0] == '#':
                continue
            fields = body.split()
            if len(fields) > 1 and expected is not None:
                raise ValueError('{}: "{}" carries more than one value per node. A vertex '
                                 'file with per-node data is a different format and is not '
                                 'read here'.format(fname, body))
            try:
                value = int(fields[0])
            except ValueError:
                # the intra / extra keyword, or any other non-numeric line
                if self.__domain is None and fields[0] in ('intra', 'extra'):
                    self.__domain = fields[0]
                continue
            if expected is None:
                expected = value
                continue
            indices.append(value)
            if len(indices) == expected:
                break
        if expected is None:
            raise ValueError('{}: no vertex count found'.format(fname))
        if len(indices) < expected:
            raise ValueError('{}: declares {} vertices but carries {}'.format(
                fname, expected, len(indices)))
        self.__indices = np.array(indices, dtype=np.int32)
