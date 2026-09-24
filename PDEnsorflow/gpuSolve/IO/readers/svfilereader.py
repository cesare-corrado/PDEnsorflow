#!/usr/bin/env python
"""
    SvFileReader: reads a single-cell state file (`.sv`) in the text format of
    the reference simulator (written by `bench --save-ini-file` and read by
    `bench --read-ini-file` and by the tissue key `imp_region[].im_sv_init`):

        -81.2               # Vm          <- global quantities, one per line;
        -                   # Lambda         '-' means "not used"
        ...
        Courtemanche                      <- a section: the cell model ...
        0.31661             # Ca_rel      <- ... and one value per entry
        ...
        Electroporation_DeBruinKrassowska98   <- one section per plugin
        500082              # n

    The reference reads the values by position and ignores the `# name`
    comments. This reader keeps the names, because matching them is the only
    way to catch a file written for another model or in another order; the
    caller checks them against the layout it expects. A value line must
    therefore carry its `# name` comment, as every file the reference writes
    does.

    Values are read as float64. The reference writes them with 6 significant
    digits (%g), so a state read from its files is rounded to that precision.

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

# The reference's mark for a global quantity the model does not use.
UNUSED_VALUE : str = '-'


class SvFileReader:
    """
    class SvFileReader: utility to read a single-cell state file.
    Provided a file name, this class returns its global quantities and its
    sections.
    """

    def __init__(self):
        self.__fname : str      = None
        self.__globals : list   = None
        self.__sections : list  = None

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self.__fname)

    def global_values(self) -> list:
        """ global_values() returns the global quantities at the top of the
            file, as [(name, value)] in file order; value is None for '-'
        """
        return(self.__globals)

    def sections(self) -> list:
        """ sections() returns [(section name, [(entry name, value)])] in file
            order: the cell model first, then its plugins
        """
        return(self.__sections)

    def read(self, fname: str):
        """ read(fname) reads the state file fname """
        try:
            with open(fname, 'r') as fsv:
                lines = fsv.readlines()
            self.__fname = fname
            self.__parse(lines, fname)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __parse(self, lines: list, fname: str):
        """ splits the file into the global block and the sections """
        self.__globals  = []
        self.__sections = []
        for lineno, line in enumerate(lines, start=1):
            body = line.strip()
            if len(body) == 0:
                continue
            if '#' not in body:
                # a section header: the name of a cell model or of a plugin
                self.__sections.append((body, []))
                continue
            text, _sep, name = body.partition('#')
            text = text.strip()
            name = name.strip()
            if len(name) == 0:
                raise ValueError('{}, line {}: "{}" has no entry name after "#"'.format(
                    fname, lineno, body))
            if text == UNUSED_VALUE:
                value = None
            else:
                try:
                    value = float(text)
                except ValueError:
                    raise ValueError('{}, line {}: the value "{}" of {} is not a number'.format(
                        fname, lineno, text, name))
            if len(self.__sections) == 0:
                self.__globals.append((name, value))
            else:
                if value is None:
                    raise ValueError('{}, line {}: {} of {} has no value'.format(
                        fname, lineno, name, self.__sections[-1][0]))
                self.__sections[-1][1].append((name, value))
        if len(self.__sections) == 0:
            raise ValueError('{}: no model section found; this is not a single-cell state '
                             'file'.format(fname))
