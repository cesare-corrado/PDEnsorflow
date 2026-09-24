#!/usr/bin/env python
"""
    SvFileWriter: writes a single-cell state file (`.sv`) in the text format of
    the reference simulator (see IO/readers/svfilereader.py for the layout), so
    that `bench --read-ini-file` and `imp_region[].im_sv_init` can read it.

    Each line is the value padded to 20 characters, then `# name`, as the
    reference writes it. The values are written with full precision (the
    shortest text that reads back to the same float64), where the reference
    writes 6 significant digits: its reader scans them with %lf, so the extra
    digits are read, not cut, and a file written here restarts a run without
    the rounding of a reference file.

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

# Width the value is padded to before its `# name` comment (the reference's
# "%-20g# name").
VALUE_WIDTH : int = 20


def value_text(value) -> str:
    """ value_text(value) returns the text of one value: '-' for None, else the
        shortest decimal that reads back to the same float64 (repr), which
        for a float32 state is its exact float64 widening
    """
    if value is None:
        return(UNUSED_VALUE)
    return(repr(float(value)))


class SvFileWriter:
    """
    class SvFileWriter: utility to write a single-cell state file.
    """

    def __init__(self, config: dict = None):
        self._fname : str = 'singlecell.sv'
        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def fname(self) -> str:
        """ fname() returns the name of the file to write """
        return(self._fname)

    def set_fname(self, fname: str):
        """ set_fname(fname) sets the name of the file to write """
        self._fname = fname

    def write(self, global_values: list, sections: list):
        """ write(global_values, sections) writes the file:
              global_values: [(name, value)], value None for "not used";
              sections: [(section name, [(entry name, value)])], the cell model
              first, then its plugins, each in the reader's positional order
        """
        try:
            lines : list = []
            for name, value in global_values:
                lines.append(self.__line(name, value))
            for section, entries in sections:
                lines.append(section)
                for name, value in entries:
                    lines.append(self.__line(name, value))
                # the reference ends each section with an empty line
                lines.append('')
            with open(self._fname, 'w') as fsv:
                fsv.write('\n'.join(lines) + '\n')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __line(self, name: str, value) -> str:
        """ one `value  # name` line; a value longer than the padding keeps a
            blank before the comment so the line stays readable
        """
        text = value_text(value)
        if len(text) >= VALUE_WIDTH:
            text = text + ' '
        return('{}# {}'.format(text.ljust(VALUE_WIDTH), name))
