#!/usr/bin/env python
"""
    SingleCellWriter: the output of the `singlecell` executable, in the formats
    and under the file names of the reference single-cell tool (bench), so that
    the tools that read bench output read this too. The rules are bench's
    (limpet/src/bench_utils.cc, open_globalvec_dump / globalvec_dump /
    write_dump_header):

      * no --fout and no -v: a `Time Vm Iion` table on stdout,
        "%10.3f " for the time, then "%+.8e " per quantity;
      * --fout[=name] (default name BENCH_REG): the same table in <name>.txt,
        and nothing on stdout;
      * --fout with -B, or -v: one float64 binary per quantity,
        <name>.Vm.bin, <name>.Iion.bin, <name>.t.bin;
      * dumped state variables (-u list, or all of them with -v): one raw
        binary per entry, float32 for an entry the reference stores as a gate
        and float64 otherwise, and a <name>_header.txt that lists every binary
        with its type, size and number of samples.

    Differences from bench, deliberate: a -u entry gets a single `.bin`
    extension (bench appends it twice), and an unknown -u name is an error
    (bench skips it without a message) -- both are checked by the caller.

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
import numpy as np


# The quantities of the table, in bench's order.
TABLE_QUANTITIES = ('Vm', 'Iion')

# Type names and sizes the header gives, as bench's data_type_names.
REAL_TYPE : tuple = ('Real', 8)
GLOBAL_TYPE : tuple = ('GlobalData_t', 8)
GATE_TYPE : tuple = ('Gatetype', 4)


class SingleCellWriter:
    """
    class SingleCellWriter: writes the table and the state dumps of a
    single-cell run. Call open() once, then write_block() for each block of
    samples, then close().
    """

    def __init__(self, config: dict = None):
        self._fout : str        = 'BENCH_REG'
        self._fout_given : bool = False
        self._binary : bool     = False
        self._validate : bool   = False
        # [(file base name without extension, gate)], one per dumped entry
        self._dumps : list      = []
        self._to_file : bool    = False
        self._split : bool      = False
        self._handles : dict    = None
        self._dump_handles : list = None
        self._nsamples : int    = 0
        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def set_dumps(self, dumps: list):
        """ set_dumps(dumps) sets the dumped entries, as [(file base name,
            gate)]; the file is <base name>.bin
        """
        self._dumps = list(dumps)

    def to_stdout(self) -> bool:
        """ to_stdout() returns True when the table goes to stdout """
        return(not self._to_file)

    def file_names(self) -> list:
        """ file_names() returns the names of the files the writer creates """
        names : list = []
        if self._to_file and self._split:
            names += ['{}.{}.bin'.format(self._fout, name) for name in TABLE_QUANTITIES + ('t',)]
        elif self._to_file:
            names.append('{}.txt'.format(self._fout))
        names += ['{}.bin'.format(base) for base, _gate in self._dumps]
        if len(self._dumps) > 0:
            names.append('{}_header.txt'.format(self._fout))
        return(names)

    def open(self):
        """ open() opens the output files and prints the list of quantities,
            as bench does, on stderr
        """
        try:
            # bench: io.w2file = validate ? 1 : fout_given; io.wbin = validate ? 1 : bin
            self._to_file = self._validate or self._fout_given
            self._split   = self._to_file and (self._validate or self._binary)
            self._handles = {}
            if self._to_file and self._split:
                for name in TABLE_QUANTITIES + ('t',):
                    self._handles[name] = open('{}.{}.bin'.format(self._fout, name), 'wb')
            elif self._to_file:
                self._handles['txt'] = open('{}.txt'.format(self._fout), 'w')
            self._dump_handles = [open('{}.bin'.format(base), 'wb') for base, _gate in self._dumps]
            self._nsamples = 0
            header = '{:>10s}\t'.format('Time') + ''.join('{:>10s}\t'.format(name)
                                                           for name in TABLE_QUANTITIES)
            print('Outputting the following quantities at each time: \n{}\n'.format(header),
                  file=sys.stderr, flush=True)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def write_block(self, times: np.ndarray, Vm: np.ndarray, Iion: np.ndarray,
                    dumps: np.ndarray = None):
        """ write_block(times, Vm, Iion, dumps) writes a block of samples:
            times, Vm and Iion are (n,) float64 arrays; dumps is (n, ndumps),
            one column per entry of set_dumps(), or None when nothing is dumped
        """
        try:
            count = times.shape[0]
            if count == 0:
                return
            if not self._to_file:
                # bench prints the time through a float ("%10.3f" of (float)t)
                stamps = times.astype(np.float32)
                lines = ['{:10.3f} {:+.8e} {:+.8e} '.format(float(stamps[i]), Vm[i], Iion[i])
                         for i in range(count)]
                sys.stdout.write('\n'.join(lines) + '\n')
                sys.stdout.flush()
            elif self._split:
                Vm.astype(np.float64).tofile(self._handles['Vm'])
                Iion.astype(np.float64).tofile(self._handles['Iion'])
                times.astype(np.float64).tofile(self._handles['t'])
            else:
                lines = ['{:10.3f} {:+.8e} {:+.8e} '.format(times[i], Vm[i], Iion[i])
                         for i in range(count)]
                self._handles['txt'].write('\n'.join(lines) + '\n')
            for column, (handle, (_base, gate)) in enumerate(zip(self._dump_handles, self._dumps)):
                values = dumps[:, column]
                values.astype(np.float32 if gate else np.float64).tofile(handle)
            self._nsamples += count
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def close(self):
        """ close() closes the files and, when some state is dumped, writes
            the header that lists the binaries of the experiment
        """
        try:
            for handle in list(self._handles.values()) + self._dump_handles:
                handle.close()
            if len(self._dumps) == 0:
                return
            lines = ['{}  # is bigendian'.format(1 if sys.byteorder == 'big' else 0)]
            if self._to_file and self._split:
                for name in TABLE_QUANTITIES + ('t',):
                    lines.append(self.__header_line('{}.{}.bin'.format(self._fout, name), REAL_TYPE))
            for base, gate in self._dumps:
                lines.append(self.__header_line('{}.bin'.format(base), GATE_TYPE if gate else GLOBAL_TYPE))
            with open('{}_header.txt'.format(self._fout), 'w') as fheader:
                fheader.write('\n'.join(lines) + '\n')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __header_line(self, fname: str, dtype: tuple) -> str:
        """ one header line, "%32s %10s %2d %10d" as bench writes it """
        return('{:>32s} {:>10s} {:2d} {:10d}'.format(os.path.basename(fname), dtype[0], dtype[1],
                                                     self._nsamples))
