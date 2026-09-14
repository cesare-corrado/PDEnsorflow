#!/usr/bin/env python
"""
    OptionReader: assembles parameter files and command-line flags into one
    ordered parameter store.

    The command line is scanned strictly from LEFT TO RIGHT and every
    assignment is applied as it is met, so the last definition of a key wins no
    matter where it came from. A parameter file is expanded in place, at the
    position of the option that named it. This is why

        -dt 25 +F run.par      takes dt from run.par
        +F run.par -dt 25      takes dt = 25

    and it is the reason the scan replays an edit list instead of filling a
    namespace: "a flag beats a file" would be the wrong rule.

    Two kinds of token are recognised. Control options steer the reader itself
    and are spelled with their own names (`+F`, `+Save`, `+Help`, and the long
    `--file`, `--save`, `--help` aliases). Everything else is a parameter
    assignment, written `-key value`, `--key value` or `--key=value`; the value
    is always the following token unless it was glued on with `=`, so negative
    numbers need no escaping.

    This class validates the SHAPE of the command line only. Whether a key
    exists and what type it carries is decided by ParameterMapper.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
from gpuSolve.carp_compatibility.parfilereader import ParFileReader


# Control options: they act on the reader rather than setting a parameter.
FILE_OPTIONS = ('+F', '--file')
SAVE_OPTIONS = ('+Save', '--save')
HELP_OPTIONS = ('+Help', '--help', '-h')


def normalise_key(token: str) -> str:
    """ normalise_key(token) strips the leading dashes of a command-line option
        so that `-dt` and `--dt` name the same parameter as `dt` in a file
    """
    key = token
    while len(key) > 0 and key[0] == '-':
        key = key[1:]
    return(key)


def quote_if_needed(value: str) -> str:
    """ quote_if_needed(value) wraps value in double quotes when it carries
        blanks, commas or a `#`, which a bare value in a `.par` file cannot
    """
    if len(value) == 0:
        return('""')
    for char in (' ', '\t', ',', '#', '"'):
        if char in value:
            return('"{}"'.format(value.replace('\\', '\\\\').replace('"', '\\"')))
    return(value)


class OptionReader:
    """
    class OptionReader: builds the parameter store from a command line.
    The store is a plain dict of key -> value strings. It is filled by replaying
    every assignment in command-line order, so it holds the RESOLVED value of
    each key once the whole line has been read.
    """

    def __init__(self):
        self.__store : dict      = None
        self.__origin : dict     = None
        self.__files : list      = None
        self.__save_fname : str  = None
        self.__help : bool       = False

    def store(self) -> dict:
        """ store() returns the resolved key -> value dict; None before read """
        return(self.__store)

    def origin(self) -> dict:
        """ origin() returns, per key, a text description of where its winning
            value came from. Used by the error messages and by +Save
        """
        return(self.__origin)

    def files(self) -> list:
        """ files() returns the parameter files that were read, in order """
        return(self.__files)

    def save_fname(self) -> str:
        """ save_fname() returns the file named by +Save; None if absent """
        return(self.__save_fname)

    def help_requested(self) -> bool:
        """ help_requested() tells whether +Help / --help was given """
        return(self.__help)

    def read(self, argv: list) -> dict:
        """ read(argv) scans argv left to right and returns the resolved store.
            argv must NOT contain the program name.
        """
        try:
            self.__store      = {}
            self.__origin     = {}
            self.__files      = []
            self.__save_fname = None
            self.__help       = False
            ipos = 0
            ntok = len(argv)
            while ipos < ntok:
                token = argv[ipos]
                if token in HELP_OPTIONS:
                    self.__help = True
                    ipos += 1
                elif token in FILE_OPTIONS or token.startswith('--file='):
                    fname, ipos = self.__option_value(argv, ipos, '--file=', 'a parameter file')
                    self.__read_par_file(fname)
                elif token in SAVE_OPTIONS or token.startswith('--save='):
                    self.__save_fname, ipos = self.__option_value(argv, ipos, '--save=', 'an output file')
                elif token.startswith('+'):
                    raise ValueError('Unrecognized keyword {}'.format(token))
                elif token.startswith('-') and len(token) > 1:
                    ipos = self.__read_flag(argv, ipos)
                else:
                    raise ValueError('Unexpected argument {}: parameters are given as '
                                     '-key value, --key value or --key=value'.format(token))
            return(self.__store)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def write_save_file(self, fname: str = None):
        """ write_save_file(fname) writes the resolved store back as a `.par`
            file. Round-tripping the resolved state is the quickest way to see
            what a command line actually asked for once every file and flag has
            been applied.
        """
        outname = fname if fname is not None else self.__save_fname
        if outname is None:
            return
        try:
            with open(outname, 'w') as fout:
                fout.write('#\n# PDEnsorflow: resolved parameters\n#\n')
                for key, value in self.__store.items():
                    fout.write('{} = {}\n'.format(key, quote_if_needed(value)))
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __option_value(self, argv: list, ipos: int, glued: str, what: str) -> tuple:
        """ reads the value of a control option, in either the `--opt=value` or
            the `opt value` form, and returns (value, next position)
        """
        token = argv[ipos]
        if token.startswith(glued):
            return((token[len(glued):], ipos + 1))
        if ipos + 1 >= len(argv):
            raise ValueError('{} expects {}'.format(token, what))
        return((argv[ipos + 1], ipos + 2))

    def __read_flag(self, argv: list, ipos: int) -> int:
        """ applies one `-key value` / `--key value` / `--key=value` assignment
            and returns the position of the next token
        """
        token = argv[ipos]
        ieq   = token.find('=')
        if ieq > 0:
            key   = normalise_key(token[:ieq])
            value = token[ieq + 1:]
            nxt   = ipos + 1
        else:
            key = normalise_key(token)
            if ipos + 1 >= len(argv):
                raise ValueError('option {} expects a value'.format(token))
            value = argv[ipos + 1]
            nxt   = ipos + 2
        if len(key) == 0:
            raise ValueError('empty option name in {}'.format(token))
        self.__assign(key, value, 'command line')
        return(nxt)

    def __read_par_file(self, fname: str):
        """ expands a parameter file at the position of the option that named
            it, applying its assignments in file order
        """
        reader = ParFileReader()
        for key, value in reader.read(fname):
            self.__assign(key, value, fname)
        self.__files.append(fname)

    def __assign(self, key: str, value: str, origin: str):
        """ applies one assignment; a later one silently replaces an earlier
            one, which is exactly the resolution rule
        """
        self.__store[key]  = value
        self.__origin[key] = origin
