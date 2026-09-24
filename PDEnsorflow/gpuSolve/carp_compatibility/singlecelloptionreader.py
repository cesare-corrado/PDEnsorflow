#!/usr/bin/env python
"""
    SingleCellOptionReader: the command line of the `singlecell` executable,
    which takes the options of the reference single-cell tool (bench) with the
    same spelling and the same parsing rules:

        singlecell --imp Tomek --imp-par "GKr*0.5" --numstim 3 --bcl 800
        singlecell -I Tomek -a 2000 -o 0.5 --fout=run1 -B

    The rules are those of the option parser bench is generated with
    (gengetopt over getopt_long), checked against the installed binary:
      * `--name value` and `--name=value`; a unique prefix of a long name is
        accepted (`--dur`), an ambiguous one is an error that lists the
        candidates, and an exact name wins over a longer one (`--dt`, not
        `--dt-out`);
      * `-c value` and `-cvalue`; flags may be grouped (`-vB`);
      * an option with an OPTIONAL argument (`--fout`) takes it only in the
        glued form `--fout=name` / `-Oname`: `--fout name` leaves `name` behind;
      * an option given twice, two options of different modes (`--numstim`
        with `--stim-times`) and two stimulus types are errors.

    One rule is deliberately stricter than bench: a word that is not an option
    (a stray `foo`, the `name` of `--fout name`) stops the run. bench drops it
    silently, which turns a mistyped value into a run that is not the one that
    was asked for.

    This class reads the SHAPE of the command line and the value types. Which
    options singlecell actually implements is decided by the caller
    (SUPPORTED_OPTIONS below lists them), so that an option of bench that is
    not implemented yet is reported as such rather than as unknown.

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
# NOTE: this module must not import TensorFlow (directly or through gpuSolve
# submodules). singlecell chooses the device and the thread count from the
# options, and both have to be in the environment before TensorFlow starts.


PROGRAM_NAME : str = 'singlecell'

# The option table of bench, in its own order (the order matters: an
# ambiguous prefix lists its candidates in it). Each entry is
#   (long name, short letter or '', kind, default, family, help)
# kind: 'int', 'double', 'float', 'string', 'flag', or 'optstring' for a
# string whose argument is optional. family: the gengetopt mode or group the
# option belongs to ('' for none). Defaults are bench's, except --imp.
BENCH_OPTIONS = (
    ('help',              'h', 'flag',      False,          '', 'Print help and exit'),
    ('detailed-help',     '',  'flag',      False,          '', 'Print help, including all details and hidden options, and exit'),
    ('full-help',         '',  'flag',      False,          '', 'Print help, including hidden options, and exit'),
    ('version',           'V', 'flag',      False,          '', 'Print version and exit'),
    ('numstim',           '',  'int',       1,              'mode:regstim', 'number of stimuli'),
    ('stim-start',        'i', 'double',    1.0,            'mode:regstim', 'start of stimulation [ms]'),
    ('bcl',               'b', 'double',    1000.0,         'mode:regstim', 'basic cycle length [ms]'),
    ('stim-times',        '',  'string',    '',             'mode:neqstim', 'comma separated list of stim times [ms]'),
    ('DIA',               '',  'flag',      False,          'mode:neqstim', 'interpret stim times as diastolic intervals'),
    ('restitute',         '',  'string',    '',             'mode:restitute', 'restitution experiment (S1S2, dyn, S1S2f)'),
    ('res-file',          '',  'string',    '',             'mode:restitute', 'definition file for restitution parameters'),
    ('res-trace',         '',  'flag',      False,          'mode:restitute', 'output ionic model trace'),
    ('res-state-vector',  '',  'flag',      False,          'mode:restitute', 'save state vector'),
    ('list-imps',         '',  'flag',      False,          'mode:info', 'list all available IMPs'),
    ('plugin-outputs',    '',  'flag',      False,          'mode:info', 'list the outputs of available plugins'),
    ('imp-info',          '',  'flag',      False,          'mode:info', 'print tunable parameters and state variables for particular IMPs'),
    ('buildinfo',         '',  'flag',      False,          'mode:info', '(deprecated) print build information'),
    ('clamp-ini',         '',  'double',    -80.0,          'mode:vclamp', 'outside of clamp pulse, clamp Vm to this value'),
    ('clamp-file',        '',  'string',    '',             'mode:vclamp', 'clamp voltage to a signal read from a file'),
    ('RRC-delay',         '',  'float',     10.0,           'mode:RRC', 'time between stimulus and clamp [ms]'),
    ('RRC-clamp-dur',     '',  'float',     1000.0,         'mode:RRC', 'duration of current clamp [ms]'),
    ('RRC-repol',         '',  'float',     -50.0,          'mode:RRC', 'repolarization voltage [mV]'),
    ('RRC-tol',           '',  'float',     0.02,           'mode:RRC', 'relative convergence criterion'),
    ('stim-curr',         'c', 'float',     60.0,           'group:stimtype', 'stimulation current [pA/pF]=[uA/cm^2]'),
    ('stim-volt',         '',  'float',     None,           'group:stimtype', 'stimulation voltage'),
    ('stim-file',         '',  'string',    None,           'group:stimtype', 'use signal from a .trc file for current stimulus'),
    ('duration',          'a', 'double',    None,           '', 'duration of simulation [ms]'),
    ('past-stim',         '',  'double',    1000.0,         '', 'duration after last stim [ms]'),
    ('dt',                'D', 'double',    0.01,           '', 'time step [ms]'),
    ('num',               'n', 'int',       1,              '', 'number of cells'),
    ('ext-vm-update',     '',  'flag',      False,          '', 'update Vm externally'),
    ('rseed',             '',  'int',       1,              '', 'random number seed'),
    ('target',            '',  'string',    'auto',         '', 'target to run the simulation on'),
    ('stim-dur',          'T', 'double',    1.0,            '', 'duration of stimulus pulse [ms]'),
    ('stim-assign',       'A', 'flag',      False,          '', 'stimulus current assignment'),
    ('stim-species',      '',  'string',    'Ki',           '', 'concentrations that should be affected by stimuli'),
    ('stim-ratios',       '',  'string',    '1.0',          '', 'proportions of stimulus current carried by each species'),
    ('resistance',        '',  'double',    100.0,          '', 'coupling resistance [kOhm]'),
    ('surface-to-volume', '',  'double',    0.14,           '', 'cell surface to cell volume ratio'),
    ('light-irrad',       '',  'double',    0.0,            '', 'unattenuated irradiance of illumination pulse'),
    ('light-dur',         '',  'double',    10.0,           '', 'duration of illumination pulses'),
    ('light-numstim',     '',  'int',       1,              '', 'number of illumination pulses (0: no limit)'),
    ('light-bcl',         '',  'double',    1000.0,         '', 'basic cycle length of illumination'),
    ('light-start',       '',  'double',    10.0,           '', 'start time for illumination pulse'),
    ('light-times',       '',  'string',    '',             '', 'comma-separated list of stim times'),
    ('light-file',        '',  'string',    '',             '', 'overrides ALL --light- vars with a signal from a file'),
    ('imp',               'I', 'string',    'Courtemanche', '', 'IMP to use'),
    ('imp-par',           'p', 'string',    '',             '', 'params to modify IMP'),
    ('plug-in',           'P', 'string',    '',             '', "plugins to use, separate with ':'"),
    ('plug-par',          'm', 'string',    '',             '', "params to modify plug-ins, separate params with ',' and plugin params with ':'"),
    ('load-module',       '',  'string',    '',             '', 'load a module for use with bench (implies --imp)'),
    ('clamp',             'l', 'double',    0.0,            '', 'clamp Vm to this value [mV]'),
    ('clamp-dur',         'L', 'double',    0.0,            '', 'duration of Vm clamp pulse [ms]'),
    ('clamp-start',       '',  'double',    10.0,           '', 'start time of Vm clamp pulse [ms]'),
    ('clamp-SVs',         '',  'string',    '',             '', 'colon separated list of state variable to clamp'),
    ('SV-clamp-files',    '',  'string',    '',             '', ': separated list of files from which to read state variables'),
    ('SV-I-trigger',      '',  'flag',      True,           '', 'apply SV clamps at each current stim'),
    ('AP-clamp-file',     '',  'string',    '',             '', 'action potential trace applied at each stimulus'),
    ('strain',            'N', 'double',    0.0,            '', 'amount of strain to apply'),
    ('strain-time',       't', 'double',    200.0,          '', 'time to strain [ms]'),
    ('strain-dur',        'y', 'float',     None,           '', 'duration of strain (default=tonic) [ms]'),
    ('strain-rate',       '',  'double',    2.0,            '', 'time to apply/remove strain [ms]'),
    ('dt-out',            'o', 'double',    1.0,            '', 'temporal output granularity [ms]'),
    ('start-out',         '',  'double',    0.0,            '', 'start time of output [ms]'),
    ('fout',              'O', 'optstring', 'BENCH_REG',    '', 'output to files'),
    ('no-trace',          '',  'flag',      False,          '', 'do not output trace'),
    ('trace-no',          '',  'int',       0,              '', 'number for trace file name'),
    ('bin',               'B', 'flag',      False,          '', 'write binary files'),
    ('imp-sv-dump',       'u', 'string',    '',             '', 'sv dump list'),
    ('plug-sv-dump',      'g', 'string',    '',             '', ': separated sv dump list for plug-ins'),
    ('dump-lut',          'd', 'flag',      False,          '', 'dump lookup tables'),
    ('APstatistics',      '',  'flag',      False,          '', 'compute AP statistics'),
    ('validate',          'v', 'flag',      False,          '', 'output all SVs'),
    ('save-time',         's', 'double',    0.0,            '', 'time at which to save binary state [ms]'),
    ('save-file',         'f', 'optstring', 'a.sv',         '', 'file in which to save binary state'),
    ('restore',           'r', 'string',    'a.sv',         '', 'restore saved binary state file'),
    ('save-ini-file',     'F', 'string',    'singlecell.sv', '', 'text file in which to save state of a single cell'),
    ('save-ini-time',     'S', 'double',    0.0,            '', 'time at which to save single cell state [ms]'),
    ('read-ini-file',     'R', 'string',    'singlecell.sv', '', 'text file from which to read state of a single cell'),
    ('SV-init',           '',  'string',    '',             '', 'colon separated list of comma separated SV=initial_values'),
    ('doppel-on',         '',  'float',     None,           '', 'time to switch to dopple [ms]'),
    ('doppel-dur',        '',  'float',     100.0,          '', 'duration of dopple [ms]'),
)

# Options singlecell adds; they are not in bench and are listed apart in --help.
GPUSOLVE_OPTIONS = (
    ('reference-scheme',  '',  'flag',      False,          '', 'integrate as the reference does where gpuSolve '
                                                                'differs: forward Euler for the Tomek gates and IKr '
                                                                'chain, the reference form of Defib_AshiharaTrayanova'),
)

# Options that may be given more than once (their values are collected).
MULTIPLE_OPTIONS = ('load-module',)

# The allowed values of an option with a closed set of values.
OPTION_VALUES = {'restitute': ('S1S2', 'dyn', 'S1S2f')}

# The options singlecell implements. Every other option of BENCH_OPTIONS is
# recognised and stops the run with "not supported yet".
SUPPORTED_OPTIONS = ('help', 'detailed-help', 'full-help', 'version',
                     'numstim', 'stim-start', 'bcl', 'stim-times', 'DIA',
                     'list-imps', 'plugin-outputs', 'imp-info', 'buildinfo',
                     'stim-curr', 'duration', 'past-stim', 'dt', 'num', 'rseed', 'target',
                     'stim-dur', 'imp', 'imp-par', 'plug-in', 'plug-par',
                     'dt-out', 'start-out', 'fout', 'no-trace', 'trace-no', 'bin',
                     'imp-sv-dump', 'validate',
                     'save-ini-file', 'save-ini-time', 'read-ini-file',
                     'reference-scheme')

# Numeric kinds and how their values are read. gengetopt reads them with
# strtol/strtod and refuses trailing characters, which int() and float() do
# as well; int() also refuses "1.5", as bench does.
NUMERIC_KINDS = {'int': int, 'double': float, 'float': float}


class SingleCellOptionReader:
    """
    class SingleCellOptionReader: reads the singlecell command line.
    read(argv) must be called once before the accessors.
    """

    def __init__(self, config: dict = None):
        self._program : str   = PROGRAM_NAME
        self._table   : dict  = None
        self._shorts  : dict  = None
        self._values  : dict  = None
        self._given   : list  = None
        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])
        self._table  = {entry[0]: entry for entry in BENCH_OPTIONS + GPUSOLVE_OPTIONS}
        self._shorts = {entry[1]: entry[0] for entry in BENCH_OPTIONS + GPUSOLVE_OPTIONS
                        if len(entry[1]) > 0}

    # ---- reading ------------------------------------------------------------
    def read(self, argv: list):
        """ read(argv) parses the command line (argv excludes the program name)
            and checks the mode and group rules. A malformed command line
            raises ValueError with the message bench prints for it.
        """
        self._values = {}
        self._given  = []
        position = 0
        while position < len(argv):
            token = argv[position]
            position += 1
            if token == '--':
                # getopt: everything after "--" is a plain argument
                if position < len(argv):
                    self.__stray(argv[position])
                break
            if token.startswith('--'):
                position = self.__read_long(token[2:], argv, position)
            elif token.startswith('-') and len(token) > 1:
                position = self.__read_short(token[1:], argv, position)
            else:
                self.__stray(token)
        self.__check_families()
        for name, allowed in OPTION_VALUES.items():
            if self.given(name) and self._values[name] not in allowed:
                raise ValueError('{}: invalid argument "{}" for option --{} (possible values: {})'.format(
                    self._program, self._values[name], name, ', '.join(allowed)))

    # ---- accessors ------------------------------------------------------------
    def given(self, name: str) -> bool:
        """ given(name) returns True when option name is on the command line """
        return(name in self._values)

    def value(self, name: str):
        """ value(name) returns the value of option name: the one given on the
            command line, or its default. A flag is True or False.
        """
        if name not in self._table:
            raise KeyError('no option named {}'.format(name))
        if name in self._values:
            return(self._values[name])
        return(self._table[name][3])

    def given_options(self) -> list:
        """ given_options() returns the long names of the options on the
            command line, in the order they were given
        """
        return(list(self._given))

    def unsupported_options(self) -> list:
        """ unsupported_options() returns the options on the command line that
            are bench options singlecell does not implement yet
        """
        return([name for name in self._given if name not in SUPPORTED_OPTIONS])

    def help_requested(self) -> bool:
        """ help_requested() returns True for --help, --detailed-help and --full-help """
        return(any(self.given(name) for name in ('help', 'detailed-help', 'full-help')))

    def version_requested(self) -> bool:
        """ version_requested() returns True for --version """
        return(self.given('version'))

    def usage_text(self) -> str:
        """ usage_text() returns the --help text: the options singlecell
            implements, with bench's layout and singlecell's defaults, then the
            options that are bench's but not implemented yet
        """
        lines = ['Usage: {} [OPTIONS]...'.format(self._program),
                 'Single-cell experiments with the gpuSolve ionic models, with the options',
                 'of the reference single-cell tool (bench). All times in ms; all voltages in',
                 'mV; currents in uA/cm^2', '']
        for entry in BENCH_OPTIONS:
            if entry[0] in SUPPORTED_OPTIONS:
                lines.append(self.__help_line(entry))
        lines.append('')
        lines.append('gpuSolve options (not in bench):')
        for entry in GPUSOLVE_OPTIONS:
            lines.append(self.__help_line(entry))
        lines.append('')
        lines.append('bench options that are recognised but not supported yet (they stop the run):')
        pending = ['--{}'.format(entry[0]) for entry in BENCH_OPTIONS if entry[0] not in SUPPORTED_OPTIONS]
        row = ' '
        for name in pending:
            if len(row) + len(name) + 1 > 78:
                lines.append(row)
                row = ' '
            row = row + ' ' + name
        lines.append(row)
        return('\n'.join(lines))

    # ---- parsing helpers ------------------------------------------------------
    def __read_long(self, body: str, argv: list, position: int) -> int:
        """ reads one `--name[=value]` token; returns the next position """
        name, has_value, glued = body.partition('=')[0], '=' in body, body.partition('=')[2]
        name = self.__resolve_long(name, body)
        kind = self._table[name][2]
        if kind == 'flag':
            if has_value:
                raise ValueError("{}: option '--{}' doesn't allow an argument".format(self._program, name))
            self.__store(name, True)
            return(position)
        if kind == 'optstring':
            # an optional argument is only taken in the glued form
            self.__store(name, glued if has_value else self._table[name][3])
            return(position)
        if not has_value:
            if position >= len(argv):
                raise ValueError("{}: option '--{}' requires an argument".format(self._program, name))
            glued = argv[position]
            position += 1
        self.__store(name, self.__convert(name, glued))
        return(position)

    def __read_short(self, body: str, argv: list, position: int) -> int:
        """ reads one `-x...` token (grouped flags, or a glued value); returns
            the next position
        """
        index = 0
        while index < len(body):
            letter = body[index]
            if letter not in self._shorts:
                raise ValueError("{}: invalid option -- '{}'".format(self._program, letter))
            name = self._shorts[letter]
            kind = self._table[name][2]
            rest = body[index + 1:]
            if kind == 'flag':
                self.__store(name, True)
                index += 1
                continue
            if kind == 'optstring':
                self.__store(name, rest if len(rest) > 0 else self._table[name][3])
                return(position)
            if len(rest) == 0:
                if position >= len(argv):
                    raise ValueError("{}: option requires an argument -- '{}'".format(self._program, letter))
                rest = argv[position]
                position += 1
            self.__store(name, self.__convert(name, rest))
            return(position)
        return(position)

    def __resolve_long(self, name: str, body: str) -> str:
        """ the full long name for name: itself, or the one option it is a
            prefix of (getopt_long accepts any unambiguous abbreviation)
        """
        if name in self._table:
            return(name)
        matches = [entry[0] for entry in BENCH_OPTIONS + GPUSOLVE_OPTIONS if entry[0].startswith(name)]
        if len(name) > 0 and len(matches) == 1:
            return(matches[0])
        if len(name) > 0 and len(matches) > 1:
            raise ValueError("{}: option '--{}' is ambiguous; possibilities: {}".format(
                self._program, name, ' '.join("'--{}'".format(match) for match in matches)))
        raise ValueError("{}: unrecognized option '--{}'".format(self._program, body))

    def __convert(self, name: str, text: str):
        """ the typed value of option name from its argument text """
        kind = self._table[name][2]
        if kind not in NUMERIC_KINDS:
            return(text)
        try:
            return(NUMERIC_KINDS[kind](text))
        except ValueError:
            raise ValueError('{}: invalid numeric value: {}'.format(self._program, text))

    def __store(self, name: str, value):
        """ records option name; a second occurrence is an error unless the
            option collects several values
        """
        if name in MULTIPLE_OPTIONS:
            self._values.setdefault(name, []).append(value)
        elif name in self._values:
            short = self._table[name][1]
            spelled = "`--{}' (`-{}')".format(name, short) if len(short) > 0 else "`--{}'".format(name)
            raise ValueError('{}: {} option given more than once'.format(self._program, spelled))
        else:
            self._values[name] = value
        if name not in self._given:
            self._given.append(name)

    def __stray(self, token: str):
        """ a word that is not an option. bench ignores it; singlecell stops,
            because it is almost always a value that lost its option
        """
        raise ValueError("{}: unexpected argument '{}': it is not an option, and no option "
                         "takes it as its value (an optional value must be glued on, as in "
                         "--fout=name)".format(self._program, token))

    def __check_families(self):
        """ the gengetopt mode and group rules: options of two different modes
            cannot be combined, and at most one option of a group is given
        """
        first_mode : dict = {}
        for name in self._given:
            family = self._table[name][4]
            if family.startswith('mode:'):
                first_mode.setdefault(family, name)
        if len(first_mode) > 1:
            names = list(first_mode.values())
            raise ValueError('{}: option --{} conflicts with option --{}'.format(
                self._program, names[1], names[0]))
        groups : dict = {}
        for name in self._given:
            family = self._table[name][4]
            if family.startswith('group:'):
                groups.setdefault(family[len('group:'):], []).append(name)
        for group, names in groups.items():
            if len(names) > 1:
                raise ValueError('{}: {} options of group {} were given. At most one is '
                                 'required.'.format(self._program, len(names), group))

    def __help_line(self, entry: tuple) -> str:
        """ one --help line, laid out as bench's """
        name, short, kind, default, _family, text = entry
        flag = '-{}, '.format(short) if len(short) > 0 else '    '
        arg = {'int': '=INT', 'double': '=DOUBLE', 'float': '=FLOAT', 'string': '=STRING',
               'optstring': '[=STRING]', 'flag': ''}[kind]
        left = '  {}--{}{}'.format(flag, name, arg)
        if kind == 'flag':
            shown = '  (default={})'.format('on' if default else 'off')
        elif default is None:
            shown = ''
        else:
            shown = "  (default=`{}')".format(default)
        text = text + shown
        if len(left) >= 32:
            return('{}\n{}{}'.format(left, ' ' * 32, text))
        return('{:32s}{}'.format(left, text))
