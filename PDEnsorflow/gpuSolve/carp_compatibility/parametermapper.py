#!/usr/bin/env python
"""
    ParameterMapper: turns a resolved parameter store into gpuSolve settings.

    This is the only class that speaks both vocabularies. It owns

      * the registry of known keys, with their type and their default. A key
        that is not in the registry is an error, so a typo is reported instead
        of being silently ignored;
      * the defaults, which are the reference simulator's, not gpuSolve's, so a
        file that omits a key means what the same file would mean elsewhere.
        There are two exceptions. The ionic parameters: a cell parameter that no
        `im_param` mentions keeps the gpuSolve class default, because the ionic
        models are gpuSolve's own implementations and a script and a parameter
        file should agree. And `renumbering`, which defaults to 1: it only
        reorders the unknowns, and the time step is several times faster with
        it (see the REGISTRY entry);
      * the unit conversions, which are concentrated here so the rest of the
        package works in gpuSolve's units throughout.

    Units in, units out
    -------------------
    Conductivities arrive in S/m, node coordinates in micrometres, `dt` in
    microseconds and every other time in milliseconds. The reference monodomain
    reads

        div(sigma_m grad V) = beta (Cm dV/dt + I_ion) + beta I_tr

    so dividing by beta Cm leaves a diffusion coefficient sigma_m/(beta Cm) and
    an ionic term that carries no beta. Cm is not a separate parameter (it is
    folded into the cell model, i.e. 1 uF/cm^2), so with lengths in micrometres
    and time in milliseconds

        sigma [um^2/ms] = 1.0e5 * g [S/m] * g_mult / (beta [um^-1] * volFrac)

    The 1.0e5 is the product of the S/m -> mS/um conversion (1.0e-3) and the
    um^2 -> cm^2 factor (1.0e-8) that turns beta Cm into uF/um^3. beta and
    volFrac only ever appear multiplied together, so they are folded into ONE
    element property named `beta`, and the tensor function divides by it: that
    keeps beta visible and per-element rather than pre-multiplied away.

    A transmembrane stimulus needs no conversion at all. Under operator
    splitting the reference adds `dt * pulse.strength` straight to Vm, which is
    what gpuSolve's forcing term I0 already does, so the strength passes
    through as the Stimulus intensity.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import re

from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.ionic.ms2v import MitchellSchaeffer2v
from gpuSolve.ionic.fenton4v import Fenton4v
from gpuSolve.ionic.courtemanche_ramirez_nattel import CourtemancheRamirezNattel
from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov
from gpuSolve.ionic.tomek import Tomek
from gpuSolve.ionic.plugins.electroporation_debruin_krassowska98 import ElectroporationDeBruinKrassowska98
from gpuSolve.ionic.plugins.defib_ashihara_trayanova import DefibAshiharaTrayanova
from gpuSolve.ionic.ionicmodelwithplugins import PLUGIN_SEPARATOR
from gpuSolve.ionic.ionicmodelwithplugins import ACTIVE_PARAMETER


# S/m and micrometres to um^2/ms, once beta (um^-1) has divided it out.
CONDUCTIVITY_TO_UM2_PER_MS : float = 1.0e5

# dt arrives in microseconds, every other time in milliseconds.
DT_MICROSECONDS_TO_MS : float = 1.0e-3

# Counter key of each indexed family. The names are irregular, so they are
# listed rather than derived.
ARRAY_COUNTERS = {'gregion': 'num_gregions',
                  'imp_region': 'num_imp_regions',
                  'stim': 'num_stim',
                  'stimulus': 'num_stim',
                  'tsav': 'num_tsav',
                  'tsav_ext': 'num_tsav'}

# The format accepts at most this many save times.
MAX_SAVE_TIMES : int = 50

# Base name of the files written by interval checkpointing (chkpt_intv); the
# time of the checkpoint is appended to it.
CHECKPOINT_BASENAME : str = 'checkpoint'

# Cell models. `MitchellSchaeffer` covers both variants: a_crit is 0 by default,
# which is the plain model, and non-zero selects the modified one.
IONIC_MODELS = {'mMS': ModifiedMS2v,
                'MitchellSchaeffer': None,
                'Courtemanche': CourtemancheRamirezNattel,
                'tenTusscherPanfilov': TenTusscherPanfilov,
                'Tomek': Tomek,
                'Fenton': Fenton4v}

# Cell types a model can be switched to with the `flags=<TYPE>` item of
# im_param, by model name. The reference re-initialises the model parameters
# for the flag BEFORE it applies the other modifiers, so `GKs*1.5,flags=ENDO`
# scales the ENDO default. Each region may ask for its own type: the types
# reach the model as the per-node parameter CELL_TYPE_PARAMETER, and each
# modifier is resolved against the default of the region's own type (see
# ParameterMapper.ionic_parameter_maps), which gives the same order. A model that is
# not listed here has no cell types, and a flags item on it is an error, as it
# is in the reference.
IONIC_CELL_TYPES = {'tenTusscherPanfilov': ('EPI', 'MCELL', 'ENDO')}

# Cell type of a model listed in IONIC_CELL_TYPES when im_param has no flags
# item: the model's own default.
DEFAULT_CELL_TYPE = {'tenTusscherPanfilov': 'EPI'}

# The per-node parameter that carries the cell type of a model listed in
# IONIC_CELL_TYPES. The model provides cell_type_default(pname, type), which
# gives the numeric id of a type for this name and each parameter's default.
CELL_TYPE_PARAMETER : str = 'celltype'

# The im_param item that selects a cell type rather than modifying a parameter.
IM_PARAM_FLAGS : str = 'flags='

# Ionic plugins for imp_region[].plugins, by the name the parameter file uses.
IONIC_PLUGINS = {'Electroporation_DeBruinKrassowska98': ElectroporationDeBruinKrassowska98,
                 'Defib_AshiharaTrayanova':             DefibAshiharaTrayanova}

# Separates the plugin names in imp_region[].plugins, and their parameter
# lists in imp_region[].plug_param.
PLUGIN_LIST_SEPARATOR : str = ':'

# parab_solve value of the theta method (the reference's Crank-Nicolson,
# its default); the only scheme gpuSolve implements for the diffusion step.
PARAB_SOLVE_THETA : int = 1

# The range of theta the reference accepts. Values outside it but inside
# (0, 1] are run, with a note, because theta = 1 (implicit Euler) is useful.
THETA_REFERENCE_MIN : float = 0.1
THETA_REFERENCE_MAX : float = 0.99

# Label of a stimulus that the input leaves unnamed; the index is appended.
# It is the reference simulator's own label, for either stimulus family, so a
# message about a stimulus names the same one in the logs of both solvers.
DEFAULT_STIM_NAME : str = 'Stimulus_'

# Cell-parameter names that differ between the two vocabularies. Everything
# else (tau_in, tau_out, tau_open, tau_close) is spelled the same way.
IM_PARAM_ALIASES = {'V_gate': 'u_gate',
                    'a_crit': 'u_crit',
                    'V_min': 'vmin',
                    'V_max': 'vmax'}

# The operator characters that end a cell-parameter name in an `im_param` item.
# `=` assigns, the other four modify the cell model default in place, which is
# why the name is cut at the FIRST of them: `GNa*0.3` is a scaling of GNa, not a
# parameter called "GNa*0".
PARAM_MOD_OPERATORS : str = '=+-/*'

# The registry. Each entry is (type, default, actuated). `actuated` False means
# the key is accepted and reported but cannot change what gpuSolve computes.
# `None` as a default marks one that is derived from other keys at resolve time.
REGISTRY = {
    'meshname':                     ('str',   'project', True),
    'simID':                        ('str',   'OUTPUT',  True),
    'dt':                           ('float', 5.0,       True),
    'tend':                         ('float', 100.0,     True),
    'spacedt':                      ('float', 3.0,       True),
    'timedt':                       ('float', 1.0,       True),
    'vofile':                       ('str',   'vm',      True),
    'gridout_i':                    ('int',   0,         True),
    # the one default that is NOT the reference's (0): node renumbering only
    # reorders the unknowns, and the time step is several times faster with it
    # on a mesh in the mesher's own order (see HeatSolver). Output keeps the
    # mesh numbering. renumbering = 0 restores the previous numerics.
    'renumbering':                  ('int',   1,         True),
    'num_gregions':                 ('int',   1,         True),
    'num_imp_regions':              ('int',   1,         True),
    'num_stim':                     ('int',   0,         True),
    'cg_tol_parab':                 ('float', 1.0e-8,    True),
    'cg_maxit_parab':               ('int',   100,       True),
    'cg_norm_parab':                ('int',   0,         True),
    'bidm_eqv_mono':                ('int',   1,         True),
    'bidomain':                     ('int',   0,         False),
    'mass_lumping':                 ('int',   1,         False),
    'parab_solve':                  ('int',   1,         True),
    # weight of the new time level for parab_solve = 1. The reference accepts
    # 0.1 to 0.99; 1.0 (implicit Euler) is accepted here as well
    'theta':                        ('float', 0.5,       True),
    'operator_splitting':           ('int',   1,         False),
    'gregion[].name':               ('str',   '',        True),
    'gregion[].g_il':               ('float', 0.174,     True),
    'gregion[].g_it':               ('float', 0.019,     True),
    'gregion[].g_in':               ('float', 0.019,     False),
    'gregion[].g_el':               ('float', 0.625,     True),
    'gregion[].g_et':               ('float', 0.236,     True),
    'gregion[].g_en':               ('float', 0.236,     False),
    'gregion[].g_mult':             ('float', 1.0,       True),
    'gregion[].num_IDs':            ('int',   0,         True),
    'gregion[].ID':                 ('idset', '',        True),
    'gregion[].ID[]':               ('int',   -1,        True),
    'imp_region[].name':            ('str',   '',        True),
    'imp_region[].im':              ('str',   '',        True),
    'imp_region[].im_param':        ('str',   '',        True),
    'imp_region[].plugins':         ('str',   '',        True),
    'imp_region[].im_sv_init':      ('str',   '',        True),
    'imp_region[].plug_param':      ('str',   '',        True),
    'imp_region[].cellSurfVolRatio': ('float', 0.14,     True),
    'imp_region[].volFrac':         ('float', 1.0,       True),
    'imp_region[].num_IDs':         ('int',   0,         True),
    'imp_region[].ID':              ('idset', '',        True),
    'imp_region[].ID[]':            ('int',   -1,        True),
    'stim[].name':                  ('str',   '',        True),
    'stim[].crct.type':             ('int',   0,         True),
    'stim[].pulse.strength':        ('float', 0.0,       True),
    'stim[].ptcl.start':            ('float', 0.0,       True),
    'stim[].ptcl.duration':         ('float', None,      True),
    'stim[].ptcl.npls':             ('int',   None,      True),
    'stim[].ptcl.bcl':              ('float', None,      True),
    'stim[].elec.p0[]':             ('float', 0.0,       True),
    'stim[].elec.p1[]':             ('float', 0.0,       True),
    'stim[].elec.vtx_file':         ('str',   '',        True),
    # the legacy stimulus family. It is translated onto the same electrode and
    # protocol as stim[] (see ParameterMapper.stimuli); the defaults are the
    # reference's, a 100 um box with its corner at the origin.
    'stimulus[].name':              ('str',   '',        True),
    'stimulus[].stimtype':          ('int',   0,         True),
    'stimulus[].strength':          ('float', 0.0,       True),
    'stimulus[].start':             ('float', 0.0,       True),
    'stimulus[].duration':          ('float', None,      True),
    'stimulus[].npls':              ('int',   None,      True),
    'stimulus[].bcl':               ('float', None,      True),
    'stimulus[].x0':                ('float', 0.0,       True),
    'stimulus[].y0':                ('float', 0.0,       True),
    'stimulus[].z0':                ('float', 0.0,       True),
    'stimulus[].xd':                ('float', 100.0,     True),
    'stimulus[].yd':                ('float', 100.0,     True),
    'stimulus[].zd':                ('float', 100.0,     True),
    'stimulus[].ctr_def':           ('int',   0,         True),
    'stimulus[].vtx_file':          ('str',   '',        True),
    'num_tsav':                     ('int',   0,         True),
    'tsav[]':                       ('float', None,      True),
    'tsav_ext[]':                   ('str',   None,      True),
    'write_statef':                 ('str',   'state',   True),
    'start_statef':                 ('str',   '',        True),
    'chkpt_start':                  ('float', 0.0,       True),
    'chkpt_intv':                   ('float', 0.0,       True),
    'chkpt_stop':                   ('float', None,      True),
    'prepacing_lats':               ('str',   '',        True),
    'prepacing_beats':              ('int',   0,         True),
    'prepacing_bcl':                ('float', -1.0,      True),
    'prepacing_stimdur':            ('float', 1.0,       True),
    'prepacing_stimstr':            ('float', 60.0,      True),
    # accepted so that files which set them run, but not acted upon: the mesh
    # reader takes no format switch, and local activation times are not
    # computed by this front end (see the notes in __collect_notes)
    'meshformat':                   ('int',   0,         False),
    'num_LATs':                     ('int',   0,         False),
    'lats[].ID':                    ('str',   '',        False),
    'lats[].all':                   ('int',   1,         False),
    'lats[].measurand':             ('int',   0,         False),
    'lats[].threshold':             ('float', -10.0,     False),
    'lats[].mode':                  ('int',   0,         False),
}

# Keys of the legacy stimulus family that openCARP reads with the same meaning
# as a stim[] key, by the stim[] key they are translated onto.
LEGACY_STIM_KEYS = {'name': 'name',
                    'stimtype': 'crct.type',
                    'strength': 'pulse.strength',
                    'start': 'ptcl.start',
                    'duration': 'ptcl.duration',
                    'npls': 'ptcl.npls',
                    'bcl': 'ptcl.bcl',
                    'vtx_file': 'elec.vtx_file'}

_INDEX = re.compile(r'\[(\d+)\]')


def pattern_of(key: str) -> str:
    """ pattern_of(key) replaces every [<integer>] with [] so a concrete key
        such as gregion[1].ID[2] can be looked up in the registry
    """
    return(_INDEX.sub('[]', key))


def indices_of(key: str) -> list:
    """ indices_of(key) returns the bracket indices of key, in order """
    return([int(m) for m in _INDEX.findall(key)])


def expand_idset(text: str) -> list:
    """ expand_idset(text) expands an aggregate tag list such as
        "100:200,203,300:2:400" into the explicit tags it names.
        The `from:to` form is INCLUSIVE at both ends, and `from:step:to` walks
        it with a stride. Blanks separate entries just as commas do.
    """
    tags : list = []
    for chunk in text.replace(',', ' ').split():
        parts = chunk.split(':')
        if len(parts) == 1:
            tags.append(int(parts[0]))
        elif len(parts) == 2:
            tags += list(range(int(parts[0]), 1 + int(parts[1])))
        elif len(parts) == 3:
            start, step, stop = int(parts[0]), int(parts[1]), int(parts[2])
            if step == 0:
                raise ValueError('zero stride in tag range "{}"'.format(chunk))
            tags += list(range(start, 1 + stop if step > 0 else stop - 1, step))
        else:
            raise ValueError('cannot read tag range "{}"'.format(chunk))
    return(tags)


def time_label(ctime: float) -> str:
    """ time_label(ctime) writes a time in ms for a file name, without trailing
        zeros: 100.0 -> "100", 12.5 -> "12.5". Six decimals are kept before the
        zeros are trimmed, so two save times one microsecond apart still get
        different names.
    """
    return('{:.6f}'.format(ctime).rstrip('0').rstrip('.'))


def parse_im_param(text: str, aliases: dict = None) -> dict:
    """ parse_im_param(text, aliases) reads a "name<op>value,name<op>value"
        cell-parameter list and returns {name: (op, value, percent)} keyed by
        the gpuSolve parameter name. The value is NOT resolved here: `GNa*0.3`
        means "the model default scaled by 0.3", so the modifier has to travel
        as far as the point where the cell model, and therefore that default,
        is known. Use apply_param_mod() there.
        aliases renames parameters on the way in; it defaults to
        IM_PARAM_ALIASES, the cell-model renames. Plugin parameters pass {}:
        the plugins use the reference's own names.
    """
    params : dict = {}
    for chunk in text.split(','):
        item = chunk.strip().replace(' ', '')
        # a flags item selects a cell type, not a parameter; im_flags() reads it
        if len(item) == 0 or item.startswith(IM_PARAM_FLAGS):
            continue
        params.update([split_param_mod(item, aliases)])
    return(params)


def im_flags(text: str) -> str:
    """ im_flags(text) returns the value of the `flags=<TYPE>` item of an
        im_param list, or '' when there is none. Two flags items are an error:
        the reference would apply both in turn, and which one wins is not
        something a parameter file should rely on.
    """
    found : list = []
    for chunk in text.split(','):
        item = chunk.strip().replace(' ', '')
        if item.startswith(IM_PARAM_FLAGS):
            found.append(item[len(IM_PARAM_FLAGS):])
    if len(found) > 1:
        raise ValueError('im_param "{}" has more than one flags item'.format(text))
    return(found[0] if len(found) == 1 else '')


def split_param_mod(item: str, aliases: dict = None) -> tuple:
    """ split_param_mod(item, aliases) cuts a cell-parameter item at its FIRST
        operator and returns (gpuSolve name, (op, value, percent)).
        The name ends at the first of `= + - / *`, so `GNa*0.3` is the parameter
        GNa scaled by 0.3, and `GNa=0.3` assigns it outright. A trailing `%`
        makes the operand that percentage OF THE CURRENT VALUE. aliases is as
        in parse_im_param().
    """
    if aliases is None:
        aliases = IM_PARAM_ALIASES
    for ipos, char in enumerate(item):
        if char in PARAM_MOD_OPERATORS:
            name    = item[:ipos]
            operand = item[1 + ipos:]
            if len(name) == 0:
                raise ValueError('cannot read cell parameter "{}": no parameter name '
                                 'before "{}"'.format(item, char))
            percent = operand.endswith('%')
            if percent:
                operand = operand[:-1]
            try:
                value = float(operand)
            except ValueError:
                # strict on purpose: the reference simulator logs a malformed
                # modifier and silently keeps the default, which turns a typo
                # into a run that looks fine and is not the one that was asked
                # for.
                raise ValueError('cannot read cell parameter "{}": "{}" is not a '
                                 'number'.format(item, operand))
            return((aliases.get(name, name), (char, value, percent)))
    raise ValueError('cannot read cell parameter "{}": expected name=value, or a '
                     'modifier such as name*0.3'.format(item))


def apply_param_mod(base: float, modifier: tuple) -> float:
    """ apply_param_mod(base, modifier) resolves one (op, value, percent) triple
        against the cell model default `base` and returns the parameter value.
        A percentage operand is first turned into that fraction of `base`, so
        `GNa-10%` is base - 0.1 base and `GNa=10%` is 0.1 base.
    """
    op, value, percent = modifier
    if percent:
        value = base * value / 100.0
    if op == '=':
        return(value)
    if op == '*':
        return(base * value)
    if op == '/':
        return(base / value)
    if op == '+':
        return(base + value)
    return(base - value)


class ParameterMapper:
    """
    class ParameterMapper: maps a resolved parameter store onto gpuSolve.
    resolve() must be called once before any of the accessors below.
    """

    def __init__(self):
        self.__store : dict  = None
        self.__notes : list  = None
        self.__counts : dict = None

    # ---- resolution ---------------------------------------------------------
    def resolve(self, store: dict):
        """ resolve(store) validates every key of store and prepares the typed
            lookups. An unknown key stops the run here rather than being
            ignored, so a misspelled parameter is never silently inert.
        """
        try:
            self.__store  = store
            self.__notes  = []
            self.__validate()
            self.__counts = {}
            for prefix in ARRAY_COUNTERS:
                self.__counts[prefix] = self.__resolve_count(prefix)
            self.__collect_notes()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def notes(self) -> list:
        """ notes() returns the lines to print in the run banner: every place
            where the resolved parameters ask for something gpuSolve does
            differently
        """
        return(self.__notes)

    def add_note(self, note: str):
        """ add_note(note) records a line for the run banner. The simulation
            runner uses it for what only shows up while the run is being built
            (a restart that overwrites an existing output file)
        """
        self.__notes.append(note)

    def count(self, prefix: str) -> int:
        """ count(prefix) returns the number of entries of an indexed family
            ('gregion', 'imp_region' or 'stim')
        """
        return(self.__counts[prefix])

    def value(self, key: str):
        """ value(key) returns the typed value of key, falling back to the
            documented default when the key was never assigned
        """
        pattern = pattern_of(key)
        if pattern not in REGISTRY:
            raise ValueError('Unrecognized keyword {}'.format(key))
        vtype, default, _actuated = REGISTRY[pattern]
        raw = self.__store.get(key)
        if raw is None:
            return(default)
        return(self.__cast(raw, vtype, key))

    # ---- what the solver needs ---------------------------------------------
    def solver_config(self) -> dict:
        """ solver_config() returns the config dict of HeatSolver /
            MonodomainSolver. dt arrives in microseconds and is converted here,
            which is also why dt_per_plot is derived from spacedt in ms.
        """
        dt_ms   = self.value('dt') * DT_MICROSECONDS_TO_MS
        spacedt = self.value('spacedt')
        # at least one step between two recorded frames, whatever spacedt says
        dt_per_plot = max(1, int(round(spacedt / dt_ms)))
        return({'mesh_file_name': self.value('meshname'),
                'dt': dt_ms,
                'Tend': self.value('tend'),
                'dt_per_plot': dt_per_plot,
                'use_renumbering': self.value('renumbering') != 0,
                'theta': self.diffusion_theta()})

    def diffusion_theta(self) -> float:
        """ diffusion_theta() returns the theta of the diffusion step.
            parab_solve = 1 (the default) is the theta method with the `theta`
            key (0.5, Crank-Nicolson). The explicit (0) and second-order (2)
            schemes are not implemented; they fall back to implicit Euler
            (theta = 1), what every value gave before, with a note.
        """
        if self.value('parab_solve') != PARAB_SOLVE_THETA:
            return(1.0)
        theta = self.value('theta')
        if not (0.0 < theta <= 1.0):
            raise ValueError('theta = {}: must lie in (0, 1] (0.5 Crank-Nicolson, '
                             '1 implicit Euler)'.format(theta))
        return(theta)

    def solver_settings(self) -> dict:
        """ solver_settings() returns the ConjGrad settings. cg_norm_parab
            selects which stopping test is armed: 0 and 1 are absolute, 2 is
            relative and 3 arms both, which maps one to one onto the absolute
            and relative tolerances ConjGrad already carries. The disarmed one
            is set to 0.0, the value that can never fire.
        """
        norm = self.value('cg_norm_parab')
        tol  = self.value('cg_tol_parab')
        return({'toll': tol if norm in (0, 1, 3) else 0.0,
                'toll_rel': tol if norm in (2, 3) else 0.0,
                'maxiter': self.value('cg_maxit_parab')})

    def output_settings(self) -> dict:
        """ output_settings() returns where and how often to write results """
        return({'simID': self.value('simID'),
                'vofile': self.value('vofile'),
                'gridout_i': self.value('gridout_i'),
                'timedt': self.value('timedt')})

    def prepacing_settings(self) -> dict:
        """ prepacing_settings() returns the single-cell prepacing parameters,
            with every time in ms:
              'lats_file': the activation-time file that guides the distribution
              'beats':   how many beats to pace
              'bcl':     the basic cycle length of the prepacing train
              'stimdur': the duration of a prepacing stimulus
              'stimstr': its strength, in uA/uF
              'dt':      the integration step, the solver's own
            Prepacing is off unless bcl is positive, which is the reference's
            switch, and unless a file of activation times names where each cell
            sits in the activation sequence.
        """
        return({'lats_file': self.value('prepacing_lats'),
                'beats': self.value('prepacing_beats'),
                'bcl': self.value('prepacing_bcl'),
                'stimdur': self.value('prepacing_stimdur'),
                'stimstr': self.value('prepacing_stimstr'),
                'dt': self.value('dt') * DT_MICROSECONDS_TO_MS,
                'tend': self.value('tend')})

    def savestate_settings(self) -> dict:
        """ savestate_settings() returns when to save the state and what to resume
            from, with every time in ms:
              'save_times':   [(time, file name without extension)], one per tsav
              'start_statef': the checkpoint to resume from ('' for a fresh run)
              'chkpt_start', 'chkpt_intv', 'chkpt_stop': interval checkpointing,
                              off when chkpt_intv is 0
            The derived defaults are the reference's: a tsav that is not given is
            one time step before tend, and its file-name suffix is the save time.
        """
        tend  = self.value('tend')
        dt_ms = self.value('dt') * DT_MICROSECONDS_TO_MS
        nsav  = self.count('tsav')
        if nsav > MAX_SAVE_TIMES:
            raise ValueError('num_tsav = {}: at most {} save times are accepted'.format(
                nsav, MAX_SAVE_TIMES))
        basename = self.value('write_statef')
        saves : list = []
        for index in range(nsav):
            tsav = self.value('tsav[{}]'.format(index))
            if tsav is None:
                tsav = tend - dt_ms
            if tsav < 0.0:
                raise ValueError('tsav[{}] = {}: a save time cannot be negative'.format(index, tsav))
            suffix = self.value('tsav_ext[{}]'.format(index))
            if suffix is None or len(suffix.strip()) == 0:
                suffix = time_label(tsav)
            saves.append((tsav, '{}.{}'.format(basename, suffix.strip())))
        chkpt_start = self.value('chkpt_start')
        chkpt_intv  = self.value('chkpt_intv')
        chkpt_stop  = self.value('chkpt_stop')
        if chkpt_stop is None:
            chkpt_stop = tend
        for key, val in (('chkpt_start', chkpt_start), ('chkpt_intv', chkpt_intv),
                         ('chkpt_stop', chkpt_stop)):
            if val < 0.0 or val > tend:
                raise ValueError('{} = {}: must lie between 0 and tend = {}'.format(key, val, tend))
        return({'save_times': saves,
                'start_statef': self.value('start_statef').strip(),
                'chkpt_start': chkpt_start,
                'chkpt_intv': chkpt_intv,
                'chkpt_stop': chkpt_stop})

    def region_tags(self, prefix: str, index: int) -> list:
        """ region_tags(prefix, index) returns the element tags claimed by one
            entry. An empty list means the entry claims every tag implicitly.
            Both spellings are accepted: the aggregate `ID = 1:3,7` and the
            indexed `ID[0] = 1` with its num_IDs counter.
        """
        aggregate = self.__store.get('{}[{}].ID'.format(prefix, index))
        if aggregate is not None:
            return(expand_idset(aggregate))
        tags : list = []
        nid = self.value('{}[{}].num_IDs'.format(prefix, index))
        if nid > 0:
            for jid in range(nid):
                tags.append(self.value('{}[{}].ID[{}]'.format(prefix, index, jid)))
        else:
            # no counter given: take the indexed entries that are actually there
            jid = 0
            while '{}[{}].ID[{}]'.format(prefix, index, jid) in self.__store:
                tags.append(self.value('{}[{}].ID[{}]'.format(prefix, index, jid)))
                jid += 1
        return(tags)

    def tag_to_entry(self, prefix: str, tags: set) -> dict:
        """ tag_to_entry(prefix, tags) maps each mesh tag to the index of the
            entry that governs it. An entry that lists no tag claims all of them
            implicitly, an explicit claim overrides an implicit one, and a tag
            claimed by nobody is absent from the map so the caller falls back to
            the documented defaults. A tag claimed by an entry but missing from
            the mesh is reported and otherwise ignored.
        """
        assignment : dict = {}
        for index in range(self.count(prefix)):
            if len(self.region_tags(prefix, index)) == 0:
                for tag in tags:
                    assignment[tag] = index
        for index in range(self.count(prefix)):
            for tag in self.region_tags(prefix, index):
                if tag in tags:
                    assignment[tag] = index
                else:
                    self.__notes.append('region tag {} in {}[{}] is not in the '
                                        'element list'.format(tag, prefix, index))
        return(assignment)

    def element_property_maps(self, tags: set) -> dict:
        """ element_property_maps(tags) returns the {tag: value} maps of the
            three element properties the diffusion tensor needs: sigma_l,
            sigma_t (already in um^2/ms) and beta.
        """
        sigma_l : dict = {}
        sigma_t : dict = {}
        beta : dict    = {}
        gassign = self.tag_to_entry('gregion', tags)
        iassign = self.tag_to_entry('imp_region', tags)
        equivalent = self.value('bidm_eqv_mono') != 0
        for tag in tags:
            gindex = gassign.get(tag)
            if gindex is None:
                g_il, g_it = REGISTRY['gregion[].g_il'][1], REGISTRY['gregion[].g_it'][1]
                g_el, g_et = REGISTRY['gregion[].g_el'][1], REGISTRY['gregion[].g_et'][1]
                g_mult     = REGISTRY['gregion[].g_mult'][1]
            else:
                g_il   = self.value('gregion[{}].g_il'.format(gindex))
                g_it   = self.value('gregion[{}].g_it'.format(gindex))
                g_el   = self.value('gregion[{}].g_el'.format(gindex))
                g_et   = self.value('gregion[{}].g_et'.format(gindex))
                g_mult = self.value('gregion[{}].g_mult'.format(gindex))
            if equivalent:
                # the bidomain-equivalent monodomain conductivity is the
                # half harmonic mean of the two domains, eigenvalue by eigenvalue
                g_il = self.__parallel(g_il, g_el)
                g_it = self.__parallel(g_it, g_et)
            sigma_l[tag] = CONDUCTIVITY_TO_UM2_PER_MS * g_mult * g_il
            sigma_t[tag] = CONDUCTIVITY_TO_UM2_PER_MS * g_mult * g_it
            iindex = iassign.get(tag)
            if iindex is None:
                beta[tag] = (REGISTRY['imp_region[].cellSurfVolRatio'][1]
                             * REGISTRY['imp_region[].volFrac'][1])
            else:
                beta[tag] = (self.value('imp_region[{}].cellSurfVolRatio'.format(iindex))
                             * self.value('imp_region[{}].volFrac'.format(iindex)))
            if beta[tag] <= 0.0:
                raise ValueError('imp_region[{}]: cellSurfVolRatio * volFrac must be '
                                 'positive, got {}'.format(iindex, beta[tag]))
        return({'sigma_l': sigma_l, 'sigma_t': sigma_t, 'beta': beta})

    def ionic_model_name(self) -> str:
        """ ionic_model_name() returns the cell model named by the imp_regions,
            or '' when none is named and the run is pure diffusion
        """
        names = set()
        for index in range(self.count('imp_region')):
            name = self.value('imp_region[{}].im'.format(index)).strip()
            if len(name) > 0:
                names.add(name)
        if len(names) == 0:
            return('')
        if len(names) > 1:
            raise ValueError('this front end solves one cell model at a time, but the '
                             'imp_regions name {}'.format(', '.join(sorted(names))))
        return(names.pop())

    def ionic_model_class(self):
        """ ionic_model_class() returns the gpuSolve class for the named model;
            None for a pure diffusion run. `MitchellSchaeffer` selects the
            modified variant as soon as any region gives it a non-zero a_crit,
            which is exactly what that parameter means.
        """
        name = self.ionic_model_name()
        if len(name) == 0:
            return(None)
        if name not in IONIC_MODELS:
            raise ValueError('unknown cell model "{}"; this front end provides {}'.format(
                name, ', '.join(sorted(IONIC_MODELS.keys()))))
        if IONIC_MODELS[name] is not None:
            return(IONIC_MODELS[name])
        modified = False
        for index in range(self.count('imp_region')):
            params = parse_im_param(self.value('imp_region[{}].im_param'.format(index)))
            if 'u_crit' in params:
                # no model exists yet, so a modifier has nothing to modify. It is
                # resolved against the modified variant's own default, because
                # the plain one has no u_crit at all: naming the parameter only
                # makes sense against the model that owns it.
                reference = float(ModifiedMS2v().get_parameter('u_crit'))
                if apply_param_mod(reference, params['u_crit']) != 0.0:
                    modified = True
        return(ModifiedMS2v if modified else MitchellSchaeffer2v)

    def ionic_model_options(self) -> dict:
        """ ionic_model_options() returns the extra constructor arguments of the
            cell model, as {argument: value}: today only {'cell_type': <TYPE>}
            when the model has cell types (IONIC_CELL_TYPES). The type comes
            from the `flags=<TYPE>` item of im_param; a region with no flags
            item asks for the model default, as it would in the reference.
            When every region asks for the same type, that type is the
            constructor default; when they differ, the constructor gets the
            model default (it then only governs tags no region claims) and the
            per-region types reach the model node by node through
            ionic_parameter_maps.
        """
        types, requested = self.__region_cell_types()
        if types is None:
            return({})
        name = self.ionic_model_name()
        distinct = set(requested.values())
        if len(distinct) == 1:
            return({'cell_type': distinct.pop()})
        return({'cell_type': DEFAULT_CELL_TYPE[name]})

    def ionic_parameter_maps(self, model, tags: set) -> dict:
        """ ionic_parameter_maps(model, tags) returns {parameter: {tag: value}}
            for every cell parameter that some im_param names. A tag whose
            region does not mention a parameter keeps the model's own default,
            so a parameter file and a hand-written script agree wherever the
            file is silent.
            For a model with cell types, when the regions ask for different
            types, the map starts with CELL_TYPE_PARAMETER (the numeric type of
            each tag; it comes first because setting it resets the type-
            dependent parameters), and every other parameter is resolved
            against the default of the tag's own cell type: flags first, then
            modifiers, as in the reference. Tags no region claims take the
            constructor type (ionic_model_options).
        """
        assignment = self.tag_to_entry('imp_region', tags)
        per_region : dict = {}
        named : set       = set()
        for index in range(self.count('imp_region')):
            per_region[index] = parse_im_param(self.value('imp_region[{}].im_param'.format(index)))
            named |= set(per_region[index].keys())
        maps : dict = {}
        tag_types = self.__tag_cell_types(assignment, tags)
        # the cell model itself, also when plugins wrap it
        cell_model = model.model() if hasattr(model, 'plugin_names') else model
        if tag_types is not None and len(set(tag_types.values())) > 1:
            maps[CELL_TYPE_PARAMETER] = {tag: cell_model.cell_type_default(CELL_TYPE_PARAMETER, ctype)
                                         for tag, ctype in tag_types.items()}
        for pname in sorted(named):
            reference = model.get_parameter(pname)
            if reference is None:
                raise ValueError('cell model {} has no parameter "{}"'.format(
                    model.model_name(), pname))
            maps[pname] = {}
            for tag in tags:
                if tag_types is None:
                    fallback = float(reference)
                else:
                    fallback = cell_model.cell_type_default(pname, tag_types[tag])
                index = assignment.get(tag)
                modifier = per_region.get(index, {}).get(pname)
                if modifier is None:
                    maps[pname][tag] = fallback
                else:
                    maps[pname][tag] = apply_param_mod(fallback, modifier)
        return(maps)

    def __region_cell_types(self) -> tuple:
        """ (types, {imp_region index: type}) for the regions that name a cell
            model; types is None, and the dict empty, for a model without cell
            types. Validates the flags items.
        """
        name = self.ionic_model_name()
        types = IONIC_CELL_TYPES.get(name)
        requested : dict = {}
        for index in range(self.count('imp_region')):
            if len(self.value('imp_region[{}].im'.format(index)).strip()) == 0:
                continue
            flag = im_flags(self.value('imp_region[{}].im_param'.format(index)))
            if types is None:
                if len(flag) > 0:
                    raise ValueError('imp_region[{}].im_param: flags={} is given, but cell model '
                                     '{} has no cell types in this front end'.format(
                                         index, flag, name))
                continue
            if len(flag) == 0:
                flag = DEFAULT_CELL_TYPE[name]
            if flag not in types:
                # a "|"-separated list lands here too: a node has exactly one
                # cell type, so a combination has no meaning
                raise ValueError('imp_region[{}].im_param: flags={} is not a cell type of {}; '
                                 'use one of {}'.format(index, flag, name, ', '.join(types)))
            requested[index] = flag
        return((types, requested))

    def __tag_cell_types(self, assignment: dict, tags: set) -> dict:
        """ {tag: cell type} for a model with cell types, None otherwise. A tag
            no region claims takes the constructor type.
        """
        types, requested = self.__region_cell_types()
        if types is None:
            return(None)
        default = self.ionic_model_options()['cell_type']
        return({tag: requested.get(assignment.get(tag), default) for tag in tags})

    def region_plugins(self, index: int) -> list:
        """ region_plugins(index) returns the plugin names imp_region[index].plugins
            lists, in order. An unknown name and a name listed twice are errors:
            the reference accepts a repeated plugin but tunes only its first
            copy, so two copies cannot differ there and the file is refused
            rather than read differently.
        """
        text  = self.value('imp_region[{}].plugins'.format(index))
        names = [name.strip() for name in text.split(PLUGIN_LIST_SEPARATOR) if len(name.strip()) > 0]
        for name in names:
            if name not in IONIC_PLUGINS:
                raise ValueError('imp_region[{}].plugins: unknown plugin "{}"; this front end '
                                 'provides {}'.format(index, name,
                                                      ', '.join(sorted(IONIC_PLUGINS.keys()))))
        repeated = sorted(set(name for name in names if names.count(name) > 1))
        if len(repeated) > 0:
            raise ValueError('imp_region[{}].plugins lists {} more than once. The reference '
                             'simulator accepts this but applies every plug_param entry to the '
                             'first copy and leaves the others at their defaults, so the copies '
                             'cannot be tuned apart; list each plugin once and scale its '
                             'parameters instead'.format(index, ', '.join(repeated)))
        return(names)

    def state_init_files(self, tags: set) -> list:
        """ state_init_files(tags) returns the single-cell state files that
            imp_region[].im_sv_init names, one entry per region that names one:
              'index':   the index of the imp_region
              'name':    the name of the region, for the messages
              'file':    the state file
              'tags':    the element tags the region governs, sorted
              'plugins': the plugin names the region lists, in order
            A region whose file is empty, and one that governs no tag of the
            mesh, are left out; the second is noted, because a file that reaches
            no node is almost always a mistake in the tag list.
        """
        assignment = self.tag_to_entry('imp_region', tags)
        entries : list = []
        for index in range(self.count('imp_region')):
            fname = self.value('imp_region[{}].im_sv_init'.format(index)).strip()
            if len(fname) == 0:
                continue
            claimed = sorted(tag for tag, owner in assignment.items() if owner == index)
            if len(claimed) == 0:
                self.add_note('imp_region[{}].im_sv_init = "{}" is ignored: the region governs no '
                              'element tag of this mesh'.format(index, fname))
                continue
            entries.append({'index':   index,
                            'name':    self.value('imp_region[{}].name'.format(index)),
                            'file':    fname,
                            'tags':    claimed,
                            'plugins': self.region_plugins(index)})
        return(entries)

    def ionic_plugin_classes(self) -> list:
        """ ionic_plugin_classes() returns the plugin classes of the run: every
            plugin that some imp_region lists, once, in the order they are first
            listed. A plugin can then be switched on in some regions only (see
            plugin_parameter_maps). Plugins need a cell model to attach to.
        """
        classes : list = []
        for index in range(self.count('imp_region')):
            for name in self.region_plugins(index):
                if IONIC_PLUGINS[name] not in classes:
                    classes.append(IONIC_PLUGINS[name])
        if len(classes) > 0 and len(self.ionic_model_name()) == 0:
            raise ValueError('imp_region[].plugins names a plugin but no imp_region names a cell '
                             'model (imp_region[].im): a plugin adds a current to a cell model '
                             'and cannot run on its own')
        return(classes)

    def plugin_parameter_maps(self, model, tags: set) -> dict:
        """ plugin_parameter_maps(model, tags) returns {parameter: {tag: value}}
            for the plugins of an IonicModelWithPlugins, with the parameters
            named '<plugin class>.<name>':
              * '<plugin class>.active', 1 on the tags of the regions that list
                the plugin and 0 elsewhere (omitted when it is 1 everywhere);
              * every parameter that some plug_param mentions. Entry i of a
                region's plug_param (':'-separated) tunes plugin i of that
                region's plugins list; a tag whose region does not mention the
                parameter keeps the plugin default.
        """
        assignment = self.tag_to_entry('imp_region', tags)
        per_region : dict = {}
        for index in range(self.count('imp_region')):
            names = self.region_plugins(index)
            text  = self.value('imp_region[{}].plug_param'.format(index)).strip()
            lists = text.split(PLUGIN_LIST_SEPARATOR) if len(text) > 0 else []
            if len(lists) > len(names):
                raise ValueError('imp_region[{}].plug_param has {} entries but plugins lists {} '
                                 'plugin{}; entry i tunes plugin i'.format(
                                     index, len(lists), len(names), '' if len(names) == 1 else 's'))
            per_region[index] = {}
            for iplug, name in enumerate(names):
                text_i = lists[iplug] if iplug < len(lists) else ''
                per_region[index][IONIC_PLUGINS[name].__name__] = parse_im_param(text_i, {})
        maps : dict = {}
        for classname in model.plugin_names():
            active = {}
            for tag in tags:
                index = assignment.get(tag)
                active[tag] = 1.0 if classname in per_region.get(index, {}) else 0.0
            if any(value != 1.0 for value in active.values()):
                maps['{}{}{}'.format(classname, PLUGIN_SEPARATOR, ACTIVE_PARAMETER)] = active
            named : set = set()
            for params in per_region.values():
                named |= set(params.get(classname, {}).keys())
            for pname in sorted(named):
                full = '{}{}{}'.format(classname, PLUGIN_SEPARATOR, pname)
                reference = model.get_parameter(full)
                if reference is None or pname == ACTIVE_PARAMETER:
                    raise ValueError('plugin {} has no parameter "{}"'.format(classname, pname))
                fallback = float(reference)
                maps[full] = {}
                for tag in tags:
                    modifier = per_region.get(assignment.get(tag), {}).get(classname, {}).get(pname)
                    if modifier is None:
                        maps[full][tag] = fallback
                    else:
                        maps[full][tag] = apply_param_mod(fallback, modifier)
        return(maps)

    def stimuli(self) -> list:
        """ stimuli() returns one (props, geometry) pair per stimulus: the dict
            Stimulus is built from, and how the stimulated nodes are selected.

            The electrode is described either by a vertex file, as
            {'vtx_file': name}, or geometrically by the corners of a box in
            micrometres, as {'p0': [...], 'p1': [...]}. A vertex file names the
            nodes outright and therefore wins over a box, which is what a
            non-empty vtx_file means in this format.
        """
        stims : list = []
        tend = self.value('tend')
        legacy = self.__uses_legacy_stimuli()
        # both families share num_stim; without it, each infers its own size
        for index in range(self.count('stimulus' if legacy else 'stim')):
            if legacy:
                entry = self.__legacy_stimulus(index)
            else:
                entry = {member: self.value('stim[{}].{}'.format(index, member))
                         for member in LEGACY_STIM_KEYS.values()}
                entry['elec.p0'] = [self.value('stim[{}].elec.p0[{}]'.format(index, k)) for k in range(3)]
                entry['elec.p1'] = [self.value('stim[{}].elec.p1[{}]'.format(index, k)) for k in range(3)]
            ctype = entry['crct.type']
            if ctype != 0:
                raise ValueError('stim[{}].crct.type = {}: this front end applies transmembrane '
                                 'stimuli (type 0) only, and silently treating an intra- or '
                                 'extracellular electrode as one would change the '
                                 'physics'.format(index, ctype))
            start    = entry['ptcl.start']
            duration = entry['ptcl.duration']
            npls     = entry['ptcl.npls']
            bcl      = entry['ptcl.bcl']
            name     = entry['name']
            # the derived defaults of the reference: a protocol that says
            # nothing is one pulse covering the rest of the simulation
            if duration is None:
                duration = tend - start
            if npls is None:
                npls = 1 if tend > 0.0 else 0
            if bcl is None:
                bcl = tend - start
            props = {'tstart': start,
                     'nstim': npls,
                     'period': bcl if bcl > 0.0 else 1.0,
                     'duration': duration,
                     'intensity': entry['pulse.strength'],
                     'name': name if len(name) > 0 else '{}{}'.format(DEFAULT_STIM_NAME, index)}
            p0 = entry['elec.p0']
            p1 = entry['elec.p1']
            vtx_file = entry['elec.vtx_file'].strip()
            if len(vtx_file) > 0:
                if any(corner != 0.0 for corner in p0 + p1):
                    self.__notes.append('stim[{}] names both a vertex file and a box: the '
                                        'vertex file defines the electrode'.format(index))
                stims.append((props, {'vtx_file': vtx_file}))
            else:
                stims.append((props, {'p0': p0, 'p1': p1}))
        return(stims)

    # ---- internals ----------------------------------------------------------
    def __uses_legacy_stimuli(self) -> bool:
        """ whether the stimuli are written with the legacy stimulus[] keys.
            Both families share num_stim and their indices. The reference picks
            ONE family for the whole run, and when both are set it keeps stim[]
            and drops every stimulus[] entry with only a warning
            (simulator/sim_utils.cc, the legacy_stim_set / new_stim_set test).
            A file that mixes them is refused here instead: silently dropping
            the stimuli someone wrote is the kind of change a run must not hide.
        """
        legacy = any(key.startswith('stimulus[') for key in self.__store)
        modern = any(key.startswith('stim[') for key in self.__store)
        if legacy and modern:
            raise ValueError('both stim[] and the legacy stimulus[] keys are set. The reference '
                             'would use stim[] only and drop every stimulus[] entry; write all '
                             'the stimuli with one of the two families')
        return(legacy)

    def __legacy_stimulus(self, index: int) -> dict:
        """ translates stimulus[index] onto the stim[] members, keyed as in
            LEGACY_STIM_KEYS plus 'elec.p0' and 'elec.p1'. The box follows the
            reference translation (physics/stimulate.cc, stimulus::translate):
            p0 = x0 - (ctr_def ? xd/2 : 0), p1 = p0 + xd, and the same in y, z,
            so ctr_def makes (x0, y0, z0) the centre of the box rather than its
            corner.
        """
        entry : dict = {}
        for legacy, member in LEGACY_STIM_KEYS.items():
            entry[member] = self.value('stimulus[{}].{}'.format(index, legacy))
        centred = self.value('stimulus[{}].ctr_def'.format(index)) != 0
        p0 : list = []
        p1 : list = []
        for axis in ('x', 'y', 'z'):
            origin = self.value('stimulus[{}].{}0'.format(index, axis))
            extent = self.value('stimulus[{}].{}d'.format(index, axis))
            corner = origin - (0.5 * extent if centred else 0.0)
            p0.append(corner)
            p1.append(corner + extent)
        entry['elec.p0'] = p0
        entry['elec.p1'] = p1
        return(entry)

    def __parallel(self, g_intra: float, g_extra: float) -> float:
        """ half the harmonic mean of the two domain conductivities """
        total = g_intra + g_extra
        if total <= 0.0:
            return(0.0)
        return((g_intra * g_extra) / total)

    def __cast(self, raw: str, vtype: str, key: str):
        """ turns the text of an assignment into the type the registry declares """
        try:
            if vtype == 'int':
                # "1" and "1.0" both name the integer 1
                return(int(float(raw)))
            if vtype == 'float':
                return(float(raw))
            if vtype == 'idset':
                return(expand_idset(raw))
            return(raw)
        except ValueError:
            raise ValueError('{}: cannot read "{}" as {}'.format(key, raw, vtype))

    def __validate(self):
        """ every assigned key must be in the registry """
        unknown = sorted(k for k in self.__store if pattern_of(k) not in REGISTRY)
        if len(unknown) > 0:
            raise ValueError('Unrecognized keyword{}: {}'.format(
                '' if len(unknown) == 1 else 's', ', '.join(unknown)))

    def __resolve_count(self, prefix: str) -> int:
        """ how many entries an indexed family has. The counter wins when it is
            given, because that is what it is for; an entry beyond it is an
            error rather than a silent omission. With no counter the count is
            inferred from the highest index that appears.
        """
        counter = ARRAY_COUNTERS[prefix]
        highest = -1
        head    = '{}['.format(prefix)
        for key in self.__store:
            if key.startswith(head):
                highest = max(highest, indices_of(key)[0])
        if counter in self.__store:
            declared = int(float(self.__store[counter]))
            if highest >= declared:
                raise ValueError('{} = {} but {}[{}] is assigned: raise the counter or '
                                 'drop the entry'.format(counter, declared, prefix, highest))
            return(declared)
        return(1 + highest)

    def __collect_notes(self):
        """ record every resolved value that asks for something gpuSolve does
            differently. Silence means the request and the implementation agree.
        """
        if self.value('bidomain') != 0:
            self.__notes.append('bidomain = {} {}: gpuSolve solves the monodomain equation '
                                'only'.format(self.value('bidomain'), self.__origin_of('bidomain')))
        if self.value('mass_lumping') != 0:
            self.__notes.append('mass_lumping = {} {}: gpuSolve assembles the consistent mass '
                                'matrix, i.e. it behaves as mass_lumping = 0'.format(
                                    self.value('mass_lumping'), self.__origin_of('mass_lumping')))
        if self.value('operator_splitting') == 0:
            self.__notes.append('operator_splitting = 0 {}: gpuSolve always splits the ionic '
                                'and the diffusion update'.format(
                                    self.__origin_of('operator_splitting')))
        if self.value('parab_solve') != PARAB_SOLVE_THETA:
            self.__notes.append('parab_solve = {} {}: only the theta method (1) is implemented; '
                                'the diffusion term is advanced with implicit Euler '
                                '(theta = 1)'.format(self.value('parab_solve'),
                                                     self.__origin_of('parab_solve')))
        elif not (THETA_REFERENCE_MIN <= self.value('theta') <= THETA_REFERENCE_MAX):
            self.__notes.append('theta = {} {}: outside the reference range [{}, {}]; accepted '
                                'here (1 is implicit Euler)'.format(
                                    self.value('theta'), self.__origin_of('theta'),
                                    THETA_REFERENCE_MIN, THETA_REFERENCE_MAX))
        for index in range(self.__count_or_zero('tsav')):
            tsav = self.value('tsav[{}]'.format(index))
            if tsav is not None and tsav > self.value('tend'):
                self.__notes.append('tsav[{}] = {} is after tend = {}: that state is never '
                                    'saved'.format(index, tsav, self.value('tend')))
        if any(key.startswith('stimulus[') for key in self.__store):
            self.__notes.append('legacy stimulus[] keys: the reference shapes every legacy pulse '
                                'as a truncated exponential (tau_edge, tau_plateau); gpuSolve '
                                'applies a square pulse of the same strength and duration')
        if self.value('num_LATs') > 0 or any(key.startswith('lats[') for key in self.__store):
            self.__notes.append('num_LATs / lats[] are set: local activation times are not '
                                'computed, so no LAT file is written')
        if self.value('prepacing_bcl') > 0.0:
            # every one of these leaves prepacing switched on but unable to do
            # anything, and the run then starts from the resting state without
            # a word unless it is said here
            if self.value('prepacing_beats') <= 0:
                self.__notes.append('prepacing_bcl = {} is set but prepacing_beats = {}: there '
                                    'is nothing to pace, so no prepacing is done'.format(
                                        self.value('prepacing_bcl'),
                                        self.value('prepacing_beats')))
            if len(self.value('prepacing_lats')) == 0:
                self.__notes.append('prepacing_bcl = {} is set but prepacing_lats names no '
                                    'file: the activation times say where each cell sits in '
                                    'the activation sequence, so no prepacing is done'.format(
                                        self.value('prepacing_bcl')))
        elif self.value('prepacing_beats') > 0:
            self.__notes.append('prepacing_beats = {} is set but prepacing_bcl = {}: prepacing '
                                'is switched on by a positive cycle length, so no prepacing is '
                                'done'.format(self.value('prepacing_beats'),
                                              self.value('prepacing_bcl')))
        if 'meshformat' in self.__store:
            self.__notes.append('meshformat = {} is accepted but not used: the mesh reader '
                                'does not take a format switch'.format(self.value('meshformat')))
        for index in range(self.__count_or_zero('gregion')):
            for member in ('g_in', 'g_en'):
                if 'gregion[{}].{}'.format(index, member) in self.__store:
                    self.__notes.append('gregion[{}].{} is set: the diffusion tensor is '
                                        'transversely isotropic, so the sheet-normal '
                                        'conductivity is not used'.format(index, member))

    def __origin_of(self, key: str) -> str:
        """ says whether a value was written down or inherited. A note about a
            key the user never typed is otherwise baffling: most of these fire
            on the default, not on anything the input asked for.
        """
        if key in self.__store:
            return('(set in the input)')
        return('(the default of this format; the input does not set it)')

    def __count_or_zero(self, prefix: str) -> int:
        """ the family size during note collection, before count() is safe """
        return(self.__counts.get(prefix, 0) if self.__counts is not None else 0)
