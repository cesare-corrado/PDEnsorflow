#!/usr/bin/env python
"""
    ParameterMapper: turns a resolved parameter store into gpuSolve settings.

    This is the only class that speaks both vocabularies. It owns

      * the registry of known keys, with their type and their default. A key
        that is not in the registry is an error, so a typo is reported instead
        of being silently ignored;
      * the defaults, which are the reference simulator's, not gpuSolve's, so a
        file that omits a key means what the same file would mean elsewhere.
        The ONE exception is the ionic parameters: a cell parameter that no
        `im_param` mentions keeps the gpuSolve class default, because the ionic
        models are gpuSolve's own implementations and a script and a parameter
        file should agree;
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


# S/m and micrometres to um^2/ms, once beta (um^-1) has divided it out.
CONDUCTIVITY_TO_UM2_PER_MS : float = 1.0e5

# dt arrives in microseconds, every other time in milliseconds.
DT_MICROSECONDS_TO_MS : float = 1.0e-3

# Counter key of each indexed family. The names are irregular, so they are
# listed rather than derived.
ARRAY_COUNTERS = {'gregion': 'num_gregions',
                  'imp_region': 'num_imp_regions',
                  'stim': 'num_stim',
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
                'Fenton': Fenton4v}

# Cell-parameter names that differ between the two vocabularies. Everything
# else (tau_in, tau_out, tau_open, tau_close) is spelled the same way.
IM_PARAM_ALIASES = {'V_gate': 'u_gate',
                    'a_crit': 'u_crit',
                    'V_min': 'vmin',
                    'V_max': 'vmax'}

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
    'renumbering':                  ('int',   0,         True),
    'num_gregions':                 ('int',   1,         True),
    'num_imp_regions':              ('int',   1,         True),
    'num_stim':                     ('int',   0,         True),
    'cg_tol_parab':                 ('float', 1.0e-8,    True),
    'cg_maxit_parab':               ('int',   100,       True),
    'cg_norm_parab':                ('int',   0,         True),
    'bidm_eqv_mono':                ('int',   1,         True),
    'bidomain':                     ('int',   0,         False),
    'mass_lumping':                 ('int',   1,         False),
    'parab_solve':                  ('int',   1,         False),
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
    'num_tsav':                     ('int',   0,         True),
    'tsav[]':                       ('float', None,      True),
    'tsav_ext[]':                   ('str',   None,      True),
    'write_statef':                 ('str',   'state',   True),
    'start_statef':                 ('str',   '',        True),
    'chkpt_start':                  ('float', 0.0,       True),
    'chkpt_intv':                   ('float', 0.0,       True),
    'chkpt_stop':                   ('float', None,      True),
}

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


def parse_im_param(text: str) -> dict:
    """ parse_im_param(text) reads a "name=value,name=value" cell-parameter list
        and returns it keyed by the gpuSolve parameter name
    """
    params : dict = {}
    for chunk in text.split(','):
        item = chunk.strip()
        if len(item) == 0:
            continue
        if '=' not in item:
            raise ValueError('cannot read cell parameter "{}": expected name=value'.format(item))
        name, value = item.split('=', 1)
        name  = name.strip()
        params[IM_PARAM_ALIASES.get(name, name)] = float(value.strip())
    return(params)


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
                'use_renumbering': self.value('renumbering') != 0})

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
            if params.get('u_crit', 0.0) != 0.0:
                modified = True
        return(ModifiedMS2v if modified else MitchellSchaeffer2v)

    def ionic_parameter_maps(self, model, tags: set) -> dict:
        """ ionic_parameter_maps(model, tags) returns {parameter: {tag: value}}
            for every cell parameter that some im_param names. A tag whose
            region does not mention a parameter keeps the model's own default,
            so a parameter file and a hand-written script agree wherever the
            file is silent.
        """
        assignment = self.tag_to_entry('imp_region', tags)
        per_region : dict = {}
        named : set       = set()
        for index in range(self.count('imp_region')):
            per_region[index] = parse_im_param(self.value('imp_region[{}].im_param'.format(index)))
            named |= set(per_region[index].keys())
        maps : dict = {}
        for pname in sorted(named):
            reference = model.get_parameter(pname)
            if reference is None:
                raise ValueError('cell model {} has no parameter "{}"'.format(
                    type(model).__name__, pname))
            fallback = float(reference)
            maps[pname] = {}
            for tag in tags:
                index = assignment.get(tag)
                maps[pname][tag] = per_region.get(index, {}).get(pname, fallback)
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
        for index in range(self.count('stim')):
            ctype = self.value('stim[{}].crct.type'.format(index))
            if ctype != 0:
                raise ValueError('stim[{}].crct.type = {}: this front end applies transmembrane '
                                 'stimuli (type 0) only, and silently treating an intra- or '
                                 'extracellular electrode as one would change the '
                                 'physics'.format(index, ctype))
            start    = self.value('stim[{}].ptcl.start'.format(index))
            duration = self.value('stim[{}].ptcl.duration'.format(index))
            npls     = self.value('stim[{}].ptcl.npls'.format(index))
            bcl      = self.value('stim[{}].ptcl.bcl'.format(index))
            name     = self.value('stim[{}].name'.format(index))
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
                     'intensity': self.value('stim[{}].pulse.strength'.format(index)),
                     'name': name if len(name) > 0 else 'stim{}'.format(index)}
            p0 = [self.value('stim[{}].elec.p0[{}]'.format(index, k)) for k in range(3)]
            p1 = [self.value('stim[{}].elec.p1[{}]'.format(index, k)) for k in range(3)]
            vtx_file = self.value('stim[{}].elec.vtx_file'.format(index)).strip()
            if len(vtx_file) > 0:
                if any(corner != 0.0 for corner in p0 + p1):
                    self.__notes.append('stim[{}] names both a vertex file and a box: the '
                                        'vertex file defines the electrode'.format(index))
                stims.append((props, {'vtx_file': vtx_file}))
            else:
                stims.append((props, {'p0': p0, 'p1': p1}))
        return(stims)

    # ---- internals ----------------------------------------------------------
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
        self.__notes.append('parab_solve = {} {}: gpuSolve advances the diffusion term with '
                            'implicit Euler, which is none of the three values this key '
                            'offers'.format(self.value('parab_solve'),
                                            self.__origin_of('parab_solve')))
        for index in range(self.__count_or_zero('tsav')):
            tsav = self.value('tsav[{}]'.format(index))
            if tsav is not None and tsav > self.value('tend'):
                self.__notes.append('tsav[{}] = {} is after tend = {}: that state is never '
                                    'saved'.format(index, tsav, self.value('tend')))
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
