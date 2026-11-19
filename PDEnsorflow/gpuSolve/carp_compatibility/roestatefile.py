#!/usr/bin/env python
"""
    RoeStateFile: the binary tissue state file of the reference simulator
    (`.roe`, see gpuSolve.IO.readers.roereader) matched to the ionic regions of
    a run. It builds the file content from a MonodomainSolver, and turns a file
    back into the potential, the state variables and the stored cell
    parameters of the run's nodes, named as gpuSolve names them.

    What a file must agree on
    -------------------------
    The reference restores a file only into the configuration that wrote it
    (MULTI_IF::restore_state), and this class applies the same checks, in the
    same order, each with its own message:
      1. the number of nodes;
      2. the number of ionic regions (imp_region entries);
      3. per region, the cell model name, the record size and, in format 3,
         the fingerprint of the field list;
      4. per region, the plugins, matched one to one;
      5. the region of every node (the mask).
    A model selected by an alias (Tomek19) is written and expected under the
    alias, as the reference names it; a file written under the canonical name
    does not restore into a run selected by the alias, nor the reverse.

    What is taken from a file
    -------------------------
    Vm, every 'state' entry of every region and plugin, and every 'parameter'
    entry. The reference copies the whole record back into its state
    structure, parameter fields included, so after a restart its cells run
    with the STORED parameters, not those of the parameter file. This class
    keeps that behaviour (the caller applies stored_parameters()) and the
    caller reports every parameter whose stored value differs from the run's,
    so that the difference is not silent. A 'constant' entry is compared only.
    Iion is not taken: the solver computes it again at the first step.

    The records follow the layouts of svlayouts.py (the reference's state
    structures), in the C layout of format 1 or the packed one of format 3.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import numpy as np

from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.ionic.ionicmodelwithplugins import PLUGIN_SEPARATOR
from gpuSolve.IO.readers.roereader import RoeReader
from gpuSolve.IO.readers.roereader import fingerprint
from gpuSolve.IO.readers.roereader import record_dtype
from gpuSolve.IO.writers.roewriter import encode_records
from gpuSolve.carp_compatibility.parametermapper import IONIC_PLUGINS
from gpuSolve.carp_compatibility.svlayouts import roe_fields
from gpuSolve.carp_compatibility.svlayouts import sv_layout


# Relative difference above which a stored constant or parameter is reported
# as different from the run's. The file stores float64; gpuSolve may hold a
# parameter in float32, which agrees to about 6e-8.
ROE_PARAMETER_RTOL : float = 1.0e-6


class RoeStateFile:
    """
    class RoeStateFile: writes and reads the binary state file of a run.
    Configured with a dict, as SvStateFile is:

        {'model': <the run's ionic model (bare, or IonicModelWithPlugins)>,
         'imp_name': 'Tomek', 'region_plugins': [[], ['Electroporation...']],
         'node_regions': <imp_region index of every node, user order>,
         'format': 1}

    'region_plugins' holds, per imp_region in order, the plugin names the
    region lists; its length is the number of regions of the file.
    """

    def __init__(self, config: dict = None):
        self._model                  = None
        self._imp_name : str         = ''
        self._region_plugins : list  = []
        self._node_regions : np.ndarray = None
        self._format : int           = 1
        self._fname : str            = ''
        self._time : float           = None
        self._Vm : np.ndarray        = None
        self._states : dict          = None
        self._parameters : dict      = None
        self._mismatches : list      = None

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    # ---- accessors ----------------------------------------------------------
    def model(self):
        """ model() returns the run's ionic model """
        return(self._model)

    def set_model(self, model):
        """ set_model(model) sets the run's ionic model (bare, or with plugins) """
        self._model = model

    def imp_name(self) -> str:
        """ imp_name() returns the name the run selected the cell model by """
        return(self._imp_name)

    def set_imp_name(self, imp_name: str):
        """ set_imp_name(imp_name) sets the name the run selected the cell model by """
        self._imp_name = imp_name

    def region_plugins(self) -> list:
        """ region_plugins() returns the plugin names of each imp_region """
        return(self._region_plugins)

    def set_region_plugins(self, region_plugins: list):
        """ set_region_plugins(region_plugins) sets, per imp_region in order,
            the plugin names the region lists
        """
        self._region_plugins = region_plugins

    def node_regions(self) -> np.ndarray:
        """ node_regions() returns the imp_region index of every node """
        return(self._node_regions)

    def set_node_regions(self, node_regions: np.ndarray):
        """ set_node_regions(node_regions) sets the imp_region index of every
            node, in the user's node order
        """
        self._node_regions = np.reshape(np.asarray(node_regions), (-1,)).astype(np.int8)

    def format(self) -> int:
        """ format() returns the format written: 1 or 3 """
        return(self._format)

    def set_format(self, fmt: int):
        """ set_format(fmt) sets the format written: 1 (release 19.0) or 3 """
        self._format = fmt

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self._fname)

    def time(self) -> float:
        """ time() returns the time stored in the file read, in ms (float32) """
        return(self._time)

    def Vm(self) -> np.ndarray:
        """ Vm() returns the potential of the file read, one value per node """
        return(self._Vm)

    def states(self) -> dict:
        """ states() returns {state variable: (nodes, values)} of the file read,
            in gpuSolve units and names; a plugin's variables cover the nodes of
            the regions that list it only
        """
        return(self._states)

    def stored_parameters(self) -> dict:
        """ stored_parameters() returns {parameter: (nodes, values)} of the file
            read: the parameter fields of the records, in gpuSolve units and
            names, which the reference restores along with the state
        """
        return(self._parameters)

    def mismatches(self) -> list:
        """ mismatches() returns one message per constant entry of the file read
            whose value differs from the run's
        """
        return(self._mismatches)

    # ---- layout -------------------------------------------------------------
    def regions(self) -> list:
        """ regions() returns, per imp_region, (model name, entries, sections)
            where sections is [(object name, entries, prefix)] for the cell
            model then each plugin the region lists, in order; prefix is how
            the run's model names that object's states and parameters
        """
        cell = self.__cell()
        name, entries = sv_layout(cell, self._imp_name)
        attached = {}
        if isinstance(self._model, IonicModelWithPlugins):
            attached = {type(plugin).__name__: plugin for plugin in self._model.plugins()}
        out : list = []
        for plugin_names in self._region_plugins:
            sections = [(name, entries, '')]
            for pname in plugin_names:
                plugin = self.__plugin(attached, pname)
                pfile, pentries = sv_layout(plugin, pname)
                sections.append((pfile, pentries,
                                 '{}{}'.format(type(plugin).__name__, PLUGIN_SEPARATOR)))
            out.append((name, entries, sections))
        return(out)

    # ---- writing ------------------------------------------------------------
    def content(self, solver) -> dict:
        """ content(solver) returns the file content (the dict RoeWriter
            writes) of the solver's current state: the potential, the ionic
            current of the last step, and per node the record of its region,
            with the parameter fields holding the values the run uses there
        """
        try:
            checkpoint = solver.checkpoint()
            npt    = int(checkpoint['num_nodes'])
            states = checkpoint['state_variables']
            packed = self._format >= 3
            Iion   = solver.ionic_current()
            if Iion is None:
                Iion = np.zeros(npt, dtype=np.float64)
            self.__check_node_regions(npt)
            regions : list = []
            records : list = []
            for index, (name, entries, sections) in enumerate(self.regions()):
                nodes = np.flatnonzero(self._node_regions == index)
                parts : list = []
                plugins : list = []
                for jsec, (oname, oentries, prefix) in enumerate(sections):
                    fields = roe_fields(oentries)
                    values = {entry: self.__entry_values(solver, states, kind, prefix, source,
                                                         scale, nodes)
                              for entry, kind, source, scale, _gate in oentries}
                    parts.append(encode_records(values, fields, packed))
                    if jsec > 0:
                        plugins.append({'name': oname,
                                        'size': record_dtype(fields, packed).itemsize,
                                        'fingerprint': fingerprint(fields)})
                fields = roe_fields(entries)
                regions.append({'model': name, 'size': record_dtype(fields, packed).itemsize,
                                'fingerprint': fingerprint(fields), 'plugins': plugins})
                records.append(np.concatenate(parts, axis=1))
            return({'format': self._format,
                    'time': float(checkpoint['time']),
                    'num_nodes': npt,
                    'globals': {'Vm': np.asarray(checkpoint['Vm'], dtype=np.float64),
                                'Iion': np.asarray(Iion, dtype=np.float64)},
                    'regions': regions,
                    'mask': self._node_regions,
                    'records': records})
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __entry_values(self, solver, states: dict, kind: str, prefix: str, source,
                       scale: float, nodes: np.ndarray) -> np.ndarray:
        """ the file values of one entry at the nodes of a region """
        if kind == 'state':
            values = np.asarray(states[prefix + source], dtype=np.float64)[nodes]
        elif kind == 'parameter':
            values = solver.cell_parameter(prefix + source)
            if values is None:
                raise ValueError('the layout entry {} names {}, which this model does not '
                                 'have'.format(source, prefix + source))
            values = values[nodes]
        else:
            values = np.full(nodes.size, float(source), dtype=np.float64)
        return(values * scale)

    # ---- reading ------------------------------------------------------------
    def read(self, fname: str):
        """ read(fname) reads a state file and matches it to the run: it fills
            time(), Vm(), states(), stored_parameters() and mismatches(), and
            raises, naming the first difference, when the file does not
            describe this run's regions
        """
        try:
            reader  = RoeReader()
            content = reader.read(fname)
            npt     = self._node_regions.size
            self.__check_node_regions(npt)
            if content['num_nodes'] != npt:
                raise ValueError('{}: the state holds {} nodes, but the mesh has {}'.format(
                    fname, content['num_nodes'], npt))
            layout = self.regions()
            if len(content['regions']) != len(layout):
                raise ValueError('{}: the state was written with {} ionic region(s), this run '
                                 'has {}'.format(fname, len(content['regions']), len(layout)))
            if 'Vm' not in content['globals']:
                raise ValueError('{}: the state file has no Vm'.format(fname))
            packed = content['format'] >= 3
            order  = [self.__match_region(fname, index, content['regions'][index], layout[index],
                                          packed)
                      for index in range(len(layout))]
            if not np.array_equal(content['mask'], self._node_regions):
                differ = int(np.count_nonzero(content['mask'] != self._node_regions))
                raise ValueError('{}: the state assigns a different ionic region than this run '
                                 'does at {} node(s)'.format(fname, differ))
            self._states     = {}
            self._parameters = {}
            self._mismatches = []
            for index, (_name, _entries, sections) in enumerate(layout):
                nodes = reader.region_nodes(index)
                for jsec, (oname, oentries, prefix) in enumerate(sections):
                    plugin = None if jsec == 0 else order[index][jsec - 1]
                    decoded = reader.region_state(index, roe_fields(oentries), plugin=plugin)
                    self.__take(fname, oname, oentries, prefix, decoded, nodes)
            self._Vm    = np.asarray(content['globals']['Vm'], dtype=np.float64)
            self._time  = float(content['time'])
            self._fname = fname
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __match_region(self, fname: str, index: int, stored: dict, expected: tuple,
                       packed: bool) -> list:
        """ checks one region of the file against the run (checks 3 and 4) and
            returns, for each plugin the run lists, the position of its records
            in the file: the reference matches plugins one to one, in any order
        """
        name, entries, sections = expected
        if stored['model'] != name:
            raise ValueError('{}: the state holds cell model {} for region {}, this run uses '
                             '{}'.format(fname, stored['model'], index, name))
        fields = roe_fields(entries)
        size   = record_dtype(fields, packed).itemsize
        if stored['size'] != size:
            raise ValueError('{}: the state holds {} byte(s) per node for region {} ({}), this '
                             'run needs {}'.format(fname, stored['size'], index, name, size))
        if packed and stored['fingerprint'] != fingerprint(fields):
            raise ValueError('{}: the state-variable layout of region {} ({}) does not match '
                             'this model'.format(fname, index, name))
        wanted = sections[1:]
        if len(stored['plugins']) != len(wanted):
            raise ValueError('{}: the state holds {} plugin(s) for region {} ({}), this run has '
                             '{}'.format(fname, len(stored['plugins']), index, name, len(wanted)))
        used  = [False] * len(stored['plugins'])
        order : list = []
        for pname, pentries, _prefix in wanted:
            pfields = roe_fields(pentries)
            psize   = record_dtype(pfields, packed).itemsize
            found   = None
            for jplug, plug in enumerate(stored['plugins']):
                if (not used[jplug] and plug['name'] == pname and plug['size'] == psize
                        and (not packed or plug['fingerprint'] == fingerprint(pfields))):
                    found = jplug
                    break
            if found is None:
                raise ValueError('{}: plugin {} of region {} has no counterpart in the state '
                                 '(it holds {})'.format(fname, pname, index, ', '.join(
                                     p['name'] for p in stored['plugins'])))
            used[found] = True
            order.append(found)
        return(order)

    def __take(self, fname: str, oname: str, entries: list, prefix: str, decoded: dict,
               nodes: np.ndarray):
        """ files the decoded fields of one object of one region """
        for entry, kind, source, scale, _gate in entries:
            values = np.asarray(decoded[entry], dtype=np.float64) / scale
            if kind == 'constant':
                expected = float(source)
                if np.any(np.abs(values - expected) > ROE_PARAMETER_RTOL * np.maximum(
                        np.abs(values), abs(expected))):
                    self._mismatches.append('state file {} stores {} = {:g} ({}) where this run '
                                            'has the constant {:g}'.format(
                                                fname, entry, float(values[0]), oname, expected))
                continue
            target = self._states if kind == 'state' else self._parameters
            key    = prefix + source
            if key in target:
                old_nodes, old_values = target[key]
                target[key] = (np.concatenate((old_nodes, nodes)),
                               np.concatenate((old_values, values)))
            else:
                target[key] = (nodes, values)

    # ---- internals ----------------------------------------------------------
    def __cell(self):
        """ the bare cell model, whether the caller gave it or the wrapper """
        if isinstance(self._model, IonicModelWithPlugins):
            return(self._model.model())
        return(self._model)

    def __plugin(self, attached: dict, pname: str):
        """ the run's plugin object selected by the name pname """
        classname = IONIC_PLUGINS[pname].__name__
        if classname not in attached:
            raise ValueError('plugin {} is listed by a region but not attached to the '
                             'model'.format(pname))
        return(attached[classname])

    def __check_node_regions(self, npt: int):
        """ the region of every node must be known before a file is matched """
        if self._node_regions is None or self._node_regions.size != npt:
            raise ValueError('the region of every node must be set (node_regions) before a '
                             'state file is written or read')
