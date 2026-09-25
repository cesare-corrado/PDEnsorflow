#!/usr/bin/env python
"""
    SvStateFile: a single-cell state file (`.sv`) matched to the cell model of a
    run. It turns the file the reference simulator writes (see
    IO/readers/svfilereader.py for the layout) into the potential and the state
    variables of one cell, named as gpuSolve names them.

    It is what both front ends read a state file with:
      * `singlecell --read-ini-file <file>`, the equivalent of bench's option of
        the same name, which starts the cell from that state;
      * `imp_region[].im_sv_init = <file>` in a parameter file, which starts
        every node of that ionic region from it.

    What the file carries and what is taken from it
    -----------------------------------------------
    The file holds one section per object (the cell model, then each of its
    plugins), and the reference reads a section by POSITION: every entry of its
    internal state structure is in the file, whether or not gpuSolve keeps it as
    state (svlayouts.py explains the three kinds). Only the `state` entries are
    taken, in the gpuSolve units of the layout; a `parameter` entry is NOT
    applied, because the parameters of a run come from the model defaults and
    the `im_param` / `--imp-par` modifiers, and a file written by another run
    would otherwise change the model behind the caller's back. Each parameter
    entry whose value differs from the one the run uses is reported instead, so
    the difference is visible rather than silent.

    A file whose sections are not the run's model and plugins is refused. The
    reference's own reader refuses it too: its generated read_svs() compares the
    section name with the region's model, logs "IMPs do not match region" and
    returns 2, which the tissue solver turns into a fatal error
    (physics/ionics.cc). bench instead carries on from rest.

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
import numpy as np

from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.ionic.ionicmodelwithplugins import PLUGIN_SEPARATOR
from gpuSolve.IO.readers.svfilereader import SvFileReader
from gpuSolve.carp_compatibility.svlayouts import sv_layout


# Relative difference above which a parameter stored in a state file is
# reported as different from the one the run uses: the reference writes 6
# significant digits, so its own values agree to about 5e-6.
SV_PARAMETER_RTOL : float = 1.0e-5


class SvStateFile:
    """
    class SvStateFile: reads a single-cell state file and matches it to a run.
    Configured with a dict, as the writers are:

        {'model': <the run's cell model>, 'imp_name': 'Courtemanche',
         'plugins': [(<plugin object>, 'Electroporation')], 'nodes': None}

    'model' is the object the parameters are read from: the cell model itself,
    or the IonicModelWithPlugins that wraps it; 'plugins' are the objects whose
    sections follow the model's, in file order, each with the name the run
    selected it by; 'nodes' restricts the parameter comparison to the nodes the
    state is meant for, for a model whose parameters vary over the mesh.

    Usage:
        svfile = SvStateFile({'model': model, 'imp_name': 'Tomek'})
        svfile.read('state.sv')
        svfile.Vm()      -> the potential, in mV
        svfile.states()  -> {state variable name: value}, in gpuSolve units
    """

    def __init__(self, config: dict = None):
        self._model                   = None
        self._imp_name : str          = ''
        self._plugins : list          = []
        self._nodes : np.ndarray      = None
        self._fname : str             = ''
        self._Vm : float              = None
        self._states : dict           = None
        self._mismatches : list       = None

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    # ---- accessors ----------------------------------------------------------
    def model(self):
        """ model() returns the object the parameters are compared against """
        return(self._model)

    def set_model(self, model):
        """ set_model(model) sets the cell model of the run (the bare model, or
            the IonicModelWithPlugins that wraps it)
        """
        self._model = model

    def imp_name(self) -> str:
        """ imp_name() returns the name the run selected the cell model by """
        return(self._imp_name)

    def set_imp_name(self, imp_name: str):
        """ set_imp_name(imp_name) sets the name the run selected the cell model
            by. It is the section name of a model the reference does not have,
            and unused for one it does (svlayouts.py holds the reference's name)
        """
        self._imp_name = imp_name

    def plugins(self) -> list:
        """ plugins() returns [(plugin object, name)], in file order """
        return(self._plugins)

    def set_plugins(self, plugins: list):
        """ set_plugins(plugins) sets the plugins whose sections follow the
            model's, as [(plugin object, name)] in the order they are listed
        """
        self._plugins = plugins

    def nodes(self) -> np.ndarray:
        """ nodes() returns the node indices the state is meant for, None when
            the state is a single cell's
        """
        return(self._nodes)

    def set_nodes(self, nodes: np.ndarray):
        """ set_nodes(nodes) restricts the parameter comparison to these nodes:
            a tissue run holds one parameter value per node, and only the values
            of the nodes the state is written to can be compared with the file
        """
        self._nodes = np.reshape(np.asarray(nodes), (-1,))

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self._fname)

    def Vm(self) -> float:
        """ Vm() returns the potential of the file, in mV; None before read() """
        return(self._Vm)

    def states(self) -> dict:
        """ states() returns {state variable name: value} in gpuSolve units and
            with the plugin prefixes of the run's model; None before read()
        """
        return(self._states)

    def mismatches(self) -> list:
        """ mismatches() returns one message per parameter entry of the file
            whose value differs from the one the run uses; those entries are not
            applied. Empty when the file agrees with the run
        """
        return(self._mismatches)

    # ---- reading ------------------------------------------------------------
    def read(self, fname: str):
        """ read(fname) reads the state file fname and matches it to the model:
            it fills Vm(), states() and mismatches(), and raises when the file
            does not describe this model and these plugins
        """
        try:
            reader = SvFileReader()
            reader.read(fname)
            self._fname      = fname
            self._states     = {}
            self._mismatches = []
            found = dict(reader.global_values())
            if found.get('Vm') is None:
                raise ValueError('{}: the state file has no Vm'.format(fname))
            expected = self.sections()
            given    = reader.sections()
            if [section[0] for section in given] != [section[0] for section in expected]:
                raise ValueError('{}: the state file holds {}, but this run has {}. The reference '
                                 'would skip such a file and start from rest without an error; '
                                 'this front end stops instead'.format(
                                     fname, ' + '.join(section[0] for section in given),
                                     ' + '.join(section[0] for section in expected)))
            for (section, entries, prefix), (_name, values) in zip(expected, given):
                self.__read_section(fname, section, entries, prefix, values)
            self._Vm = float(found['Vm'])
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def sections(self) -> list:
        """ sections() returns [(section name, entries, prefix)] for the cell
            model, then each plugin; prefix is how the run's model names that
            object's states and parameters
        """
        sections = [sv_layout(self.__cell(), self._imp_name) + ('',)]
        for plugin, name in self._plugins:
            prefix = '{}{}'.format(type(plugin).__name__, PLUGIN_SEPARATOR)
            sections.append(sv_layout(plugin, name) + (prefix,))
        return(sections)

    # ---- internals ----------------------------------------------------------
    def __cell(self):
        """ the bare cell model, whether the caller gave it or the wrapper """
        if isinstance(self._model, IonicModelWithPlugins):
            return(self._model.model())
        return(self._model)

    def __read_section(self, fname: str, section: str, entries: list, prefix: str, values: list):
        """ one section: the state entries are kept, the others compared """
        names = [entry[0] for entry in entries]
        if [value[0] for value in values] != names:
            raise ValueError('{}: section {} lists {}, but the model expects {} (in this '
                             'order)'.format(fname, section, ', '.join(v[0] for v in values),
                                             ', '.join(names)))
        for (entry, kind, source, scale, _gate), (_n, value) in zip(entries, values):
            if kind == 'state':
                self._states[prefix + source] = value / scale
                continue
            used = self.entry_value(kind, prefix, source)
            if used is None:
                self._mismatches.append('state file {} sets {} = {:g} ({}); ignored, this run '
                                        'does not hold it as one value'.format(
                                            fname, entry, value, section))
                continue
            used = used * scale
            if abs(value - used) > SV_PARAMETER_RTOL * max(abs(value), abs(used)):
                self._mismatches.append('state file {} sets {} = {:g} ({}); ignored, this run uses '
                                        '{:g}'.format(fname, entry, value, section, used))

    def entry_value(self, kind: str, prefix: str, source):
        """ entry_value(kind, prefix, source) returns the value the run uses for
            a parameter ('parameter', source is its name) or a fixed entry
            ('constant', source is the value), in gpuSolve units. It is None
            when the run holds no single value for it, i.e. when the parameter
            varies over the nodes the state is written to
        """
        if kind == 'constant':
            return(float(source))
        value = self._model.get_parameter(prefix + source)
        if value is None:
            raise ValueError('the state file entry {} names {}, which this model does not '
                             'have'.format(source, prefix + source))
        values = np.reshape(np.asarray(value, dtype=np.float64), (-1,))
        # a parameter registered per node holds one value per node of the whole
        # mesh; only the nodes this state is written to are relevant here
        if self._nodes is not None and values.size > 1:
            values = values[self._nodes]
        if values.size > 1 and not np.all(values == values[0]):
            return(None)
        return(float(values[0]))
