#!/usr/bin/env python
"""
    A TensorFlow-based Cardiac Electrophysiology Modeler

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
import tensorflow as tf

from gpuSolve.ionic.ionicmodel import IonicModel
from gpuSolve.ionic.plugins.ionicplugin import IonicPlugin
from gpuSolve.ionic.plugins.ionicplugin import PLUGIN_DTYPE

# Separates a plugin's class name from the name of one of its parameters or
# state variables ('ElectroporationDeBruinKrassowska98.n'), so that a plugin
# variable can never collide with a variable of the cell model.
PLUGIN_SEPARATOR : str = '.'

# Joins the class names of the model and of its plugins in model_name(), the
# name a checkpoint records ('Tomek+ElectroporationDeBruinKrassowska98').
PLUGIN_JOINER : str = '+'

# The per-node parameter that switches a plugin on (1) or off (0) at each node:
# '<plugin class>.active'. It lets one plugin cover only some regions.
ACTIVE_PARAMETER : str = 'active'


class IonicModelWithPlugins(IonicModel):
    """
    A cell model together with the ionic plugins attached to it.

    The step follows the reference implementation: the model computes its
    current first, then each plugin adds its own, in the order the plugins
    were attached, all at the potential of the start of the step, and the
    potential is updated with the sum. The model never sees a plugin's state.

    To the solver this is one IonicModel. Names are routed as follows:
      * a bare name ('GNa', 'm', 'V_init') belongs to the model;
      * '<plugin class>.<name>' belongs to that plugin, for parameters and
        state variables alike ('ElectroporationDeBruinKrassowska98.sigma');
      * '<plugin class>.active' is the per-node switch of that plugin: 1 where
        it adds its current, 0 where it does not (default 1 everywhere). Where
        it is 0 the plugin's state is still advanced, but its current is not
        used.
    Each plugin class may be attached once: see add_plugin().

    Usage:
        model = IonicModelWithPlugins(dt)
        model.set_model(Tomek(dt))
        model.add_plugin(ElectroporationDeBruinKrassowska98(dt))
    """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)
        self._model   : IonicModel = None
        self._plugins : list       = []
        self._active  : dict       = {}

    # ---- composition --------------------------------------------------------
    def set_model(self, model: IonicModel):
        """ set_model(model) sets the cell model the plugins are attached to """
        try:
            if isinstance(model, IonicPlugin):
                raise ValueError('{} is a plugin, not a cell model'.format(type(model).__name__))
            if isinstance(model, IonicModelWithPlugins):
                raise ValueError('the cell model of IonicModelWithPlugins cannot itself hold '
                                 'plugins; attach all of them to one IonicModelWithPlugins')
            self._model = model
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def add_plugin(self, plugin: IonicPlugin):
        """
        add_plugin(plugin) attaches plugin after the ones already attached.
        A plugin class may be attached only once. The reference implementation
        accepts the same plugin twice in a region, but it applies every
        parameter string to the first copy and leaves the second at its
        defaults, so two copies cannot be tuned independently there; it is
        refused here rather than reproduced.
        """
        try:
            if not isinstance(plugin, IonicPlugin):
                raise ValueError('{} is not an ionic plugin'.format(type(plugin).__name__))
            name = type(plugin).__name__
            if name in self.plugin_names():
                raise ValueError('plugin {} is already attached: each plugin may be used once per '
                                 'model (the reference implementation would tune only the first '
                                 'copy and leave the second at its defaults)'.format(name))
            self._plugins.append(plugin)
            self._active[name] = tf.constant(1.0, dtype=PLUGIN_DTYPE)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def model(self) -> IonicModel:
        """ model() returns the cell model """
        return(self._model)

    def plugins(self) -> list:
        """ plugins() returns the attached plugins, in the order they run """
        return(list(self._plugins))

    def plugin_names(self) -> list:
        """ plugin_names() returns the class names of the attached plugins, in order """
        return([type(plugin).__name__ for plugin in self._plugins])

    def model_name(self) -> str:
        """
        model_name() returns the name a checkpoint records: the class name of the
        model followed by those of the plugins, in order, so a checkpoint
        restores only into a run with the same model and the same plugins. With
        no plugin it is the model's own name.
        """
        return(PLUGIN_JOINER.join([self._model.model_name()] + self.plugin_names()))

    # ---- time step ----------------------------------------------------------
    def set_dt(self, dt: float):
        """ set_dt(dt) sets the time step (ms) of the model and of every plugin """
        self._dt = dt
        self._model.set_dt(dt)
        for plugin in self._plugins:
            plugin.set_dt(dt)

    # ---- parameters ---------------------------------------------------------
    def set_parameter(self, pname: str, pvalue: np.ndarray):
        """
        set_parameter(pname, pvalue) sets a model parameter (bare name), a plugin
        parameter ('<plugin>.<name>') or a plugin switch ('<plugin>.active',
        0 or 1, one value or one per node)
        """
        try:
            plugin, name = self.__split(pname)
            if plugin is None:
                self._model.set_parameter(pname, pvalue)
            elif name == ACTIVE_PARAMETER:
                values = np.asarray(pvalue, dtype=np.float64)
                if not np.all(np.isin(values, [0.0, 1.0])):
                    raise ValueError('{} must be 0 or 1 at every node'.format(pname))
                self._active[type(plugin).__name__] = tf.constant(values, dtype=PLUGIN_DTYPE)
            else:
                plugin.set_parameter(name, pvalue)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def get_parameter(self, pname: str) -> tf.Tensor:
        """
        get_parameter(pname) returns a model parameter (bare name), a plugin
        parameter ('<plugin>.<name>') or a plugin switch ('<plugin>.active');
        None if there is no such parameter
        """
        plugin, name = self.__split(pname, strict=False)
        if plugin is None:
            if PLUGIN_SEPARATOR in pname:
                return(None)
            return(self._model.get_parameter(pname))
        if name == ACTIVE_PARAMETER:
            return(self._active[type(plugin).__name__])
        if name not in plugin.tunable_parameter_names():
            return(None)
        return(plugin.get_parameter(name))

    # ---- state --------------------------------------------------------------
    def initialize_state_variables(self, U: tf.Variable):
        """
        initialize_state_variables(U) initializes the model, then each plugin, at
        the potential U. This is the order of the reference implementation,
        where the plugins start from the potential the model has just set.
        The time step is handed down first: the solver sets it on this object
        only, and some models fold it into their lookup tables here.
        """
        if not self._initialized:
            try:
                if self._model is None:
                    raise ValueError('IonicModelWithPlugins has no cell model; call set_model() first')
                self.set_dt(self._dt)
                self._model.initialize_state_variables(U)
                for plugin in self._plugins:
                    plugin.initialize_state_variables(U)
                self._initialized = True
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

    def state_variable_names(self) -> tuple:
        """
        state_variable_names() returns the model's state variables, then each
        plugin's, prefixed with the plugin class name
        """
        names = list(self._model.state_variable_names())
        for plugin in self._plugins:
            prefix = type(plugin).__name__
            names += ['{}{}{}'.format(prefix, PLUGIN_SEPARATOR, name)
                      for name in plugin.state_variable_names()]
        return(tuple(names))

    def state_variable(self, name: str) -> tf.Variable:
        """ state_variable(name) returns the tf.Variable of a model or plugin state variable """
        plugin, local = self.__split(name, strict=False)
        if plugin is None:
            if PLUGIN_SEPARATOR in name:
                return(None)
            return(self._model.state_variable(name))
        return(plugin.state_variable(local))

    # ---- dynamics -----------------------------------------------------------
    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """
        differentiate(U) returns dU = -(Iion of the model + the current of every
        active plugin) and advances the model and the plugins by dt. The model
        and each plugin run as their own compiled steps; each plugin current is
        computed in float64 and cast to the dtype of dU when it is added.
        """
        dU = self._model.differentiate(U)
        for plugin in self._plugins:
            current = plugin.compute_current(U)
            # the switch is one value or one per node; both are flattened so
            # an (n, 1) column cannot broadcast against an (n,) current
            active  = tf.reshape(self._active[type(plugin).__name__], [-1])
            current = tf.reshape(tf.reshape(current, [-1]) * active, tf.shape(dU))
            dU = dU - tf.cast(current, dU.dtype)
        return(dU)

    # ---- internals ----------------------------------------------------------
    def __split(self, pname: str, strict: bool = True) -> tuple:
        """
        splits '<plugin class>.<name>' into (plugin, name). A bare name gives
        (None, pname). An unknown plugin raises when strict, and gives
        (None, pname) otherwise.
        """
        if PLUGIN_SEPARATOR not in pname:
            return((None, pname))
        prefix, name = pname.split(PLUGIN_SEPARATOR, 1)
        for plugin in self._plugins:
            if type(plugin).__name__ == prefix:
                return((plugin, name))
        if strict:
            raise ValueError('"{}" names plugin {}, which is not attached; the attached plugins '
                             'are {}'.format(pname, prefix, ', '.join(self.plugin_names()) or 'none'))
        return((None, pname))
