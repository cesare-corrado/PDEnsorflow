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

# Plugins are integrated in float64 whatever the precision of the potential:
# their formulas are typically evaluated far from rest (shock-induced potentials
# of several hundred mV) where float32 overflows or cancels. The current is
# cast back to the dtype of U by the model that holds the plugin.
PLUGIN_DTYPE = tf.float64


class IonicPlugin(IonicModel):
    """
    A base class for the ionic plugins.

    A plugin is a current with its own state variables that is added to the
    current of a cell model: it does not define the potential, it cannot run
    on its own, and it is attached to a model with IonicModelWithPlugins.
    Within a time step the model runs first, then each plugin in the order it
    was attached, all with the potential at the start of the step; the
    potential is updated with the sum of the currents afterwards.

    A plugin reuses the parameter and state-variable interface of IonicModel
    (set_parameter, get_parameter, state_variable_names, get/set_state_variables)
    and replaces differentiate() with compute_current().

    Parameters are held as float64 tf.constants, either one value or one value
    per node as an (n_nodes, 1) column. They must be set before the first
    compute_current() call, which captures them when it is first traced.
    """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)

    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """
        differentiate(U) is not available on a plugin: a plugin adds a current
        to a cell model and is advanced by the model that holds it
        """
        raise NotImplementedError('{} is a plugin: it adds a current to a cell model and cannot '
                                  'run on its own; attach it to a model with '
                                  'IonicModelWithPlugins'.format(type(self).__name__))

    def compute_current(self, U: tf.Variable) -> tf.Tensor:
        """
        compute_current(U) returns the current the plugin adds to Iion at the
        potential U (mV), in uA/uF, positive outward (the sign of Iion), as a
        float64 tensor of U's shape. It is evaluated with the state at the start
        of the step, then the state is advanced by dt. Override in subclasses.
        """
        raise NotImplementedError('compute_current must be implemented in subclass')

    def tunable_parameter_names(self) -> tuple:
        """ tunable_parameter_names() returns the names set_parameter() accepts. Override in subclasses. """
        return(())

    def set_parameter(self, pname: str, pvalue: np.ndarray):
        """
        set_parameter(pname, pvalue) sets the parameter pname to pvalue, a scalar
        or a per-node (n_nodes, 1) column. Only the names listed by
        tunable_parameter_names() are accepted: a name that is not a parameter
        raises instead of being silently ignored, as a typo would be otherwise.
        """
        try:
            if pname not in self.tunable_parameter_names():
                raise ValueError('{}: "{}" is not a tunable parameter; the plugin accepts {}'.format(
                    type(self).__name__, pname, ', '.join(self.tunable_parameter_names())))
            values = np.asarray(pvalue, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError('{}: parameter {} has non-finite values'.format(
                    type(self).__name__, pname))
            setattr(self, '_{}'.format(pname), tf.constant(values, dtype=PLUGIN_DTYPE))
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def _flat_parameter(self, pname: str) -> tf.Tensor:
        """
        _flat_parameter(pname) returns the parameter pname as a flat tensor: one
        entry per node for a column, a single entry otherwise, which broadcasts
        against the flat potential. Flattening both sides keeps an (n, 1) column
        from broadcasting against an (n,) potential into an (n, n) matrix.
        """
        return(tf.reshape(getattr(self, '_{}'.format(pname)), [-1]))
