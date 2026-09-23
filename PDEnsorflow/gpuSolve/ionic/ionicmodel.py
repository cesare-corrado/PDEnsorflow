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



class IonicModel:
    """
    A base class for all the ionic models to gather functions common to each model
    """

    def __init__(self, dt=0.0, n_nodes=0):
        self._dt = dt
        self._n_nodes = n_nodes
        self._initialized = False

    def initialize_state_variables(self, U: tf.Variable):
        """initialize_state_variables(U) initializes the internal state variables matching U's shape.
        Override in subclasses.
        """
        self._initialized = True

    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """differentiate(U) computes the ionic current derivative dU and updates internal state variables.
        Override in subclasses.
        """
        raise NotImplementedError("differentiate must be implemented in subclass")

    def set_dt(self, dt: float):
        """
        set_dt(dt) sets the time step (ms) the model advances by. It must be set
        before initialize_state_variables(): some models fold dt into their
        lookup tables there.
        """
        self._dt = dt

    def dt(self) -> float:
        """ dt() returns the time step (ms) the model advances by """
        return(self._dt)

    def model_name(self) -> str:
        """
        model_name() returns the name that identifies the model in a checkpoint:
        the class name, which is unambiguous where a parameter-file name is not
        """
        return(type(self).__name__)

    def state_variable(self, name: str) -> tf.Variable:
        """
        state_variable(name) returns the tf.Variable that holds the state
        variable name (as listed by state_variable_names()); None if the model
        has no such variable or it is not initialised yet. Callers go through
        this accessor rather than getattr(model, '_' + name) because a model
        that holds other models (IonicModelWithPlugins) names their variables
        with a prefix that is not an attribute of its own.
        """
        return(getattr(self, '_{}'.format(name), None))

    def set_parameter(self,pname:str, pvalue: np.ndarray):
        """
        set_parameter(pname,pvalue) if pname exists, sets the parameter value to pvalue
        """
        internal_name = '_{}'.format(pname)
        if internal_name in self.__dict__.keys():
            setattr(self, internal_name, tf.constant(pvalue))
 
    def get_parameter(self,pname:str) -> tf.constant:
        """
        get_parameter(pname) returns the parameter values of pname  in pname exists; None otherwise
        """
        internal_name = '_{}'.format(pname)
        return( getattr(self, internal_name, None))

    def state_variable_names(self) -> tuple:
        """
        state_variable_names() returns the names (without the leading underscore)
        of the state variables that differentiate() advances in time.
        The list is declared by each model rather than discovered by scanning the
        tf.Variable attributes: some models also hold per-node conductances as
        tf.Variables that differentiate() never assigns, and those are parameters
        (rebuilt from the parameter file), not state. Override in subclasses.
        """
        return(())

    def get_state_variables(self) -> dict:
        """
        get_state_variables() returns {name: values} for every state variable, each
        value a flat numpy array with one entry per node, in the order the model
        holds them (the caller undoes any node renumbering)
        """
        try:
            states : dict = {}
            for name in self.state_variable_names():
                variable = self.state_variable(name)
                if variable is None:
                    raise ValueError('{}: state variable {} is not initialised; call '
                                     'initialize_state_variables(U) first'.format(
                                         type(self).__name__, name))
                states[name] = np.reshape(variable.numpy(), (-1,))
            return(states)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def set_state_variables(self, states: dict):
        """
        set_state_variables(states) overwrites every state variable with the flat
        per-node arrays in states ({name: values}). The names must match
        state_variable_names() exactly: a missing variable would silently restart
        from its initial value, and an extra one belongs to another model.
        Values are assigned into the existing tf.Variables, so a differentiate()
        that has already been traced keeps updating the same objects.
        """
        try:
            expected = set(self.state_variable_names())
            given    = set(states.keys())
            if given != expected:
                missing = sorted(expected - given)
                extra   = sorted(given - expected)
                raise ValueError('{}: the state variables do not match the model (missing: {}; '
                                 'not in the model: {})'.format(type(self).__name__,
                                                              ', '.join(missing) or 'none',
                                                              ', '.join(extra) or 'none'))
            for name in self.state_variable_names():
                variable = self.state_variable(name)
                if variable is None:
                    raise ValueError('{}: state variable {} is not initialised; call '
                                     'initialize_state_variables(U) first'.format(
                                         type(self).__name__, name))
                values = np.asarray(states[name])
                if values.size != int(np.prod(variable.shape)):
                    raise ValueError('{}: state variable {} has {} values, the model holds {}'.format(
                        type(self).__name__, name, values.size, int(np.prod(variable.shape))))
                variable.assign(tf.constant(np.reshape(values, variable.shape),
                                            dtype=variable.dtype))
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


