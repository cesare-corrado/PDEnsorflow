#!/usr/bin/env python
"""
    A TensorFlow-based Cardiac Electrophysiology Modeler

    Copyright 2022-2023 Cesare Corrado (cesare.corrado@kcl.ac.uk)

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
    

