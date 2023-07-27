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

    def __init__(self):
        self._vmin : tf.constant = tf.constant(0.0, name = "vmin",dtype=tf.float32)
        self._vmax : tf.constant = tf.constant(1.0, name = "vmax",dtype=tf.float32)
        self._DV   : tf.constant = tf.subtract(self._vmax,self._vmin)

    def set_vmin(self, vmin:float = 0.0):
        """ set_vmin(vmin = 0.0): sets the minimum value of the potential for rescaling to vmin
        """
        self._vmin = tf.constant(vmin, name="vmin",dtype=tf.float32)
        self._DV   = tf.subtract(self._vmax,self._vmin)

    def set_vmax(self, vmax:float = 1.0):
        """ set_vmax(vmax = 1.0): sets the maximum value of the potential for rescaling to vmax
        """
        self._vmax = tf.constant(vmax, name="vmax",dtype=tf.float32)
        self._DV   = tf.subtract(self._vmax,self._vmin)

    def vmin(self) ->tf.constant:
        """ vmin(): returns the minimum value of the potential vmin
        """
        return(self._vmin)

    def vmax(self) ->tf.constant:
        """ vmax(): returns the maximum value of the potential vmax
        """
        return(self._vmax)

    def set_parameter(self,pname:str, pvalue: np.ndarray):
        """
        set_parameter(pname,pvalue) if pname exists, sets the parameter value to pvalue
        """
        internal_name = '_{}'.format(pname)
        if internal_name in self.__dict__.keys():
            setattr(self, internal_name, tf.constant(pvalue,dtype=tf.float32))
 
    def get_parameter(self,pname:str) -> tf.constant:
        """
        get_parameter(pname) returns the parameter values of pname  in pname exists; None otherwise
        """
        internal_name = '_{}'.format(pname)
        return( getattr(self, internal_name, None))
    
    @tf.function
    def to_dimensionless(self,U: tf.Variable) -> tf.constant:
        """ to_dimensionless(U) rescales U to its dimensionless values (range [0,1])
        """
        return tf.divide(tf.subtract(U,self._vmin),self._DV)
    
    @tf.function
    def to_dimensional(self,U: tf.Variable) -> tf.constant:
        """ to_dimensional(U) rescales U to its dimensional values (range [vmin,vmax])
        """
        return tf.add(tf.scalar_mul(self._DV,U),self._vmin)

    @tf.function
    def derivative_to_dimensionless(self,dU: tf.Variable) -> tf.constant:
        """ derivative_to_dimensionless(U) rescales the derivative of U (dU) to dimensionless units
        """
        return tf.divide(dU,self._DV)
    
    @tf.function
    def derivative_to_dimensional(self,dU: tf.Variable) -> tf.constant:
        """ derivative_to_dimensional(U) rescales the derivative of U (dU) to dimensional values
        """
        return tf.scalar_mul(self._DV,dU)

    
