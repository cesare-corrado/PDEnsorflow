#!/usr/bin/env python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler

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
import time


import tensorflow as tf
tf.config.run_functions_eagerly(True)





class ModifiedMS2v:
    """
        The modified Mitchell Schaeffer (2v) left-atrial model
        Corrado C, Niederer S. A two-variable model 
        robust to pacemaker behaviour for the dynamics of the cardiac action potential. 
        Math Biosci. 216 Nov;281:46-54.
        This class implements the transmembrane potential within the interval [0,1]
    """

    def __init__(self, props=None):
        rankmax = 0
        self._tau_in    = tf.constant(0.1)
        self._tau_out   = tf.constant(9.0)
        self._tau_open  = tf.constant(100.0)
        self._tau_close = tf.constant(120.0)
        self._u_gate    = tf.constant(0.13)
        self._u_crit    = tf.constant(0.13)
        if props is not None:
            for key in self.__dict__.keys():
                prop_key = key[1:]
                if prop_key in props.keys():
                    prop_value = tf.constant(props[prop_key])
                    prop_rank  = tf.rank(prop_value).numpy()
                    rankmax    = max(rankmax,prop_rank)
                     setattr(self, key, prop_value) 
            if rankmax>0:
                newrank = np.ones(shape=rankmax,dtype=int)
                for key in self.__dict__.keys():
                    prop_value = getattr(self,key)
                    prop_rank  = tf.rank(prop_value).numpy()
                    if prop_rank<rankmax:
                        prop_value = tf.constant(prop_value.numpy(),shape= newrank,dtype=np.float32 )
                        setattr(self, key, prop_value)
                
    def tau_in(self):
        return(self._tau_in)        

    def tau_out(self):
        return(self._tau_out)        

    def tau_open(self):
        return(self._tau_open)        

    def tau_close(self):
        return(self._tau_close)        

    def u_gate(self):
        return(self._u_gate)        

    def u_crit(self):
        return(self.u_crit)        

 
    @tf.function
    def differentiate(self, U, H):
        """ the state differentiation for the 2v model """
        # constants for the modified Mitchell Schaeffer 2v left atrial action potential model
        J_in  =  -1.0 * H * U * (U-self._u_crit) * (1.0-U)/self._tau_in
        J_out =  (1.0-H)*U/self._tau_out
        dU    = - (J_in +J_out)
        dH = tf.where(U > self._u_gate, -H / self._tau_close, (1.0 - H) / self._tau_open)        
        return dU, dH


