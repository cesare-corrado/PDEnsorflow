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
        

    @tf.function
    def differentiate(self, U, H):
        """ the state differentiation for the 2v model """
        # constants for the modified Mitchell Schaeffer 2v left atrial action potential model
        tau_in    = tf.constant(0.1)
        tau_out   = tf.constant(9.0)
        tau_open  = tf.constant(100.0)
        tau_close = tf.constant(120.0)
        u_gate    = tf.constant(0.13)
        u_crit    = tf.constant(0.13)
        J_in  =  -1.0 * H * U * (U-u_crit) * (1.0-U)/tau_in
        J_out =  (1.0-H)*U/tau_out
        dU    = - (J_in +J_out)
        dH = tf.where(U > u_gate, -H / tau_close, (1.0 - H) / tau_open)        
        return dU, dH


