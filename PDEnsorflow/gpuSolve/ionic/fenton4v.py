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



@tf.function
def H(x):
    """ the step function """
    return (1.0 + tf.sign(x)) * 0.5

@tf.function
def G(x):
    """ the step function """
    return (1.0 - tf.sign(x)) * 0.5





class Fenton4v:
    """
        The Cherry-Ehrlich-Nattel-Fenton (4v) canine left-atrial model
        Cherry EM, Ehrlich JR, Nattel S, Fenton FH. Pulmonary vein reentry--
        properties and size matter: insights from a computational analysis.
        Heart Rhythm. 2007 Dec;4(12):1553-62.
    """

    #def __init__(self, props):
    #    for key, val in config.items():
    #        setattr(self, key, val)



        

    @tf.function
    def differentiate(self, U, V, W, S):
        """ the state differentiation for the 4v model """
        # constants for the Fenton 4v left atrial action potential model
        tau_vp = tf.constant(3.33)
        tau_vn = tf.constant(19.2)
        tau_wp = tf.constant(160.0)
        tau_wn1 = tf.constant(75.0)
        tau_wn2 = tf.constant(75.0)
        tau_d = tf.constant(0.065)
        tau_si = tf.constant(31.8364)
        tau_so = tau_si
        tau_a = tf.constant(0.009)
        u_c = tf.constant(0.23)
        u_w = tf.constant(0.146)
        u_0 = tf.constant(0.0)
        u_m = tf.constant(1.0)
        u_csi = tf.constant(0.8)
        u_so = tf.constant(0.3)
        r_sp = tf.constant(0.02)
        r_sn = tf.constant(1.2)
        k_ = tf.constant(3.0)
        a_so = tf.constant(0.115)
        b_so = tf.constant(0.84)
        c_so = tf.constant(0.02)
        I_fi = -V * H(U - u_c) * (U - u_c) * (u_m - U) / tau_d
        I_si = -W * S / tau_si
        I_so = (0.5 * (a_so - tau_a) * (1 + tf.tanh((U - b_so) / c_so)) +
               (U - u_0) * G(U - u_so) / tau_so + H(U - u_so) * tau_a)
        dU = -(I_fi + I_si + I_so)
        dV = tf.where(U > u_c, -V / tau_vp, (1 - V) / tau_vn)
        dW = tf.where(U > u_c, -W / tau_wp, tf.where(U > u_w, (1 - W) / tau_wn2, (1 - W) / tau_wn1)   )
        r_s = (r_sp - r_sn) * H(U - u_c) + r_sn
        dS = r_s * (0.5 * (1 + tf.tanh((U - u_csi) * k_)) - S)
        return dU, dV, dW, dS


