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




from gpuSolve.ionic.ionicmodel import IonicModel
import tensorflow as tf




@tf.function
def H(x: tf.Variable) ->  tf.Variable:
    """ the step function """
    return (1.0 + tf.sign(x)) * 0.5

@tf.function
def G(x: tf.Variable) ->  tf.Variable:
    """ the step function """
    return (1.0 - tf.sign(x)) * 0.5


class Fenton4v(IonicModel):
    """
        The Cherry-Ehrlich-Nattel-Fenton (4v) canine left-atrial model
        Cherry EM, Ehrlich JR, Nattel S, Fenton FH. Pulmonary vein reentry--
        properties and size matter: insights from a computational analysis.
        Heart Rhythm. 2007 Dec;4(12):1553-62.
    """

    def __init__(self):
        super().__init__()
        self._tau_vp = tf.constant(3.33)
        self._tau_vn = tf.constant(19.2)
        self._tau_wp = tf.constant(160.0)
        self._tau_wn1 = tf.constant(75.0)
        self._tau_wn2 = tf.constant(75.0)
        self._tau_d = tf.constant(0.065)
        self._tau_si = tf.constant(31.8364)
        self._tau_so = tf.constant(31.8364)
        self._tau_a = tf.constant(0.009)
        self._u_c = tf.constant(0.23)
        self._u_w = tf.constant(0.146)
        self._u_0 = tf.constant(0.0)
        self._u_m = tf.constant(1.0)
        self._u_csi = tf.constant(0.8)
        self._u_so = tf.constant(0.3)
        self._r_sp = tf.constant(0.02)
        self._r_sn = tf.constant(1.2)
        self._k_ = tf.constant(3.0)
        self._a_so = tf.constant(0.115)
        self._b_so = tf.constant(0.84)
        self._c_so = tf.constant(0.02)
        self._vmin : tf.constant = tf.constant(0.0, name = "vmin")
        self._vmax : tf.constant = tf.constant(1.0, name = "vmax")
        self._DV   : tf.constant = self._vmax-self._vmin

    def tau_vp(self)  -> tf.constant:
        return(self._tau_vp)

    def tau_vn(self) -> tf.constant:
        return(self._tau_vn)

    def tau_wp(self) -> tf.constant:
        return(self._tau_wp)

    def tau_wn1(self) -> tf.constant:
        return(self._tau_wn1)

    def tau_wn2(self) -> tf.constant:
        return(self._tau_wn2)

    def tau_d(self) -> tf.constant:
        return(self._tau_d)

    def tau_si(self) -> tf.constant:
        return(self._tau_si)
        
    def tau_so(self) -> tf.constant:
        return(self._tau_so)

    def tau_a(self) -> tf.constant:
        return(self._tau_a)

    def u_c(self) -> tf.constant:
        return(self._u_c)

    def u_w(self) -> tf.constant:
        return(self._u_w)

    def u_0(self) -> tf.constant:
        return(self._u_0)

    def u_m(self) -> tf.constant:
        return(self._u_m)

    def u_csi(self) -> tf.constant:
        return(self._u_csi)

    def u_so(self) -> tf.constant:
        return(self._u_so)

    def r_sp(self) -> tf.constant:
        return(self._r_sp)

    def r_sn(self) -> tf.constant:
        return(self._r_sn)

    def k_(self) -> tf.constant:
        return(self._k_)

    def a_so(self) -> tf.constant:
        return(self._a_so)

    def b_so(self) -> tf.constant:
        return(self._b_so)

    def c_so(self) -> tf.constant:
        return(self._c_so)

    @tf.function
    def to_dimensionless(self,U: tf.Variable) -> tf.Variable:
        """ to_dimensionless(U) rescales U to its dimensionless values (range [0,1])
        """
        return(U-self._vmin)/self._DV
        
    @tf.function
    def derivative_to_dimensional(self,dU: tf.Variable) -> tf.Variable:
        """ derivative_to_dimensional(U) rescales the derivative of U (dU) to dimensional values
        """
        return(self._DV*dU)

    @tf.function
    def differentiate(self, U: tf.Variable, V: tf.Variable, W: tf.Variable, S: tf.Variable)->(tf.Variable, tf.Variable,tf.Variable,tf.Variable):
        """ the state differentiation for the 4v model """
        Uad   = self.to_dimensionless(U)
        # constants for the Fenton 4v left atrial action potential model
        I_fi = -V * H(Uad - self._u_c) * (Uad - self._u_c) * (self._u_m - Uad) / self._tau_d
        I_si = -W * S / self._tau_si
        I_so = (0.5 * (self._a_so - self._tau_a) * (1 + tf.tanh((Uad - self._b_so) / self._c_so)) +
               (Uad - self._u_0) * G(Uad - self._u_so) / self._tau_so + H(Uad - self._u_so) * self._tau_a)
        dU = -self.derivative_to_dimensional(I_fi + I_si + I_so)
        dV = tf.where(Uad > self._u_c, -V / self._tau_vp, (1 - V) / self._tau_vn)
        dW = tf.where(Uad > self._u_c, -W / self._tau_wp, tf.where(Uad > self._u_w, (1 - W) / self._tau_wn2, (1 - W) / self._tau_wn1)   )
        r_s = (self._r_sp - self._r_sn) * H(Uad - self._u_c) + self._r_sn
        dS = r_s * (0.5 * (1 + tf.tanh((Uad - self._u_csi) * self._k_)) - S)
        return dU, dV, dW, dS


