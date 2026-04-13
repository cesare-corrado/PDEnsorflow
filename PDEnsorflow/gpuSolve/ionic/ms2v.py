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





class MitchellSchaeffer2v(IonicModel):
    """
        The Mitchell Schaeffer (2v) cell model
        C.C. Mitchell, D.G. Schaeffer. A two current model for the dynamics of cardiac membrane.
        Bull. Math. Biol. 2003 Sep;65(5):767-93.
        This class implements the transmembrane potential within the interval [0,1]
    """

    def __init__(self, dt=0.0, n_nodes=0):
        super().__init__(dt, n_nodes)
        self._tau_in    = tf.constant(0.3)
        self._tau_out   = tf.constant(6.0)
        self._tau_open  = tf.constant(120.0)
        self._tau_close = tf.constant(150.0)
        self._u_gate    = tf.constant(0.13)
        self._vmin : tf.constant = tf.constant(-80.0, name = "vmin")
        self._vmax : tf.constant = tf.constant(20.0, name = "vmax")
        self._DV   : tf.constant = self._vmax-self._vmin
        self._H_state = None

    def initialize_state_variables(self, U: tf.Variable):
        # create internal H gate variable matching U's shape
        """initialize_state_variables(U) creates H gate variable with the same shape as U"""
        if not self._initialized:
            self._H_state = tf.Variable(tf.ones_like(U), name="H_state")
            self._initialized = True

    def tau_in(self) -> tf.constant:
        return(self._tau_in)        

    def tau_out(self) -> tf.constant:
        return(self._tau_out)        

    def tau_open(self) -> tf.constant:
        return(self._tau_open)        

    def tau_close(self) -> tf.constant:
        return(self._tau_close)        

    def u_gate(self) -> tf.constant:
        return(self._u_gate)        

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
    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """ the state differentiation for the 2v model """
        # constants for the modified Mitchell Schaeffer 2v left atrial action potential model
        Uad   = self.to_dimensionless(U)
        J_in  =  -1.0 * self._H_state * Uad * Uad * (1.0-Uad)/self._tau_in
        J_out =  Uad/self._tau_out
        dU    = - self.derivative_to_dimensional(J_in +J_out)
        dH = tf.where(Uad > self._u_gate, -self._H_state / self._tau_close, (1.0 - self._H_state) / self._tau_open)
        self._H_state.assign(self._H_state + self._dt * dH)
        return dU


