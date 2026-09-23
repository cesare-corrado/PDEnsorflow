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

from gpuSolve.ionic.plugins.ionicplugin import IonicPlugin
from gpuSolve.ionic.plugins.ionicplugin import PLUGIN_DTYPE

# Physical constants, with the values (and the truncated pi) of the reference
# model description, so that both codes evaluate the same expressions.
_PI : float = 3.14159
_K  : float = 1.38066e-20     # Boltzmann constant, mJ/K
_T  : float = 310.0           # absolute temperature, K
_E  : float = 1.6021765e-19   # elementary charge, C

# The pore conductance gp(V) is a quotient that is 0/0 at three potentials:
# V = 0, and V = +-V_w where one of the two entrance terms divides by zero
# (V_w = w0 / (nn e/kT), about 935 mV with the default parameters). The limits
# are finite and are used within this distance of the singular point, where
# cancellation has already cost the quotient most of its digits: at 1e-6 mV the
# direct formula is still correct to about 1e-8, at 1e-12 mV it is off by 1%,
# and at the point itself it is NaN, which a single node would spread to the
# whole mesh through the implicit diffusion solve.
_SINGULAR_BAND : float = 1.0e-6   # mV


class ElectroporationDeBruinKrassowska98(IonicPlugin):
    """
        Membrane electroporation current (ionic plugin).
        DeBruin KA, Krassowska W. Electroporation and shock-induced
        transmembrane potential in a cardiac fiber during defibrillation
        strength shocks. Ann Biomed Eng 1998;26:584-596. doi:10.1114/1.101

        Pores open in the membrane at a rate that grows with V^2. The pore
        density n (cm^-2) obeys

            dn/dt = alpha exp(beta V^2) (1 - (n/N0) exp(-q beta V^2))

        and each pore conducts gp(V), the conductance of a toroidal pore with
        an energy barrier w0 and relative entrance length nn, so the plugin
        adds I_ep = gp(V) n V (uA/uF) to Iion. At rest n sits at its steady
        state N0 exp(q beta V^2), which is also its initial value, computed from
        the potential handed to initialize_state_variables() (the resting
        potential of the cell model the plugin is attached to).

        Selected in a parameter file with imp_region[].plugins =
        Electroporation_DeBruinKrassowska98.

        Parameters (set_parameter / plug_param), one value or one per node:
          * alpha (cm^-2 ms^-1), beta (mV^-2), q, N0 (cm^-2): pore creation
          * sigma (mS/cm), h (cm), nn, w0: pore conductance
        They must be set before initialize_state_variables(), which computes
        the initial pore density from them, as the reference does.

        Integration: n is advanced by forward Euler, the scheme of the
        reference implementation; its time constant is (N0/alpha)
        exp((q - 1) beta V^2), 750 ms or more with the default parameters.
        Everything is computed in float64, and the current is evaluated with
        the pore density at the start of the step before n is advanced.
    """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)

        # ---- tunable parameters (tf.constant, float64) ------------------------
        self._alpha : tf.Tensor = tf.constant(200.0, dtype=PLUGIN_DTYPE)     # cm^-2 ms^-1
        self._beta  : tf.Tensor = tf.constant(6.25e-5, dtype=PLUGIN_DTYPE)   # mV^-2
        self._q     : tf.Tensor = tf.constant(2.46, dtype=PLUGIN_DTYPE)      # unitless
        self._N0    : tf.Tensor = tf.constant(1.5e5, dtype=PLUGIN_DTYPE)     # cm^-2
        self._sigma : tf.Tensor = tf.constant(13.0, dtype=PLUGIN_DTYPE)      # mS/cm
        self._h     : tf.Tensor = tf.constant(5.0e-7, dtype=PLUGIN_DTYPE)    # cm
        self._nn    : tf.Tensor = tf.constant(0.15, dtype=PLUGIN_DTYPE)      # unitless
        self._w0    : tf.Tensor = tf.constant(5.25, dtype=PLUGIN_DTYPE)      # unitless

        # ---- constants ------------------------------------------------------
        # e/kT, per mV: turns the potential into units of the thermal voltage
        self._nd_f : float = _E / (_K * _T)

        # ---- state variables (initialized via initialize_state_variables) ----
        self._n : tf.Variable = None


    # ---- parameters ---------------------------------------------------------
    def tunable_parameter_names(self) -> tuple:
        """ tunable_parameter_names() returns the names set_parameter() accepts """
        return(('alpha', 'beta', 'q', 'N0', 'sigma', 'h', 'nn', 'w0'))

    def set_parameter(self, pname: str, pvalue: np.ndarray):
        """
        set_parameter(pname, pvalue) sets the parameter pname to pvalue, a scalar
        or a per-node (n_nodes, 1) column. w0 must be positive: the limit of the
        pore conductance at V = 0 divides by it.
        """
        try:
            if pname == 'w0' and np.any(np.asarray(pvalue, dtype=np.float64) <= 0.0):
                raise ValueError('{}: w0 is the energy barrier of a pore and must be '
                                 'positive'.format(type(self).__name__))
            super().set_parameter(pname, pvalue)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    # ---- state --------------------------------------------------------------
    def initialize_state_variables(self, U: tf.Variable):
        """
        initialize_state_variables(U) creates the pore density n with U's shape,
        at its steady state N0 exp(q beta V^2) for the potential U (mV)
        """
        if not self._initialized:
            try:
                V = tf.reshape(tf.cast(U, PLUGIN_DTYPE), [-1])
                q, beta, N0 = self._flat_parameter('q'), self._flat_parameter('beta'), self._flat_parameter('N0')
                n_init = N0 * tf.exp((q * beta) * (V * V))
                self._n = tf.Variable(tf.reshape(n_init, tf.shape(U)), name='n')
                self._initialized = True
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

    def state_variable_names(self) -> tuple:
        """ state_variable_names() returns the variable advanced by compute_current(): n """
        return(('n',))


    # ---- dynamics -----------------------------------------------------------
    @tf.function(jit_compile=True)
    def compute_current(self, U: tf.Variable) -> tf.Tensor:
        """
        compute_current(U) returns the electroporation current gp(V) n V (uA/uF)
        at the potential U (mV) as a float64 tensor of U's shape, and advances
        the pore density n by dt with forward Euler
        """
        V     = tf.reshape(tf.cast(U, PLUGIN_DTYPE), [-1])
        n     = tf.reshape(self._n, [-1])
        alpha = self._flat_parameter('alpha')
        beta  = self._flat_parameter('beta')
        q     = self._flat_parameter('q')
        N0    = self._flat_parameter('N0')
        nn    = self._flat_parameter('nn')
        w0    = self._flat_parameter('w0')
        one   = tf.constant(1.0, dtype=PLUGIN_DTYPE)

        # ---- pore conductance (mS per pore) ----
        # The expressions and their order of evaluation are those of the
        # reference, so the two codes round alike away from the singular points.
        gp_f = ((_PI * self._flat_parameter('sigma')) * self._flat_parameter('h')) * 0.25
        nd_vm = V * self._nd_f          # potential in units of kT/e
        nvm   = nn * nd_vm
        nvmm  = w0 - nvm
        nvmp  = w0 + nvm
        # singular points of gp, see _SINGULAR_BAND. Each quotient is evaluated
        # with a harmless denominator inside its band and then replaced by its
        # limit, so the branch that is not selected never produces a NaN.
        V_w     = w0 / (nn * self._nd_f)
        near_0  = tf.abs(V) < _SINGULAR_BAND
        near_wp = tf.abs(V - V_w) < _SINGULAR_BAND
        near_wm = tf.abs(V + V_w) < _SINGULAR_BAND
        # (w0 e^y - nvm)/y with y = w0 - nvm is w0 (e^y - 1)/y + 1: w0 + 1 at y = 0
        inner = tf.where(near_wp, w0 + one,
                         ((w0 * tf.exp(nvmm)) - nvm) / tf.where(near_wp, one, nvmm))
        # (w0 e^z + nvm)/z with z = w0 + nvm is w0 (e^z - 1)/z + 1: w0 + 1 at z = 0
        outer = tf.where(near_wm, w0 + one,
                         ((w0 * tf.exp(nvmp)) + nvm) / tf.where(near_wm, one, nvmp))
        e_vm  = tf.exp(nd_vm)
        den   = (e_vm * inner) - outer
        gp    = (gp_f * (e_vm - one)) / tf.where(near_0, one, den)
        # at V = 0 numerator and denominator both vanish; the ratio of their
        # derivatives gives gp(0) = gp_f / (e^w0 (1 - 2nn + 2nn/w0) - 2nn/w0)
        gp_0  = gp_f / (tf.exp(w0) * (one - 2.0 * nn + 2.0 * nn / w0) - 2.0 * nn / w0)
        gp    = tf.where(near_0, gp_0, gp)

        # ---- current, with the pore density at the start of the step ----
        I_ep = (gp * n) * V

        # ---- pore density: forward Euler ----
        dvm_2  = V * V
        dN1    = alpha * tf.exp(beta * dvm_2)
        dN2    = (-dN1 / N0) * tf.exp((-q * beta) * dvm_2)
        diff_n = dN1 + dN2 * n
        self._n.assign(tf.reshape(n + diff_n * self._dt, tf.shape(self._n)))

        return(tf.reshape(I_ep, tf.shape(U)))
