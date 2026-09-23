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

# Ia = exp(_RATE (V - _V_HALF)) below the take-off potential: the constants of
# Cheng et al. (1999), Eq. 1a, which the reference keeps fixed (only the
# take-off potential and the slope above it are parameters).
_RATE   : float = 0.09      # mV^-1
_V_HALF : float = 100.0     # mV, where Ia = 1 uA/uF


class DefibAshiharaTrayanova(IonicPlugin):
    """
        Hypothetical outward current activated by strong shock-induced
        depolarization (ionic plugin).
        Ashihara T, Trayanova NA. Asymmetry in membrane responses to electric
        shocks: insights from bidomain simulations. Biophys J 2004;87:2271-2282.
        doi:10.1529/biophysj.104.043091
        The current is the one formulated by Cheng DK, Tung L, Sobie EA.
        Nonuniform responses of transmembrane potential during electric field
        stimulation of single cardiac cells. Am J Physiol 1999;277:H351-H362,
        Eq. 1 (VtakeOff = 160 mV, slopeFac = 1 there):

            Ia = exp(0.09 (V - 100))                                V <= VtakeOff
            Ia = exp(0.09 (VtakeOff - 100)) (0.09 slopeFac (V - VtakeOff) + 1)
                                                                    V >  VtakeOff

        in uA/uF, positive outward, added to Iion. It is negligible at
        physiological potentials (1e-7 uA/uF at -80 mV, 1 uA/uF at +100 mV)
        and grows exponentially, then linearly, under a strong depolarization.
        Both branches equal exp(0.09 (VtakeOff - 100)) at VtakeOff, so the
        current is continuous for any VtakeOff; its slope is continuous for
        slopeFac = 1, where the linear branch is the tangent of the exponential.

        Ashihara and Trayanova add Ia to the K+ component of the L-type Ca2+
        current of the LRd model. A plugin cannot reach inside its cell model,
        so here, as in the reference, Ia is added to the total Iion (the
        setting of Cheng et al.).

        Selected in a parameter file with imp_region[].plugins =
        Defib_AshiharaTrayanova.

        Parameters (set_parameter / plug_param), one value or one per node:
          * VtakeOff (mV): where the exponential turns into a straight line
          * slopeFac: scales the slope of the linear branch

        Differences from the reference model description, both deliberate:
          * Lower branch. The reference writes exp(0.09 (V - VtakeOff)), not
            exp(0.09 (V - 100)): at the default VtakeOff the current is then
            221 times smaller than in Cheng et al. below VtakeOff and jumps
            from 1 to 221 uA/uF at VtakeOff, which pins a shocked cell near
            160 mV. The two forms agree only for VtakeOff = 100 mV. The
            reference form is available with set_use_reference_form(True), to
            compare step by step with the reference or to reproduce its
            results; it is off by default.
          * No state variable. The reference declares Ki, with
            dKi/dt = sl_i2c Ia and Ki(0) = 5.4 mM, but sl_i2c, a cell-geometry
            factor, is never set for a plugin and stays 0, so Ki never
            changes. Ia is time-independent in both papers, and 5.4 mM (with an
            outward current that raises it) describes the extracellular, not
            the intracellular, K+ concentration. The variable is left out.

        Everything is computed in float64. There is no division, hence no
        removable singularity.
    """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)

        # ---- tunable parameters (tf.constant, float64) ------------------------
        self._VtakeOff : tf.Tensor = tf.constant(160.0, dtype=PLUGIN_DTYPE)  # mV
        self._slopeFac : tf.Tensor = tf.constant(1.0, dtype=PLUGIN_DTYPE)    # unitless

        # ---- options ------------------------------------------------------------
        # False: the lower branch of Cheng et al., continuous at VtakeOff.
        # True: the lower branch of the reference model description (see the
        # class docstring), for step-by-step comparisons with the reference.
        self._use_reference_form : bool = False


    # ---- parameters ---------------------------------------------------------
    def tunable_parameter_names(self) -> tuple:
        """ tunable_parameter_names() returns the names set_parameter() accepts """
        return(('VtakeOff', 'slopeFac'))

    def set_use_reference_form(self, flag: bool):
        """
        set_use_reference_form(flag) selects the lower branch of the reference
        model description, exp(0.09 (V - VtakeOff)), when flag is True, instead
        of exp(0.09 (V - 100)) of Cheng et al. (the default). Set it before the
        first compute_current() call, which captures it when it is traced.
        """
        self._use_reference_form = bool(flag)

    def use_reference_form(self) -> bool:
        """ use_reference_form() returns True if the reference lower branch is used """
        return(self._use_reference_form)


    # ---- dynamics -----------------------------------------------------------
    @tf.function(jit_compile=True)
    def compute_current(self, U: tf.Variable) -> tf.Tensor:
        """
        compute_current(U) returns the outward current Ia (uA/uF) at the
        potential U (mV) as a float64 tensor of U's shape. Ia has no state, so
        nothing is advanced.
        """
        V        = tf.reshape(tf.cast(U, PLUGIN_DTYPE), [-1])
        VtakeOff = self._flat_parameter('VtakeOff')
        slopeFac = self._flat_parameter('slopeFac')

        # The expressions and their order of evaluation are those of the
        # reference, so the two codes round alike.
        above = V > VtakeOff
        upper = tf.exp(_RATE * (VtakeOff - _V_HALF)) * (((_RATE * slopeFac) * (V - VtakeOff)) + 1.0)
        # The lower branch is evaluated at min(V, VtakeOff): unchanged where it
        # is selected, and finite where it is not (exp(0.09 (V - 100)) overflows
        # above about +7990 mV). tf.where discards that inf today, so the result
        # does not depend on this; it keeps a NaN (0*inf) out of a gradient
        # through tf.where, or out of a selection rewritten as
        # mask*upper + (1 - mask)*lower.
        V_low = tf.minimum(V, VtakeOff)
        if self._use_reference_form:
            lower = tf.exp(_RATE * (V_low - VtakeOff))
        else:
            lower = tf.exp(_RATE * (V_low - _V_HALF))
        Ia = tf.where(above, upper, lower)

        return(tf.reshape(Ia, tf.shape(U)))
