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
import numpy as np
import tensorflow as tf
from math import exp, sqrt

# To switch to float64, change this line and update all tf.constant/tf.Variable dtype args below
_DTYPE = tf.float32


class TenTusscherPanfilov(IonicModel):
    """
        The ten Tusscher-Panfilov (2006) human ventricular action potential model.
        ten Tusscher KHWJ, Panfilov AV. Alternans and spiral breakup in a human
        ventricular tissue model. Am J Physiol Heart Circ Physiol. 2006;291(3):H1088-H1100.

        This model uses lookup tables for voltage-dependent and CaSS-dependent
        quantities, with Rush-Larsen integration for gating variables.

        Cell types: "EPI" (epicardial), "ENDO" (endocardial), "MCELL" (midmyocardial)
    """

    def __init__(self, dt=0.0, n_nodes=0, cell_type="EPI"):
        super().__init__(dt, n_nodes)

        self._cell_type = cell_type if cell_type is not None else "EPI"

        # Constants
        self._CaSR_init = 1.3
        self._CaSS_init = 0.00007
        self._Cai_init = 0.00007
        self._EC = 1.5
        self._F2_init = 1.0
        self._FCaSS_init = 1.0
        self._F_state_init = 1.0
        self._H_init = 0.75
        self._J_init = 0.75
        self._Ki_init = 138.3
        self._M_init = 0.0
        self._Nai_init = 7.67
        self._O_init = 0.0
        self._R_bar_init = 1.0
        self._R_init = 0.0
        self._S_init = 1.0
        self._V_init = -86.2
        self._Vxfer = 0.0038
        self._Xr1_init = 0.0
        self._Xr2_init = 1.0
        self._Xs_init = 0.0
        self._D_init = 0.0
        self._k1_ = 0.15
        self._k2_ = 0.045
        self._k3 = 0.060
        self._k4 = 0.005
        self._maxsr = 2.5
        self._minsr = 1.0

        # Parameters
        self._Bufc = 0.2
        self._Bufsr = 10.0
        self._Bufss = 0.4
        self._CAPACITANCE = 0.185
        self._Cao = 2.0
        self._D_CaL_off = 0.0
        self._Fconst = 96485.3415
        self._GK1 = 5.405
        self._GNa = 14.838
        self._GbCa = 0.000592
        self._GbNa = 0.00029
        self._GpCa = 0.1238
        self._GpK = 0.0146
        self._Kbufc = 0.001
        self._Kbufsr = 0.3
        self._Kbufss = 0.00025
        self._KmCa = 1.38
        self._KmK = 1.0
        self._KmNa = 40.0
        self._KmNai = 87.5
        self._Ko = 5.4
        self._KpCa = 0.0005
        self._Kup = 0.00025
        self._Nao = 140.0
        self._Rconst = 8314.472
        self._T = 310.0
        self._Vc = 0.016404
        self._Vleak = 0.00036
        self._Vmaxup = 0.006375
        self._Vrel = 0.102
        self._Vsr = 0.001094
        self._Vss = 0.00005468
        self._knaca = 1000.0
        self._knak = 2.724
        self._ksat = 0.1
        self._n = 0.35
        self._pKNa = 0.03
        self._scl_tau_f = 1.0
        self._vHalfXs = 5.0
        self._xr2_off = 0.0
        self._GCaL_init = 0.00003980
        self._GKr_init = 0.153
        self._GKs_init = 0.392 if self._cell_type == "EPI" else 0.098 if self._cell_type == "MCELL" else 0.392
        self._Gto_init = 0.294 if self._cell_type == "EPI" else 0.294 if self._cell_type == "MCELL" else 0.073

        # CaSS_TableIndex
        self._CaSS2_idx = 0
        self._FCaSS_rush_larsen_A_idx = 1
        self._FCaSS_rush_larsen_B_idx = 2
        self._CaSS_NROWS = 3

        # V_TableIndex
        self._D_rush_larsen_A_idx = 0
        self._D_rush_larsen_B_idx = 1
        self._F2_rush_larsen_A_idx = 2
        self._F2_rush_larsen_B_idx = 3
        self._F_rush_larsen_A_idx = 4
        self._F_rush_larsen_B_idx = 5
        self._H_rush_larsen_A_idx = 6
        self._H_rush_larsen_B_idx = 7
        self._INaCa_A_idx = 8
        self._INaCa_B_idx = 9
        self._J_rush_larsen_A_idx = 10
        self._J_rush_larsen_B_idx = 11
        self._M_rush_larsen_A_idx = 12
        self._M_rush_larsen_B_idx = 13
        self._R_rush_larsen_A_idx = 14
        self._R_rush_larsen_B_idx = 15
        self._S_rush_larsen_A_idx = 16
        self._S_rush_larsen_B_idx = 17
        self._Xr1_rush_larsen_A_idx = 18
        self._Xr1_rush_larsen_B_idx = 19
        self._Xr2_rush_larsen_A_idx = 20
        self._Xr2_rush_larsen_B_idx = 21
        self._Xs_rush_larsen_A_idx = 22
        self._Xs_rush_larsen_B_idx = 23
        self._a2_idx = 24
        self._rec_iNaK_idx = 25
        self._rec_ipK_idx = 26
        self._V_NROWS = 27

        # VEk_TableIndex
        self._rec_iK1_idx = 0
        self._VEk_NROWS = 1

        # CaSS_TableParam
        self._CaSS_T_mn = 0.00001
        self._CaSS_T_mx = 10.0
        self._CaSS_T_res = 0.00001
        self._CaSS_T_step = 1.0 / self._CaSS_T_res
        self._CaSS_T_mx_idx = int((self._CaSS_T_mx - self._CaSS_T_mn) * self._CaSS_T_step) - 1

        # V_TableParam
        self._V_T_mn = -800.0
        self._V_T_mx = 800.0
        self._V_T_res = 0.05
        self._V_T_step = 1.0 / self._V_T_res
        self._V_T_mx_idx = int((self._V_T_mx - self._V_T_mn) * self._V_T_step) - 1

        # VEk_TableParam
        self._VEk_T_mn = -800.0
        self._VEk_T_mx = 800.0
        self._VEk_T_res = 0.05
        self._VEk_T_step = 1.0 / self._VEk_T_res
        self._VEk_T_mx_idx = int((self._VEk_T_mx - self._VEk_T_mn) * self._VEk_T_step) - 1

        # Lookup tables (initialized by construct_tables)
        self._CaSS_tab = None
        self._V_tab_lut = None
        self._VEk_tab = None

        # 22 internal state variables (initialized via initialize_state_variables)
        self._GCaL = None
        self._GKr = None
        self._GKs = None
        self._Gto = None
        self._CaSR = None
        self._CaSS = None
        self._Cai = None
        self._F_state = None
        self._F2_state = None
        self._FCaSS_state = None
        self._H_state = None
        self._J_state = None
        self._Ki = None
        self._M_state = None
        self._Nai = None
        self._R_state = None
        self._R_bar = None
        self._S_state = None
        self._Xr1_state = None
        self._Xr2_state = None
        self._Xs_state = None
        self._D_state = None


    def _interpolate(self, X, table, mn, mx, res, step, mx_idx):
        """Lookup table interpolation (linear). Works on 1D or column vectors."""
        # To switch to float64, change _DTYPE at the top of this file
        X_clamped = tf.clip_by_value(X, mn, mx)
        idx_float = (X_clamped - mn) * step
        idx = tf.cast(idx_float, tf.int32)
        lower_idx = tf.clip_by_value(idx, 0, mx_idx - 1)
        higher_idx = lower_idx + 1
        # To switch to float64, change _DTYPE at the top of this file
        lower_pos = tf.cast(lower_idx, _DTYPE) * res + mn
        w = tf.expand_dims((X_clamped - lower_pos) / res, axis=-1)
        return (1.0 - w) * tf.gather(table, lower_idx) + w * tf.gather(table, higher_idx)


    def construct_tables(self):
        """Build the 3 lookup tables (V, CaSS, VEk) used for Rush-Larsen integration."""
        # To switch to float64, change _DTYPE at the top of this file
        KmNai3 = self._KmNai * self._KmNai * self._KmNai
        Nao3 = self._Nao * self._Nao * self._Nao
        RTONF = (self._Rconst * self._T) / self._Fconst
        invKmCa_Cao = 1.0 / (self._KmCa + self._Cao)
        F_RT = 1.0 / RTONF
        invKmNai3_Nao3 = 1.0 / (KmNai3 + Nao3)
        pmf_INaCa = (self._knaca * invKmNai3_Nao3) * invKmCa_Cao

        # ---- CaSS Lookup Table ----
        np_dtype = np.float32 if _DTYPE == tf.float32 else np.float64
        CaSS_np = np.arange(self._CaSS_T_mn, self._CaSS_T_mx, self._CaSS_T_res).astype(np_dtype)
        CaSS_tab_np = np.zeros((CaSS_np.shape[0], self._CaSS_NROWS), dtype=np_dtype)

        FCaSS_inf = (0.6 / (1.0 + (CaSS_np / 0.05) * (CaSS_np / 0.05))) + 0.4
        tau_FCaSS = (80.0 / (1.0 + (CaSS_np / 0.05) * (CaSS_np / 0.05))) + 2.0
        CaSS_tab_np[:, self._FCaSS_rush_larsen_B_idx] = np.exp(-self._dt / tau_FCaSS)
        FCaSS_rush_larsen_C = np.expm1(-self._dt / tau_FCaSS)
        CaSS_tab_np[:, self._FCaSS_rush_larsen_A_idx] = (-FCaSS_inf) * FCaSS_rush_larsen_C

        self._CaSS_tab = tf.constant(CaSS_tab_np, dtype=_DTYPE)

        # ---- V Lookup Table ----
        V_np = np.arange(self._V_T_mn, self._V_T_mx, self._V_T_res).astype(np_dtype)
        V_tab_np = np.zeros((V_np.shape[0], self._V_NROWS), dtype=np_dtype)

        D_inf = 1.0 / (1.0 + np.exp(((-8.0 + self._D_CaL_off) - V_np) / 7.5))
        F2_inf = (0.67 / (1.0 + np.exp((V_np + 35.0) / 7.0))) + 0.33
        F_inf = 1.0 / (1.0 + np.exp((V_np + 20.0) / 7.0))
        H_inf = 1.0 / ((1.0 + np.exp((V_np + 71.55) / 7.43)) * (1.0 + np.exp((V_np + 71.55) / 7.43)))
        M_inf = 1.0 / ((1.0 + np.exp((-56.86 - V_np) / 9.03)) * (1.0 + np.exp((-56.86 - V_np) / 9.03)))
        R_inf = 1.0 / (1.0 + np.exp((20.0 - V_np) / 6.0))
        S_inf = 1.0 / (1.0 + np.exp((V_np + 20.0) / 5.0))
        Xr1_inf = 1.0 / (1.0 + np.exp((-26.0 - V_np) / 7.0))
        Xr2_inf = 1.0 / (1.0 + np.exp((V_np - (-88.0 + self._xr2_off)) / 24.0))
        Xs_inf = 1.0 / (1.0 + np.exp((-self._vHalfXs - V_np) / 14.0))

        V_tab_np[:, self._a2_idx] = 0.25 * np.exp(2.0 * (V_np - 15.0) * F_RT)

        aa_D = (1.4 / (1.0 + np.exp((-35.0 - V_np) / 13.0))) + 0.25
        aa_F = 1102.5 * np.exp(-(V_np + 27.0) * (V_np + 27.0) / 225.0)
        aa_F2 = 562.0 * np.exp(-(V_np + 27.0) * (V_np + 27.0) / 240.0)
        aa_H = np.where(V_np >= -40.0, 0.0, 0.057 * np.exp(-(V_np + 80.0) / 6.8))
        aa_J = np.where(V_np >= -40.0, 0.0, ((-2.5428e4 * np.exp(0.2444 * V_np) - 6.948e-6 * np.exp(-0.04391 * V_np)) * (V_np + 37.78)) / (1.0 + np.exp(0.311 * (V_np + 79.23))))
        aa_M = 1.0 / (1.0 + np.exp((-60.0 - V_np) / 5.0))
        aa_Xr1 = 450.0 / (1.0 + np.exp((-45.0 - V_np) / 10.0))
        aa_Xr2 = 3.0 / (1.0 + np.exp((-60.0 - V_np) / 20.0))
        aa_Xs = 1400.0 / np.sqrt(1.0 + np.exp((5.0 - V_np) / 6.0))
        bb_D = 1.4 / (1.0 + np.exp((V_np + 5.0) / 5.0))
        bb_F = 200.0 / (1.0 + np.exp((13.0 - V_np) / 10.0))
        bb_F2 = 31.0 / (1.0 + np.exp((25.0 - V_np) / 10.0))
        bb_H = np.where(V_np >= -40.0, 0.77 / (0.13 * (1.0 + np.exp(-(V_np + 10.66) / 11.1))), 2.7 * np.exp(0.079 * V_np) + 3.1e5 * np.exp(0.3485 * V_np))
        bb_J = np.where(V_np >= -40.0, (0.6 * np.exp(0.057 * V_np)) / (1.0 + np.exp(-0.1 * (V_np + 32.0))), (0.02424 * np.exp(-0.01052 * V_np)) / (1.0 + np.exp(-0.1378 * (V_np + 40.14))))
        bb_M = (0.1 / (1.0 + np.exp((V_np + 35.0) / 5.0))) + (0.10 / (1.0 + np.exp((V_np - 50.0) / 200.0)))
        bb_Xr1 = 6.0 / (1.0 + np.exp((V_np + 30.0) / 11.5))
        bb_Xr2 = 1.12 / (1.0 + np.exp((V_np - 60.0) / 20.0))
        bb_Xs = 1.0 / (1.0 + np.exp((V_np - 35.0) / 15.0))
        cc_D = 1.0 / (1.0 + np.exp((50.0 - V_np) / 20.0))
        cc_F = (180.0 / (1.0 + np.exp((V_np + 30.0) / 10.0))) + 20.0
        cc_F2 = 80.0 / (1.0 + np.exp((V_np + 30.0) / 10.0))
        den = pmf_INaCa / (1.0 + self._ksat * np.exp((self._n - 1.0) * V_np * F_RT))

        V_tab_np[:, self._rec_iNaK_idx] = 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V_np * F_RT) + 0.0353 * np.exp(-V_np * F_RT))
        V_tab_np[:, self._rec_ipK_idx] = 1.0 / (1.0 + np.exp((25.0 - V_np) / 5.98))

        tau_R = 9.5 * np.exp(-(V_np + 40.0) * (V_np + 40.0) / 1800.0) + 0.8
        if self._cell_type == "ENDO":
            tau_S = 1000.0 * np.exp(-(V_np + 67.0) * (V_np + 67.0) / 1000.0) + 8.0
        else:
            tau_S = 85.0 * np.exp(-(V_np + 45.0) * (V_np + 45.0) / 320.0) + 5.0 / (1.0 + np.exp((V_np - 20.0) / 5.0)) + 3.0

        V_tab_np[:, self._INaCa_A_idx] = (den * self._Cao) * np.exp(self._n * V_np * F_RT)
        V_tab_np[:, self._INaCa_B_idx] = (den * np.exp((self._n - 1.0) * V_np * F_RT)) * Nao3 * 2.5

        J_inf = H_inf
        V_tab_np[:, self._R_rush_larsen_B_idx] = np.exp(-self._dt / tau_R)
        R_rush_larsen_C = np.expm1(-self._dt / tau_R)
        V_tab_np[:, self._S_rush_larsen_B_idx] = np.exp(-self._dt / tau_S)
        S_rush_larsen_C = np.expm1(-self._dt / tau_S)

        tau_D = aa_D * bb_D + cc_D
        tau_F2 = aa_F2 + bb_F2 + cc_F2
        tau_F_factor = aa_F + bb_F + cc_F
        tau_H = 1.0 / (aa_H + bb_H)
        tau_J = 1.0 / (aa_J + bb_J)
        tau_M = aa_M * bb_M
        tau_Xr1 = aa_Xr1 * bb_Xr1
        tau_Xr2 = aa_Xr2 * bb_Xr2
        tau_Xs = aa_Xs * bb_Xs + 80.0

        V_tab_np[:, self._D_rush_larsen_B_idx] = np.exp(-self._dt / tau_D)
        D_rush_larsen_C = np.expm1(-self._dt / tau_D)
        V_tab_np[:, self._F2_rush_larsen_B_idx] = np.exp(-self._dt / tau_F2)
        F2_rush_larsen_C = np.expm1(-self._dt / tau_F2)
        V_tab_np[:, self._H_rush_larsen_B_idx] = np.exp(-self._dt / tau_H)
        H_rush_larsen_C = np.expm1(-self._dt / tau_H)
        V_tab_np[:, self._J_rush_larsen_B_idx] = np.exp(-self._dt / tau_J)
        J_rush_larsen_C = np.expm1(-self._dt / tau_J)
        V_tab_np[:, self._M_rush_larsen_B_idx] = np.exp(-self._dt / tau_M)
        M_rush_larsen_C = np.expm1(-self._dt / tau_M)
        V_tab_np[:, self._R_rush_larsen_A_idx] = (-R_inf) * R_rush_larsen_C
        V_tab_np[:, self._S_rush_larsen_A_idx] = (-S_inf) * S_rush_larsen_C
        V_tab_np[:, self._Xr1_rush_larsen_B_idx] = np.exp(-self._dt / tau_Xr1)
        Xr1_rush_larsen_C = np.expm1(-self._dt / tau_Xr1)
        V_tab_np[:, self._Xr2_rush_larsen_B_idx] = np.exp(-self._dt / tau_Xr2)
        Xr2_rush_larsen_C = np.expm1(-self._dt / tau_Xr2)
        V_tab_np[:, self._Xs_rush_larsen_B_idx] = np.exp(-self._dt / tau_Xs)
        Xs_rush_larsen_C = np.expm1(-self._dt / tau_Xs)

        tau_F = np.where(V_np > 0.0, tau_F_factor * self._scl_tau_f, tau_F_factor)
        V_tab_np[:, self._D_rush_larsen_A_idx] = (-D_inf) * D_rush_larsen_C
        V_tab_np[:, self._F2_rush_larsen_A_idx] = (-F2_inf) * F2_rush_larsen_C
        V_tab_np[:, self._F_rush_larsen_B_idx] = np.exp(-self._dt / tau_F)
        F_rush_larsen_C = np.expm1(-self._dt / tau_F)
        V_tab_np[:, self._H_rush_larsen_A_idx] = (-H_inf) * H_rush_larsen_C
        V_tab_np[:, self._J_rush_larsen_A_idx] = (-J_inf) * J_rush_larsen_C
        V_tab_np[:, self._M_rush_larsen_A_idx] = (-M_inf) * M_rush_larsen_C
        V_tab_np[:, self._Xr1_rush_larsen_A_idx] = (-Xr1_inf) * Xr1_rush_larsen_C
        V_tab_np[:, self._Xr2_rush_larsen_A_idx] = (-Xr2_inf) * Xr2_rush_larsen_C
        V_tab_np[:, self._Xs_rush_larsen_A_idx] = (-Xs_inf) * Xs_rush_larsen_C
        V_tab_np[:, self._F_rush_larsen_A_idx] = (-F_inf) * F_rush_larsen_C

        # Replace NaN/Inf that may arise from singularities
        V_tab_np = np.nan_to_num(V_tab_np, nan=0.0, posinf=0.0, neginf=0.0)

        self._V_tab_lut = tf.constant(V_tab_np, dtype=_DTYPE)

        # ---- VEk Lookup Table ----
        VEk_np = np.arange(self._VEk_T_mn, self._VEk_T_mx, self._VEk_T_res).astype(np_dtype)
        VEk_tab_np = np.zeros((VEk_np.shape[0], self._VEk_NROWS), dtype=np_dtype)

        a_K1 = 0.1 / (1.0 + np.exp(0.06 * (VEk_np - 200.0)))
        b_K1 = (3.0 * np.exp(0.0002 * (VEk_np + 100.0)) + np.exp(0.1 * (VEk_np - 10.0))) / (1.0 + np.exp(-0.5 * VEk_np))
        VEk_tab_np[:, self._rec_iK1_idx] = a_K1 / (a_K1 + b_K1)

        self._VEk_tab = tf.constant(VEk_tab_np, dtype=_DTYPE)


    def initialize_state_variables(self, U: tf.Variable):
        """Initialize 22 internal state variables matching U's shape."""
        if not self._initialized:
            self.construct_tables()
            shape = tf.shape(U)
            # To switch to float64, change _DTYPE at the top of this file
            self._GCaL = tf.Variable(tf.fill(shape, tf.constant(self._GCaL_init, dtype=_DTYPE)), name="GCaL")
            self._GKr = tf.Variable(tf.fill(shape, tf.constant(self._GKr_init, dtype=_DTYPE)), name="GKr")
            self._GKs = tf.Variable(tf.fill(shape, tf.constant(self._GKs_init, dtype=_DTYPE)), name="GKs")
            self._Gto = tf.Variable(tf.fill(shape, tf.constant(self._Gto_init, dtype=_DTYPE)), name="Gto")
            self._CaSR = tf.Variable(tf.fill(shape, tf.constant(self._CaSR_init, dtype=_DTYPE)), name="CaSR")
            self._CaSS = tf.Variable(tf.fill(shape, tf.constant(self._CaSS_init, dtype=_DTYPE)), name="CaSS")
            self._Cai = tf.Variable(tf.fill(shape, tf.constant(self._Cai_init, dtype=_DTYPE)), name="Cai")
            self._F_state = tf.Variable(tf.fill(shape, tf.constant(self._F_state_init, dtype=_DTYPE)), name="F_state")
            self._F2_state = tf.Variable(tf.fill(shape, tf.constant(self._F2_init, dtype=_DTYPE)), name="F2_state")
            self._FCaSS_state = tf.Variable(tf.fill(shape, tf.constant(self._FCaSS_init, dtype=_DTYPE)), name="FCaSS_state")
            self._H_state = tf.Variable(tf.fill(shape, tf.constant(self._H_init, dtype=_DTYPE)), name="H_state")
            self._J_state = tf.Variable(tf.fill(shape, tf.constant(self._J_init, dtype=_DTYPE)), name="J_state")
            self._Ki = tf.Variable(tf.fill(shape, tf.constant(self._Ki_init, dtype=_DTYPE)), name="Ki")
            self._M_state = tf.Variable(tf.fill(shape, tf.constant(self._M_init, dtype=_DTYPE)), name="M_state")
            self._Nai = tf.Variable(tf.fill(shape, tf.constant(self._Nai_init, dtype=_DTYPE)), name="Nai")
            self._R_state = tf.Variable(tf.fill(shape, tf.constant(self._R_init, dtype=_DTYPE)), name="R_state")
            self._R_bar = tf.Variable(tf.fill(shape, tf.constant(self._R_bar_init, dtype=_DTYPE)), name="R_bar")
            self._S_state = tf.Variable(tf.fill(shape, tf.constant(self._S_init, dtype=_DTYPE)), name="S_state")
            self._Xr1_state = tf.Variable(tf.fill(shape, tf.constant(self._Xr1_init, dtype=_DTYPE)), name="Xr1_state")
            self._Xr2_state = tf.Variable(tf.fill(shape, tf.constant(self._Xr2_init, dtype=_DTYPE)), name="Xr2_state")
            self._Xs_state = tf.Variable(tf.fill(shape, tf.constant(self._Xs_init, dtype=_DTYPE)), name="Xs_state")
            self._D_state = tf.Variable(tf.fill(shape, tf.constant(self._D_init, dtype=_DTYPE)), name="D_state")
            self._initialized = True


    @tf.function
    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """Compute ionic current dV/dt and update all 22 internal state variables.

        Args:
            U: transmembrane potential (in mV), shape (n_nodes, 1) or (n_nodes,)

        Returns:
            dU: the ionic current contribution to dV/dt (= -Iion)
        """
        # Flatten to 1D for table lookups and arithmetic
        V = tf.reshape(U, [-1])
        GCaL = tf.reshape(self._GCaL, [-1])
        GKr = tf.reshape(self._GKr, [-1])
        GKs = tf.reshape(self._GKs, [-1])
        Gto = tf.reshape(self._Gto, [-1])
        CaSR = tf.reshape(self._CaSR, [-1])
        CaSS = tf.reshape(self._CaSS, [-1])
        Cai = tf.reshape(self._Cai, [-1])
        F_state = tf.reshape(self._F_state, [-1])
        F2_state = tf.reshape(self._F2_state, [-1])
        FCaSS_state = tf.reshape(self._FCaSS_state, [-1])
        H_state = tf.reshape(self._H_state, [-1])
        J_state = tf.reshape(self._J_state, [-1])
        Ki = tf.reshape(self._Ki, [-1])
        M_state = tf.reshape(self._M_state, [-1])
        Nai = tf.reshape(self._Nai, [-1])
        R_state = tf.reshape(self._R_state, [-1])
        R_bar = tf.reshape(self._R_bar, [-1])
        S_state = tf.reshape(self._S_state, [-1])
        Xr1_state = tf.reshape(self._Xr1_state, [-1])
        Xr2_state = tf.reshape(self._Xr2_state, [-1])
        Xs_state = tf.reshape(self._Xs_state, [-1])
        D_state = tf.reshape(self._D_state, [-1])

        RTONF = (self._Rconst * self._T) / self._Fconst
        inverseVcF = 1.0 / (self._Vc * self._Fconst)
        inverseVcF2 = 1.0 / (2.0 * self._Vc * self._Fconst)
        inverseVssF2 = 1.0 / (2.0 * self._Vss * self._Fconst)
        pmf_INaK = self._knak * (self._Ko / (self._Ko + self._KmK))
        sqrt_Ko = sqrt(self._Ko / 5.4)
        F_RT = 1.0 / RTONF
        invVcF_Cm = inverseVcF * self._CAPACITANCE

        # Lookup table interpolations
        V_row = self._interpolate(V, self._V_tab_lut, self._V_T_mn, self._V_T_mx, self._V_T_res, self._V_T_step, self._V_T_mx_idx)
        CaSS_row = self._interpolate(CaSS, self._CaSS_tab, self._CaSS_T_mn, self._CaSS_T_mx, self._CaSS_T_res, self._CaSS_T_step, self._CaSS_T_mx_idx)

        # Nernst potentials
        Eca = 0.5 * RTONF * tf.math.log(self._Cao / Cai)
        Ek = RTONF * tf.math.log(self._Ko / Ki)
        Eks = RTONF * tf.math.log((self._Ko + self._pKNa * self._Nao) / (Ki + self._pKNa * Nai))
        Ena = RTONF * tf.math.log(self._Nao / Nai)

        # Ionic currents
        IpCa = (self._GpCa * Cai) / (self._KpCa + Cai)
        a1 = (GCaL * self._Fconst * F_RT * 4.0) * tf.where(
            tf.abs(V - 15.0) < 1e-10,
            0.5 * F_RT,
            (V - 15.0) / tf.math.expm1(2.0 * (V - 15.0) * F_RT)
        )
        ICaL_A = a1 * V_row[:, self._a2_idx]
        ICaL_B = a1 * self._Cao
        IKr = GKr * sqrt_Ko * Xr1_state * Xr2_state * (V - Ek)
        IKs = GKs * Xs_state * Xs_state * (V - Eks)
        INa = self._GNa * M_state * M_state * M_state * H_state * J_state * (V - Ena)
        INaK = pmf_INaK * (Nai / (Nai + self._KmNa)) * V_row[:, self._rec_iNaK_idx]
        IbCa = self._GbCa * (V - Eca)
        IbNa = self._GbNa * (V - Ena)
        IpK = self._GpK * V_row[:, self._rec_ipK_idx] * (V - Ek)
        Ito = Gto * R_state * S_state * (V - Ek)
        VEk = V - Ek

        VEk_row = self._interpolate(VEk, self._VEk_tab, self._VEk_T_mn, self._VEk_T_mx, self._VEk_T_res, self._VEk_T_step, self._VEk_T_mx_idx)
        ICaL = D_state * F_state * F2_state * FCaSS_state * (ICaL_A * CaSS - ICaL_B)
        INaCa = V_row[:, self._INaCa_A_idx] * Nai * Nai * Nai - V_row[:, self._INaCa_B_idx] * Cai
        IK1 = self._GK1 * VEk_row[:, self._rec_iK1_idx] * (V - Ek)
        Iion = IKr + IKs + IK1 + Ito + INa + IbNa + ICaL + IbCa + INaK + INaCa + IpCa + IpK

        # Forward Euler update for concentration variables
        Ileak = self._Vleak * (CaSR - Cai)
        Iup = self._Vmaxup / (1.0 + (self._Kup * self._Kup) / (Cai * Cai))
        Ixfer = self._Vxfer * (CaSS - Cai)
        diff_Ki = (-(IK1 + Ito + IKr + IKs - 2.0 * INaK + IpK)) * invVcF_Cm
        diff_Nai = (-(INa + IbNa + 3.0 * INaK + 3.0 * INaCa)) * invVcF_Cm
        kCaSR = self._maxsr - (self._maxsr - self._minsr) / (1.0 + (self._EC / CaSR) * (self._EC / CaSR))
        diff_Cai = ((Ixfer - ((IbCa + IpCa - 2.0 * INaCa) * inverseVcF2 * self._CAPACITANCE)) - ((Iup - Ileak) * (self._Vsr / self._Vc))) / (1.0 + (self._Bufc * self._Kbufc / (self._Kbufc + Cai)) / (self._Kbufc + Cai))
        diff_R_bar = self._k4 * (1.0 - R_bar) - self._k2_ * kCaSR * CaSS * R_bar
        k1 = self._k1_ / kCaSR
        O = (k1 * CaSS * CaSS * R_bar) / (self._k3 + k1 * CaSS * CaSS)
        Irel = self._Vrel * O * (CaSR - CaSS)
        diff_CaSR = ((Iup - Irel - Ileak) / (1.0 + (self._Bufsr * self._Kbufsr / (CaSR + self._Kbufsr)) / (CaSR + self._Kbufsr)))
        diff_CaSS = ((-Ixfer * (self._Vc / self._Vss) + Irel * (self._Vsr / self._Vss) + (-ICaL * inverseVssF2 * self._CAPACITANCE)) / (1.0 + (self._Bufss * self._Kbufss / (CaSS + self._Kbufss)) / (CaSS + self._Kbufss)))

        orig_shape = tf.shape(self._CaSR)
        self._CaSR.assign(tf.reshape(CaSR + diff_CaSR * self._dt, orig_shape))
        self._CaSS.assign(tf.reshape(CaSS + diff_CaSS * self._dt, orig_shape))
        self._Cai.assign(tf.reshape(Cai + diff_Cai * self._dt, orig_shape))
        self._Ki.assign(tf.reshape(Ki + diff_Ki * self._dt, orig_shape))
        self._Nai.assign(tf.reshape(Nai + diff_Nai * self._dt, orig_shape))
        self._R_bar.assign(tf.reshape(R_bar + diff_R_bar * self._dt, orig_shape))

        # Rush-Larsen update for gating variables
        self._D_state.assign(tf.reshape(V_row[:, self._D_rush_larsen_A_idx] + V_row[:, self._D_rush_larsen_B_idx] * D_state, orig_shape))
        self._F_state.assign(tf.reshape(V_row[:, self._F_rush_larsen_A_idx] + V_row[:, self._F_rush_larsen_B_idx] * F_state, orig_shape))
        self._F2_state.assign(tf.reshape(V_row[:, self._F2_rush_larsen_A_idx] + V_row[:, self._F2_rush_larsen_B_idx] * F2_state, orig_shape))
        self._FCaSS_state.assign(tf.reshape(CaSS_row[:, self._FCaSS_rush_larsen_A_idx] + CaSS_row[:, self._FCaSS_rush_larsen_B_idx] * FCaSS_state, orig_shape))
        self._H_state.assign(tf.reshape(V_row[:, self._H_rush_larsen_A_idx] + V_row[:, self._H_rush_larsen_B_idx] * H_state, orig_shape))
        self._J_state.assign(tf.reshape(V_row[:, self._J_rush_larsen_A_idx] + V_row[:, self._J_rush_larsen_B_idx] * J_state, orig_shape))
        self._M_state.assign(tf.reshape(V_row[:, self._M_rush_larsen_A_idx] + V_row[:, self._M_rush_larsen_B_idx] * M_state, orig_shape))
        self._R_state.assign(tf.reshape(V_row[:, self._R_rush_larsen_A_idx] + V_row[:, self._R_rush_larsen_B_idx] * R_state, orig_shape))
        self._S_state.assign(tf.reshape(V_row[:, self._S_rush_larsen_A_idx] + V_row[:, self._S_rush_larsen_B_idx] * S_state, orig_shape))
        self._Xr1_state.assign(tf.reshape(V_row[:, self._Xr1_rush_larsen_A_idx] + V_row[:, self._Xr1_rush_larsen_B_idx] * Xr1_state, orig_shape))
        self._Xr2_state.assign(tf.reshape(V_row[:, self._Xr2_rush_larsen_A_idx] + V_row[:, self._Xr2_rush_larsen_B_idx] * Xr2_state, orig_shape))
        self._Xs_state.assign(tf.reshape(V_row[:, self._Xs_rush_larsen_A_idx] + V_row[:, self._Xs_rush_larsen_B_idx] * Xs_state, orig_shape))

        # Reshape back to match U's shape
        dU = tf.reshape(-Iion, tf.shape(U))
        return dU
