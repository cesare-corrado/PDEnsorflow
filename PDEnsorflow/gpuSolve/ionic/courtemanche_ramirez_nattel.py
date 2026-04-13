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
from math import log, exp, expm1

# To switch to float64, change this line and update all tf.constant/tf.Variable dtype args below
_DTYPE = tf.float32


class CourtemancheRamirezNattel(IonicModel):
    """
        The Courtemanche-Ramirez-Nattel (1998) human atrial action potential model.
        Courtemanche M, Ramirez RJ, Nattel S. Ionic mechanisms underlying human
        atrial action potential properties: insights from a mathematical model.
        Am J Physiol. 1998 Jul;275(1):H301-21.

        This model uses lookup tables for voltage-dependent and calcium-dependent
        quantities, with Rush-Larsen integration for gating variables.
    """

    def __init__(self, dt=0.0, n_nodes=0):
        super().__init__(dt, n_nodes)

        # Constants
        self._C_B1a = 3.79138232501097e-05
        self._C_B1b = 0.0811764705882353
        self._C_B1c = 0.00705882352941176
        self._C_B1d = 0.00537112496043221
        self._C_B1e = 11.5
        self._C_Fn1 = 9.648e-13
        self._C_Fn2 = 2.5910306809e-13
        self._C_dCa_rel = 8.0
        self._C_dCaup = 0.0869565217391304
        self._Ca_rel_init = 1.49
        self._Ca_up_init = 1.49
        self._Cai_init = 1.02e-1
        self._F = 96.4867
        self._K_Q10 = 3.0
        self._K_up = 0.00092
        self._Ki_init = 139.0
        self._KmCa = 1.38
        self._KmCmdn = 0.00238
        self._KmCsqn = 0.8
        self._KmKo = 1.5
        self._KmNa = 87.5
        self._KmNa3 = 669921.875
        self._KmNai = 10.0
        self._KmTrpn = 0.0005
        self._Nai = 11.2
        self._R_gas = 8.3143
        self._T = 310.0
        self._V_init = -81.2
        self._Volcell = 20100.0
        self._Voli = 13668.0
        self._Volrel = 96.48
        self._Volup = 1109.52
        self._d_init = 1.37e-4
        self._f_Ca_init = 0.775
        self._f_init = 0.999
        self._gamma = 0.35
        self._h_init = 0.965
        self._j_init = 0.978
        self._k_rel = 30.0
        self._k_sat = 0.1
        self._m_init = 2.91e-3
        self._maxCmdn = 0.05
        self._maxCsqn = 10.0
        self._maxTrpn = 0.07
        self._oa_init = 3.04e-2
        self._oi_init = 0.999
        self._tau_f_Ca = 2.0
        self._tau_tr = 180.0
        self._tau_u = 8.0
        self._u_init = 0.0
        self._ua_init = 4.96e-3
        self._ui_init = 0.999
        self._v_init = 1.0
        self._w_init = 0.999
        self._xr_init = 3.29e-5
        self._xs_init = 1.87e-2

        # Parameters (can be modified per-region via set_parameter)
        self._ACh = 0.000001
        self._Cao = 1.8
        self._Cm = 100.0
        self._GACh = 0.0
        self._GCaL = 0.1238
        self._GK1 = 0.09
        self._GKr = 0.0294
        self._GKs = 0.129
        self._GNa = 7.8
        self._GbCa = 0.00113
        self._GbNa = 0.000674
        self._Gto = 0.1652
        self._Ko = 5.4
        self._Nao = 140.0
        self._factorGKur = 1.0
        self._factorGrel = 1.0
        self._factorGtr = 1.0
        self._factorGup = 1.0
        self._factorhGate = 0.0
        self._factormGate = 0.0
        self._factoroaGate = 0.0
        self._factorxrGate = 1.0
        self._maxCaup = 15.0
        self._maxINaCa = 1600.0
        self._maxINaK = 0.60
        self._maxIpCa = 0.275
        self._maxIup = 0.005

        # Cai_TableIndex
        self._carow_1_idx = 0
        self._carow_2_idx = 1
        self._carow_3_idx = 2
        self._conCa_idx = 3
        self._f_Ca_rush_larsen_A_idx = 4
        self._Cai_NROWS = 5

        # V_TableIndex
        self._GKur_idx = 0
        self._INaK_idx = 1
        self._IbNa_idx = 2
        self._d_rush_larsen_A_idx = 3
        self._d_rush_larsen_B_idx = 4
        self._f_rush_larsen_A_idx = 5
        self._f_rush_larsen_B_idx = 6
        self._h_rush_larsen_A_idx = 7
        self._h_rush_larsen_B_idx = 8
        self._j_rush_larsen_A_idx = 9
        self._j_rush_larsen_B_idx = 10
        self._m_rush_larsen_A_idx = 11
        self._m_rush_larsen_B_idx = 12
        self._oa_rush_larsen_A_idx = 13
        self._oa_rush_larsen_B_idx = 14
        self._oi_rush_larsen_A_idx = 15
        self._oi_rush_larsen_B_idx = 16
        self._ua_rush_larsen_A_idx = 17
        self._ua_rush_larsen_B_idx = 18
        self._ui_rush_larsen_A_idx = 19
        self._ui_rush_larsen_B_idx = 20
        self._vrow_29_idx = 21
        self._vrow_31_idx = 22
        self._vrow_32_idx = 23
        self._vrow_36_idx = 24
        self._vrow_7_idx = 25
        self._w_rush_larsen_A_idx = 26
        self._w_rush_larsen_B_idx = 27
        self._xr_rush_larsen_A_idx = 28
        self._xr_rush_larsen_B_idx = 29
        self._xs_rush_larsen_A_idx = 30
        self._xs_rush_larsen_B_idx = 31
        self._V_NROWS = 32

        # fn_TableIndex
        self._u_rush_larsen_A_idx = 0
        self._v_rush_larsen_A_idx = 1
        self._v_rush_larsen_B_idx = 2
        self._fn_NROWS = 3

        # Cai_TableParam
        self._Cai_T_mn = 3e-4
        self._Cai_T_mx = 30.0
        self._Cai_T_res = 3e-4
        self._Cai_T_step = 1.0 / self._Cai_T_res
        self._Cai_T_mn_idx = 0
        self._Cai_T_mx_idx = int((self._Cai_T_mx - self._Cai_T_mn) * self._Cai_T_step) - 1

        # V_TableParam
        self._V_T_mn = -200.0
        self._V_T_mx = 200.0
        self._V_T_res = 0.1
        self._V_T_step = 1.0 / self._V_T_res
        self._V_T_mn_idx = 0.0
        self._V_T_mx_idx = int((self._V_T_mx - self._V_T_mn) * self._V_T_step) - 1

        # fn_TableParam
        self._fn_T_mn = -2e-11
        self._fn_T_mx = 10.0e-11
        self._fn_T_res = 2e-15
        self._fn_T_step = 1.0 / self._fn_T_res
        self._fn_T_mn_idx = 0.0
        self._fn_T_mx_idx = int((self._fn_T_mx - self._fn_T_mn) * self._fn_T_step) - 1

        # Lookup tables (initialized by construct_tables)
        self._Cai_tab = None
        self._V_tab = None
        self._fn_tab = None

        # Rush-Larsen constants computed once from dt
        self._f_Ca_rush_larsen_B = None
        self._u_rush_larsen_B = None

        # 19 internal state variables (initialized via initialize_state_variables)
        self._Ca_rel = None
        self._Ca_up = None
        self._Cai = None
        self._Ki = None
        self._d_state = None
        self._f_state = None
        self._f_Ca_state = None
        self._h_state = None
        self._j_state = None
        self._m_state = None
        self._oa_state = None
        self._oi_state = None
        self._u_state = None
        self._ua_state = None
        self._ui_state = None
        self._v_state = None
        self._w_state = None
        self._xr_state = None
        self._xs_state = None


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
        """Build the 3 lookup tables (V, Cai, fn) used for Rush-Larsen integration."""
        # To switch to float64, change _DTYPE at the top of this file
        E_Na = ((self._R_gas * self._T) / self._F) * log(self._Nao / self._Nai)
        f_Ca_rush_larsen_C = expm1((-self._dt) / self._tau_f_Ca)
        sigma = (exp(self._Nao / 67.3) - 1.0) / 7.0
        u_rush_larsen_C = expm1((-self._dt) / self._tau_u)

        self._f_Ca_rush_larsen_B = exp((-self._dt) / self._tau_f_Ca)
        self._u_rush_larsen_B = exp((-self._dt) / self._tau_u)

        # ---- Cai Lookup Table ----
        Cai_np = np.arange(self._Cai_T_mn, self._Cai_T_mx, self._Cai_T_res).astype(np.float32 if _DTYPE == tf.float32 else np.float64)
        Cai_tab_np = np.zeros((Cai_np.shape[0], self._Cai_NROWS), dtype=Cai_np.dtype)

        conCa = Cai_np / 1000.0
        Cai_tab_np[:, self._conCa_idx] = conCa
        Cai_tab_np[:, self._carow_1_idx] = (self._factorGup * self._maxIup) / (1.0 + (self._K_up / conCa))
        Cai_tab_np[:, self._carow_2_idx] = (((((self._maxTrpn * self._KmTrpn) / (conCa + self._KmTrpn)) / (conCa + self._KmTrpn) + ((self._maxCmdn * self._KmCmdn) / (conCa + self._KmCmdn)) / (conCa + self._KmCmdn)) + 1.0) / self._C_B1c) / 1000.0
        Cai_tab_np[:, self._carow_3_idx] = (self._maxIpCa * conCa) / (0.0005 + conCa) - (((self._GbCa * self._R_gas * self._T) / 2.0) / self._F) * np.log(self._Cao / conCa)
        f_Ca_inf = 1.0 / (1.0 + (conCa / 0.00035))
        Cai_tab_np[:, self._f_Ca_rush_larsen_A_idx] = (-f_Ca_inf) * f_Ca_rush_larsen_C

        self._Cai_tab = tf.constant(Cai_tab_np, dtype=_DTYPE)

        # ---- V Lookup Table ----
        V_np = np.arange(self._V_T_mn, self._V_T_mx, self._V_T_res).astype(np.float32 if _DTYPE == tf.float32 else np.float64)
        V_tab_np = np.zeros((V_np.shape[0], self._V_NROWS), dtype=V_np.dtype)

        V_tab_np[:, self._GKur_idx] = 0.005 + 0.05 / (1.0 + np.exp((15.0 - V_np) / 13.0))
        V_tab_np[:, self._IbNa_idx] = self._GbNa * (V_np - E_Na)

        a_h = np.where(V_np >= -40.0, 0.0, 0.135 * np.exp(((V_np + 80.0) - self._factorhGate) / -6.8))
        a_j = np.where(V_np < -40.0, (((-127140.0 * np.exp(0.2444 * V_np)) - (3.474e-5 * np.exp(-0.04391 * V_np))) * (V_np + 37.78)) / (1.0 + np.exp(0.311 * (V_np + 79.23))), 0.0)
        a_m = np.where(np.abs(V_np + 47.13) < 1e-10, 3.2, (0.32 * (V_np + 47.13)) / (1.0 - np.exp(-0.1 * (V_np + 47.13))))
        aa_oa = 0.65 / (np.exp((V_np + 10.0) / -8.5) + np.exp((30.0 - V_np) / 59.0))
        aa_oi = 1.0 / (18.53 + np.exp((V_np + 113.7) / 10.95))
        aa_ua = 0.65 / (np.exp((V_np + 10.0) / -8.5) + np.exp((V_np - 30.0) / -59.0))
        aa_ui = 1.0 / (21.0 + np.exp((V_np - 185.0) / -28.0))
        aa_xr = self._factorxrGate * ((0.0003 * (V_np + 14.1)) / (1.0 - np.exp((V_np + 14.1) / -5.0)))
        aa_xs = (4.0e-5 * (V_np - 19.9)) / (1.0 - np.exp((19.9 - V_np) / 17.0))
        b_h = np.where(V_np >= -40.0, (1.0 / 0.13) / (1.0 + np.exp((-(V_np + 10.66)) / 11.1)), 3.56 * np.exp(0.079 * V_np) + 3.1e5 * np.exp(0.35 * V_np))
        b_j = np.where(V_np >= -40.0, (0.3 * np.exp(-2.535e-7 * V_np)) / (1.0 + np.exp(-0.1 * (V_np + 32.0))), (0.1212 * np.exp(-0.01052 * V_np)) / (1.0 + np.exp(-0.1378 * (V_np + 40.14))))
        b_m = 0.08 * np.exp((-(V_np - self._factormGate)) / 11.0)
        bb_oa = 0.65 / (2.5 + np.exp((V_np + 82.0) / 17.0))
        bb_oi = 1.0 / (35.56 + np.exp((-(V_np + 1.26)) / 7.44))
        bb_ua = 0.65 / (2.5 + np.exp((V_np + 82.0) / 17.0))
        bb_ui = np.exp((V_np - 158.0) / 16.0)
        bb_xr = (1.0 / self._factorxrGate) * ((7.3898e-5 * (V_np - 3.3328)) / (np.exp((V_np - 3.3328) / 5.1237) - 1.0))
        bb_xs = (3.5e-5 * (V_np - 19.9)) / (np.exp((V_np - 19.9) / 9.0) - 1.0)
        d_inf = 1.0 / (1.0 + np.exp((-(V_np + 10.0)) / 8.0))
        f_NaK = 1.0 / (1.0 + 0.1245 * np.exp((-0.1 * self._F * V_np / self._R_gas) / self._T) + 0.0365 * sigma * np.exp((-self._F * V_np / self._R_gas) / self._T))
        f_inf = 1.0 / (1.0 + np.exp((V_np + 28.0) / 6.9))
        oa_inf = 1.0 / (1.0 + np.exp((-((V_np + 20.47) - self._factoroaGate)) / 17.54))
        oi_inf = 1.0 / (1.0 + np.exp((V_np + 43.1) / 5.3))
        tau_d = np.where(np.abs(V_np + 10.0) < 1e-10, ((1.0 / 6.24) / 0.035) / 2.0, ((1.0 - np.exp((-(V_np + 10.0)) / 6.24)) / 0.035 / (V_np + 10.0)) / (1.0 + np.exp((-(V_np + 10.0)) / 6.24)))
        tau_f = 9.0 / (0.0197 * np.exp(-0.0337 * 0.0337 * (V_np + 10.0) * (V_np + 10.0)) + 0.02)
        # Avoid division by zero at V=7.9
        tau_w_denom = V_np - 7.9
        tau_w_denom = np.where(np.abs(tau_w_denom) < 1e-10, 1e-10, tau_w_denom)
        tau_w = ((6.0 * (1.0 - np.exp((7.9 - V_np) / 5.0))) / (1.0 + 0.3 * np.exp((7.9 - V_np) / 5.0))) / tau_w_denom
        ua_inf = 1.0 / (1.0 + np.exp((V_np + 30.3) / -9.6))
        ui_inf = 1.0 / (1.0 + np.exp((V_np - 99.45) / 27.48))

        V_tab_np[:, self._vrow_29_idx] = self._GCaL * (V_np - 65.0)
        V_tab_np[:, self._vrow_31_idx] = (((((self._maxINaCa * np.exp((self._gamma * self._F * V_np / self._R_gas) / self._T)) * self._Nai * self._Nai * self._Nai * self._Cao) / (self._KmNa3 + self._Nao**3)) / (self._KmCa + self._Cao)) / (1.0 + self._k_sat * np.exp(((self._gamma - 1.0) * self._F * V_np / self._R_gas) / self._T)))
        V_tab_np[:, self._vrow_32_idx] = (((((self._maxINaCa * np.exp(((self._gamma - 1.0) * self._F * V_np / self._R_gas) / self._T)) * self._Nao**3) / (self._KmNa3 + self._Nao**3)) / (self._KmCa + self._Cao)) / (1.0 + self._k_sat * np.exp(((self._gamma - 1.0) * self._F * V_np / self._R_gas) / self._T))) / 1000.0
        V_tab_np[:, self._vrow_36_idx] = V_np * self._GbCa
        V_tab_np[:, self._vrow_7_idx] = self._GNa * (V_np - E_Na)
        w_inf = 1.0 - 1.0 / (1.0 + np.exp((40.0 - V_np) / 17.0))
        xr_inf = 1.0 / (1.0 + np.exp((V_np + 14.1) / -6.5))
        xs_inf = 1.0 / np.sqrt(1.0 + np.exp((V_np - 19.9) / -12.7))

        V_tab_np[:, self._INaK_idx] = ((self._maxINaK * f_NaK / (1.0 + pow(self._KmNai / self._Nai, 1.5))) * self._Ko) / (self._Ko + self._KmKo)

        V_tab_np[:, self._d_rush_larsen_B_idx] = np.exp(-self._dt / tau_d)
        d_rush_larsen_C = np.expm1(-self._dt / tau_d)
        V_tab_np[:, self._f_rush_larsen_B_idx] = np.exp(-self._dt / tau_f)
        f_rush_larsen_C = np.expm1(-self._dt / tau_f)
        V_tab_np[:, self._h_rush_larsen_A_idx] = ((-a_h) / (a_h + b_h)) * np.expm1(-self._dt * (a_h + b_h))
        V_tab_np[:, self._h_rush_larsen_B_idx] = np.exp(-self._dt * (a_h + b_h))
        V_tab_np[:, self._j_rush_larsen_A_idx] = ((-a_j) / (a_j + b_j)) * np.expm1(-self._dt * (a_j + b_j))
        V_tab_np[:, self._j_rush_larsen_B_idx] = np.exp(-self._dt * (a_j + b_j))
        V_tab_np[:, self._m_rush_larsen_A_idx] = ((-a_m) / (a_m + b_m)) * np.expm1(-self._dt * (a_m + b_m))
        V_tab_np[:, self._m_rush_larsen_B_idx] = np.exp(-self._dt * (a_m + b_m))

        tau_oa = (1.0 / (aa_oa + bb_oa)) / self._K_Q10
        tau_oi = (1.0 / (aa_oi + bb_oi)) / self._K_Q10
        tau_ua = (1.0 / (aa_ua + bb_ua)) / self._K_Q10
        tau_ui = (1.0 / (aa_ui + bb_ui)) / self._K_Q10
        tau_xr = 1.0 / (aa_xr + bb_xr)
        tau_xs = 0.5 / (aa_xs + bb_xs)

        V_tab_np[:, self._w_rush_larsen_B_idx] = np.exp(-self._dt / tau_w)
        w_rush_larsen_C = np.expm1(-self._dt / tau_w)
        V_tab_np[:, self._d_rush_larsen_A_idx] = (-d_inf) * d_rush_larsen_C
        V_tab_np[:, self._f_rush_larsen_A_idx] = (-f_inf) * f_rush_larsen_C
        V_tab_np[:, self._oa_rush_larsen_B_idx] = np.exp(-self._dt / tau_oa)
        oa_rush_larsen_C = np.expm1(-self._dt / tau_oa)
        V_tab_np[:, self._oi_rush_larsen_B_idx] = np.exp(-self._dt / tau_oi)
        oi_rush_larsen_C = np.expm1(-self._dt / tau_oi)
        V_tab_np[:, self._ua_rush_larsen_B_idx] = np.exp(-self._dt / tau_ua)
        ua_rush_larsen_C = np.expm1(-self._dt / tau_ua)
        V_tab_np[:, self._ui_rush_larsen_B_idx] = np.exp(-self._dt / tau_ui)
        ui_rush_larsen_C = np.expm1(-self._dt / tau_ui)
        V_tab_np[:, self._w_rush_larsen_A_idx] = (-w_inf) * w_rush_larsen_C
        V_tab_np[:, self._xr_rush_larsen_B_idx] = np.exp(-self._dt / tau_xr)
        xr_rush_larsen_C = np.expm1(-self._dt / tau_xr)
        V_tab_np[:, self._xs_rush_larsen_B_idx] = np.exp(-self._dt / tau_xs)
        xs_rush_larsen_C = np.expm1(-self._dt / tau_xs)
        V_tab_np[:, self._oa_rush_larsen_A_idx] = (-oa_inf) * oa_rush_larsen_C
        V_tab_np[:, self._oi_rush_larsen_A_idx] = (-oi_inf) * oi_rush_larsen_C
        V_tab_np[:, self._ua_rush_larsen_A_idx] = (-ua_inf) * ua_rush_larsen_C
        V_tab_np[:, self._ui_rush_larsen_A_idx] = (-ui_inf) * ui_rush_larsen_C
        V_tab_np[:, self._xr_rush_larsen_A_idx] = (-xr_inf) * xr_rush_larsen_C
        V_tab_np[:, self._xs_rush_larsen_A_idx] = (-xs_inf) * xs_rush_larsen_C

        # Replace NaN/Inf that may arise from singularities
        V_tab_np = np.nan_to_num(V_tab_np, nan=0.0, posinf=0.0, neginf=0.0)

        self._V_tab = tf.constant(V_tab_np, dtype=_DTYPE)

        # ---- fn Lookup Table ----
        fn_np = np.arange(self._fn_T_mn, self._fn_T_mx, self._fn_T_res).astype(np.float32 if _DTYPE == tf.float32 else np.float64)
        fn_tab_np = np.zeros((fn_np.shape[0], self._fn_NROWS), dtype=fn_np.dtype)

        tau_v_fn = 1.91 + 2.09 / (1.0 + np.exp((3.4175e-13 - fn_np) / 13.67e-16))
        u_inf_fn = 1.0 / (1.0 + np.exp((3.4175e-13 - fn_np) / 13.67e-16))
        v_inf_fn = 1.0 - 1.0 / (1.0 + np.exp((6.835e-14 - fn_np) / 13.67e-16))

        u_rush_larsen_C_val = expm1(-self._dt / self._tau_u)
        fn_tab_np[:, self._u_rush_larsen_A_idx] = (-u_inf_fn) * u_rush_larsen_C_val
        fn_tab_np[:, self._v_rush_larsen_B_idx] = np.exp(-self._dt / tau_v_fn)
        v_rush_larsen_C_fn = np.expm1(-self._dt / tau_v_fn)
        fn_tab_np[:, self._v_rush_larsen_A_idx] = (-v_inf_fn) * v_rush_larsen_C_fn

        self._fn_tab = tf.constant(fn_tab_np, dtype=_DTYPE)


    def initialize_state_variables(self, U: tf.Variable):
        """Initialize 19 internal state variables matching U's shape."""
        if not self._initialized:
            self.construct_tables()
            # Flatten to get number of nodes
            shape = tf.shape(U)
            # To switch to float64, change _DTYPE at the top of this file
            self._Ca_rel = tf.Variable(tf.fill(shape, tf.constant(self._Ca_rel_init, dtype=_DTYPE)), name="Ca_rel")
            self._Ca_up = tf.Variable(tf.fill(shape, tf.constant(self._Ca_up_init, dtype=_DTYPE)), name="Ca_up")
            self._Cai = tf.Variable(tf.fill(shape, tf.constant(self._Cai_init, dtype=_DTYPE)), name="Cai")
            self._Ki = tf.Variable(tf.fill(shape, tf.constant(self._Ki_init, dtype=_DTYPE)), name="Ki")
            self._d_state = tf.Variable(tf.fill(shape, tf.constant(self._d_init, dtype=_DTYPE)), name="d_state")
            self._f_state = tf.Variable(tf.fill(shape, tf.constant(self._f_init, dtype=_DTYPE)), name="f_state")
            self._f_Ca_state = tf.Variable(tf.fill(shape, tf.constant(self._f_Ca_init, dtype=_DTYPE)), name="f_Ca_state")
            self._h_state = tf.Variable(tf.fill(shape, tf.constant(self._h_init, dtype=_DTYPE)), name="h_state")
            self._j_state = tf.Variable(tf.fill(shape, tf.constant(self._j_init, dtype=_DTYPE)), name="j_state")
            self._m_state = tf.Variable(tf.fill(shape, tf.constant(self._m_init, dtype=_DTYPE)), name="m_state")
            self._oa_state = tf.Variable(tf.fill(shape, tf.constant(self._oa_init, dtype=_DTYPE)), name="oa_state")
            self._oi_state = tf.Variable(tf.fill(shape, tf.constant(self._oi_init, dtype=_DTYPE)), name="oi_state")
            self._u_state = tf.Variable(tf.fill(shape, tf.constant(self._u_init, dtype=_DTYPE)), name="u_state")
            self._ua_state = tf.Variable(tf.fill(shape, tf.constant(self._ua_init, dtype=_DTYPE)), name="ua_state")
            self._ui_state = tf.Variable(tf.fill(shape, tf.constant(self._ui_init, dtype=_DTYPE)), name="ui_state")
            self._v_state = tf.Variable(tf.fill(shape, tf.constant(self._v_init, dtype=_DTYPE)), name="v_state")
            self._w_state = tf.Variable(tf.fill(shape, tf.constant(self._w_init, dtype=_DTYPE)), name="w_state")
            self._xr_state = tf.Variable(tf.fill(shape, tf.constant(self._xr_init, dtype=_DTYPE)), name="xr_state")
            self._xs_state = tf.Variable(tf.fill(shape, tf.constant(self._xs_init, dtype=_DTYPE)), name="xs_state")
            self._initialized = True


    @tf.function
    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """Compute ionic current dV/dt and update all 19 internal state variables.

        Args:
            U: transmembrane potential (in mV), shape (n_nodes, 1) or (n_nodes,)

        Returns:
            dU: the ionic current contribution to dV/dt (= -Iion)
        """
        # Flatten to 1D for table lookups and arithmetic
        V = tf.reshape(U, [-1])
        Ca_rel = tf.reshape(self._Ca_rel, [-1])
        Ca_up = tf.reshape(self._Ca_up, [-1])
        Cai = tf.reshape(self._Cai, [-1])
        Ki = tf.reshape(self._Ki, [-1])
        d_state = tf.reshape(self._d_state, [-1])
        f_state = tf.reshape(self._f_state, [-1])
        f_Ca_state = tf.reshape(self._f_Ca_state, [-1])
        h_state = tf.reshape(self._h_state, [-1])
        j_state = tf.reshape(self._j_state, [-1])
        m_state = tf.reshape(self._m_state, [-1])
        oa_state = tf.reshape(self._oa_state, [-1])
        oi_state = tf.reshape(self._oi_state, [-1])
        u_state = tf.reshape(self._u_state, [-1])
        ua_state = tf.reshape(self._ua_state, [-1])
        ui_state = tf.reshape(self._ui_state, [-1])
        v_state = tf.reshape(self._v_state, [-1])
        w_state = tf.reshape(self._w_state, [-1])
        xr_state = tf.reshape(self._xr_state, [-1])
        xs_state = tf.reshape(self._xs_state, [-1])

        # Lookup table interpolations
        V_row = self._interpolate(V, self._V_tab, self._V_T_mn, self._V_T_mx, self._V_T_res, self._V_T_step, self._V_T_mx_idx)
        Cai_row = self._interpolate(Cai, self._Cai_tab, self._Cai_T_mn, self._Cai_T_mx, self._Cai_T_res, self._Cai_T_step, self._Cai_T_mx_idx)

        # Compute ionic currents
        E_K = ((self._R_gas * self._T) / self._F) * tf.math.log(self._Ko / Ki)
        ICaL = V_row[:, self._vrow_29_idx] * d_state * f_state * f_Ca_state
        IKACh = (self._GACh * (10.0 / (1.0 + (9.13652 / (pow(self._ACh, 0.477811))))) * (0.0517 + 0.4516 / (1.0 + tf.exp((V + 59.53) / 17.18)))) * (V - E_K)
        INa = V_row[:, self._vrow_7_idx] * m_state * m_state * m_state * h_state * j_state
        INaCa = V_row[:, self._vrow_31_idx] - Cai * V_row[:, self._vrow_32_idx]
        vrow_13 = self._Gto * (V - E_K)
        vrow_18 = (self._factorGKur * V_row[:, self._GKur_idx]) * (V - E_K)
        vrow_21 = (self._GKr * (V - E_K)) / (1.0 + tf.exp((V + 15.0) / 22.4))
        vrow_24 = self._GKs * (V - E_K)
        vrow_8 = (self._GK1 * (V - E_K)) / (1.0 + tf.exp(0.07 * (V + 80.0)))
        IK1 = vrow_8
        IKr = vrow_21 * xr_state
        IKs = vrow_24 * xs_state * xs_state
        IKur = vrow_18 * ua_state * ua_state * ua_state * ui_state
        IpCa = Cai_row[:, self._carow_3_idx] + V_row[:, self._vrow_36_idx]
        Ito = vrow_13 * oa_state * oa_state * oa_state * oi_state
        Iion = INa + IK1 + Ito + IKur + IKr + IKs + ICaL + IpCa + INaCa + V_row[:, self._IbNa_idx] + V_row[:, self._INaK_idx] + IKACh

        # Forward Euler update for concentration variables
        Itr = (self._factorGtr * (Ca_up - Ca_rel)) / self._tau_tr
        Irel = self._factorGrel * u_state * u_state * v_state * self._k_rel * w_state * (Ca_rel - Cai_row[:, self._conCa_idx])
        dIups = Cai_row[:, self._carow_1_idx] - (self._maxIup / self._maxCaup) * Ca_up
        diff_Ca_rel = (Itr - Irel) / (1.0 + (self._C_dCa_rel / (Ca_rel + self._KmCsqn)) / (Ca_rel + self._KmCsqn))
        diff_Ki = (-(((Ito + IKr + IKur + IKs + IK1 + IKACh) - 2.0 * V_row[:, self._INaK_idx]))) / (self._F * self._Voli)
        diff_Ca_up = dIups - Itr * self._C_dCaup
        diff_Cai = ((self._C_B1d * ((INaCa + INaCa) - IpCa - ICaL)) - (self._C_B1e * dIups) + Irel) / Cai_row[:, self._carow_2_idx]

        self._Ca_rel.assign(tf.reshape(Ca_rel + diff_Ca_rel * self._dt, tf.shape(self._Ca_rel)))
        self._Ca_up.assign(tf.reshape(Ca_up + diff_Ca_up * self._dt, tf.shape(self._Ca_up)))
        self._Cai.assign(tf.reshape(Cai + diff_Cai * self._dt, tf.shape(self._Cai)))
        self._Ki.assign(tf.reshape(Ki + diff_Ki * self._dt, tf.shape(self._Ki)))

        # Rush-Larsen update for gating variables
        fn = (self._C_Fn1 * Irel) - (self._C_Fn2 * (ICaL - 0.4 * INaCa))
        fn_row = self._interpolate(fn, self._fn_tab, self._fn_T_mn, self._fn_T_mx, self._fn_T_res, self._fn_T_step, self._fn_T_mx_idx)

        self._d_state.assign(tf.reshape(V_row[:, self._d_rush_larsen_A_idx] + V_row[:, self._d_rush_larsen_B_idx] * d_state, tf.shape(self._d_state)))
        self._f_state.assign(tf.reshape(V_row[:, self._f_rush_larsen_A_idx] + V_row[:, self._f_rush_larsen_B_idx] * f_state, tf.shape(self._f_state)))
        self._f_Ca_state.assign(tf.reshape(Cai_row[:, self._f_Ca_rush_larsen_A_idx] + self._f_Ca_rush_larsen_B * f_Ca_state, tf.shape(self._f_Ca_state)))
        self._h_state.assign(tf.reshape(V_row[:, self._h_rush_larsen_A_idx] + V_row[:, self._h_rush_larsen_B_idx] * h_state, tf.shape(self._h_state)))
        self._j_state.assign(tf.reshape(V_row[:, self._j_rush_larsen_A_idx] + V_row[:, self._j_rush_larsen_B_idx] * j_state, tf.shape(self._j_state)))
        self._m_state.assign(tf.reshape(V_row[:, self._m_rush_larsen_A_idx] + V_row[:, self._m_rush_larsen_B_idx] * m_state, tf.shape(self._m_state)))
        self._oa_state.assign(tf.reshape(V_row[:, self._oa_rush_larsen_A_idx] + V_row[:, self._oa_rush_larsen_B_idx] * oa_state, tf.shape(self._oa_state)))
        self._oi_state.assign(tf.reshape(V_row[:, self._oi_rush_larsen_A_idx] + V_row[:, self._oi_rush_larsen_B_idx] * oi_state, tf.shape(self._oi_state)))
        self._u_state.assign(tf.reshape(fn_row[:, self._u_rush_larsen_A_idx] + self._u_rush_larsen_B * u_state, tf.shape(self._u_state)))
        self._ua_state.assign(tf.reshape(V_row[:, self._ua_rush_larsen_A_idx] + V_row[:, self._ua_rush_larsen_B_idx] * ua_state, tf.shape(self._ua_state)))
        self._ui_state.assign(tf.reshape(V_row[:, self._ui_rush_larsen_A_idx] + V_row[:, self._ui_rush_larsen_B_idx] * ui_state, tf.shape(self._ui_state)))
        self._v_state.assign(tf.reshape(fn_row[:, self._v_rush_larsen_A_idx] + fn_row[:, self._v_rush_larsen_B_idx] * v_state, tf.shape(self._v_state)))
        self._w_state.assign(tf.reshape(V_row[:, self._w_rush_larsen_A_idx] + V_row[:, self._w_rush_larsen_B_idx] * w_state, tf.shape(self._w_state)))
        self._xr_state.assign(tf.reshape(V_row[:, self._xr_rush_larsen_A_idx] + V_row[:, self._xr_rush_larsen_B_idx] * xr_state, tf.shape(self._xr_state)))
        self._xs_state.assign(tf.reshape(V_row[:, self._xs_rush_larsen_A_idx] + V_row[:, self._xs_rush_larsen_B_idx] * xs_state, tf.shape(self._xs_state)))

        # Reshape back to match U's shape
        dU = tf.reshape(-Iion, tf.shape(U))
        return dU
