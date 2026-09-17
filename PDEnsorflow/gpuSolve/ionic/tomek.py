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


from gpuSolve.ionic.ionicmodel import IonicModel
import numpy as np
import tensorflow as tf
from math import exp, log, sqrt

# The model is integrated in float64. It is stiff and spans many orders of
# magnitude (Jrel_p starts at 1.2e-20, Cai rests near 8e-5 mM) and its voltage
# table has a 0.01 mV step over +-1000 mV, which float32 cannot resolve at the
# ends of the range. The potential handed in by the solver may still be
# float32: it is cast on the way in, and dU is cast back to U's dtype.
_DTYPE = tf.float64

# Cell types, as integers: this is the only form the parameter file accepts
# (celltype=1). A name such as celltype=EPI is silently read as ENDO by the
# single-cell reference tool, so any value other than these three is rejected.
_ENDO  : int = 0
_EPI   : int = 1
_MCELL : int = 2

# Extracellular concentrations: tunable, but one value for the whole tissue.
_EXTRACELLULAR : tuple = ('Ko', 'Nao', 'Cao')


class Tomek(IonicModel):
    """
        The ToR-ORd human ventricular action potential model.
        Tomek J, Bueno-Orovio A, Passini E, Zhou X, Minchole A, Britton O,
        Bartolucci C, Severi S, Shrier A, Virag L, Varro A, Rodriguez B.
        Development, calibration, and validation of a novel human ventricular
        myocyte model in health, disease, and drug block. eLife 2019;8:e48890.
        doi:10.7554/eLife.48890

        43 state variables; potential in mV, time in ms, concentrations in mM
        (Cai included: it is stored in the unit of the equations, and a
        comparison with tools that report it in uM must multiply by 1e3).

        Parameters (set_parameter / im_param):
          * celltype: 0 ENDO (default), 1 EPI, 2 MCELL. May differ node by node.
          * GNa, GNaL_b, PCa_b, Gto_b, GKr_b, GKs_b, GK1_b: base conductances
            (permeability for PCa_b), before the cell-type factor. May differ
            node by node, e.g. to remodel a border zone or a scar.
          * CoefGNa, CoefGNaL, CoefCaL, CoefK1, CoefKr, CoefKs, Coefto: drug
            block factors in [0, 1], all 1 by default. They multiply the
            conductance (CoefCaL multiplies PCa), so the block reaches every
            quantity that depends on the channel, as in the authors' code.
          * Ko, Nao, Cao: extracellular K, Na and Ca concentrations (mM), 5, 140
            and 1.8 by default. One value for all the nodes.
          * V_init: resting potential used as initial condition.

        The effective conductance of a node is Coef x cell-type factor x base.
        These products are built once, as per-node columns, whenever one of
        their inputs changes, so differentiate() only reads them.
        Parameters must be set before the first differentiate() call: it runs
        inside a tf.function that captures them when it is first traced.

        Gating variables are advanced through precomputed voltage tables as
        x_new = A(V) + B(V) x. With use_rush_larsen True (default) A and B
        encode the Rush-Larsen exponential update; with False they encode
        forward Euler, the scheme of the reference implementation, which is
        useful to compare with it step for step. All other variables use
        forward Euler.

        differentiate() is compiled with XLA. The model has several hundred
        small kernels per step; fused, a step on the RTX A2000 takes 2.7 ms for
        63001 nodes instead of 10.8 ms as a plain graph (0.48 ms instead of
        10.1 ms for 101 nodes). Under tf.config.run_functions_eagerly(True) the
        compilation is skipped and the step runs eagerly.
    """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)

        # ---- integration ----------------------------------------------------
        # Rush-Larsen by default. Several gates have time constants shorter than
        # a practical dt: near rest tm is about 0.005 ms, so forward Euler with
        # dt = 0.01 ms has B = 1 - dt/tm of about -1.2, and m oscillates and is
        # held only by the [0, 1] clamp (1e-3 away from a dt = 0.001 ms solution
        # after repolarisation, against 8e-7 with Rush-Larsen). Measured against
        # that converged solution over one beat, Rush-Larsen is as accurate or
        # better at every dt from 0.01 to 0.1 ms, and halves the upstroke error.
        self._use_rush_larsen : bool = True

        # ---- tunable parameters (tf.constant, float64) ------------------------
        self._celltype : tf.Tensor = tf.constant(float(_ENDO), dtype=_DTYPE)
        self._GNa      : tf.Tensor = tf.constant(11.7802, dtype=_DTYPE)       # mS/uF
        self._GNaL_b   : tf.Tensor = tf.constant(0.0279, dtype=_DTYPE)        # mS/uF
        self._PCa_b    : tf.Tensor = tf.constant(8.3757e-05, dtype=_DTYPE)    # unitless
        self._Gto_b    : tf.Tensor = tf.constant(0.16, dtype=_DTYPE)          # mS/uF
        self._GKr_b    : tf.Tensor = tf.constant(0.0321, dtype=_DTYPE)        # mS/uF
        self._GKs_b    : tf.Tensor = tf.constant(0.0011, dtype=_DTYPE)        # mS/uF
        self._GK1_b    : tf.Tensor = tf.constant(0.6992, dtype=_DTYPE)        # mS/uF
        self._CoefGNa  : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._CoefGNaL : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._CoefCaL  : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._CoefK1   : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._CoefKr   : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._CoefKs   : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._Coefto   : tf.Tensor = tf.constant(1.0, dtype=_DTYPE)
        self._Ko       : tf.Tensor = tf.constant(5.0, dtype=_DTYPE)           # mM
        self._Nao      : tf.Tensor = tf.constant(140.0, dtype=_DTYPE)         # mM
        self._Cao      : tf.Tensor = tf.constant(1.8, dtype=_DTYPE)           # mM
        self._V_init   : float     = -88.7638                                 # mV

        # ---- constants ------------------------------------------------------
        # extracellular chloride (mM) and physical constants
        self._Clo   : float = 150.0
        self._Cli   : float = 24.0
        self._R     : float = 8314.0
        self._T     : float = 310.0
        self._F     : float = 96485.0
        self._zNa   : float = 1.0
        self._zCa   : float = 2.0
        self._zK    : float = 1.0
        self._zcl   : float = -1.0
        # cell geometry (cm)
        self._L     : float = 0.01
        self._rad   : float = 0.0011
        # CaMK
        self._KmCaMK : float = 0.15
        self._aCaMK  : float = 0.05
        self._bCaMK  : float = 0.00068
        self._CaMKo  : float = 0.05
        self._KmCaM  : float = 0.0015
        # buffers
        self._cmdnmax_b : float = 0.05
        self._Kmcmdn    : float = 0.00238
        self._trpnmax   : float = 0.07
        self._Kmtrpn    : float = 0.0005
        self._BSRmax    : float = 0.047
        self._KmBSR     : float = 0.00087
        self._BSLmax    : float = 1.124
        self._KmBSL     : float = 0.0087
        self._csqnmax   : float = 10.0
        self._Kmcsqn    : float = 0.8
        # potassium currents
        self._PKNa    : float = 0.01833
        self._gKatp   : float = 4.3195
        self._fKatp   : float = 0.0
        self._K_o_n   : float = 5.0
        self._A_atp   : float = 2.0
        self._K_atp   : float = 0.25
        self._EKshift : float = 0.0
        self._GKb_b   : float = 0.0189
        self._alpha_1 : float = 0.154375
        self._beta_1  : float = 0.1911
        # sodium currents and Ito
        self._thL     : float = 200.0
        # L-type calcium current
        self._Kmn          : float = 0.002
        self._K2n          : float = 500.0
        self._Aff          : float = 0.6
        self._tjCa         : float = 75.0
        self._VShift       : float = 0.0
        self._offset       : float = 0.0
        self._dielConstant : float = 74.0
        self._ICaL_fractionSS : float = 0.8
        # Na/Ca exchanger
        self._INaCa_fractionSS : float = 0.35
        self._KNa1    : float = 15.0
        self._KNa2    : float = 5.0
        self._KNa3    : float = 88.12
        self._Kasymm  : float = 12.5
        self._wNa     : float = 6.0e4
        self._wCa     : float = 6.0e4
        self._wNaCa   : float = 5.0e3
        self._KCaon   : float = 1.5e6
        self._KCaoff  : float = 5.0e3
        self._qNa     : float = 0.5224
        self._qCa     : float = 0.167
        self._KmCaAct : float = 150.0e-6
        self._Gncx_b  : float = 0.0034
        # Na/K pump
        self._K1p     : float = 949.5
        self._K1m     : float = 182.4
        self._K2p     : float = 687.2
        self._K2m     : float = 39.4
        self._K3p     : float = 1899.0
        self._K3m     : float = 79300.0
        self._K4p     : float = 639.0
        self._K4m     : float = 40.0
        self._KNai0   : float = 9.073
        self._KNao0   : float = 27.78
        self._delta   : float = -0.155
        self._KKi     : float = 0.5
        self._KKo     : float = 0.3582
        self._MgADP   : float = 0.05
        self._MgATP   : float = 9.8
        self._Kmgatp  : float = 1.698e-7
        self._H       : float = 1.0e-7
        self._eP      : float = 4.2
        self._Khp     : float = 1.698e-7
        self._KNap    : float = 224.0
        self._KxKur   : float = 292.0
        self._PNaK_b  : float = 15.4509
        # background, pump and chloride currents
        self._PNab    : float = 1.9239e-09
        self._PCab    : float = 5.9194e-08
        self._GpCa    : float = 5.0e-04
        self._KmCap   : float = 0.0005
        self._GClCa   : float = 0.2843
        self._GClb    : float = 1.98e-3
        self._KdClCa  : float = 0.1
        self._Fjunc   : float = 1.0
        # diffusion, release and uptake
        self._tauNa      : float = 2.0
        self._tauK       : float = 2.0
        self._tauCa      : float = 0.2
        self._bt         : float = 4.75
        self._Cajsr_half : float = 1.7
        self._Jrel_b     : float = 1.5378
        self._Jup_b      : float = 1.0

        # ---- initial values -------------------------------------------------
        self._init_values : dict = {'CaMKt': 0.0111, 'Nai': 12.1025, 'Nass': 12.1029,
                                    'Ki': 142.3002, 'Kss': 142.3002, 'Cass': 7.0305e-5,
                                    'Cansr': 1.5211, 'Cajsr': 1.5214, 'Cai': 8.1583e-05,
                                    'm': 8.0572e-4, 'h': 0.8286, 'j': 0.8284, 'hp': 0.6707,
                                    'jp': 0.8281, 'mL': 1.629e-4, 'hL': 0.5255, 'hLp': 0.2872,
                                    'a': 9.5098e-4, 'iF': 0.9996, 'iS': 0.5936, 'ap': 4.8454e-4,
                                    'iFp': 0.9996, 'iSp': 0.6538, 'd': 8.1084e-9, 'ff': 1.0,
                                    'fs': 0.939, 'fCaf': 1.0, 'fCas': 0.9999, 'jCa': 1.0,
                                    'ffp': 1.0, 'fCafp': 1.0, 'nCa_ss': 6.6462e-4,
                                    'nCa_i': 0.0012, 'C1': 7.0344e-4, 'C2': 8.5109e-4,
                                    'C3': 0.9981, 'I': 1.3289e-5, 'O': 3.7585e-4,
                                    'xs1': 0.248, 'xs2': 1.7707e-4,
                                    'Jrel_np': 1.6129e-22, 'Jrel_p': 1.2475e-20}

        # ---- lookup tables --------------------------------------------------
        # Same ranges and steps as the reference implementation. The step is
        # held as a 32-bit float there, and the grid and the interpolation
        # weight are computed from that float: the parameters below are
        # derived the same way (see construct_tables) so that both codes
        # interpolate between the same grid points.
        self._V_T_mn    : float = -1000.0
        self._V_T_mx    : float = 1000.0
        self._V_T_res   : float = 1.0e-2
        self._Cai_T_mn  : float = 1.0e-6
        self._Cai_T_mx  : float = 1.0e-2
        self._Cai_T_res : float = 1.0e-6
        self._V_lut     : dict      = None
        self._Cai_lut   : dict      = None
        self._V_tab     : tf.Tensor = None
        self._Cai_tab   : tf.Tensor = None
        self._V_col     : dict      = None
        self._Cai_col   : dict      = None

        # ---- per-node effective parameters (built by __build_cell_columns) ----
        self._cell_GNa     : tf.Tensor = None
        self._cell_GNaL    : tf.Tensor = None
        self._cell_PCa     : tf.Tensor = None
        self._cell_Gto     : tf.Tensor = None
        self._cell_GKr     : tf.Tensor = None
        self._cell_GKs     : tf.Tensor = None
        self._cell_GK1     : tf.Tensor = None
        self._cell_GKb     : tf.Tensor = None
        self._cell_Gncx    : tf.Tensor = None
        self._cell_PNaK    : tf.Tensor = None
        self._cell_upScale : tf.Tensor = None
        self._cell_Jrel    : tf.Tensor = None
        self._cell_is_epi  : tf.Tensor = None

        # ---- state variables (initialized via initialize_state_variables) ----
        self._CaMKt   : tf.Variable = None
        self._Nai     : tf.Variable = None
        self._Nass    : tf.Variable = None
        self._Ki      : tf.Variable = None
        self._Kss     : tf.Variable = None
        self._Cass    : tf.Variable = None
        self._Cansr   : tf.Variable = None
        self._Cajsr   : tf.Variable = None
        self._Cai     : tf.Variable = None
        self._m       : tf.Variable = None
        self._h       : tf.Variable = None
        self._j       : tf.Variable = None
        self._hp      : tf.Variable = None
        self._jp      : tf.Variable = None
        self._mL      : tf.Variable = None
        self._hL      : tf.Variable = None
        self._hLp     : tf.Variable = None
        self._a       : tf.Variable = None
        self._iF      : tf.Variable = None
        self._iS      : tf.Variable = None
        self._ap      : tf.Variable = None
        self._iFp     : tf.Variable = None
        self._iSp     : tf.Variable = None
        self._d       : tf.Variable = None
        self._ff      : tf.Variable = None
        self._fs      : tf.Variable = None
        self._fCaf    : tf.Variable = None
        self._fCas    : tf.Variable = None
        self._jCa     : tf.Variable = None
        self._ffp     : tf.Variable = None
        self._fCafp   : tf.Variable = None
        self._nCa_ss  : tf.Variable = None
        self._nCa_i   : tf.Variable = None
        self._C1      : tf.Variable = None
        self._C2      : tf.Variable = None
        self._C3      : tf.Variable = None
        self._I       : tf.Variable = None
        self._O       : tf.Variable = None
        self._xs1     : tf.Variable = None
        self._xs2     : tf.Variable = None
        self._Jrel_np : tf.Variable = None
        self._Jrel_p  : tf.Variable = None


    # ---- parameters ---------------------------------------------------------
    def tunable_parameter_names(self) -> tuple:
        """ tunable_parameter_names() returns the names set_parameter() accepts """
        return(('celltype', 'GNa', 'GNaL_b', 'PCa_b', 'Gto_b', 'GKr_b', 'GKs_b', 'GK1_b',
                'CoefGNa', 'CoefGNaL', 'CoefCaL', 'CoefK1', 'CoefKr', 'CoefKs', 'Coefto',
                'Ko', 'Nao', 'Cao', 'V_init'))

    def set_parameter(self, pname: str, pvalue: np.ndarray):
        """
        set_parameter(pname, pvalue) sets the parameter pname to pvalue, a scalar
        or a per-node (n_nodes, 1) column. Only the names listed by
        tunable_parameter_names() are accepted: the other constants are folded
        into the lookup tables, so changing one afterwards would be silently
        ignored, and it raises instead. celltype must be 0, 1 or 2.
        Ko, Nao and Cao take one value for the whole tissue: a column is
        accepted when all its entries are equal (the parameter-file front end
        passes region parameters as columns), and refused otherwise.
        """
        try:
            if pname not in self.tunable_parameter_names():
                raise ValueError('Tomek: "{}" is not a tunable parameter; the model accepts {}'.format(
                    pname, ', '.join(self.tunable_parameter_names())))
            if pname == 'V_init':
                self._V_init = float(pvalue)
                return
            values = np.asarray(pvalue, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError('Tomek: parameter {} has non-finite values'.format(pname))
            if pname in _EXTRACELLULAR:
                self.__set_extracellular(pname, values)
                return
            if pname == 'celltype':
                if not np.all(np.isin(values, [_ENDO, _EPI, _MCELL])):
                    raise ValueError('Tomek: celltype must be 0 (ENDO), 1 (EPI) or 2 (MCELL), '
                                     'got {}'.format(sorted(set(np.reshape(values, (-1,)).tolist()))))
            setattr(self, '_{}'.format(pname), tf.constant(values, dtype=_DTYPE))
            if self._initialized:
                self.__build_cell_columns()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __set_extracellular(self, pname: str, values: np.ndarray):
        """ stores an extracellular concentration as a single positive value """
        unique = np.unique(values)
        if unique.size != 1:
            raise ValueError('Tomek: {} is an extracellular concentration and takes one value for '
                             'all the nodes, got {}'.format(pname, unique.tolist()))
        if unique[0] <= 0.0:
            raise ValueError('Tomek: {} must be positive, got {}'.format(pname, unique[0]))
        setattr(self, '_{}'.format(pname), tf.constant(float(unique[0]), dtype=_DTYPE))
        # Nao enters the voltage table (Na/Ca exchanger, Na/K pump)
        if self._initialized and pname == 'Nao':
            self.construct_tables()

    def set_use_rush_larsen(self, use_rush_larsen: bool):
        """
        set_use_rush_larsen(use_rush_larsen) selects the update of the gating
        variables: Rush-Larsen (True, default) or forward Euler (False). The
        choice is baked into the voltage table, so it must be made before
        initialize_state_variables().
        """
        self._use_rush_larsen = use_rush_larsen

    def effective_parameter(self, pname: str) -> tf.Tensor:
        """
        effective_parameter(pname) returns the per-node value used by
        differentiate() for pname (GNa, GNaL, PCa, Gto, GKr, GKs, GK1, GKb,
        Gncx, PNaK, upScale, Jrel): Coef x cell-type factor x base. It is
        available after initialize_state_variables().
        """
        return(getattr(self, '_cell_{}'.format(pname), None))

    def use_rush_larsen(self) -> bool:
        """ use_rush_larsen() returns True if the gates use the Rush-Larsen update """
        return(self._use_rush_larsen)

    def __build_cell_columns(self):
        """
        builds the per-node effective parameters from celltype, the base
        conductances and the drug-block coefficients. Each result is a flat
        tensor with one entry per node, or a single entry when all its inputs
        are scalars (it then broadcasts).
        """
        celltype = np.reshape(self._celltype.numpy(), (-1,))
        is_epi   = (celltype == _EPI)
        is_mcell = (celltype == _MCELL)

        def factor(epi: float, mcell: float) -> np.ndarray:
            return(np.where(is_epi, epi, np.where(is_mcell, mcell, 1.0)))

        def column(values) -> tf.Tensor:
            return(tf.constant(np.reshape(np.asarray(values, dtype=np.float64), (-1,)), dtype=_DTYPE))

        def value(tensor: tf.Tensor) -> np.ndarray:
            return(np.reshape(tensor.numpy(), (-1,)))

        # cell-type factors of the reference model
        self._cell_GNa     = column(value(self._CoefGNa) * value(self._GNa))
        self._cell_GNaL    = column(value(self._CoefGNaL) * factor(0.6, 1.0) * value(self._GNaL_b))
        self._cell_PCa     = column(value(self._CoefCaL) * factor(1.2, 2.0) * value(self._PCa_b))
        self._cell_Gto     = column(value(self._Coefto) * factor(2.0, 2.0) * value(self._Gto_b))
        self._cell_GKr     = column(value(self._CoefKr) * factor(1.3, 0.8) * value(self._GKr_b))
        self._cell_GKs     = column(value(self._CoefKs) * factor(1.4, 1.0) * value(self._GKs_b))
        self._cell_GK1     = column(value(self._CoefK1) * factor(1.2, 1.3) * value(self._GK1_b))
        self._cell_GKb     = column(factor(0.6, 1.0) * self._GKb_b)
        self._cell_Gncx    = column(factor(1.1, 1.4) * self._Gncx_b)
        self._cell_PNaK    = column(factor(0.9, 0.7) * self._PNaK_b)
        self._cell_upScale = column(factor(1.3, 1.0))
        self._cell_Jrel    = column(factor(1.0, 1.7))
        self._cell_is_epi  = tf.constant(is_epi, dtype=tf.bool)


    # ---- lookup tables ------------------------------------------------------
    def __table_layout(self, mn: float, mx: float, res: float) -> dict:
        """
        returns the grid of a lookup table as the reference implementation
        builds it: the step is a 32-bit float, the index of a value x is
        int(x/step) truncated toward zero, and the grid point of index i is
        i*step evaluated in 32-bit arithmetic.
        """
        res32  = np.float32(res)
        step32 = np.float32(1.0) / res32
        mn_ind = int(np.float32(mn) * step32)
        mx_ind = int(np.float32(mx) * step32)
        grid   = (np.arange(mn_ind, mx_ind + 1).astype(np.float32) * res32).astype(np.float64)
        return({'mn': float(np.float32(mn)), 'mx': float(np.float32(mx)), 'res': float(res32),
                'step': float(step32), 'mn_ind': mn_ind, 'mx_ind': mx_ind, 'grid': grid})

    def __gate_coefficients(self, ss: np.ndarray, tau: np.ndarray) -> tuple:
        """ returns (A, B) such that x_new = A + B x advances a gate by dt """
        dt = self._dt
        if self._use_rush_larsen:
            B = np.exp(-dt / tau)
            A = -ss * np.expm1(-dt / tau)
        else:
            B = 1.0 - dt / tau
            A = dt * ss / tau
        return(A, B)

    def construct_tables(self):
        """ construct_tables() builds the voltage and Cai lookup tables """
        try:
            self.__construct_V_table()
            self.__construct_Cai_table()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __construct_V_table(self):
        """ builds the voltage table: V-only rates and the gate coefficients """
        self._V_lut = self.__table_layout(self._V_T_mn, self._V_T_mx, self._V_T_res)
        V   = self._V_lut['grid']
        EKs = self._EKshift
        Nao = self._Nao.numpy().item()
        RT_F  = self._R * self._T / self._F
        ECl   = (RT_F / self._zcl) * log(self._Clo / self._Cli)
        cols  : dict = {}
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Vfrt = V / RT_F
            cols['Vfrt']  = Vfrt
            cols['Vffrt'] = V * self._F / RT_F
            cols['AfCaf'] = 0.3 + 0.6 / (1.0 + np.exp((V - 10.0) / 10.0))
            cols['AiF']   = 1.0 / (1.0 + np.exp(((V + EKs) - 213.6) / 151.2))
            cols['IClb']  = self._GClb * (V - ECl)
            cols['xKb']   = 1.0 / (1.0 + np.exp(-(V - 10.8968) / 23.9871))
            # IKr Markov rates
            alpha   = 0.1161 * np.exp(0.2990 * Vfrt)
            beta    = 0.2442 * np.exp(-1.604 * Vfrt)
            alpha_2 = 0.0578 * np.exp(0.9710 * Vfrt)
            beta_2  = 0.349e-3 * np.exp(-1.062 * Vfrt)
            alpha_i = 0.2533 * np.exp(0.5953 * Vfrt)
            beta_i  = 0.06525 * np.exp(-0.8209 * Vfrt)
            alpha_C2ToI = 0.52e-4 * np.exp(1.525 * Vfrt)
            cols['alpha']   = alpha
            cols['beta']    = beta
            cols['alpha_2'] = alpha_2
            cols['beta_2']  = beta_2
            cols['alpha_i'] = alpha_i
            cols['beta_i']  = beta_i
            cols['alpha_C2ToI'] = alpha_C2ToI
            cols['beta_ItoC2']  = (beta_2 * beta_i * alpha_C2ToI) / (alpha_2 * alpha_i)
            # Na/Ca exchanger. The extracellular terms h7..h11 are identical for
            # the myoplasm and the subspace, so K3, K3pp and K8 are held once.
            hNa = np.exp(self._qNa * Vfrt)
            cols['hNa'] = hNa
            cols['hCa'] = np.exp(self._qCa * Vfrt)
            h7  = 1.0 + (Nao / self._KNa3) * (1.0 + 1.0 / hNa)
            h8  = Nao / (self._KNa3 * hNa * h7)
            h9  = 1.0 / h7
            h10 = self._Kasymm + 1.0 + (Nao / self._KNa1) * (1.0 + Nao / self._KNa2)
            h11 = (Nao * Nao) / (h10 * self._KNa1 * self._KNa2)
            cols['K3pp'] = h8 * self._wNaCa
            cols['K3']   = h9 * self._wCa + cols['K3pp']
            cols['K8']   = h8 * h11 * self._wNa
            # Na/K pump. Only the voltage-dependent ratio Nao/KNao is tabulated:
            # a3 and b2 also depend on Ko, which can then be changed without
            # rebuilding the table, so they are assembled from this ratio.
            KNao = self._KNao0 * np.exp(((1.0 - self._delta) * Vfrt) / 3.0)
            cols['KNai']     = self._KNai0 * np.exp((self._delta * Vfrt) / 3.0)
            cols['Nao_KNao'] = Nao / KNao

            # INa
            mss  = 1.0 / ((1.0 + np.exp(-(V + 56.86) / 9.03)) ** 2)
            tm   = 0.1292 * np.exp(-((V + 45.79) / 15.54) ** 2) + 0.06487 * np.exp(-((V - 4.823) / 51.12) ** 2)
            hss  = 1.0 / ((1.0 + np.exp((V + 71.55) / 7.43)) ** 2)
            hssp = 1.0 / ((1.0 + np.exp((V + 77.55) / 7.43)) ** 2)
            ah   = np.where(V >= -40.0, 0.0, 0.057 * np.exp(-(V + 80.0) / 6.8))
            bh   = np.where(V >= -40.0, 0.77 / (0.13 * (1.0 + np.exp(-(V + 10.66) / 11.1))),
                            2.7 * np.exp(0.079 * V) + 3.1e5 * np.exp(0.3485 * V))
            aj   = np.where(V >= -40.0, 0.0,
                            ((-2.5428e4 * np.exp(0.2444 * V) - 6.948e-6 * np.exp(-0.04391 * V)) * (V + 37.78))
                            / (1.0 + np.exp(0.311 * (V + 79.23))))
            bj   = np.where(V >= -40.0, (0.6 * np.exp(0.057 * V)) / (1.0 + np.exp(-0.1 * (V + 32.0))),
                            (0.02424 * np.exp(-0.01052 * V)) / (1.0 + np.exp(-0.1378 * (V + 40.14))))
            th   = 1.0 / (ah + bh)
            tj   = 1.0 / (aj + bj)
            # INaL
            mLss  = 1.0 / (1.0 + np.exp(-(V + 42.85) / 5.264))
            hLss  = 1.0 / (1.0 + np.exp((V + 87.61) / 7.488))
            hLssp = 1.0 / (1.0 + np.exp((V + 93.81) / 7.488))
            thL   = np.full_like(V, self._thL)
            # Ito
            ass  = 1.0 / (1.0 + np.exp(-((V + EKs) - 14.34) / 14.82))
            assp = 1.0 / (1.0 + np.exp(-((V + EKs) - 24.34) / 14.82))
            ta   = 1.0515 / (1.0 / (1.2089 * (1.0 + np.exp(-((V + EKs) - 18.4099) / 29.3814)))
                             + 3.5 / (1.0 + np.exp((V + EKs + 100.0) / 29.3814)))
            iss  = 1.0 / (1.0 + np.exp((V + EKs + 43.94) / 5.711))
            tiF_b = 4.562 + 1.0 / (0.3933 * np.exp(-(V + EKs + 100.0) / 100.0)
                                   + 0.08004 * np.exp((V + EKs + 50.0) / 16.59))
            tiS_b = 23.62 + 1.0 / (0.001416 * np.exp(-(V + EKs + 96.52) / 59.05)
                                   + 1.78e-8 * np.exp((V + EKs + 114.1) / 8.079))
            dti   = ((1.354 + 1.0e-4 / (np.exp(((V + EKs) - 167.4) / 15.89) + np.exp(-((V + EKs) - 12.23) / 0.2154)))
                     * (1.0 - 0.5 / (1.0 + np.exp((V + EKs + 70.0) / 20.0))))
            delta_epi = 1.0 - 0.95 / (1.0 + np.exp((V + EKs + 70.0) / 5.0))
            # ICaL
            dss   = np.where(V >= 31.4978, 1.0, 1.0763 * np.exp(-1.0070 * np.exp(-0.0829 * V)))
            td    = self._offset + 0.6 + 1.0 / (np.exp(-0.05 * (V + self._VShift + 6.0))
                                                + np.exp(0.09 * (V + self._VShift + 14.0)))
            fss   = 1.0 / (1.0 + np.exp((V + 19.58) / 3.696))
            tff   = 7.0 + 1.0 / (0.0045 * np.exp(-(V + 20.0) / 10.0) + 0.0045 * np.exp((V + 20.0) / 10.0))
            tfs   = 1000.0 + 1.0 / (0.000035 * np.exp(-(V + 5.0) / 4.0) + 0.000035 * np.exp((V + 5.0) / 6.0))
            tfCaf = 7.0 + 1.0 / (0.04 * np.exp(-(V - 4.0) / 7.0) + 0.04 * np.exp((V - 4.0) / 7.0))
            tfCas = 100.0 + 1.0 / (0.00012 * np.exp(-V / 3.0) + 0.00012 * np.exp(V / 7.0))
            jCass = 1.0 / (1.0 + np.exp((V + 18.08) / 2.7916))
            tjCa  = np.full_like(V, self._tjCa)
            # IKs
            xs1ss = 1.0 / (1.0 + np.exp(-(V + 11.6) / 8.932))
            txs1  = 817.3 + 1.0 / (2.326e-4 * np.exp((V + 48.28) / 17.8) + 0.001292 * np.exp(-(V + 210.0) / 230.0))
            txs2  = 1.0 / (0.01 * np.exp((V - 50.0) / 20.0) + 0.0193 * np.exp(-(V + 66.54) / 31.0))

            # (steady state, time constant) of every gate. The four Ito
            # inactivation gates have a second version for EPI cells, whose
            # time constants are scaled by a function of V (delta_epi): a
            # per-node scalar cannot express that, so both versions are
            # tabulated and differentiate() picks one per node.
            gates = {'m': (mss, tm), 'h': (hss, th), 'j': (hss, tj), 'hp': (hssp, th),
                     'jp': (hss, 1.46 * tj), 'mL': (mLss, tm), 'hL': (hLss, thL),
                     'hLp': (hLssp, 3.0 * thL), 'a': (ass, ta), 'ap': (assp, ta),
                     'iF': (iss, tiF_b), 'iS': (iss, tiS_b),
                     'iFp': (iss, dti * tiF_b), 'iSp': (iss, dti * tiS_b),
                     'iF_epi': (iss, tiF_b * delta_epi), 'iS_epi': (iss, tiS_b * delta_epi),
                     'iFp_epi': (iss, dti * tiF_b * delta_epi), 'iSp_epi': (iss, dti * tiS_b * delta_epi),
                     'd': (dss, td), 'ff': (fss, tff), 'fs': (fss, tfs), 'fCaf': (fss, tfCaf),
                     'fCas': (fss, tfCas), 'jCa': (jCass, tjCa), 'ffp': (fss, 2.5 * tff),
                     'fCafp': (fss, 2.5 * tfCaf), 'xs1': (xs1ss, txs1), 'xs2': (xs1ss, txs2)}
            for name, (ss, tau) in gates.items():
                A, B = self.__gate_coefficients(ss, tau)
                cols['{}_A'.format(name)] = A
                cols['{}_B'.format(name)] = B

        # A table row is only used near the potential it tabulates, but linear
        # interpolation multiplies the neighbouring row by a weight that can be
        # exactly 0, and 0*inf is NaN. Every entry must therefore be finite.
        names = sorted(cols.keys())
        table = np.column_stack([np.broadcast_to(cols[name], V.shape) for name in names])
        if not np.all(np.isfinite(table)):
            bad = [name for name in names if not np.all(np.isfinite(cols[name]))]
            raise ValueError('Tomek: non-finite entries in the voltage table columns {}'.format(bad))
        self._V_col = {name: index for index, name in enumerate(names)}
        self._V_tab = tf.constant(table, dtype=_DTYPE)

    def __construct_Cai_table(self):
        """ builds the Cai table (Cai in mM) """
        self._Cai_lut = self.__table_layout(self._Cai_T_mn, self._Cai_T_mx, self._Cai_T_res)
        Cai  = self._Cai_lut['grid']
        cols : dict = {}

        def BCai(cmdnmax: float) -> np.ndarray:
            return(1.0 / (1.0 + (cmdnmax * self._Kmcmdn) / ((self._Kmcmdn + Cai) ** 2)
                          + (self._trpnmax * self._Kmtrpn) / ((self._Kmtrpn + Cai) ** 2)))
        # EPI cells have 1.3 times more calmodulin, which enters BCai non-linearly
        cols['BCai']     = BCai(self._cmdnmax_b)
        cols['BCai_epi'] = BCai(1.3 * self._cmdnmax_b)
        cols['IpCa']     = (self._GpCa * Cai) / (self._KmCap + Cai)
        # SERCA uptake without the cell-type scale, which is linear and applied per node
        cols['Jupnp']    = (0.005425 * Cai) / (Cai + 0.00092)
        cols['Jupp']     = (2.75 * 0.005425 * Cai) / ((Cai + 0.00092) - 0.00017)
        cols['KsCa']     = 1.0 + 0.6 / (1.0 + (3.8e-5 / Cai) ** 1.4)
        cols['allo_i']   = 1.0 / (1.0 + (self._KmCaAct / Cai) ** 2)
        names = sorted(cols.keys())
        table = np.column_stack([cols[name] for name in names])
        if not np.all(np.isfinite(table)):
            raise ValueError('Tomek: non-finite entries in the Cai table')
        self._Cai_col = {name: index for index, name in enumerate(names)}
        self._Cai_tab = tf.constant(table, dtype=_DTYPE)

    def _interpolate(self, X: tf.Tensor, table: tf.Tensor, lut: dict) -> tf.Tensor:
        """
        _interpolate(X, table, lut) returns the table rows interpolated at X.
        The index is X*step truncated toward zero and the weight is measured
        from that grid point, as in the reference implementation: for a
        negative X the truncation lands on the upper neighbour and the value
        is extrapolated from the next interval. The error is second order in
        the step in both cases (below 1e-8 relative for the 0.01 mV step), and
        matching the rule keeps a comparison with that implementation free of
        interpolation differences. Outside [mn, mx] the edge row is returned.
        """
        Xc     = tf.clip_by_value(X, lut['mn'], lut['mx'])
        idx    = tf.cast(Xc * lut['step'], tf.int64)
        idx    = tf.clip_by_value(idx, lut['mn_ind'], lut['mx_ind'])
        # idx*res in 32-bit arithmetic, as the grid was built
        gridpt = tf.cast(tf.cast(idx, tf.float32) * tf.constant(lut['res'], dtype=tf.float32), _DTYPE)
        derr   = tf.expand_dims((X - gridpt) / lut['res'], axis=-1)
        lo     = idx - lut['mn_ind']
        hi     = tf.minimum(lo + 1, lut['mx_ind'] - lut['mn_ind'])
        row_lo = tf.gather(table, lo)
        row    = (1.0 - derr) * row_lo + derr * tf.gather(table, hi)
        oob    = tf.expand_dims(tf.logical_or(X < lut['mn'], X > lut['mx']), axis=-1)
        return(tf.where(oob, row_lo, row))


    # ---- state --------------------------------------------------------------
    def initialize_state_variables(self, U: tf.Variable):
        """
        initialize_state_variables(U) builds the lookup tables and the per-node
        parameters, and creates the 43 state variables with U's shape
        """
        if not self._initialized:
            try:
                self.construct_tables()
                shape = tf.shape(U)
                for name in self.state_variable_names():
                    init = tf.fill(shape, tf.constant(self._init_values[name], dtype=_DTYPE))
                    setattr(self, '_{}'.format(name), tf.Variable(init, name=name))
                self._initialized = True
                self.__build_cell_columns()
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

    def state_variable_names(self) -> tuple:
        """ state_variable_names() returns the 43 variables advanced by differentiate() """
        return(('CaMKt', 'Nai', 'Nass', 'Ki', 'Kss', 'Cass', 'Cansr', 'Cajsr', 'Cai',
                'm', 'h', 'j', 'hp', 'jp', 'mL', 'hL', 'hLp', 'a', 'iF', 'iS', 'ap', 'iFp', 'iSp',
                'd', 'ff', 'fs', 'fCaf', 'fCas', 'jCa', 'ffp', 'fCafp', 'nCa_ss', 'nCa_i',
                'C1', 'C2', 'C3', 'I', 'O', 'xs1', 'xs2', 'Jrel_np', 'Jrel_p'))


    # ---- dynamics -----------------------------------------------------------
    @tf.function(jit_compile=True)
    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """
        differentiate(U) returns dU = -Iion for the potential U (mV) and
        advances all 43 state variables by dt
        """
        V  = tf.reshape(tf.cast(U, _DTYPE), [-1])
        s  = {name: tf.reshape(getattr(self, '_{}'.format(name)), [-1])
              for name in self.state_variable_names()}
        Vc = self._V_col
        Cc = self._Cai_col
        Vr = self._interpolate(V, self._V_tab, self._V_lut)
        Cr = self._interpolate(s['Cai'], self._Cai_tab, self._Cai_lut)

        def vt(name: str) -> tf.Tensor:
            return(Vr[:, Vc[name]])

        def ct(name: str) -> tf.Tensor:
            return(Cr[:, Cc[name]])

        is_epi = self._cell_is_epi

        # ---- constants ----
        # extracellular concentrations: one value for the tissue, read when the
        # function is traced (.numpy() rather than float(), which autograph
        # would turn into a tensor cast)
        Ko, Nao, Cao = self._Ko.numpy().item(), self._Nao.numpy().item(), self._Cao.numpy().item()
        R, T, F = self._R, self._T, self._F
        RT_F    = R * T / F
        Vcell   = 1000.0 * 3.14 * self._rad * self._rad * self._L
        Ageo    = 2.0 * 3.14 * self._rad * self._rad + 2.0 * 3.14 * self._rad * self._L
        ACap    = 2.0 * Ageo
        Vmyo    = 0.68 * Vcell
        Vnsr    = 0.0552 * Vcell
        Vjsr    = 0.0048 * Vcell
        Vss     = 0.02 * Vcell
        ECl     = (RT_F / self._zcl) * log(self._Clo / self._Cli)
        Afs     = 1.0 - self._Aff
        bKiK    = 1.0 / (1.0 + (self._A_atp / self._K_atp) ** 2)
        constA  = 1.82e6 * (self._dielConstant * T) ** (-1.5)
        h10     = self._Kasymm + 1.0 + (Nao / self._KNa1) * (1.0 + Nao / self._KNa2)
        K1_ncx  = (1.0 / h10) * Cao * self._KCaon
        K2_ncx  = self._KCaoff
        K5_ncx  = self._KCaoff
        b1      = self._K1m * self._MgADP
        a2      = self._K2p
        a4      = ((self._K4p * self._MgATP) / self._Kmgatp) / (1.0 + self._MgATP / self._Kmgatp)

        # ---- per-node parameters ----
        GNa, GNaL, PCa = self._cell_GNa, self._cell_GNaL, self._cell_PCa
        Gto, GKr, GKs  = self._cell_Gto, self._cell_GKr, self._cell_GKs
        GK1, GKb       = self._cell_GK1, self._cell_GKb
        Gncx, PNaK     = self._cell_Gncx, self._cell_PNaK

        # ---- terms of the extracellular concentrations ----
        aKiK      = (Ko / self._K_o_n) ** 0.24
        sqrtKo    = sqrt(Ko / 5.0)
        Io        = 0.5 * (Nao + Ko + self._Clo + 4.0 * Cao) / 1000.0
        dh_o      = sqrt(Io) / (1.0 + sqrt(Io)) - 0.3 * Io
        gamma_o1  = exp(-constA * dh_o)
        gamma_Cao = exp(-constA * 4.0 * dh_o)

        Nai, Nass, Ki, Kss  = s['Nai'], s['Nass'], s['Ki'], s['Kss']
        Cai, Cass, Cajsr    = s['Cai'], s['Cass'], s['Cajsr']

        # The GHK terms below are 0/0 at V = 0 exactly, which a potential can
        # hit (the grid point is exactly 0). Moving Vfrt off zero by 1e-12 gives
        # the correct limit; Vffrt = F*Vfrt is kept consistent with it.
        Vfrt  = vt('Vfrt')
        Vffrt = vt('Vffrt')
        at_zero = tf.abs(Vfrt) < 1.0e-20
        Vfrt  = tf.where(at_zero, tf.constant(1.0e-12, dtype=_DTYPE), Vfrt)
        Vffrt = tf.where(at_zero, tf.constant(1.0e-12 * F, dtype=_DTYPE), Vffrt)
        exp1  = tf.exp(Vfrt)
        exp2  = tf.exp(2.0 * Vfrt)
        em1   = tf.math.expm1(Vfrt)
        em2   = tf.math.expm1(2.0 * Vfrt)

        # ---- CaMK ----
        CaMKb = (self._CaMKo * (1.0 - s['CaMKt'])) / (1.0 + self._KmCaM / Cass)
        CaMKa = CaMKb + s['CaMKt']
        # one phosphorylated fraction serves INa, INaL, Ito, ICaL, Jrel and Jup
        fCaMKp = 1.0 / (1.0 + self._KmCaMK / CaMKa)

        # ---- reversal potentials ----
        ENa = (RT_F / self._zNa) * tf.math.log(Nao / Nai)
        EK  = (RT_F / self._zK) * tf.math.log(Ko / Ki)
        EKs = (RT_F / self._zK) * tf.math.log((Ko + self._PKNa * Nao) / (Ki + self._PKNa * Nai))

        # ---- sodium currents ----
        m = s['m']
        INa  = GNa * (V - ENa) * m * m * m * ((1.0 - fCaMKp) * s['h'] * s['j'] + fCaMKp * s['hp'] * s['jp'])
        INaL = GNaL * (V - ENa) * s['mL'] * ((1.0 - fCaMKp) * s['hL'] + fCaMKp * s['hLp'])

        # ---- Ito ----
        AiF = vt('AiF')
        AiS = 1.0 - AiF
        i_t = AiF * s['iF'] + AiS * s['iS']
        ip  = AiF * s['iFp'] + AiS * s['iSp']
        Ito = Gto * (V - EK) * ((1.0 - fCaMKp) * s['a'] * i_t + fCaMKp * s['ap'] * ip)

        # ---- ICaL, ICaNa, ICaK ----
        f      = self._Aff * s['ff'] + Afs * s['fs']
        fp     = self._Aff * s['ffp'] + Afs * s['fs']
        AfCaf  = vt('AfCaf')
        AfCas  = 1.0 - AfCaf
        fCa    = AfCaf * s['fCaf'] + AfCas * s['fCas']
        fCap   = AfCaf * s['fCafp'] + AfCas * s['fCas']
        jCa    = s['jCa']
        d      = s['d']
        Ii     = 0.5 * (Nai + Ki + self._Cli + 4.0 * Cai) / 1000.0
        Iss    = 0.5 * (Nass + Kss + self._Cli + 4.0 * Cass) / 1000.0
        dh_i   = tf.sqrt(Ii) / (1.0 + tf.sqrt(Ii)) - 0.3 * Ii
        dh_ss  = tf.sqrt(Iss) / (1.0 + tf.sqrt(Iss)) - 0.3 * Iss
        gamma1_i  = tf.exp(-constA * dh_i)
        gamma4_i  = tf.exp(-constA * 4.0 * dh_i)
        gamma1_ss = tf.exp(-constA * dh_ss)
        gamma4_ss = tf.exp(-constA * 4.0 * dh_ss)
        PhiCaL_ss  = (4.0 * Vffrt * (gamma4_ss * Cass * exp2 - gamma_Cao * Cao)) / em2
        PhiCaNa_ss = (Vffrt * (gamma1_ss * Nass * exp1 - gamma_o1 * Nao)) / em1
        PhiCaK_ss  = (Vffrt * (gamma1_ss * Kss * exp1 - gamma_o1 * Ko)) / em1
        PhiCaL_i   = (4.0 * Vffrt * (gamma4_i * Cai * exp2 - gamma_Cao * Cao)) / em2
        PhiCaNa_i  = (Vffrt * (gamma1_i * Nai * exp1 - gamma_o1 * Nao)) / em1
        PhiCaK_i   = (Vffrt * (gamma1_i * Ki * exp1 - gamma_o1 * Ko)) / em1
        PCap   = 1.1 * PCa
        open_ss  = d * (f * (1.0 - s['nCa_ss']) + jCa * fCa * s['nCa_ss'])
        openp_ss = d * (fp * (1.0 - s['nCa_ss']) + jCa * fCap * s['nCa_ss'])
        open_i   = d * (f * (1.0 - s['nCa_i']) + jCa * fCa * s['nCa_i'])
        openp_i  = d * (fp * (1.0 - s['nCa_i']) + jCa * fCap * s['nCa_i'])
        # the Na and K permeabilities are fixed fractions of PCa, so CoefCaL
        # (inside PCa) blocks the three currents through the channel together
        chan_ss = self._ICaL_fractionSS * ((1.0 - fCaMKp) * PCa * open_ss + fCaMKp * PCap * openp_ss)
        chan_i  = (1.0 - self._ICaL_fractionSS) * ((1.0 - fCaMKp) * PCa * open_i + fCaMKp * PCap * openp_i)
        ICaL_ss  = chan_ss * PhiCaL_ss
        ICaNa_ss = 0.00125 * chan_ss * PhiCaNa_ss
        ICaK_ss  = 3.574e-4 * chan_ss * PhiCaK_ss
        ICaL_i   = chan_i * PhiCaL_i
        ICaNa_i  = 0.00125 * chan_i * PhiCaNa_i
        ICaK_i   = 3.574e-4 * chan_i * PhiCaK_i
        ICaL  = ICaL_ss + ICaL_i
        ICaNa = ICaNa_ss + ICaNa_i
        ICaK  = ICaK_ss + ICaK_i

        # ---- potassium currents ----
        IKr = GKr * sqrtKo * s['O'] * (V - EK)
        IKs = GKs * ct('KsCa') * s['xs1'] * s['xs2'] * (V - EKs)
        VEK = V - EK
        aK1 = 4.094 / (1.0 + tf.exp(0.1217 * (VEK - 49.934)))
        bK1 = ((15.72 * tf.exp(0.0674 * (VEK - 3.257)) + tf.exp(0.0618 * (VEK - 594.31)))
               / (1.0 + tf.exp(-0.1629 * (VEK + 14.207))))
        IK1 = GK1 * sqrtKo * (aK1 / (aK1 + bK1)) * VEK
        IKb = GKb * vt('xKb') * VEK
        IKATP = self._fKatp * self._gKatp * aKiK * bKiK * VEK

        # ---- Na/Ca exchanger ----
        hNa  = vt('hNa')
        hCa  = vt('hCa')
        K3   = vt('K3')
        K3pp = vt('K3pp')
        K8   = vt('K8')

        def ncx_fluxes(Na: tf.Tensor, Ca: tf.Tensor) -> tuple:
            h1 = 1.0 + (Na / self._KNa3) * (1.0 + hNa)
            h2 = (Na * hNa) / (self._KNa3 * h1)
            h3 = 1.0 / h1
            h4 = 1.0 + (Na / self._KNa1) * (1.0 + Na / self._KNa2)
            h5 = (Na * Na) / (h4 * self._KNa1 * self._KNa2)
            h6 = 1.0 / h4
            K4pp = h2 * self._wNaCa
            K4   = (h3 * self._wCa) / hCa + K4pp
            K6   = h6 * Ca * self._KCaon
            K7   = h5 * h2 * self._wNa
            x1 = K2_ncx * K4 * (K7 + K6) + K5_ncx * K7 * (K2_ncx + K3)
            x2 = K1_ncx * K7 * (K4 + K5_ncx) + K4 * K6 * (K1_ncx + K8)
            x3 = K1_ncx * K3 * (K7 + K6) + K8 * K6 * (K2_ncx + K3)
            x4 = K2_ncx * K8 * (K4 + K5_ncx) + K3 * K5_ncx * (K1_ncx + K8)
            xs = x1 + x2 + x3 + x4
            E1, E2, E3, E4 = x1 / xs, x2 / xs, x3 / xs, x4 / xs
            JncxNa = 3.0 * (E4 * K7 - E1 * K8) + E3 * K4pp - E2 * K3pp
            JncxCa = E2 * K2_ncx - E1 * K1_ncx
            return(self._zNa * JncxNa + self._zCa * JncxCa)

        allo_ss  = 1.0 / (1.0 + (self._KmCaAct / Cass) ** 2)
        INaCa_i  = (1.0 - self._INaCa_fractionSS) * Gncx * ct('allo_i') * ncx_fluxes(Nai, Cai)
        INaCa_ss = self._INaCa_fractionSS * Gncx * allo_ss * ncx_fluxes(Nass, Cass)

        # ---- Na/K pump ----
        KNai  = vt('KNai')
        r_o   = vt('Nao_KNao')
        den_o = (1.0 + r_o) ** 3 + (1.0 + Ko / self._KKo) ** 2 - 1.0
        a3    = (self._K3p * (Ko / self._KKo) ** 2) / den_o
        b2    = (self._K2m * r_o ** 3) / den_o
        P    = self._eP / (1.0 + self._H / self._Khp + Nai / self._KNap + Ki / self._KxKur)
        den_i = (1.0 + Nai / KNai) ** 3 + (1.0 + Ki / self._KKi) ** 2 - 1.0
        a1   = (self._K1p * (Nai / KNai) ** 3) / den_i
        b3   = (self._K3m * P * self._H) / (1.0 + self._MgATP / self._Kmgatp)
        b4   = (self._K4m * (Ki / self._KKi) ** 2) / den_i
        x1   = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
        x2   = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
        x3   = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
        x4   = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1
        xs   = x1 + x2 + x3 + x4
        JNaKNa = 3.0 * ((x1 / xs) * a3 - (x2 / xs) * b3)
        JNaKK  = 2.0 * ((x4 / xs) * b1 - (x3 / xs) * a1)
        INaK   = PNaK * (self._zNa * JNaKNa + self._zK * JNaKK)

        # ---- background, pump and chloride currents ----
        INab  = (self._PNab * Vffrt * (Nai * exp1 - Nao)) / em1
        ICab  = (self._PCab * 4.0 * Vffrt * (gamma4_i * Cai * exp2 - gamma_Cao * Cao)) / em2
        IpCa  = ct('IpCa')
        IClCa = ((self._Fjunc * self._GClCa) / (1.0 + self._KdClCa / Cass)
                 + ((1.0 - self._Fjunc) * self._GClCa) / (1.0 + self._KdClCa / Cai)) * (V - ECl)
        IClb  = vt('IClb')

        Iion = (INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1 + INaCa_i + INaCa_ss
                + INaK + INab + IKb + IpCa + ICab + IClCa + IClb + IKATP)

        # ---- fluxes ----
        JdiffNa = (Nass - Nai) / self._tauNa
        JdiffK  = (Kss - Ki) / self._tauK
        Jdiff   = (Cass - Cai) / self._tauCa
        Jtr     = (s['Cansr'] - Cajsr) / 60.0
        # SR release: non-phosphorylated and CaMK-phosphorylated components
        jsr_gate = 1.0 + (self._Cajsr_half / Cajsr) ** 8
        btp      = 1.25 * self._bt
        Jrel_inf  = self._cell_Jrel * (-(0.5 * self._bt) * ICaL_ss) / jsr_gate
        Jrel_infp = self._cell_Jrel * (-(0.5 * btp) * ICaL_ss) / jsr_gate
        tau_rel   = tf.maximum(self._bt / (1.0 + 0.0123 / Cajsr), 0.001)
        tau_relp  = tf.maximum(btp / (1.0 + 0.0123 / Cajsr), 0.001)
        Jrel = self._Jrel_b * ((1.0 - fCaMKp) * s['Jrel_np'] + fCaMKp * s['Jrel_p'])
        # SERCA uptake and leak
        Jup  = self._Jup_b * (self._cell_upScale * ((1.0 - fCaMKp) * ct('Jupnp') + fCaMKp * ct('Jupp'))
                              - (0.0048825 * s['Cansr']) / 15.0)

        # ---- time derivatives (forward Euler variables) ----
        dt = self._dt
        diff : dict = {}
        diff['CaMKt'] = self._aCaMK * CaMKb * (CaMKb + s['CaMKt']) - self._bCaMK * s['CaMKt']
        diff['Nai']   = (-(INa + INaL + 3.0 * INaCa_i + ICaNa_i + 3.0 * INaK + INab) * ACap / (F * Vmyo)
                         + JdiffNa * Vss / Vmyo)
        diff['Nass']  = -(ICaNa_ss + 3.0 * INaCa_ss) * ACap / (F * Vss) - JdiffNa
        diff['Ki']    = (-((Ito + IKr + IKs + IK1 + IKb + IKATP - 2.0 * INaK) + ICaK_i) * ACap / (F * Vmyo)
                         + JdiffK * Vss / Vmyo)
        diff['Kss']   = -ICaK_ss * ACap / (F * Vss) - JdiffK
        BCai = tf.where(is_epi, ct('BCai_epi'), ct('BCai'))
        diff['Cai']   = BCai * (-(ICaL_i + IpCa + ICab - 2.0 * INaCa_i) * ACap / (2.0 * F * Vmyo)
                                - Jup * Vnsr / Vmyo + Jdiff * Vss / Vmyo)
        BCass = 1.0 / (1.0 + (self._BSRmax * self._KmBSR) / ((self._KmBSR + Cass) ** 2)
                       + (self._BSLmax * self._KmBSL) / ((self._KmBSL + Cass) ** 2))
        diff['Cass']  = BCass * (-(ICaL_ss - 2.0 * INaCa_ss) * ACap / (2.0 * F * Vss)
                                 + Jrel * Vjsr / Vss - Jdiff)
        diff['Cansr'] = Jup - Jtr * Vjsr / Vnsr
        BCajsr = 1.0 / (1.0 + (self._csqnmax * self._Kmcsqn) / ((self._Kmcsqn + Cajsr) ** 2))
        diff['Cajsr'] = BCajsr * (Jtr - Jrel)
        diff['Jrel_np'] = (Jrel_inf - s['Jrel_np']) / tau_rel
        diff['Jrel_p']  = (Jrel_infp - s['Jrel_p']) / tau_relp
        Km2n = jCa
        anCa_ss = 1.0 / (self._K2n / Km2n + (1.0 + self._Kmn / Cass) ** 4)
        anCa_i  = 1.0 / (self._K2n / Km2n + (1.0 + self._Kmn / Cai) ** 4)
        diff['nCa_ss'] = anCa_ss * self._K2n - s['nCa_ss'] * Km2n
        diff['nCa_i']  = anCa_i * self._K2n - s['nCa_i'] * Km2n
        alpha, beta     = vt('alpha'), vt('beta')
        alpha_2, beta_2 = vt('alpha_2'), vt('beta_2')
        alpha_i, beta_i = vt('alpha_i'), vt('beta_i')
        alpha_C2ToI, beta_ItoC2 = vt('alpha_C2ToI'), vt('beta_ItoC2')
        C1, C2, C3, Iinact, O = s['C1'], s['C2'], s['C3'], s['I'], s['O']
        diff['C3'] = beta * C2 - alpha * C3
        diff['C2'] = alpha * C3 + self._beta_1 * C1 - (beta + self._alpha_1) * C2
        diff['C1'] = (self._alpha_1 * C2 + beta_2 * O + beta_ItoC2 * Iinact
                      - (self._beta_1 + alpha_2 + alpha_C2ToI) * C1)
        diff['O']  = alpha_2 * C1 + beta_i * Iinact - (beta_2 + alpha_i) * O
        diff['I']  = alpha_C2ToI * C1 + alpha_i * O - (beta_ItoC2 + beta_i) * Iinact

        new : dict = {name: s[name] + dt * rate for name, rate in diff.items()}

        # ---- gates: x_new = A(V) + B(V) x ----
        for name in ('m', 'h', 'j', 'hp', 'jp', 'mL', 'hL', 'hLp', 'a', 'ap', 'd', 'ff', 'fs',
                     'fCaf', 'fCas', 'jCa', 'ffp', 'fCafp', 'xs1', 'xs2'):
            new[name] = vt('{}_A'.format(name)) + vt('{}_B'.format(name)) * s[name]
        for name in ('iF', 'iS', 'iFp', 'iSp'):
            A = tf.where(is_epi, vt('{}_epi_A'.format(name)), vt('{}_A'.format(name)))
            B = tf.where(is_epi, vt('{}_epi_B'.format(name)), vt('{}_B'.format(name)))
            new[name] = A + B * s[name]

        # ---- bounds of the reference model, then store ----
        concentrations = ('CaMKt', 'Nai', 'Nass', 'Ki', 'Kss', 'Cai', 'Cass', 'Cansr', 'Cajsr')
        for name in self.state_variable_names():
            if name in concentrations:
                value = tf.maximum(new[name], 1.0e-9)
            else:
                value = tf.clip_by_value(new[name], 0.0, 1.0)
            variable = getattr(self, '_{}'.format(name))
            variable.assign(tf.reshape(value, tf.shape(variable)))

        dU = tf.reshape(-Iion, tf.shape(U))
        return(tf.cast(dU, U.dtype))
