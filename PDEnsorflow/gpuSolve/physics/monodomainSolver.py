#!/usr/bin/env python
"""
    MonodomainSolver: one-step implicit FEM monodomain solver (TensorFlow).

    Inherits HeatSolver and adds an ionic-model term to the RHS:
        (MASS + dt * STIFFNESS) U^{n+1} = MASS (U^n + dt (differentiate(U) + I0))

    The ionic model is held by composition. Permutation of nodal properties
    and ionic-state variables (when reverse Cuthill-McKee renumbering is on)
    is handled here so the per-step kernel stays clean.
"""
import numpy as np
import tensorflow as tf

from gpuSolve.physics.heatSolver import HeatSolver
from gpuSolve.ionic.ionicmodel import IonicModel


class MonodomainSolver(HeatSolver):
    """ Monodomain step: heat-equation step + ionic forward Euler. """

    _U_DEFAULT = -80.0

    def __init__(self, ionic_model: IonicModel, cfgdict=None):
        super().__init__(cfgdict)
        self._ionic_model = ionic_model
        # Keep the ionic model's dt aligned with the solver's dt.
        if hasattr(self._ionic_model, '_dt'):
            self._ionic_model._dt = self._dt

    # ---- nodal/material extensions -----------------------------------------
    def add_nodal_material_property(self, pname: str, ptype: str, prop: dict):
        self._materials.add_nodal_property(pname, ptype, prop)

    def assign_nodal_properties(self):
        """ Push nodal material properties into the ionic model.
            Mirrors the assign_nodal_properties from the original demo
            scripts (mMS.py / fenton.py).
        """
        uniform_only = True
        nodal_properties = self._materials.nodal_property_names()
        if nodal_properties is not None:
            point_region_ids = self._Domain.point_region_ids()
            npt = point_region_ids.shape[0]
            for mat_prop in nodal_properties:
                prtype = self._materials.nodal_property_type(mat_prop)
                refval = self._ionic_model.get_parameter(mat_prop)
                if refval is not None:
                    if prtype == 'uniform':
                        pvals = self._materials.NodalProperty(mat_prop, -1, -1)
                    else:
                        uniform_only = False
                        pvals = np.full(shape=(npt, 1), fill_value=refval.numpy())
                        for pointID, regionID in enumerate(point_region_ids):
                            new_val = self._materials.NodalProperty(
                                mat_prop, pointID, regionID)
                            pvals[pointID] = new_val
                    self._ionic_model.set_parameter(mat_prop, pvals)
        if uniform_only or (not self._use_renumbering):
            self._materials.remove_all_nodal_properties()

    # ---- setup overrides ---------------------------------------------------
    def set_initial_condition(self, U0: np.ndarray = None):
        super().set_initial_condition(U0)
        self._ionic_model.initialize_state_variables(self._U)

    def finalize_for_run(self):
        if self._use_renumbering:
            perm = self._renumbering['perm']
            self._U = tf.Variable(tf.gather(self._U, perm), name=self._U.name)
            # every variable the model advances in time is permuted in lockstep
            # with U. The list comes from the model itself: a variable left out
            # would be harmless while all nodes share the initial value, and
            # silently wrong as soon as a restored state varies from node to node.
            # The permuted values are written into the existing variables, not
            # into new ones: a compiled differentiate() keeps the variable
            # objects it was traced with, so a model compiled before this call
            # would otherwise go on advancing the old, unpermuted copies.
            for name in self._ionic_model.state_variable_names():
                sv = getattr(self._ionic_model, '_{}'.format(name), None)
                if sv is not None:
                    sv.assign(tf.gather(sv, perm))
            if self._StimulusDict is not None:
                for _key, stim in self._StimulusDict.items():
                    stim.apply_indices_permutation(perm)
            nodal_properties = self._materials.nodal_property_names()
            if nodal_properties is not None:
                for mat_prop in nodal_properties:
                    prtype = self._materials.nodal_property_type(mat_prop)
                    refval = self._ionic_model.get_parameter(mat_prop)
                    if refval is not None and prtype != 'uniform':
                        pvals = tf.gather(refval, perm).numpy()
                        self._ionic_model.set_parameter(mat_prop, pvals)
                self._materials.remove_all_nodal_properties()
        self._ready_for_run = True

    # ---- per-step kernel ----------------------------------------------------
    def solve_step(self, U: tf.Variable, I0: tf.constant) -> tf.Variable:
        """ Forward Euler for the ionic ODEs + implicit step for diffusion.
            Not a tf.function, for the reason given in HeatSolver.solve_step;
            the ionic step (differentiate) is compiled with XLA.
        """
        dU   = self._ionic_model.differentiate(U)
        dU   = tf.add(dU, I0)
        RHS0 = tf.add(U, tf.math.scalar_mul(self._dt, dU))
        RHS  = tf.raw_ops.SparseMatrixMatMul(a=self._MASS._matrix, b=RHS0)
        self._Solver.set_X0(self._warm_start_X0(U))
        self._Solver.set_RHS(RHS)
        self._Solver.solve()
        return self._Solver.X()

    # ---- accessors ----------------------------------------------------------
    def ionic_model(self) -> IonicModel:
        return self._ionic_model

    def ionic_state(self, attr: str) -> tf.Variable:
        """ Return an ionic state variable (e.g. '_H_state') in user-space
            (undoing the renumbering permutation if it is active).
        """
        sv = getattr(self._ionic_model, attr, None)
        if sv is None:
            return None
        if self._use_renumbering:
            return tf.gather(sv, self._renumbering['iperm'])
        return sv

    # ---- checkpoint hooks ---------------------------------------------------
    def checkpoint_model_name(self) -> str:
        """ checkpoint_model_name() returns the cell-model name recorded in a
            checkpoint: the class name, which is unambiguous where a parameter
            file name is not (`MitchellSchaeffer` selects one of two classes)
        """
        return(type(self._ionic_model).__name__)

    def _checkpoint_state_variables(self) -> dict:
        """ the cell-model state variables, in the user's node order """
        states = self._ionic_model.get_state_variables()
        return({name: self._to_user_order(values) for name, values in states.items()})

    def _restore_state_variables(self, states: dict):
        """ pushes user-ordered state variables into the cell model """
        self._ionic_model.set_state_variables(
            {name: self._to_solver_order(np.asarray(values)) for name, values in states.items()})
