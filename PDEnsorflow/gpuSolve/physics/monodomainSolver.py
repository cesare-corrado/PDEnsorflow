#!/usr/bin/env python
"""
    MonodomainSolver: FEM monodomain solver (TensorFlow), operator splitting.

    Inherits HeatSolver and adds an ionic-model term: the source of the theta
    step of HeatSolver is S = differentiate(U^n) + I0 (forward Euler for the
    cell model), so by default
        (MASS + theta dt STIFFNESS) U^{n+1} = (MASS - (1 - theta) dt STIFFNESS) (U^n + dt S)
    theta = 0.5 (the default) is Crank-Nicolson, theta = 1 implicit Euler;
    split_source = False takes the source unsplit, (...) U^n + dt MASS S.

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
        # Which cell parameters ended up varying from node to node, and whether
        # any of them does so *within* a region ('nodal' rather than 'region').
        # Both are recorded by assign_nodal_properties() while it still knows,
        # because it clears the property registry on its way out: a later caller
        # that needs to know how the parameters vary, such as the prepacer
        # choosing how many cells to pace, would otherwise read an empty dict
        # and quietly conclude the model is uniform.
        self._nodal_cell_parameters : list       = []
        self._has_nodal_cell_parameters : bool   = False
        # Keep the ionic model's dt aligned with the solver's dt.
        if hasattr(self._ionic_model, '_dt'):
            self._ionic_model._dt = self._dt

    # ---- nodal/material extensions -----------------------------------------
    def add_nodal_material_property(self, pname: str, ptype: str, prop: dict):
        self._materials.add_nodal_property(pname, ptype, prop)

    def assign_nodal_properties(self):
        """ assign_nodal_properties() pushes the nodal material properties into
            the ionic model, as a (n_nodes, 1) column (or the single value of a
            'uniform' property). The column has the dtype of the model's current
            value of the parameter.
            The columns are built with NumPy indexing, not a Python loop over
            the nodes: a 'region' property is looked up once per region and
            spread with the inverse index of point_region_ids, a 'nodal' one is
            gathered at once. On a 3.27M-node mesh with five properties the
            loop made 16M lookups. The values are the ones NodalProperty()
            returns node by node, and a missing region or node still raises.
        """
        uniform_only = True
        nodal_properties = self._materials.nodal_property_names()
        if nodal_properties is not None:
            point_region_ids = np.asarray(self._Domain.point_region_ids())
            npt = point_region_ids.shape[0]
            regions, inverse = np.unique(point_region_ids, return_inverse=True)
            inverse = np.reshape(inverse, (-1,))
            for mat_prop in nodal_properties:
                prtype = self._materials.nodal_property_type(mat_prop)
                refval = self._ionic_model.get_parameter(mat_prop)
                if refval is not None:
                    if prtype == 'uniform':
                        pvals = self._materials.NodalProperty(mat_prop, -1, -1)
                    else:
                        uniform_only = False
                        self._nodal_cell_parameters.append(mat_prop)
                        if prtype == 'nodal':
                            self._has_nodal_cell_parameters = True
                        # the dtype of the model's value, as the loop that
                        # filled np.full(..., refval) had
                        dtype = np.asarray(refval).dtype
                        if prtype == 'region':
                            per_region = np.array([self._materials.NodalProperty(mat_prop, -1, region)
                                                   for region in regions.tolist()], dtype=dtype)
                            values = per_region[inverse]
                        elif prtype == 'nodal':
                            values = self.__nodal_values(mat_prop, npt, dtype)
                        else:
                            # NodalProperty raises the unknown-type error
                            values = np.asarray(self._materials.NodalProperty(mat_prop, 0, point_region_ids[0]))
                        pvals = np.reshape(values, (npt, 1)).astype(dtype, copy=False)
                    self._ionic_model.set_parameter(mat_prop, pvals)
        if uniform_only or (not self._use_renumbering):
            self._materials.remove_all_nodal_properties()

    def __nodal_values(self, mat_prop: str, npt: int, dtype) -> np.ndarray:
        """ the values of a 'nodal' property at points 0..npt-1. An array or a
            list is gathered at once (np.take raises, as idmap[pointID] did,
            when it holds fewer than npt values); a dict is read key by key.
        """
        idmap = self._materials.nodal_property_map(mat_prop)
        if isinstance(idmap, dict):
            return(np.array([idmap[point] for point in range(npt)], dtype=dtype))
        flat = np.asarray(idmap, dtype=dtype)
        flat = np.reshape(flat, (flat.shape[0], -1))[:, 0] if flat.ndim > 1 else flat
        return(np.take(flat, np.arange(npt), mode='raise'))

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
            # The variable is looked up through state_variable(): a plugin's
            # variables carry a prefix that is not an attribute of the model,
            # and getattr would return None and skip them without a word.
            for name in self._ionic_model.state_variable_names():
                sv = self._ionic_model.state_variable(name)
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
        """ Forward Euler for the ionic ODEs + theta-method step for diffusion;
            the ionic and forcing current is the source of HeatSolver's step.
            Not a tf.function, for the reason given in HeatSolver.solve_step;
            the ionic step (differentiate) is compiled with XLA.
        """
        dU   = self._ionic_model.differentiate(U)
        dU   = tf.add(dU, I0)
        RHS  = self._diffusion_rhs(U, dU)
        self._Solver.set_X0(self._warm_start_X0(U))
        self._Solver.set_RHS(RHS)
        self._Solver.solve()
        return self._Solver.X()

    # ---- accessors ----------------------------------------------------------
    def ionic_model(self) -> IonicModel:
        return self._ionic_model

    def nodal_cell_parameters(self) -> list:
        """ nodal_cell_parameters() returns the names of the cell parameters
            assign_nodal_properties() pushed as one value per node; empty before
            that call, and for a model whose parameters are all uniform
        """
        return(self._nodal_cell_parameters)

    def has_nodal_cell_parameters(self) -> bool:
        """ has_nodal_cell_parameters() returns True when a cell parameter was
            registered as a per-node ('nodal') property, i.e. when it may vary
            between two nodes of the same region. False when every parameter is
            uniform or constant per region.
        """
        return(self._has_nodal_cell_parameters)

    def ionic_state(self, attr: str) -> tf.Variable:
        """ Return an ionic state variable (e.g. '_H_state') in user-space
            (undoing the renumbering permutation if it is active).
        """
        sv = getattr(self._ionic_model, attr, None)
        if sv is None:
            return None
        # the same test U() makes: the state variables are permuted by
        # finalize_for_run(), so before it they are already in the user's order
        if self._use_renumbering and self._ready_for_run:
            return tf.gather(sv, self._renumbering['iperm'])
        return sv

    # ---- checkpoint hooks ---------------------------------------------------
    def checkpoint_model_name(self) -> str:
        """ checkpoint_model_name() returns the cell-model name recorded in a
            checkpoint: the class name, which is unambiguous where a parameter
            file name is not (`MitchellSchaeffer` selects one of two classes).
            A model with plugins adds their class names, so a checkpoint only
            restores into a run with the same plugins.
        """
        return(self._ionic_model.model_name())

    def _checkpoint_state_variables(self) -> dict:
        """ the cell-model state variables, in the user's node order """
        states = self._ionic_model.get_state_variables()
        return({name: self._to_user_order(values) for name, values in states.items()})

    def _restore_state_variables(self, states: dict):
        """ pushes user-ordered state variables into the cell model """
        self._ionic_model.set_state_variables(
            {name: self._to_solver_order(np.asarray(values)) for name, values in states.items()})
