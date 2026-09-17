# ionic
This package implements the ionic models for cardiac simulations. All the models inherit from the base class `IonicModel` that implements the following methods:

* `set_parameter(pname,pvalue)`  sets the parameter `pname` to the value specified in `pvalue` (if `pname` exists; otherwise id does nothing)
* `get_parameter(pname)` returns the parameter values of `pname` in `pname` exists; `None` otherwise
* `state_variable_names()` returns the names of the variables `differentiate()` advances in time. Each model declares its own list; per-node conductances held as `tf.Variable`s are parameters and are not listed
* `get_state_variables()` returns `{name: flat per-node array}` for every state variable
* `set_state_variables(states)` overwrites the state variables; the names must match `state_variable_names()` exactly
* `set_vmin(vmin = 0.0)` sets the minimum value of the potential for rescaling to `vmin`
* `set_vmax(vmax = 1.0)` sets the maximum value of the potential for rescaling to `vmax`
* `vmin()` returns the minimum value of the potential vmin
* `vmax()` returns the maximum value of the potential vmax
* `to_dimensionless(U)` rescales U to its dimensionless values (range [0,1])
* `to_dimensional(U)` rescales U to its dimensional values (range [vmin,vmax])
* `derivative_to_dimensionless(U)` rescales the derivative of U (*dU*) to dimensionless units
* `derivative_to_dimensional(U)` rescales the derivative of U (*dU*) to dimensional values

## Ionic models implemented

**PDEnsorflow** implements the following cell models:

* `Fenton4v`: The Cherry-Ehrlich-Nattel-Fenton (4v) canine left-atrial model (Heart Rhythm. 2007 Dec;4(12):1553-62)
* `ModifiedMS2v`: The modified Mitchell-Shaeffer (2v) human left-atrial model (Math Biosci. 2016 Nov 281:46-54)
* `MitchellSchaeffer2v`: The Mitchell-Shaeffer (2v) human left-atrial model (Bull Math Bi 2003 Sep 65(5):767-93)
* `Tomek`: The ToR-ORd human ventricular model (eLife 2019;8:e48890). Cell type (`celltype` 0 ENDO, 1 EPI, 2 MCELL) and the base conductances `GNa`, `GNaL_b`, `PCa_b`, `Gto_b`, `GKr_b`, `GKs_b`, `GK1_b` may differ by region; the drug-block factors `CoefGNa`, `CoefGNaL`, `CoefCaL`, `CoefK1`, `CoefKr`, `CoefKs`, `Coefto` multiply the conductances. The extracellular concentrations `Ko`, `Nao`, `Cao` are tunable, one value for the whole tissue. Integrated in float64, gates with Rush-Larsen by default (`set_use_rush_larsen(False)` gives the forward Euler of the reference); `differentiate()` is compiled with XLA; `Cai` is held in mM. Selected in a parameter file with `imp_region[].im = Tomek`

