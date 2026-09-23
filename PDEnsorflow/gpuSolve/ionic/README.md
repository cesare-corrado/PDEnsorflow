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
* `Tomek`: The ToR-ORd human ventricular model (eLife 2019;8:e48890). Cell type (`celltype` 0 ENDO, 1 EPI, 2 MCELL) and the base conductances `GNa`, `GNaL_b`, `PCa_b`, `Gto_b`, `GKr_b`, `GKs_b`, `GK1_b` may differ by region; the drug-block factors `CoefGNa`, `CoefGNaL`, `CoefCaL`, `CoefK1`, `CoefKr`, `CoefKs`, `Coefto` multiply the conductances. The extracellular concentrations `Ko`, `Nao`, `Cao` are tunable, one value for the whole tissue. Integrated in float64; by default the gates use Rush-Larsen and the IKr Markov chain the matrix exponential, tabulated per potential, which stays a probability distribution under shock potentials where forward Euler is unstable (above +222 mV at `dt = 0.01` ms); `set_use_rush_larsen(False)` gives the forward Euler of the reference for both; `differentiate()` is compiled with XLA; `Cai` is held in mM. Selected in a parameter file with `imp_region[].im = Tomek`

## Ionic plugins

A plugin is a current with its own state variables that is added to the current of a cell model (membrane electroporation, fibroblast coupling, ...). It does not define the potential and cannot run on its own. Plugins live in the subpackage `plugins/` and derive from `IonicPlugin` (`plugins/ionicplugin.py`), which reuses the parameter and state-variable interface of `IonicModel` and replaces `differentiate()` with `compute_current(U)`: the plugin current in uA/uF at the potential `U`, evaluated with the state at the start of the step, after which the state is advanced by `dt`. Plugins work in float64 whatever the precision of `U`.

A model and its plugins are held together by `IonicModelWithPlugins` (`ionicmodelwithplugins.py`), which the solver sees as one `IonicModel`:

```python
model = IonicModelWithPlugins(dt)
model.set_model(Tomek(dt))
model.add_plugin(ElectroporationDeBruinKrassowska98(dt))
```

Within a step the model runs first, then each plugin in the order it was attached, all at the potential of the start of the step, and `dU = -(Iion + the plugin currents)`; this is the order of the reference implementation. Names are routed by a prefix: a bare name (`GNa`, `m`, `V_init`) belongs to the model, `<plugin class>.<name>` to that plugin, for parameters and state variables alike (`ElectroporationDeBruinKrassowska98.n`). `<plugin class>.active` is a per-node 0/1 switch that confines a plugin to some regions. A plugin class can be attached once. `model_name()`, which a checkpoint records, is `Tomek+ElectroporationDeBruinKrassowska98`, so a checkpoint only restores into a run with the same plugins.

**PDEnsorflow** implements the following plugins:

* `ElectroporationDeBruinKrassowska98`: membrane electroporation current (Ann Biomed Eng 1998;26:584-596). State: the pore density `n`, initialised at its steady state for the initial potential and advanced with forward Euler, as in the reference. Parameters `alpha`, `beta`, `q`, `N0`, `sigma`, `h`, `nn`, `w0`. The pore conductance is 0/0 at V = 0 and at V = +-935 mV (default parameters); its finite limit is used within 1e-6 mV of those points, where the reference implementation returns NaN. Selected in a parameter file with `imp_region[].plugins = Electroporation_DeBruinKrassowska98` and tuned with `imp_region[].plug_param`
* `DefibAshiharaTrayanova`: outward current Ia activated by strong depolarization (Ashihara and Trayanova, Biophys J 2004;87:2271-2282), in the form of Cheng et al. (Am J Physiol 1999;277:H351-H362, Eq. 1): `exp(0.09 (V - 100))` up to `VtakeOff`, a straight line above. No state. Parameters `VtakeOff` (160 mV) and `slopeFac` (1). The reference model description writes the lower branch as `exp(0.09 (V - VtakeOff))`, which jumps 221-fold at `VtakeOff`; `set_use_reference_form(True)` selects it, to compare with or reproduce the reference. Its `Ki` state never changes in the reference (its factor `sl_i2c` is 0 for a plugin) and is left out. Selected with `imp_region[].plugins = Defib_AshiharaTrayanova`.
