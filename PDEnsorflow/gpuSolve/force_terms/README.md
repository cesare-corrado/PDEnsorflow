# force_terms

This is the sub-directory of gpuSolve contains the classes and functions to apply force terms:

* `Stimulus`: A class to apply an external stimulus.



## Stimulus

This class implements a source term with a temporal law. Its configuration parameters are:
* the source term `'intensity'` (default = 1.0)
* the source term `'name'` (default = "s2")
* the source term starting time `'tstart'` (default = 0.0)
* the source term number of stimuli `'nstim'` (default = 1)
* the source term `'period'` (default = 1.0)
* the source term stimulus `'duration'` (default = 1.0)

Methods:
* `set_intensity(Imax)`: sets the stimulus intensity to Imax
* `set_stimregion(streg)` set the stimulation region; it takes a numpy tensor as input with non-zero entries on application points.
* `stimulate_tissue_timestep(timestep: int, dt)`: tells if the stimulus is applied at a specific time step (input is an integer); it requires `dt` as input parameter
* `def stimulate_tissue_timevalue(time)`: tells if the stimulus is applied at a specific time value (input is a float)
* `is_active()`: tells if the stimulus is still active
* `deactivate()`: deactivates the stimulus
* `activate()`: activates the stimulus
* `stimApp(time)`: returns the stimulus if applied at current time; None otherwise
* The source term tensor is obtained by invoking the class name.

