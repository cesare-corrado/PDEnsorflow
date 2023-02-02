# force_terms

This is the sub-directory of gpuSolve contains the classes and functions to apply force terms:

* `Stimulus`: A class to apply an external stimulus.



## Stimulus

This class implements a source term with a temporal law. Its configuration parameters are:
* the forcing term starting time `'tstart'` (default = 0.0)
* the forcing term number of stimuli `'nstim'` (default = 1)
* the forcing term `'period'` (default = 1.0)
* the forcing term  `'duration'` (default = 1.0)
* the forcing term `'intensity'` (default = 1.0)
* the forcing term `'name'` (default = "s2")


**Methods:**

* `set_tstart(tstart)`: sets the staring time of the forcing term to *tstart*
* `set_nstim(nstim)`: sets the total nb of stimuli to *nstim*
* `set_period(period)`: sets the period of the forcing term to *period*
* `set_duration(duration)`: sets the duration of the forcing term to *duration*
* `set_intensity(Imax)`: sets the intensity of the forcing term to *Imax*
* `set_name(name)`: sets the name of the forcing term to *name*
* `set_stimregion(streg)` sets to 1 the points/voxels where the forcing term is applied (takes a boolean mask *streg* as input)
* `deactivate()`: deactivates the stimulus
* `activate()`: activates the stimulus
* `is_active()`: tells if the stimulus is still active
* `stimulate_tissue_timestep(timestep: int, dt)`: tells if the stimulus is applied at a specific time step (input is an integer); it requires `dt` as input parameter
* `def stimulate_tissue_timevalue(time)`: tells if the stimulus is applied at a specific time value (input is a float)
* `stimApp(time)`: returns the stimulus if applied at current time; a *tf.tensor* of 0 otherwise
* The source term tensor is obtained by invoking the class name.

