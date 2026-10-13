import math

import numpy as np

import tensorflow as tf



class Stimulus:
    """
    Class stimulus handles external pacing/periodic pacing
    The stimulated region is defined by the user with the function
    set_stimregion that takes a numpy bool array as input
    
    Parameters:
    tstart:    time the forcing term starts (float, default: 0.0)
    nstim:     number of forcing term stimuli (int, default: 1)
    period:    period for periodic forcing term (float, default is 1.0)
    duration:  duration of the forcing term (float, default 1.0)
    intensity: intensity of the forcing term (float, default 1.0)
    shape:     the pulse shape, "square" or "truncexp" (str, default: "square")
    tau_edge:  truncexp: time constant of the leading/trailing edges, in ms (float, default 0.01)
    tau_plateau: truncexp: time constant of the plateau decay, in ms (float, default 1000.0)
    d1:        truncexp: duration of the first sub-pulse, relative to duration (float, default 1.0)
    bias:      truncexp: constant added to the waveform (float, default 0.0)
    name:      the name of the forcing term (default: "s2")

    The "truncexp" shape is the truncated exponential (capacitive discharge)
    pulse of openCARP, which shapes every pulse of its legacy stimulus[]
    family that way and defaults to it in the stim[].pulse.shape of the
    current one: a monophasic pulse that rises with tau_edge, decays slowly
    with tau_plateau over its plateau, and falls back with tau_edge over the
    last 5*tau_edge of its duration (physics/stimulate.cc, sample_wave_form;
    numerics/signals.h, monophasicTruncExpFunc). It therefore delivers
    slightly less charge than a square pulse of the same strength and
    duration: about 2.6% less for a 2 ms pulse at the default tau_edge,
    and more for a shorter one, since the edges cost a fixed 5*tau_edge.
    """
    def __init__(self,props=None):
        self._tstart : float    = 0.0
        self._nstim  : int      = 1
        self._period : float    = 1.0
        self._duration : float  = 1.0        
        self._intensity : float = 1.0
        # The pulse shape over the duration of one stimulus. "square" is a
        # constant of unit height and stays the default, so that every caller
        # written before the shape existed keeps its numerics; the four
        # parameters below are inert while it is selected. They are openCARP's
        # own defaults, so selecting "truncexp" alone reproduces its pulse.
        self._shape       : str   = "square"
        self._tau_edge    : float = 0.01
        self._tau_plateau : float = 1000.0
        self._d1          : float = 1.0
        self._bias        : float = 0.0
        self._name      : str   = "s2"
        self.__active: bool     = True
        self.__stim : tf.constant = None

        if props:
            for attribute in self.__dict__.keys():
                if attribute[1:] in props.keys():
                    setattr(self, attribute, props[attribute[1:]])
        self.__tend = self._tstart+self._period*(self._nstim-1)+self._duration
        self.__check_shape()


    def set_tstart(self,tstart: float):
        """set_tstart(tstart): sets the staring time of the forcing term to tstart
        """
        self._tstart = tstart
        self.__tend = self._tstart+self._period*(self._nstim-1)+self._duration        

    def set_nstim(self,nstim: int):
        """set_nstim(nstim): sets the total nb of stimuli to nstim
        """
        self._nstim = nstim
        self.__tend = self._tstart+self._period*(self._nstim-1)+self._duration        

    def set_period(self,period: float):
        """set_period(period): sets the period of the forcing term to period
        """
        self._period = period
        self.__tend = self._tstart+self._period*(self._nstim-1)+self._duration        

    def set_duration(self,duration: float):
        """set_duration(duration): sets the duration of the forcing term to duration
        """
        self._duration = duration
        self.__tend = self._tstart+self._period*(self._nstim-1)+self._duration        

    def set_intensity(self,Imax: float):
        """ set_intensity(Imax): sets the intensity of the forcing term to Imax
        """
        self._intensity = Imax

    def set_name(self,name : str):
        """ set_name(name): sets the name of the forcing term to name
        """
        self._name = name

    def set_shape(self,shape : str):
        """ set_shape(shape): sets the pulse shape to shape, "square" or "truncexp"
        """
        self._shape = shape
        self.__check_shape()

    def set_tau_edge(self,tau_edge : float):
        """ set_tau_edge(tau_edge): sets the edge time constant of a truncexp pulse to tau_edge (ms)
        """
        self._tau_edge = tau_edge

    def set_tau_plateau(self,tau_plateau : float):
        """ set_tau_plateau(tau_plateau): sets the plateau time constant of a truncexp pulse to tau_plateau (ms)
        """
        self._tau_plateau = tau_plateau

    def set_d1(self,d1 : float):
        """ set_d1(d1): sets the relative duration of the first sub-pulse of a truncexp pulse to d1
        """
        self._d1 = d1
        self.__check_shape()

    def set_bias(self,bias : float):
        """ set_bias(bias): sets the constant added to a truncexp waveform to bias
        """
        self._bias = bias

    def shape(self) -> str:
        """ shape(): returns the pulse shape
        """
        return(self._shape)

    def tau_edge(self) -> float:
        """ tau_edge(): returns the edge time constant of a truncexp pulse, in ms
        """
        return(self._tau_edge)

    def tau_plateau(self) -> float:
        """ tau_plateau(): returns the plateau time constant of a truncexp pulse, in ms
        """
        return(self._tau_plateau)

    def d1(self) -> float:
        """ d1(): returns the relative duration of the first sub-pulse of a truncexp pulse
        """
        return(self._d1)

    def bias(self) -> float:
        """ bias(): returns the constant added to a truncexp waveform
        """
        return(self._bias)

    def waveform(self,ctime_sim: float) -> float:
        """ waveform(ctime_sim) returns the height of the pulse at time ctime_sim,
            as a fraction of the intensity. It is 1.0 everywhere for a square
            pulse; for a truncexp pulse it is the normalised monophasic
            truncated exponential, evaluated on the time elapsed since the
            start of the pulse the time falls in.
            The caller is expected to have established that the stimulus is on:
            outside the pulse the formula is continued, not clipped.
        """
        if self._shape == "square":
            return(1.0)
        # Time since the start of the current pulse. A train is folded onto its
        # period, so that every pulse of it carries the identical waveform.
        # A single pulse is NOT folded: it has no recurrence, and a caller that
        # has no period to give may pass one shorter than the pulse itself (the
        # .par front end stands in the rest of the simulation for a missing
        # bcl), which would otherwise restart the waveform inside the pulse.
        elapsed = float(ctime_sim) - self._tstart
        if self._nstim > 1 and self._period > 0.0:
            elapsed = math.fmod(elapsed, self._period)
        # Start of the trailing edge. The reference leaves the last five edge
        # time constants of the pulse to the decay, which is where a
        # capacitive discharge has fallen to e^-5 of its height. Clamped at
        # zero because a pulse shorter than that has no plateau at all.
        t_edge = max(self._duration - 5.0 * self._tau_edge, 0.0)
        if elapsed < t_edge:
            return(self.__trunc_exp_plateau(elapsed) + self._bias)
        # The decay restarts from the plateau height reached at t_edge, so the
        # waveform is continuous there.
        decay = math.exp(-(elapsed - t_edge) / self._tau_edge) if self._tau_edge > 0.0 else 0.0
        return(self.__trunc_exp_plateau(t_edge) * decay + self._bias)

    def __trunc_exp_plateau(self,elapsed: float) -> float:
        """ __trunc_exp_plateau(elapsed) returns the rising-plus-plateau part of a
            truncexp waveform: the exponential rise with tau_edge times the slow
            exponential decay of the plateau with tau_plateau. expm1 is used for
            the rise because it keeps the first few samples accurate, where
            elapsed/tau_edge is small and 1-exp(-x) cancels.
        """
        rise = -math.expm1(-elapsed / self._tau_edge) if self._tau_edge > 0.0 else 1.0
        return(rise * math.exp(-elapsed / self._tau_plateau))

    def __check_shape(self):
        """ __check_shape() rejects a shape this class does not implement, at the
            point the shape is set rather than at the first time step.
        """
        if self._shape not in ("square", "truncexp"):
            raise ValueError('Stimulus shape "{}" is not implemented: use "square" or '
                             '"truncexp"'.format(self._shape))
        if self._shape == "truncexp" and self._d1 != 1.0:
            # d1 < 1 selects the reference's biphasic pulse, whose second phase
            # reverses the current. That is a defibrillation waveform, applied
            # through an extracellular electrode, and this class carries a
            # transmembrane forcing term only: approximating it silently would
            # be worse than refusing it.
            raise ValueError('Stimulus d1 = {} asks for a biphasic truncated exponential, '
                             'which is not implemented: only the monophasic pulse '
                             '(d1 = 1) is'.format(self._d1))

    def set_stimregion(self,streg : np.ndarray):
        """
        set_stimregion(streg): sets to 1 the points/voxels where the forcing term is applied (takes a boolean mask streg as input)
        """
        region = np.squeeze(streg.astype(bool))
        if region.ndim==1:
            region=region[:,np.newaxis]
        self.__stim = tf.constant(region,name=self._name, dtype=np.float32 )

    def get_stimregion(self) -> tf.constant:
        """ get_stimeregion() returns the stimulus region
        """
        return(self.__stim)

    def apply_indices_permutation(self, perm_array: np.ndarray):
        """apply_indices_permutation(perm_array) applies the permutation specified 
        in perm_array to the stimulated region.
        Useful when a permutation is used to reduce the breadthwidth of the matrices.
        """
        if self.__stim is not None:
            self.__stim = tf.gather(self.__stim, perm_array,name=self._name)
        
    def deactivate(self):
        """deactivate(): sets the flag _active" to False"""
        self.__active = False   
            
    def activate(self):
        """activate(): sets the flag _active" to True """
        self.__active = True

    def is_active(self) -> bool:
        """is_active(): returns true if the stimulus is still active.
        This is a flag that is sset externally to speed-up aplllications
        to skip stimulus that are no longer active
        """
        return(self.__active)

    def stimulate_tissue_timestep(self,timestep: int, dt) -> bool:
        """
        stimulate_tissue_timestep(timestep) tells if stimulating the tissue or not
        Input is a time step (of type int)
        """
        if not hasattr(self,'_int_duration'):
            setattr(self, '_int_duration', int(self._duration/dt) )

        if not hasattr(self,'_int_period'):
            setattr(self, '_int_period', int(self._period/dt) )
            
        if not hasattr(self,'_int_tstart'):
            setattr(self, '_int_tstart', int(self._tstart/dt) )

        if not hasattr(self,'_int_tend'):
            setattr(self, '_int_tend', int((self._tstart+self._period*(self._nstim-1)+self._duration)/dt) )

        #If the stimulus is _active
        if self.__active:
            # ofset the time
            current_int_time = timestep-self._int_tstart            
            if(current_int_time>=0):
                if ((current_int_time-self._int_tend) >0) :
                    self.__active = False
                else:
                    if (current_int_time%self._int_period< self._int_duration ):
                        return True
        return False

    def stimulate_tissue_timevalue(self,ctime_sim: float) -> bool:    
        """
        stimulate_tissue_timevalue(ctime_sim) tells if stimulating the tissue or not
        Input is a ctime_sim value (of type float)
        """
        #If the stimulus is _active
        if self.__active:
            # ofset the time
            current_time = ctime_sim-self._tstart
            if(current_time>=0):
                # Passed the last stimulus
                if ((ctime_sim-self.__tend) >0.0) :
                    self.__active = False
                else:
                    if (current_time%self._period<= self._duration ):
                        return True
        return False
    
    def stimApp(self,ctime_sim: float) -> tf.constant:
        """stimApp(ctime_sim): returns the stimulus if applied at current time; None otherwise"""
        if self.stimulate_tissue_timevalue(ctime_sim):
            return(tf.math.scalar_mul(self._intensity*self.waveform(ctime_sim), self.__stim))
        else:
            return(0.0*self.__stim)

    def __call__(self, ctime_sim: float = None) -> tf.constant:
        """ __call__([ctime_sim]): returns the forcing term, without testing whether
            the stimulus is on. Given a time, the pulse is scaled by its waveform
            at that time; without one it is returned at full intensity, which is
            the whole pulse for a square shape only.
        """
        if ctime_sim is None:
            return(self._intensity*self.__stim)
        return(tf.math.scalar_mul(self._intensity*self.waveform(ctime_sim), self.__stim))
    
