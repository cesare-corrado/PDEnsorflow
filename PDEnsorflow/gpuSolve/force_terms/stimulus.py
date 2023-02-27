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
    name:      the name of the forcing term (default: "s2")
    
    """
    def __init__(self,props=None):
        self._tstart : float    = 0.0
        self._nstim  : int      = 1
        self._period : float    = 1.0
        self._duration : float  = 1.0        
        self._intensity : float = 1.0
        self._name      : str   = "s2"
        self.__active: bool     = True
        self.__stim : tf.constant = None

        if props:
            for attribute in self.__dict__.keys():
                if attribute[1:] in props.keys():
                    setattr(self, attribute, props[attribute[1:]])
        self.__tend = self._tstart+self._period*(self._nstim-1)+self._duration


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
            return(self._intensity*self.__stim)
        else:
            return(0.0*self.__stim)
        
    def __call__(self):
        return(self._intensity*self.__stim)
    
