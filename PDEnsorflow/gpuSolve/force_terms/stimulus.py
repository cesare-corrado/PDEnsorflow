import numpy as np

import tensorflow as tf



class Stimulus:
    """
    Class stimulus handles external pacing/periodic pacing
    The stimulated region is defined by the user with the function
    set_stimregion that takes a numpy bool array as input
    
    Temporal parameters:
    tstart:    time the stimulus starts (float, default: 0.0)
    nstim:     number of stimuli to apply (int, default: 1)
    period:    period for periodic stimuli (float, default is 1.0)
    duration:  duration of the stimulus (float, default 1.0)
    
    """
    def __init__(self,props=None):
        self.width     = 1
        self.height    = 1
        self.depth     = 1
        self.dx        = 1.0
        self.dy        = 1.0
        self.dz        = 1.0
        self.dt        = 1.0
        self.intensity = 1.0
        self.name      = "s2"
        self.tstart    = 0.0
        self.nstim     = 1
        self.period    = 1.0
        self.duration  = 1.0        
        self.stim      = None        
        self._active    = True

        if props:
            for attribute, val in props.items():
                if(hasattr(self,attribute) ):
                  setattr(self, attribute, val)

    
    def __call__(self):
        return(self.stim)
    
    
    def set_stimregion(self,streg):
        self.stim = tf.constant(self.intensity*streg.astype(bool),name=self.name, dtype=np.float32)
        [self.width,self.height,self.depth] = streg.shape
    
    
    
    def stimulate_tissue_timestep(self,timestep: int, dt):
        """
        stimulate_tissue_timestep(timestep) tells if stimulating the tissue or not
        Input is a time step (of type int)
        """
        
        if not hasattr(self,'_int_duration'):
            setattr(self, '_int_duration', int(self.duration/dt) )

        if not hasattr(self,'_int_period'):
            setattr(self, '_int_period', int(self.period/dt) )
            
        if not hasattr(self,'_int_tstart'):
            setattr(self, '_int_tstart', int(self.tstart/dt) )

        if not hasattr(self,'_int_tend'):
            setattr(self, '_int_tend', int((self.tstart+self.period*(self.nstim-1)+self.duration)/dt) )


        #If the stimulus is _active
        if self._active:
            # ofset the time
            current_int_time = timestep-self._int_tstart            
            if(current_int_time>=0):
                if ((current_int_time-self._int_tend) >0) :
                    self._active = False
                else:
                    if (current_int_time%self._int_period< self._int_duration ):
                        return True
        return False
    
    
    def stimulate_tissue_timevalue(self,time):    
        '''
        stimulate_tissue_timevalue(time) tells if stimulating the tissue or not
        Input is a time value (of type float)
        '''
        if not hasattr(self,'_tend'):
            setattr(self, '_tend', self.tstart+self.period*(self.nstim-1)+self.duration )

        #If the stimulus is _active
        if self._active:
            # ofset the time
            current_time = time-self.tstart
            if(current_time>=0):
                if ((current_time-self._tend) >0.0) :
                    self._active = False
                else:
                    if (current_time%self.period<= self.duration ):
                        return True
        return False
    
    
    
    
