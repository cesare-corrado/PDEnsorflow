import numpy as np

class ResultWriter:
    """
    Class ResultWriter
    Implements a writer for GPU simulations
    When the domain is not a 3D slab, it is possible to write the result in a compact form
    Secifying the domain
    """
    def __init__(self, config : dict =None):
        self.width: int            = 1
        self.height: int           = 1
        self.depth: int            = 1
        self.samples: int          = 1
        self.dt_per_plot: int      = 1
        self.not_saved: bool       = True
        self.prefix_name: str      = 'cube3D'
        self._sparse: bool         = False
        self.__save_on_exit: bool  = True
        self.initval: float        = 0.0
        self.__cube: np.ndarray    = None
        self.__cube_exists: bool   = False
        self.__counter: int        = None

        if(config):
            for attribute in self.__dict__.keys():
                if attribute in config.keys():
                    setattr(self, attribute, config[attribute])
        self._sparse  = False


    def initialise_cube(self):
        """Initialise the  cube to store results"""
        n                  = 1+int(self.samples//self.dt_per_plot)
        self.__cube        = np.full(shape=(n, self.width, self.height,  self.depth),fill_value=self.initval ,dtype=np.float32)
        self.__counter     = 0
        self.__cube_exists = True


    def disable_save_on_exit(self):
        """ disable saving theresults when the destructor is invoked"""
        self.__save_on_exit  = False


    def enable_save_on_exit(self):
        """ enable saving theresults when the destructor is invoked"""
        self.__save_on_exit  = True


    def set_sparse_domain(self,domain: np.ndarray):
        self._domain   = domain.astype(bool) 
        self._sparse   = True


    def imshow(self,VolData: np.ndarray):
        if not self.__cube_exists:
            self.initialise_cube()        
        self.__cube[self.__counter,:,:,:] = VolData
        self.__counter = self.__counter + 1


    def wait(self):
        if self.not_saved:
            self.save()
        self.not_saved = False
        for x in [0,1,2]:
            pass


    def save(self):
        if self._sparse:
            dimensions = self.__cube.shape
            indices  = np.where(self._domain)
            values   = self.__cube[:,self._domain]
            cube_sparse = {'dimensions':dimensions,'indices':indices,'values':values}
            fname = '{0}_sparse_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth)
            print('saving file {0}'.format(fname))
            np.save(fname, cube_sparse)
        else:          
            fname = '{0}_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth)
            print('saving file {0}'.format(fname))
            np.save(fname, self.__cube)


    def __del__(self):
        if self.not_saved and self.__save_on_exit:
            self.save()




