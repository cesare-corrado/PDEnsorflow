import numpy as np

class ResultWriter:
    """
    Class ResultWriter
    Implements a writer for GPU simulations
    When the domain is not a 3D slab, it is possible to write the result in a compact form
    Secifying the domain
    """
    def __init__(self, config={}):
        self.width          = 1
        self.height         = 1
        self.depth          = 1
        self.samples        = 1
        self.dt_per_plot    = 1
        self.not_saved      = True
        self.prefix_name    = 'cube3D'
        self._sparse        = False
        self._save_on_exit  = True
        self.initval        = 0.0
        self.cube           = None
        self.cube_exists    = False
        

        if(len(config)>0):
            for attribute in self.__dict__.keys():
                if attribute in config.keys():
                    setattr(self, attribute, config[attribute])
        self._sparse  = False



    def initialise_cube(self):
        """Initialise the  cube to store results"""
        n                = int(self.samples//self.dt_per_plot)
        self.cube        = np.full(shape=(n, self.width, self.height,  self.depth),fill_value=self.initval ,dtype=np.float32)
        self.counter     = 0
        self.cube_exists = True
        
    def disable_save_on_exit(self):
        """ disable saving theresults when the destructor is invoked"""
        self._save_on_exit  = False
    
    def enable_save_on_exit(self):
        """ enable saving theresults when the destructor is invoked"""
        self._save_on_exit  = True
     
    def set_sparse_domain(self,domain):
        self._domain   = domain.astype(bool) 
        self._sparse   = True

    def imshow(self,VolData):
        if not self.cube_exists:
            self.initialise_cube()
        
        self.cube[self.counter,:,:,:] = VolData
        self.counter = self.counter + 1

    def wait(self):
        if self.not_saved:
            self.save()
        self.not_saved = False
        for x in [0,1,2]:
            pass

    def save(self):
        if self._sparse:
            dimensions = self.cube.shape
            indices  = np.where(self._domain)
            values   = self.cube[:,self._domain]
            cube_sparse = {'dimensions':dimensions,'indices':indices,'values':values}
            fname = '{0}_sparse_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth)
            print('saving file {0}'.format(fname))
            np.save(fname, cube_sparse)
        else:          
            fname = '{0}_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth)
            print('saving file {0}'.format(fname))
            np.save(fname, self.cube)

    def __del__(self):
        if self.not_saved and self._save_on_exit:
            self.save()




