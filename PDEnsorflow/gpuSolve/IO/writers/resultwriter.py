import numpy as np
import tensorflow as tf
from gpuSolve.IO.writers.basewriter import BaseWriter

class ResultWriter(BaseWriter):
    """
    Class ResultWriter
    Implements a writer for GPU simulations
    When the domain is not a 3D slab, it is possible to write the result in a compact form
    Secifying the domain
    The GPU buffering/chunking is inherited from BaseWriter; this class only
    implements the NumPy specific saving logic. The legacy pre-allocated cube
    (enabled by initialise_cube()) is kept for backward compatibility.
    """
    def __init__(self, config : dict =None):
        # BaseWriter manages the GPU buffer and the every_N / max_chunk_mb flush triggers
        super().__init__(config)
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
        self.__use_cube: bool      = False    # legacy pre-allocated cube mode

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
        self.__use_cube    = True


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
        if self.__use_cube:
            # legacy pre-allocated cube path (kept for backward compatibility)
            if not self.__cube_exists:
                self.initialise_cube()
            self.__cube[self.__counter,:,:,:] = VolData
            self.__counter = self.__counter + 1
        else:
            # keep the solution on the GPU and flush it in chunks
            self.add_solution(VolData)


    def wait(self):
        if self.not_saved:
            self.save()
        self.not_saved = False
        for x in [0,1,2]:
            pass


    def save(self):
        if self.__use_cube:
            # legacy path: the whole solution already lives in the pre-allocated cube
            self.__save_array(self.__cube)
        else:
            # flush the GPU buffer and aggregate the disk chunks
            self.finalize()


    def __save_array(self,cube: np.ndarray):
        """__save_array(cube): writes a (already host-resident) cube to disk,
        either dense or in the compact sparse form."""
        if self._sparse:
            dimensions = cube.shape
            indices  = np.where(self._domain)
            values   = cube[:,self._domain]
            cube_sparse = {'dimensions':dimensions,'indices':indices,'values':values}
            fname = '{0}_sparse_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth)
            print('saving file {0}'.format(fname))
            np.save(fname, cube_sparse)
        else:
            fname = '{0}_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth)
            print('saving file {0}'.format(fname))
            np.save(fname, cube)


    def _tmp_prefix(self) -> str:
        """_tmp_prefix(): prefix used to name the temporary chunk files."""
        return('{0}_{1}_{2}_{3}'.format(self.prefix_name,self.height,self.width,self.depth))


    def _final_path(self) -> str:
        """_final_path(): full path (with .npy extension) of the final dense file."""
        return('{0}_{1}_{2}_{3}.npy'.format(self.prefix_name,self.height,self.width,self.depth))


    def _aggregate(self,chunk_files: list):
        """_aggregate(chunk_files): assembles the chunks into the final NumPy file.
        Dense output is streamed chunk-by-chunk (see BaseWriter); the sparse form
        needs the full cube to extract the active nodes."""
        if self._sparse:
            self.__save_array(self.__load_all(chunk_files))
        else:
            super()._aggregate(chunk_files)


    def __load_all(self,chunk_files: list) -> np.ndarray:
        """__load_all(chunk_files): reloads and concatenates all the disk chunks."""
        if len(chunk_files) == 0:
            return(np.empty((0,), dtype=np.float32))
        if len(chunk_files) == 1:
            return(np.load(chunk_files[0]))
        return(np.concatenate([np.load(f) for f in chunk_files], axis=0))


    def __del__(self):
        if self.not_saved and self.__save_on_exit:
            self.save()




