import numpy as np
import sys
import time
from gpuSolve.IO.readers.imagedata import ImageData


class Domain3D:
    """
    Class Domain3D
    This class defines the 3D domain for numerical simulations.
    it collects domain shape and properties such as fibre directions and conductivities.
    """

    def __init__(self,props : dict=None):
        self._width : int                 = None
        self._height : int                = None
        self._depth : int                 = None
        self._dx : float                  = 1.0
        self._dy : float                  = 1.0
        self._dz : float                  = 1.0
        self.__geometry : np.ndarray      = None
        self.__conductivity : np.ndarray  = None
        self.__fibtensor : np.ndarray     = None
        self._geometryfile : str          = ''
        self._conductivityfile : str      = ''
        self._fibtensorfile : str         = ''
        self.__timecounter : float        = 0.0
        self.__anisotropic : bool         = False
        if props:
            for attribute in self.__dict__.keys():
                attr_name = attribute[1:]
                if attr_name in props.keys():
                    setattr(self, attribute, props[attr_name])



    def load_geometry_file(self,fname : str ='', Mx : int = 0, My: int = 0, threshold : float = 1.e-4):
        ''' 
        This function loads the domain geometry from a file
        If no file name is provided, this function 
        builds a 3D cubic domain with size (width,height,depth)
        Arguments:
            fname:     the file name containing the geometry definition
            Mx,My:     the grid dimensions within thefigure of the geometry (for .png files only)
            threshold: the threshold (wthin interval [0,1]) to separate the anatomy from the background 
        '''
        then = time.time()
        if self._geometryfile =='':
            self._geometryfile = fname
        if len(self._geometryfile):
            Image = ImageData()
            print('reading the geometry from the file {0}'.format(fname))
            Image.load_image(fname,Mx,My)
            img_vox = Image.get_rescaled_data('unit')
            [self._width,self._height,self._depth] = img_vox.shape 
            img_vox[img_vox>threshold]  = 1.0
            img_vox[img_vox<=threshold] = 0.0
            self.__geometry = img_vox
            print('Sizes: ({}X{}X{})'.format(self._width,self._height,self._depth) )
        else:
            print('Cubic geometry ({}X{}X{})'.format(self._width,self._height,self._depth) )
            self.__geometry = np.ones(shape=(self._width,self._height,self._depth))
        elapsed = (time.time() - then)
        self.__timecounter += elapsed
        print('geometry initialised; elapsed: %f sec' % elapsed)



    def assign_geometry(self, geometryTensor: np.ndarray):
        ''' 
        This function assigns a pre-defined tensor to the domain geometry.
        All the pre-processing operations are supposed to be done before calling.
        input arguments:
            geometryTensor: a 3D tensor (numpy or tensorflow) with geometry voxel values
        '''
        then = time.time()    
        self.__geometry = geometryTensor
        [self._width,self._height,self._depth] = geometryTensor.shape 
        elapsed = (time.time() - then)
        self.__timecounter += elapsed


    def load_conductivity(self, fname : str = '', Mx: int = 0, My: int = 0, cond_unif: float = 1.0):
        ''' 
        This function loads the domain conductivity from a file
        If no file name is provided, this function builds an uniform conductivity 
        tensor on the domain
        Arguments:
            fname:     the file name containing the conductivity values
            Mx,My:     the grid dimensions of the domain figure (for .png files only)
            cond_unif: the uniform value for the conductivity (when no file is specified)
        '''
        if self._conductivityfile =='':
            self._conductivityfile = fname
    
        if self.__geometry is not None:
            then = time.time()
            if len(self._conductivityfile):
                Image = ImageData()
                print('reading the conductivity from the file {0}'.format(fname))
                Image.load_image(fname,Mx,My)
                img_vox = Image.get_fdata()
                if(len(img_vox.shape)==4 and img_vox.shape[-1] >1 ):
                    print('anisotropic conductivity (need fibres)')
                    self.__anisotropic = True
                else:
                    print('isotropic conductivity')
                    self.__anisotropic = False
                    # remove one dim from the tensor
                    if len(img_vox.shape)==4:
                        img_vox = img_vox[:,:,:,0]
                if self.__anisotropic:
                    img_vox[:,:,:,1] = img_vox[:,:,:,1] - img_vox[:,:,:,0]
                    self.__conductivity = img_vox
                else:
                    self.__conductivity = img_vox
            else:            
                print('homogeneous conductivity')
                self.__conductivity = cond_unif*self.__geometry
            elapsed = (time.time() - then)
            print('initialisation of conductivity tensor, elapsed: %f sec' % elapsed)
            self.__timecounter += elapsed
        else:
            sys.exit("No geometry defined! (assign a geometry first)")
	

    def assign_conductivity(self,conductivityTensor: np.ndarray):
        ''' 
        This function assigns a pre-defined tensor to the domain conductivity.
        conductivityTensor: a 3D tensor (numpy or tensorflow) with conductivity voxel values
        '''
        then = time.time() 
        if self.__geometry is not None:   
            if(len(conductivityTensor.shape)==4 and conductivityTensor.shape[-1] >1 ):
                print('anisotropic conductivity (need fibres)')
                self.__anisotropic = True
            else:
                print('isotropic conductivity')
                self.__anisotropic = False
                # remove one dim from the tensor
                if len(conductivityTensor.shape)==4:
                    conductivityTensor = conductivityTensor[:,:,:,0]
            if self.__anisotropic:
                conductivityTensor[:,:,:,1] = conductivityTensor[:,:,:,1] - conductivityTensor[:,:,:,0]
                self.__conductivity = conductivityTensor
            else:
                self.__conductivity = conductivityTensor
            elapsed = (time.time() - then)
            self.__timecounter += elapsed
        else:
            sys.exit("No geometry defined! (assign a geometry first)")


    def load_fiber_direction(self, fname : str ='', Mx: int =0, My: int =0):
        ''' 
        This function loads the fiber directions from a file
        Arguments:
            fname:     the file name containing the conductivity values
            Mx,My:     the grid dimensions of the domain figure (for .png files only)
        Note: the direction tensor has hape (D, H, W,6 ); each channel represents 
            the following components of the diadic product: 
            A0A0 A0A1 A0A2  A1A1 A1A2 A2A2
        '''

        if self._fibtensorfile == '':
            self._fibtensorfile = fname
    
        if self.__geometry is not None:
            if len(self._fibtensorfile):
                then = time.time()
                Image = ImageData()
                print('reading the fibres from the file {0}'.format(fname))
                Image.load_image(fname,Mx,My)
                img_vox = Image.get_fdata()
                [W,H,D,_] = img_vox.shape
                fibtens=np.zeros(shape=(W,H,D,6),dtype=np.float32)
                fibtens[:,:,:,0] = img_vox[:,:,:,0] *img_vox[:,:,:,0]
                fibtens[:,:,:,1] = img_vox[:,:,:,0] *img_vox[:,:,:,1]
                fibtens[:,:,:,2] = img_vox[:,:,:,0] *img_vox[:,:,:,2]
                fibtens[:,:,:,3] = img_vox[:,:,:,1] *img_vox[:,:,:,1]
                fibtens[:,:,:,4] = img_vox[:,:,:,1] *img_vox[:,:,:,2]
                fibtens[:,:,:,5] = img_vox[:,:,:,2] *img_vox[:,:,:,2]
                self.__fibtensor  = fibtens
                elapsed = (time.time() - then)
                self.__timecounter += elapsed
                print('initialisation of fibers tensor, elapsed: %f sec' % elapsed)
            else:            
                sys.exit("Invalid fiber file!")
        else:
            sys.exit("No geometry defined! (assign a geometry first)")

	
    # Set functions
    def set_dx(self, dx: float):
        self._dx = dx

    def set_dy(self, dy: float):
        self._dy = dy

    def set_dz(self, dz: float):
        self._dz = dz

    # Get functions
    def dx(self) -> float:
        return(self._dx)
    
    def dy(self)-> float:
        return(self._dy)
    
    def dz(self)-> float:
        return(self._dz)

    def width(self) -> int :
        return(self._width)

    def height(self) -> int :
        return(self._height)

    def depth(self) -> int :
        return(self._depth)

    def anisotropic(self) -> bool:
        return(self.__anisotropic)

    def geometry(self) -> np.ndarray:
        return(self.__geometry)

    def conductivity(self) -> np.ndarray:
        return(self.__conductivity)
        
    def fibtensor (self) -> np.ndarray:
        return(self.__fibtensor)

    def walltime(self) -> float:
        return(self.__timecounter)

