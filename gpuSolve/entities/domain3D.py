import tensorflow as tf
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

    def __init__(self,props={}):
        self._width             = None
        self._height            = None
        self._depth             = None
        self._dx                = 1.0
        self._dy                = 1.0
        self._dz                = 1.0
        self._geometry          = None
        self._conductivity      = None
        self._fibtensor         = None
        self._geometryfile      = ''
        self._conductivityfile  = ''
        self._fibtensorfile     = ''
        self._timecounter       = 0.0
        self._anisotropic       = False
        if(len(props)>0):
            for attribute in self.__dict__.keys():
                attr_name = attribute[1:]
                if attr_name in props.keys():
                    setattr(self, attribute, props[attr_name])



    def load_geometry_file(self,fname='',Mx=0,My=0,threshold = 1.e-4):
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
            tf.print('reading the geometry from the file {0}'.format(fname))
            Image.load_image(fname,Mx,My)
            img_vox = Image.get_rescaled_data('unit').astype(np.float32)
            [self._width,self._height,self._depth] = img_vox.shape 
            img_vox[img_vox>threshold]  = 1.0
            img_vox[img_vox<=threshold] = 0.0
            self._geometry = tf.constant(img_vox, dtype=np.float32, name='domain' )
            tf.print('Sizes: ({}X{}X{})'.format(self._width,self._height,self._depth) )
        else:
            tf.print('Cubic geometry ({}X{}X{})'.format(self._width,self._height,self._depth) )
            self._geometry = tf.constant(1.0,shape=(self._width,self._height,self._depth), dtype=np.float32, name='domain')
        elapsed = (time.time() - then)
        self._timecounter += elapsed
        tf.print('geometry initialised; elapsed: %f sec' % elapsed)



    def assign_geometry(self,geometryTensor):
        ''' 
        This function assigns a pre-defined tensor to the domain geometry.
        All the pre-processing operations are supposed to be done before calling.
        input arguments:
            geometryTensor: a 3D tensor (numpy or tensorflow) with geometry voxel values
        '''
        then = time.time()    
        self._geometry = tf.constant(geometryTensor, dtype=np.float32, name='domain' )
        [self._width,self._height,self._depth] = geometryTensor.shape 
        elapsed = (time.time() - then)
        self._timecounter += elapsed


    def load_conductivity(self, fname='', Mx=0, My=0, cond_unif=1.0):
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
    
        if self._geometry is not None:
            then = time.time()
            if len(self._conductivityfile):
                Image = ImageData()
                tf.print('reading the conductivity from the file {0}'.format(fname))
                Image.load_image(fname,Mx,My)
                img_vox = Image.get_fdata()
                if(len(img_vox.shape)==4 and img_vox.shape[-1] >1 ):
                    tf.print('anisotropic conductivity (need fibres)')
                    self._anisotropic = True
                else:
                    tf.print('isotropic conductivity')
                    self._anisotropic = False
                    # remove one dim from the tensor
                    if len(img_vox.shape)==4:
                        img_vox = img_vox[:,:,:,0]
                if self._anisotropic:
                    img_vox[:,:,:,1] = img_vox[:,:,:,1] - img_vox[:,:,:,0]
                    self._conductivity = tf.constant(img_vox, dtype=np.float32, name='diffusion' )
                else:
                    self._conductivity = tf.constant(img_vox, dtype=np.float32, name='diffusion' )
            else:            
                tf.print('homogeneous conductivity')
                self._conductivity = tf.constant(cond_unif*self._geometry, dtype=np.float32, name='diffusion' )
            elapsed = (time.time() - then)
            tf.print('initialisation of conductivity tensor, elapsed: %f sec' % elapsed)
            self._timecounter += elapsed
        else:
            sys.exit("No geometry defined! (assign a geometry first)")
	

    def assign_conductivity(self,conductivityTensor):
        ''' 
        This function assigns a pre-defined tensor to the domain conductivity.
        conductivityTensor: a 3D tensor (numpy or tensorflow) with conductivity voxel values
        '''
        then = time.time() 
        if self._geometry is not None:   
            if(len(conductivityTensor.shape)==4 and conductivityTensor.shape[-1] >1 ):
                tf.print('anisotropic conductivity (need fibres)')
                self._anisotropic = True
            else:
                tf.print('isotropic conductivity')
                self._anisotropic = False
                # remove one dim from the tensor
                if len(conductivityTensor.shape)==4:
                    conductivityTensor = conductivityTensor[:,:,:,0]
            if self._anisotropic:
                conductivityTensor[:,:,:,1] = conductivityTensor[:,:,:,1] - conductivityTensor[:,:,:,0]
                self._conductivity = tf.constant(conductivityTensor, dtype=np.float32, name='diffusion' )
            else:
                self._conductivity = tf.constant(conductivityTensor, dtype=np.float32, name='diffusion' )
            elapsed = (time.time() - then)
            self._timecounter += elapsed
        else:
            sys.exit("No geometry defined! (assign a geometry first)")


    def load_fiber_direction(self, fname='', Mx=0, My=0):
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
    
        if self._geometry is not None:
            if len(self._fibtensorfile):
                then = time.time()
                Image = ImageData()
                tf.print('reading the fibres from the file {0}'.format(fname))
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
                self._fibtensor  = tf.constant(fibtens, dtype=np.float32, name='fibtensor' )
                elapsed = (time.time() - then)
                self._timecounter += elapsed
                tf.print('initialisation of fibers tensor, elapsed: %f sec' % elapsed)
            else:            
                sys.exit("Invalid fiber file!")
        else:
            sys.exit("No geometry defined! (assign a geometry first)")

	
    # Set functions
    def set_dx(self, dx):
        self._dx = dx

    def set_dy(self, dy):
        self._dy = dy

    def set_dz(self, dz):
        self._dz = dz

    # Get functions
    def DX(self):
        return(tf.constant(self._dx, dtype=np.float32))
    
    def DY(self):
        return(tf.constant(self._dy, dtype=np.float32))
    
    def DZ(self):
        return(tf.constant(self._dz, dtype=np.float32))

    def width(self):
        return(self._width)

    def height(self):
        return(self._height)

    def depth(self):
        return(self._depth)

    def anisotropic(self):
        return(self._anisotropic)

    def geometry(self):
        return(self._geometry)

    def conductivity(self):
        return(self._conductivity)
        
    def fibtensor (self):
        return(self._fibtensor)

    def walltime(self):
        return(self._timecounter)


