import imageio
import numpy as np
import nibabel as nib


def parse_name(fname):
    """ Function to parse the file name to extraxt file type and compression """
    parsed_name = {'gzipped':False,
                   'type': 'unknown',
                   'name': '',
                   'fname': ''
                  }
    parsed_name['fname'] = fname
    lastname=fname.strip().split('/')[-1]    
    spl_fname=lastname.strip().split('.')
    
    if len(fname)==0:
        return(parsed_name)
    elif len(spl_fname)==1:
        parsed_name['name']=spl_fname[0]
    else:
        if(spl_fname[-1]=='gz'):
            parsed_name['gzipped']=True
            spl_fname=spl_fname[:-1]
        else:  
            parsed_name['gzipped']=False
        if len(spl_fname)==2:
            parsed_name['type']=spl_fname[-1]
            parsed_name['name']=spl_fname[-2]
        else:
            parsed_name['name']=spl_fname[-1]
    return(parsed_name)


def load_png_image(imgfile,mx,my): 
    """
    Function to load a domain represented in a png image with slices
    organised into a Mx X My grid
    """   
    im=imageio.imread(imgfile)
    H,L=im.shape
    #im: height, width
    h  = int(H/my)
    l  = int(L/mx)
    d  = int((H*L)/(h*l))
    img3d = np.zeros(shape=(h,l,d))
    for Y in range(H):
      for X in range(L):
          cx = np.floor(X/l) #0..mx-1
          cy = np.floor(Y/h) #0..my-1
          x = int(X - l*cx )    #[0,l-1]
          y = int(Y - h*cy )    #[0,h-1]
          z = int(cy*mx+cx)   # note: slice numeration differs from Abubu.js
          img3d[y,x,z] = im[Y,X]
    return img3d      





class ImageData:
    """ 
    class ImageData: utility to reads in image and generate the 3D domain
    This class reads in NIfTI .nii(.gz) or transform a .png image in a 
    nibabel (NIfTI) image. 
    3D tensor.    
    """
    
    def __init__(self):
        self.Mx      = 1
        self.My      = 1
        self._img     = None
        self._imgfile = None


    def load_image(self,fname,mx=1,my=1):
        """ 
        load_image(fname,mx,my)
        loads the image file fname and returns a nibabel image object
        If the image is of type png, one has to specify 
        the two grid dimensions mx and my        
        """
        print('load file {0}'.format(fname),flush=True)
        self._imgfile = parse_name(fname)
        
        if(self._imgfile['type']=='png'):
            self.Mx      = mx
            self.My      = my
            img = nib.Nifti1Image(load_png_image(fname,mx,my),np.eye(4) )
        elif(self._imgfile['type']=='nii'):
            img = nib.load(fname)
        elif(self._imgfile['type']=='npy'):
           img = nib.Nifti1Image(np.load(fname),np.eye(4) )
        else:
            print('file type {0} not wnown'.format(self._imgfile['type']))
        self._img = img


    def save_nifty(self,fout):
        """
        method save_nifty(fout) saves the NIfTI image to a file fout
        """
        if self._img:
            print('saving NIfTI image in {0}'.format(fout),flush=True)
            nib.save(self._img, fout)


    def get_data(self):
        """
        returns the image data as a sumpy tensor
        """
        if self._img:
            return(self._img.get_fdata())
        else:
            return(None)


    def get_rescaled_data(self,scaling_type='unit'):
        """
         get_rescaled_data(scaling_type) returns the image data as a sumpy tensor
         rescaled following one of the following criteria:
         * 'unit' (default): rescaled between 0 and 1 (x-x.min())/(x.max()-c.min())
         * 'mstd':           offset to mean; scaled to standard deviation (x-x.mean())/x.std()
         
        """
        if self._img:
            if scaling_type == 'unit':
                mval  = np.nanmin(self._img.get_fdata())
                Delta = np.nanmax(self._img.get_fdata()) - mval
            
            elif scaling_type == 'mstd':
                mval = np.nanmean(self._img.get_fdata())
                Delta = np.nanstd(self._img.get_fdata())

            else:
                print('unknown scaling method',flush=True)
                return(None)
        

            return( (self._img.get_fdata()-mval)/Delta )
        else:
            return(None)


    def image(self):
        """
        returns the NIfTI image (nibabel format)
        """
        return(self._img)


  


    







