import numpy as np
import struct
import os


class IGBWriter:
    """
    Class IGBWriter
    Implements a writer in IGB format
    """


    def __init__(self, config=None):
        self._fname  = 'out.igb'
        self._Tstart = 0.0
        self._nt     = 0.0
        self._nx     = 0.0
        self._dim_t  = 1000
        self.__fobj  = None
        self.__hwrt  = True
        '''
        self.not_saved      = True
        self._save_on_exit  = True
        '''

        if(config is not None):
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def set_fname(self,fname):
        """ set_fname(fname) sets the output file name
        """
        self._fname = fname
    
    def set_nt(self,nt):
        """ set_nt(nt) assigns the nb of time steps
        """
        self._nt = nt

    def set_nx(self,nx):
        """ set_nx(nx) assigns the space dimension of the problem
        """
        self._nx = nx

    def set_Tstart(self,t0):
        """ set_Tstart(t0): sets the initial time (a.k.a. org_t) to t0
        """
        self._Tstart = t0
    
    def set_dim_t(self,dim_t):
        """ set_dim_t(dim_t) sets the temporal dimension (converts from time step to actual time)
        """
        self._dim_t = dim_t

    def nx(self):
        """ nx(): returns the space dimension
        """
        return(self._nx)                  

    def nt(self):
        """ nt(): returns the number of time steps
        """
        return(self._nt)                  

    def Tstart(self):
        """ Tstart(): returns the time origin """
        return (self._Tstart)

    def dim_t(self):
        """ dim_t(): returns the factor to convert from time step to time units
        """
        return(self._dim_t)

    def imshow(self,data):
        """ imshow(data): writes the data to the file
        """
        if self.__hwrt:
            self.__write_header()  
        for ix in range(self._nx):
            self.__fobj.write(struct.pack('f', data[ix]))

    def wait(self):
        '''wait function; usually for plot on screen '''
        self.__fobj.close()
        for x in [0,1,2]:
            pass

    def __del__(self):
        '''destructor: closes the file '''
        self.__fobj.close()

    def __write_header(self):
        '''writes the header to the file.'''
        self.___openfile()
        header ='x:{} y:1 z:1 t:{} type:float systeme:little_endian org_t:{}'.format(self._nx, self._nt, self._Tstart)
        for isp in range(1022-len(header)):
            header = header+' '
        header = header+'\n\f'
        header = ''.join('\n' if i % 80 == 0 else char for i, char in enumerate(header, 1))
        self.__fobj.write(header.encode('utf8'))
        self.__hwrt = False

    def ___openfile(self):
        '''openfile(): checks that the output directory exists and
        that the file name has the extension .igb; opens a file object'''
        try:
            if self.__fobj is None:
                fdir  = os.path.split(self._fname)[0]
                fname = os.path.split(self._fname)[-1]
                tmp   = fname.rfind('.')
                if tmp ==-1:
                    self._fname = '{}.igb'.format(self._fname)
                if fdir and (not os.path.exists(fdir)):
                    os.makedirs(fdir)
                self.__fobj = open(self._fname, 'wb')            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    
    
    

