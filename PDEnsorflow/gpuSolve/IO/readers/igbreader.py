import numpy as np
import sys



class IGBReader:
    """ 
    class IGBReader: reads an igb file
    """
    
    def __init__(self):
        self.__header: dict     = None
        self.__data: np.ndarray = None
        self.__filename: str    = None
        self.__ndiff: int       = 0
        self.__head_size: int   = 256

    def read(self,igbfname: str):
        """ read(igbfname):
        reads the IGB file igbfname and fills the header and data attributes.
        """
        self.__filename = igbfname
        try:
            parsed_header = self.__parse_header()

            #now read the data and create an array        
            with open(self.__filename,'rb') as f:
                y = np.fromfile(f,'f4')
            y        = y[self.__head_size:]
            nt       = parsed_header['t']
            nx       = parsed_header['x']            
            nentries = y.shape[0]
            ntot     = nt*nx
            self.__ndiff = nentries-ntot
            
            if(nentries>ntot):
                # More values than expected
                print('Warning: problem with the igb file',flush=True)
                print('Discarding the last {} elements'.format(self.__ndiff))
                y  = y[:ntot] 
            elif(nentries<ntot):  
                #less values than expected nentries<ntot
                nt = nentries//nx
                if(nt==0):
                    print('ERROR: y too short!')
                    print('({} elements; expected {} (problems with the igb file)'.format(nentries,ntot))
                    sys.exit()
                else:
                    ntot     = nt*nx
                    y        = y[:ntot]
                    missing = nentries%nx
                    print('Warning: problem with the ifgb file',flush=True)                    
                    print('Missing {} elements to reach {}'.format(missing,parsed_header['t'])) 
                    print('Reshaping to {} time steps'.format(nt))
            else:
                print('file is ok')
            parsed_header['t'] = nt
            y = np.reshape(y,(nt,nx))
            self.__header = parsed_header
            self.__data   = np.copy(y)
        except ValueError:
            print('error with {0}'.format(self.__filename) )


    def header(self) ->dict :
        """header(): returns the header dict"""
        return(self.__header)

    def data(self) ->np.ndarray :
        """data(): returns the numpy array that contains the data"""
        return(self.__data)

    def filename(self) ->str :
        """filename(): returns the file name that contained the data"""
        return(self.__filename)

    def ndiff(self) ->int :
        """ndiff(): returns the number of entries in excess/missing in the original file"""
        return(self.__ndiff)

    def org_t(self) ->float:
        """org_t(): returns the time origin"""
        return(self.__header['org_t'])
        
    def nt(self) -> int:
        """nt(): returns the total number of time steps"""
        return(self.__header['t'])
    
    def nx(self) -> int:
        """nx(): returns the x space dimension"""
        return(self.__header['x'])

    def ny(self) -> int:
        """ny(): returns the y space dimension"""
        return(self.__header['y'])

    def nz(self) -> int:
        """nz(): returns the z space dimension"""
        return(self.__header['z'])

    def dim_t(self) ->float:
        """dim_t(): returns the temporal dimension""" 
        return(self.__header['dim_t'])
    
    def dt(self) ->float:
        """dt(): returns the time step (nan if nt=1)"""
        if(self.nt()>1):
            return(self.dim_t()/(self.nt()-1))
        else:
            return(np.nan)

    def timevalues(self,shifted:bool = False) ->np.ndarray:
        """timevalues(shifted=False): returns a numpy array with the time axis
        if shifted = True, it shifts the time values to have 0 as the first one
        """
        tline = self.dt()*np.arange(self.nt())
        if shifted:
            return(tline)
        else:
            return(self.org_t()+tline)


    def __parse_header(self) ->dict :
        ''' parse_header():
        parses the header of the IGB file.
        '''
        parsed_header = {}
        int_keys = ['x','y','z','t']
        try:
            with open(self.__filename,'rb') as f:
                header = f.read(self.__head_size)
            header = header.decode("utf-8")
            for jj in header.strip().split():
                [key,val]=jj.split(':')
                if(val.isdigit()):
                    if key in int_keys:
                        parsed_header[key]=int(val)
                    else:
                        parsed_header[key]=float(val)
                else:
                    parsed_header[key]=val            
            # Now add some keys that might miss
            if not 'y' in parsed_header.keys():
                parsed_header['y'] = int(1)            
            if not 'z' in parsed_header.keys():
                parsed_header['z'] = int(1)
            if not 'org_t' in parsed_header.keys():
                parsed_header['org_t'] = float(0)
            if not 'dim_t' in parsed_header.keys():
                parsed_header['dim_t'] = float(parsed_header['t']-1 )
            if not 'unites_x' in parsed_header.keys():
                parsed_header['unites_x'] = 'unk'
            if not 'unites_y' in parsed_header.keys():
                parsed_header['unites_y'] = 'unk'
            if not 'unites_z' in parsed_header.keys():
                parsed_header['unites_z'] = 'unk'
            if not 'unites_t' in parsed_header.keys():
                parsed_header['unites_t'] = 'unk'
            if not 'unites' in parsed_header.keys():
                parsed_header['unites'] = 'unk'
            if not 'facteur' in parsed_header.keys():
                parsed_header['facteur'] = 1
            if not 'zero' in parsed_header.keys():
                parsed_header['zero'] = 0
            return(parsed_header)
                                  
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

