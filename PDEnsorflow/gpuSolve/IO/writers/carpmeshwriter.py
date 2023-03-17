import os


def elemCode(elemName : str) -> str:
    '''
    elemcode(elemName) converts the the PDEnsorflow elemName
    into carp file name elem name.
    '''
    code_to_elem={ 'Edges': 'Cx',
                   'Trias': 'Tr',
                   'Quads': 'Qd',
                   'Tetras': 'Tt',
                   'Hexas': 'Hx',
                   'Pyras': 'Py',
                   'Prisms': 'Pr'
                   }
    return(code_to_elem[elemName])


class CarpMeshWriter:
    """ 
    class CarpMeshWriter: utility to write a mesh file in carp format
    It requires a mesh object as input and writes the carp files
    (.pts, .elem and .lon) as the output ( needs a prefix, with the path)
    """
    def __init__(self):
        self.__Mesh   = None
        self.__nElems : int = 0

    def assignMesh(self, msh):
        """ 
        assignMesh(msh) assigns the mesh msh to the writer
        """
        self.__Mesh   = msh
    
    def Mesh(self):
        """
        Mesh(): returns the mesh
        """
        return(self._Mesh)

    def writeMesh(self,fprefix : str):
        """
        writeMesh(fprefix): writes the mesh in Carp format, using fprefix as the prtefix.
        The prefix must contain the path.
        """
        try:
            fdir  = os.path.split(fprefix)[0]
            if fdir and (not os.path.exists(fdir)):
                os.makedirs(fdir)            
            self.__write_carp_nodes(fprefix)
            self.__write_carp_elements(fprefix)
            self.__write_carp_fibers(fprefix)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    
    def __count_mesh_elements(self):
        '''count_mesh_elements() counts the total nb of elements on the mesh'''
        try:
            self.__nElems = 0
            Elems = self.__Mesh.Elems()
            for elemtype,Elements in Elems.items():
                self.__nElems += Elements.shape[0]
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __write_carp_nodes(self,prefix : str):
        ''' writes the nodes in carp format (.pts file)'''
        Pts    = self.__Mesh.Pts()        
        print('writing points',flush=True)     
        with open('{0}.pts'.format(prefix), 'w') as fp:
            fp.write('{0}\n'.format(Pts.shape[0]) )
            for pt in Pts:
                fp.write('{0} {1} {2}\n'.format(pt[0],pt[1],pt[2]) )
    
    def __write_carp_elements(self,prefix :str):
        ''' writes the elements in carp format (.elem file)'''    
        Elems  = self.__Mesh.Elems()
        print('writing elements',flush=True)
        self.__count_mesh_elements()
        with open('{}.elem'.format(prefix), 'w') as fe:
                fe.write('{}\n'.format(self.__nElems) )
                for elemtype,Elements in Elems.items():
                    for Elem in Elements:
                        row = str(elemCode(elemtype))
                        for ivertex in Elem:
                            row=row+' {}'.format(ivertex)
                        row=row+'\n'
                        fe.write(row)

    def __write_carp_fibers(self,prefix : str):
        ''' writes the fibers in carp format (.lon file)'''    
        Fibres = self.__Mesh.Fibres()
        if Fibres.shape[1]==3:
            fformat = 1
        else:
            fformat = 2
        print('writing fibers',flush=True)        
        with open('{0}.lon'.format(prefix), 'w') as fe:
            fe.write('{0:d}\n'.format(fformat))
            for fib in Fibres:
                row=''
                for entry in fib:
                    row=row+'{} '.format(entry)
                row = row[:-1]
                row = row+'\n'
                fe.write(row)
        
        
        

