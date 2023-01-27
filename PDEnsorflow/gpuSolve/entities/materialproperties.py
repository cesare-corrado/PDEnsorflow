import numpy as np

class MaterialProperties:
    """
    Class MaterialProperties
    This class collects all the material properties associated to nodes or elements
    and implements some proxy functions to access the values
    """

    def __init__(self):
        self._element_properties = None
        self._nodal_properties   = None
        self._ud_functions       = None
        
    def add_ud_function(self,fname,fdef):
        if self._ud_functions is None:
            self._ud_functions = {} 
        self._ud_functions[fname]=fdef

    def remove_ud_function(self,fname):
        """
        remove_ud_function(fname) if function fname exists, it removes 
        it from the the _ud_functions dict 
        """
        self._ud_functions.pop(fname,None)
        if not self._ud_functions:
            self._ud_functions = None

    def execute_ud_func(self,fname,*kwargs):
        """ executes the function with key fname, passing the arguments *kwargs
        If the function does not exists, it returns None
        """
        if fname in self._ud_functions.keys(): 
            return(self._ud_functions[fname](*kwargs))
        else:
            return(None)

    def add_element_property(self,pname,ptype,pmap):
        """ add_element_property(pname,ptype,pmap)
        adds a material property associated to elements to 
        the element_properties dict.
        Inputs:
           pname: the name of the property (e.g.: conductivity')
           ptype: the type of the property (possible values are: 'region', 'elem')
           pmap: the property mapping that associates to a region/element ID the property value
        """
        print('adding the element property {} of type {}'.format(pname,ptype),flush=True)
        if self._element_properties is None:
            self._element_properties = {}
        self._element_properties[pname] = {'type':ptype,
                                           'idmap':pmap
                                          }        

    def add_nodal_property(self,pname,ptype,pmap):
        """ add_nodal_property(pname,ptype,pmap)
        adds a material property associated to Nodes to 
        the nodal_properties dict.
        Inputs:
           pname: the name of the property (e.g.: gNa')
           ptype: the type of the property (possible values are: 'region', 'nodal')
           pmap: the property mapping that associates to a region/point ID the property value
        """
        print('adding the nodal property {} of type {}'.format(pname,ptype),flush=True)    
        if self._nodal_properties is None:
            self._nodal_properties = {}
        self._nodal_properties[pname] = {'type':ptype,
                                           'idmap':pmap
                                        }

    def remove_element_property(self,pname):
        """
        remove_element_property(pname) if property pname exists, it removes 
        it from the the element_properties dict 
        """
        self._element_properties.pop(pname,None)
        if not self._element_properties:
            self._element_properties = None

    def remove_nodal_property(self,pname):
        """
        remove_nodal_property(pname) if property pname exists, it removes 
        it from the the nodal_properties dict 
        """
        self._nodal_properties.pop(pname,None)
        if not self._nodal_properties:
            self._nodal_properties = None        

    def remove_all_element_properties(self):
        """
        remove_all_element_properties() deletes all properties in element_properties
        """
        self._element_properties.clear()
        self._element_properties = None

    def remove_all_nodal_properties(self):
        """
        remove_all_nodal_properties() deletes all properties in nodal_properties
        """
        self._nodal_properties.clear()
        self._nodal_properties = None


    def ElementProperty(self,pname,elemtype,elemID,regionID):
        """"
        ElementProperty(pname,elemtype,elemID,regionID)
        returns the element property of an element/region
        Input:
            pname: the name of the property (e.g.: 'conductivity')
            elemtype: the type of element (e.g. 'Trias'; used when properties
                      are assigned elementwise)
            elemID: the element ID (used when properties are assigned elementwise)
            regionID: the element region ID (used when properties are assigned by region')
        Output:
            the value of property pname for the given input
        """
        try:
            value = None
            ptype = self._element_properties[pname]['type']
            if ptype=='region':
                value = self._element_properties[pname]['idmap'][regionID]
            elif ptype=='elem':
                value = self._element_properties[pname]['idmap'][elemtype][elemID]
            else:
                raise Exception('{}: unknown type'.format(ptype))
            return (value)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def NodalProperty(self,pname,pointID,regionID):
        """"
        NodalProperty(pname,pointID,regionID)
        returns the nodal property of a node/region
        Input:
            pname: the name of the property (e.g.: 'gNa')
            pointID: the point ID (used when properties are assigned at each node)
            regionID: the element region ID (used when properties are assigned by region')
        Output:
            the value of property pname for the given input
        """
        try:
            value = None        
            ptype = _nodal_properties[pname]['type']
            if ptype=='region':
                value = self._nodal_properties[pname]['idmap'][regionID]
            elif ptype=='nodal':
                value = self._nodal_properties[pname]['idmap'][pointID]
            else:
                raise Exception('{}: unknown type'.format(ptype))
            return (value)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
        

