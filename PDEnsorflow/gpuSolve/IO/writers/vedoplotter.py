import numpy as np
try:
    import vedo 
    is_vedo = True
    class VedoPlotter:  
        """
        Class VedoPlotter
        Implements a plotter based on vedo library
        Need improvements
        """

        def __init__(self, config={}):
            self.interactive  = False
            self.niso         = 50
            self.plot_isosurf = True
            self.cmap         = 'jet'
            self.cutter       = False
            self.vmin         = 0.0
            self.vmax         = 1.0
            self.update_cbar  = False
            if(len(config)>0):
                for attribute in self.__dict__.keys():
                    if attribute in config.keys():
                        setattr(self, attribute, config[attribute])
            self.Plotter = vedo.Plotter()


        def plot(self,VolData):
            if self.update_cbar:
                self.vmin  = VolData.min()
                self.vmax  = VolData.max()
            if self.plot_isosurf:
                dh         = (self.vmax-self.vmin)/self.niso
                thresholds = np.arange(self.vmin,self.vmax+dh,dh).tolist()
                pdata      = vedo.Volume(VolData).isosurface(thresholds).cmap(self.cmap,vmin=self.vmin,vmax=self.vmax).addScalarBar()
            else:
                pdata = vedo.Volume(VolData).cmap(self.cmap,vmin=self.vmin,vmax=self.vmax).addScalarBar()
  
            if self.cutter:
                self.Plotter.show(pdata, __doc__, axes=9, interactive=self.interactive).addGlobalAxes().addCutterTool(pdata)
            else:
                self.Plotter.show(pdata, __doc__, axes=9, interactive=self.interactive).addGlobalAxes()

        
        def imshow(self,VolData):
            if self.update_cbar:
                self.vmin  = min(0,VolData.min())
                self.vmax  = max(0,VolData.max())
            dh         = (self.vmax-self.vmin)/(self.niso)
            thresholds = np.arange(self.vmin,self.vmax+dh,dh).tolist()
            pdata      = vedo.Volume(VolData).isosurface(thresholds).cmap(self.cmap,vmin=self.vmin,vmax=self.vmax).addScalarBar()
            self.Plotter.show(pdata, __doc__, axes=9, interactive=False).addGlobalAxes()
            #self.Plotter.close()

        def close(self):
            self.Plotter.close()
    
        def wait(self):
            for x in [0,1,2]:
                pass

        def __del__(self):
            self.close()
  

except:  
    is_vedo = False
    print('Warning: no vedo found',flush=True)
    class VedoPlotter:
        def __init__(self, config={}):  
            prin('dummy class')

