# readers

This is the sub-directory of I/O contains functions for reading data from files and/or convert file formats.

* `ImageData`: a class to handle images in tensor format


## ImageData

This class handles images in *NIfTI* format (through the *nibabel* library). 
Input data can be eithr a *NIfTI* image, or a *png*
2D image with slices organised in a `Mx` X `My` matrix.
Members:
* `load_image(fname,mx=1,my=1)`: reads the image specified in `fname`. If this is a *png* file, `mx` and `my` specifies the slice grid sizes
* `save_nifty(fout)`: save the image to `fout` in *NIfTI* format
* `get_data` returns the image tensor
* `get_rescaled_data(scaling_type)`: rescale the image values with scaling type: `'unit'` (default): between `[0,1]`; `'mstd'`: ofset to mean value and rescaled to standard deviation.
* `image`: returnf the *nibabel* object

## CarpMeshReader
This class reads meshes in *Carp* format. The input data is formed by three files with the same prefix:
* A file with extension *.pts* that contains the coordinates of the vertices
* A file with extension *.elem* that contains the element type, the point IDs belonging to each element and the element region ID
* A file with extension *.lon* that contains the fibers.
Members:
* `read(fsuffix)`: reads the mesh files with suffix `fsuffix` (the suffix must contain the path).
* `Pts()`: returns the numpy array of the point coordinates
* `Elems()`: returns the python dict that contains the mesh Elements; each key corresponds to an element type (e.g.: `Trias` corresponds to Triangles); each entry is a numpy array of type int.
* `Edges()`, `Trias()`, `Tetras()`, etc: shortcuts to `Elems()[key]`. It provides a numpy array of type int where the first n-1 columns are the element point IDs; the last column is the lement region ID. If the mesh does not contain the type, it returns `None`.
* `Fibres()`: returns the numpy array of the fiber directions. 
