# readers

This is the sub-directory of I/O contains functions for reading data from files and/or convert file formats.

* `ImageData`: a class to handle images in tensor format


## ImageData

This class handles images in *NIfTI* format (through the *nibabel* library). input data can be eithr a *NIfTI* image, or a *png*
2D image with slices organised in a `Mx` X `My` matrix.
Members:
* `load_image(fname,mx=1,my=1)`: reads the image specified in `fname`. If this is a *png* file, `mx` and `my` specifies the slice grid sizes
* `save_nifty(fout)`: save the image to `fout` in *NIfTI* format
* `get_data` returns the image tensor
* `get_rescaled_data(scaling_type)`: rescale the image values with scaling type: `'unit'` (default): between `[0,1]`; `'mstd'`: ofset to mean value and rescaled to standard deviation.
* `image`: returnf the *nibabel* object
