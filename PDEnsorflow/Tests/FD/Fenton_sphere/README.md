# Tests/Fenton_sphere

## Description
This example implements the monodomain with Fenton model, for an S1S2 cross-filed stimulus:
* on a 3D cubic domain, with a hole (`'hole' :  True`)
* on a cylindrical/spherical domain (`'hole' :  False`)

When `'cylindric': True`, the hole/domain is cylindric with axis parallel to z; spherical otherwise

## Config

```
    config = {
        'width':  128,
        'height': 128,
        'depth':  128,
        'radius': 32,
        'hole': True,
        'cylindric':True,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 0.8,
        'samples': 10000,
        's2_time': 200
    }

```
