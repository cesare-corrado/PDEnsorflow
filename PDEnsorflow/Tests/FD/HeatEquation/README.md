# Tests/HeatEquation

## Description
This example implements the Parabolic heat equation on a 3D cubic domain, for an S1S2 cross-filed stimulus.

It discretises the Laplace operato either with a convolutional operator, or with finite differences

## Config

```
    config = {
        'width': 64,
        'height': 64,
        'depth': 64,
        'dx': 1,
        'dy': 1,
        'dz': 1,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 0.8,
        'samples': 10000,
        's2_time': 200,
         'convl': False
    }

```
