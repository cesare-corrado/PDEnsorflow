# Tests/Fenton_atria

## Description
This example implements the monodomain with Fenton model, for an S1S2 cross-filed stimulus, for a domain that 
is loaded from a image (a png file in this case)




## Config

```
    config = {
        'width':  64,
        'height': 64,
        'depth':  64,
        'dx':     1,
        'dy':     1,
        'dz':     1,
        'fname': '../../data/structure.png',
        'Mx': 16,
        'My': 8,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 0.75,
        'samples': 10000,
        's2_time': 190
    }

```
