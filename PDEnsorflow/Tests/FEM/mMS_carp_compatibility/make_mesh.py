#!/usr/bin/env python
"""
    Writes the demo sheet as the three-file external mesh format, with node
    coordinates in MICROMETRES, and removes it again.

    Tests/data/triangulated_square.pkl stores a 10 x 10 mm sheet in
    millimetres. The parameter-file front end expects the coordinate unit of
    that format, which is micrometres, so the points are scaled by 1000 on the
    way out. Scaling the mesh rather than the conductivity keeps the parameter
    file readable: it carries conductivities in S/m, as such a file should.

    The mesh is a build artefact of the example, not data: it is ~10 MB of
    text, so it is generated before the run and removed after it.

        python make_mesh.py             # write square.pts / .elem / .lon
        python make_mesh.py --remove    # delete them again

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import pickle
import sys

import numpy as np

MM_TO_UM : float = 1.0e3
BASENAME : str   = 'square'
SUFFIXES         = ('.pts', '.elem', '.lon')
MESH_PKL         = os.path.join('..', '..', 'data', 'triangulated_square.pkl')

# Type specifiers of the external element format, by internal element name.
ELEMENT_CODES = {'Edges': 'Ln', 'Trias': 'Tr', 'Quads': 'Qd',
                 'Tetras': 'Tt', 'Hexas': 'Hx', 'Pyras': 'Py', 'Prisms': 'Pr'}


def remove_mesh(basename: str = BASENAME):
    """ remove_mesh(basename) deletes the three mesh files if they are there """
    for suffix in SUFFIXES:
        path = '{}{}'.format(basename, suffix)
        if os.path.isfile(path):
            os.remove(path)
            print('removed {}'.format(path))


def write_mesh(basename: str = BASENAME):
    """ write_mesh(basename) converts the pickled demo sheet into the three
        files of the external mesh format, with coordinates in micrometres
    """
    try:
        with open(MESH_PKL, 'rb') as fmesh:
            mesh = pickle.load(fmesh)
        points = np.asarray(mesh['Pts'], dtype=float) * MM_TO_UM
        fibres = mesh['Fibres']
        with open('{}.pts'.format(basename), 'w') as fout:
            fout.write('{}\n'.format(points.shape[0]))
            fout.write('\n'.join('{:.6f} {:.6f} {:.6f}'.format(*row) for row in points))
            fout.write('\n')
        rows : list = []
        for elemtype, elements in mesh['Elems'].items():
            if elements is None or len(elements) == 0:
                continue
            code = ELEMENT_CODES[elemtype]
            for elem in np.asarray(elements, dtype=int):
                rows.append('{} {} {}'.format(code,
                                              ' '.join(str(n) for n in elem[:-1]),
                                              elem[-1]))
        with open('{}.elem'.format(basename), 'w') as fout:
            fout.write('{}\n'.format(len(rows)))
            fout.write('\n'.join(rows))
            fout.write('\n')
        with open('{}.lon'.format(basename), 'w') as fout:
            fout.write('1\n')
            fout.write('\n'.join('{:.6f} {:.6f} {:.6f}'.format(*row)
                                 for row in np.asarray(fibres, dtype=float)))
            fout.write('\n')
        extent = points.max(axis=0) - points.min(axis=0)
        print('wrote {}.pts/.elem/.lon: {} points, {} elements, '
              'extent {:.0f} x {:.0f} um'.format(basename, points.shape[0], len(rows),
                                                 extent[0], extent[1]))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--remove':
        remove_mesh()
    else:
        write_mesh()
