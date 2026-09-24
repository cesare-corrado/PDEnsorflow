#!/usr/bin/env python
"""
    Cell-type numbering shared by every cell model that has cell types
    (Tomek, TenTusscherPanfilov).

    A per-node `celltype` column is a plain number, so the same number must name
    the same cell in every model: a script that sets celltype = 0 gets an
    endocardial cell whichever model it runs. The numbering is Tomek's, whose
    parameter file only accepts the integer form (celltype=1), so it cannot be
    changed without breaking existing files. A model whose reference file
    enumerates its flags in another order (the tenTusscherPanfilov model file
    has EPI = 0, MCELL = 1, ENDO = 2) translates at the boundary: the parameter
    file names the type (flags=ENDO), never its number.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""

CELL_TYPE_IDS = {'ENDO': 0, 'EPI': 1, 'MCELL': 2}
