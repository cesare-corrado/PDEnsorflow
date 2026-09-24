#!/usr/bin/env python
"""
    The layout of the single-cell state files (`.sv`) of the reference
    simulator, for each gpuSolve ionic model and plugin, so that a state saved
    by either simulator can be read by the other.

    The reference reads a section by POSITION: one line per entry, in the order
    of the model's internal state structure; the `# name` comment on each line
    is not read (limpet/src/python/limpetcommon.py, read_svs). A file for the
    reference must therefore hold every entry of that structure, in that order,
    including entries gpuSolve does not keep as state:
      * cell parameters the reference stores in the state (the four
        conductances of tenTusscherPanfilov, the parameters of
        MitchellSchaeffer): they are written with the value the run used;
      * plugin entries gpuSolve leaves out because they never change
        (Defib_AshiharaTrayanova's Ki and __sl_i2c_local): they are written with
        the reference's constant value.

    Each entry is (file name, kind, source, scale, gate):
      kind    'state'     source is the gpuSolve state variable name;
              'parameter' source is the gpuSolve parameter name;
              'constant'  source is the value itself.
      scale   file value = gpuSolve value * scale (Cai is uM in the reference
              files and mM in gpuSolve's tenTusscherPanfilov and Tomek);
      gate    True when the reference stores the entry as a single-precision
              gate (Gatetype, 4 bytes), which is how its binary dumps write it.

    The orders and types are those of the reference's generated structures
    (limpet/src/imps_src/<Model>.h, struct <Model>_state); the names and scales
    were checked against state files written by bench at t = 0.

    A model with no counterpart in the reference (ModifiedMS2v, whose outward
    current carries a (1 - h) factor the reference's MitchellSchaeffer does not
    have, and Fenton4v) gets a layout made of its own state variables, under
    its own parameter-file name. Such a file round-trips through singlecell but
    has no meaning for the reference.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""

# The global quantities that open every state file, in the reference's order
# (its IMP data types). singlecell provides Vm and Iion; the others are written
# as '-', the reference's mark for "not used by this model".
IMP_DATA_NAMES = ('Vm', 'Lambda', 'delLambda', 'Tension', 'Ke', 'Nae', 'Cae', 'Iion',
                  'tension_component', 'illum')

# Cai: uM in the reference files, mM in gpuSolve
_UM_PER_MM : float = 1.0e3


def _gates(names: tuple) -> list:
    """ state entries for gates whose gpuSolve name is '<name>_state' """
    return([(name, 'state', '{}_state'.format(name), 1.0, True) for name in names])


# Section name and entries, by gpuSolve class name.
SV_LAYOUTS = {
    'CourtemancheRamirezNattel': ('Courtemanche', [
        ('Ca_rel', 'state', 'Ca_rel', 1.0, False),
        ('Ca_up',  'state', 'Ca_up',  1.0, False),
        ('Cai',    'state', 'Cai',    1.0, False),
        ('Ki',     'state', 'Ki',     1.0, False)]
        + _gates(('d', 'f', 'f_Ca', 'h', 'j', 'm', 'oa', 'oi', 'u', 'ua', 'ui', 'v', 'w',
                  'xr', 'xs'))),
    'TenTusscherPanfilov': ('tenTusscherPanfilov', [
        ('CaSR',  'state',     'CaSR',        1.0,        False),
        ('CaSS',  'state',     'CaSS',        1.0,        False),
        ('Cai',   'state',     'Cai',         _UM_PER_MM, False)]
        + _gates(('D', 'F', 'F2', 'FCaSS'))
        + [('GCaL',  'parameter', 'GCaL', 1.0, False),
           ('GKr',   'parameter', 'GKr',  1.0, False),
           ('GKs',   'parameter', 'GKs',  1.0, False),
           ('Gto',   'parameter', 'Gto',  1.0, False)]
        + _gates(('H', 'J'))
        + [('Ki',    'state', 'Ki',  1.0, False)]
        + _gates(('M',))
        + [('Nai',   'state', 'Nai', 1.0, False)]
        + _gates(('R',))
        + [('R_',    'state', 'R_bar', 1.0, False)]
        + _gates(('S', 'Xr1', 'Xr2', 'Xs'))),
    'MitchellSchaeffer2v': ('MitchellSchaeffer', [
        ('V_gate',    'parameter', 'u_gate',    1.0, False),
        ('V_max',     'parameter', 'vmax',      1.0, False),
        ('V_min',     'parameter', 'vmin',      1.0, False),
        # the plain model has no excitation threshold: it is the reference
        # model with a_crit = 0
        ('a_crit',    'constant',  0.0,         1.0, False),
        ('h',         'state',     'H_state',   1.0, False),
        ('tau_close', 'parameter', 'tau_close', 1.0, False),
        ('tau_in',    'parameter', 'tau_in',    1.0, False),
        ('tau_open',  'parameter', 'tau_open',  1.0, False),
        ('tau_out',   'parameter', 'tau_out',   1.0, False)]),
    'Tomek': ('Tomek', [
        (name, 'state', name, _UM_PER_MM if name == 'Cai' else 1.0, False)
        for name in ('C1', 'C2', 'C3', 'CaMKt', 'Cai', 'Cajsr', 'Cansr', 'Cass', 'I',
                     'Jrel_np', 'Jrel_p', 'Ki', 'Kss', 'Nai', 'Nass', 'O', 'a', 'ap', 'd',
                     'fCaf', 'fCafp', 'fCas', 'ff', 'ffp', 'fs', 'h', 'hL', 'hLp', 'hp',
                     'iF', 'iFp', 'iS', 'iSp', 'j', 'jCa', 'jp', 'm', 'mL', 'nCa_i',
                     'nCa_ss', 'xs1', 'xs2')]),
    'ElectroporationDeBruinKrassowska98': ('Electroporation_DeBruinKrassowska98', [
        ('n', 'state', 'n', 1.0, False)]),
    'DefibAshiharaTrayanova': ('Defib_AshiharaTrayanova', [
        # constant in the reference: its factor sl_i2c is 0 for a plugin
        ('Ki',             'constant', 5.4, 1.0, False),
        ('__sl_i2c_local', 'constant', 0.0, 1.0, False)]),
}


def sv_layout(model, section_name: str) -> tuple:
    """ sv_layout(model, section_name) returns (section name, entries) for a
        cell model or a plugin (the object, not its wrapper). A class with no
        counterpart in the reference gets its own state variables, under
        section_name, the name the run selected it by.
    """
    classname = type(model).__name__
    if classname in SV_LAYOUTS:
        return(SV_LAYOUTS[classname])
    return((section_name, [(name, 'state', name, 1.0, False)
                           for name in model.state_variable_names()]))
