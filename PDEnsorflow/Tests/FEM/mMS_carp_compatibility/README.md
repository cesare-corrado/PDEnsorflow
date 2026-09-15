# Tests/FEM/mMS_carp_compatibility

The `Tests/FEM/mMS` example, driven by a **parameter file** instead of a Python
script, and the same simulation written **both ways** so the two can be compared.

This is the worked example of the `PDEnsorflow` command: it shows what a `.par`
file looks like, how its units are converted, and that driving the library from
a file and driving it from Python give the same solver.

## Running it

```
python make_mesh.py                  # write square.pts / .elem / .lon (micrometres)
PDEnsorflow +F parameters.par        # the parameter-file route   -> OUT_mMS/vm.igb
python equivalent_script.py          # the Python API route       -> OUT_mMS_api/vm.igb
python check_result.py               # conduction velocity, and the two compared
python make_mesh.py --remove         # delete the generated mesh
```

### Saving the state and resuming from it

```
PDEnsorflow +F parameters.par -num_tsav 1 -tsav[0] 100                          # also writes OUT_mMS/state.100.pkl
PDEnsorflow +F parameters.par -simID OUT_mMS_restart -start_statef OUT_mMS/state.100
python check_restart.py              # the resumed run against the uninterrupted one
```

The resumed run starts at step 1000 (t = 100 ms) and writes frames for
[100, 250] ms. Measured on an RTX A2000:

```
state file : OUT_mMS/state.100.pkl at t = 100.0000 ms, model ModifiedMS2v, 63001 nodes, 1 state variable(s)
frames     : uninterrupted 251, restarted 151
restart frame 0 vs saved Vm: max |dV| = 0.000e+00 mV
restarted vs uninterrupted over the last 150 frames (150 ms):
            max |dV| = 5.648e-02 mV, mean |dV| = 4.740e-06 mV, all finite: True
            max |dV| on the first / last compared frame: 3.815e-05 / 2.012e-02 mV
```

The resumed run is not identical to round-off, for two reasons. The CG
warm-start history `U^{n-1}` is not stored in the state file, so the first step
after the restart starts CG from a different guess. Also, two GPU runs of the
same file already differ by about `2.4e-2` mV (see below). The restart
difference is the same size as that floor. Saving the state does not change the
uninterrupted run: `check_result.py` still gives 45.839 um/ms.

The mesh is a build artefact, not data: `make_mesh.py` converts
`Tests/data/triangulated_square.pkl` (a 10 x 10 mm sheet stored in millimetres)
into the three-file format with coordinates in **micrometres**, which is the
unit that format specifies. It is about 8 MB of text and is removed at the end.

## What the parameter file asks for

A 10 x 10 mm sheet of 63001 nodes and 125000 triangles, paced along the strip
`x < 500 um` at `t = 0`, run for 250 ms with `dt = 100 us`. Two ionic regions
share the modified Mitchell-Schaeffer model but differ in `tau_close` (120 ms on
tags 1, 2 and 4; 60 ms on tag 3), exactly as `../mMS/mMS.py` does.

Two things differ from that script, both because of the format:

* **Units.** The script sets a diffusion coefficient of `0.001 mm^2/ms`
  directly. A parameter file carries conductivities in **S/m** and the mesh in
  micrometres, so the same physics is `g_il = g_it = 0.01 S/m` with
  `cellSurfVolRatio = 1`: the mapping applies
  `sigma = 1e5 * g * g_mult / (beta * volFrac) = 1000 um^2/ms`, which is
  `0.001 mm^2/ms`.
* **Pacing.** The script starts from a depolarised block at `x < 0.5 mm`. A
  parameter file paces with a stimulus, so `S1` covers the same strip with
  `pulse.strength = 60` for 2 ms. A transmembrane stimulus needs no unit
  conversion: it is added to `dV/dt` in mV/ms, which is what the strength of a
  `crct.type = 0` electrode already means.

`bidm_eqv_mono = 0` is set on purpose. Left at its default of 1 the conductivity
would be the half harmonic mean of the intracellular and extracellular values
rather than `g_il` alone, and would not reproduce the script.

## What it should produce

Ahead of the front the fast gate is still closed, so the reaction reduces to the
Nagumo bistable form and the planar front speed is analytic:

```
CV = 0.5 (1 - 2 u_crit) sqrt(2 D / tau_in)
   = 0.5 (1 - 0.2) sqrt(2 * 1000 / 0.15)
   = 46.19 um/ms  (4.62 cm/s)
```

Measured on this mesh, `check_result.py` reports:

```
front end : 251 frames x 63001 nodes
            V in [-80.386, 23.138] mV, all finite: True
CV measured:  45.839 um/ms (4.58 cm/s)
CV analytic:  46.188 um/ms (4.62 cm/s)
rel. error : 0.76%
vs Python API: max |dV| = 2.366e-02 mV, mean |dV| = 7.339e-06 mV
```

The 0.76% is the linear-FEM and forward-Euler discretisation bias, in line with
the 1D and 2D regressions under `Tests/CICD`.

The difference between the two routes is round-off, not a difference in what was
solved. Re-running the *same* parameter file a second time gives a difference of
the same size, which settles where it comes from:

```
front end vs itself (same .par, rerun): max |dV| = 2.361e-02 mV   mean = 7.570e-06 mV
front end vs Python API               : max |dV| = 2.366e-02 mV   mean = 7.339e-06 mV
```

The sparse kernels do not reduce in a fixed order on the GPU, so two identical
runs already differ by that much. The maximum sits on the upstroke, where the
steepest gradient turns the smallest timing difference into the largest voltage
difference; the mean over 251 frames and 63001 nodes is 7e-6 mV.

Both runs take about 41 s on an RTX A2000.

## Files

| File | What it is |
|------|------------|
| `make_mesh.py` | writes and removes the micrometre mesh (`--remove` to clean up) |
| `parameters.par` | the simulation, as a parameter file |
| `equivalent_script.py` | the same simulation through the Python API |
| `check_result.py` | conduction velocity vs the analytic value, and the two runs compared |
