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
            max |dV| = 5.569e-04 mV, mean |dV| = 6.397e-06 mV, all finite: True
            max |dV| on the first / last compared frame: 6.866e-05 / 1.068e-04 mV
```

The resumed run is not identical to round-off: the CG warm-start history
`U^{n-1}` is not stored in the state file, so the first step after the restart
starts CG from a different guess, and the two runs then agree to within the CG
tolerance. Saving the state does not change the uninterrupted run:
`check_result.py` still gives 47.198 um/ms.

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
  `sigma = 1e5 * g * g_mult / (beta_ref * volFrac) = 1000 um^2/ms`, which is
  `0.001 mm^2/ms`. `beta_ref` is the `beta` of `imp_region[0]`, the only value
  the conductivity is divided by; where a file gives several regions different
  `cellSurfVolRatio`, the rest of `beta` weighs the mass matrix instead. Here
  both regions have `cellSurfVolRatio = 1`, so the weight is 1 and the two
  readings of the formula coincide.
* **Where `beta` goes.** `equivalent_script.py` registers the `beta` map, the
  reference `beta_ref` and the weighted mass function, exactly as the runner
  does, so the two routes assemble the same two matrices. With
  `cellSurfVolRatio = 1` everywhere the mass coefficient is 1, so this changes
  no number here; it keeps the script a faithful mirror of the front end.
* **Pacing.** The script starts from a depolarised block at `x < 0.5 mm`. A
  parameter file paces with a stimulus, so `S1` covers the same strip with
  `pulse.strength = 60` for 2 ms. A transmembrane stimulus needs no unit
  conversion: it is added to `dV/dt` in mV/ms, which is what the strength of a
  `crct.type = 0` electrode already means.
* **Pulse shape.** The format defaults `stim[].pulse.shape` to the truncated
  exponential, so that is the pulse the front end applies: it rises with
  `tau_edge`, creeps down with `tau_plateau` over its plateau, and falls back
  over the last `5 tau_edge` of its duration, which costs about 2.6% of the
  charge a square pulse of the same strength and duration would deliver. The
  `Stimulus` class defaults to the square pulse instead, so `equivalent_script.py`
  has to ask for `'shape': 'truncexp'` to stay equivalent.

One **LAT detector** is asked for as well (`num_LATs = 1`): the first crossing
of -30 mV on the upstroke of every node, written as
`OUT_mMS/init_acts_vm_act.dat`. It is a passive monitor, so the solution is
unchanged; it sees every time step, so its activation times are finer than the
1 ms `spacedt` of the IGB frames, and `check_result.py` fits the front speed
from them as well.

### Which region a node belongs to

The conductivities are element properties, so each triangle simply uses its own
tag. The cell parameters are read **node by node**, and a node on the boundary
between two tags has to be given one of them. The front end asks the mesh for
the assignment of the reference simulator
(`Triangulation.set_regionIDrule('carp_compatibility')`): a node takes the
highest tag an `imp_region` **claims** among the triangles it belongs to, and a
node no claimed tag reaches keeps the `-1` sentinel and is governed by
`imp_region[0]`.

Only `imp_region[1]` claims a tag here, tag 3. The IDs of `imp_region[0]` do not
take part: the reference builds its nodal mask over the regions 1..n only and
leaves region 0 to govern whatever they do not reach, which is the same set of
nodes either way. So of the 63001 nodes, 15876 are governed by `imp_region[1]`
(`tau_close = 60`) and the remaining 47125 by `imp_region[0]`
(`tau_close = 120`).

`equivalent_script.py` asks for the same rule and the same claimed set, and adds
the `-1` entry to each of its nodal maps. Without that it would use the library
default (`mostfrequent`, the most frequent tag at the node and, on a tie, the
smallest), which differs on the 130 nodes that lie on the boundary of tag 3, and
the two runs would no longer be comparable.

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
            V in [-81.080, 22.880] mV, all finite: True
CV measured:  47.198 um/ms (4.72 cm/s)
CV analytic:  46.188 um/ms (4.62 cm/s)
rel. error : 2.19%
LAT file  : 63001 of 63001 nodes activated, t in [0.669, 202.116] ms
CV from LAT:  47.196 um/ms (4.72 cm/s), rel. error 2.18%
vs Python API: max |dV| = 0.000e+00 mV, mean |dV| = 0.000e+00 mV
```

The two front speeds agree to 2e-3 um/ms although they are measured from
different data: the detector interpolates each activation inside the step it
happens in, the frame-based fit can only place it on a 1 ms grid. The bias of
that grid is almost the same at every x, so it cancels in the slope. `CV
measured` is also unchanged from the run without a detector, which is the check
that the monitoring does not perturb the solution.

The detector costs time, measured on this example (2500 steps, 63001 nodes,
RTX A2000): 51.1 and 51.9 s in the time loop against 41.1 s with the same file
and `-lats[0].method 0`, about 4 ms per step.

Where that time goes was measured, and the first answer was wrong. The guess
that the per-step read-back of the activation count is the cost is false:
deferring it (`set_use_deferred_count(True)`) made this run **slower**, 55.8 and
56.0 s, and it was slower at every mesh size tried later. Running the step as
one compiled graph (`set_use_graph_check(True)`) writes a **byte-identical**
activation file and roughly halves what the detector costs per step, but that
was not visible here: on the shared RTX A2000 of this workstation the same
comparison gave 51.3 and 51.1 s against 50.8 and 50.3 s, i.e. nothing, because
another process was holding the card at 99% while it ran.

Repeated on an **idle** RTX 5090, the detector cost per step (loop time minus
the signal-generation baseline) is:

| nodes | `all = 0` eager | `all = 0` graph | `all = 1` eager | `all = 1` graph |
|------:|----------------:|----------------:|----------------:|----------------:|
| 63 001 | 955 us | 456 us | 1138 us | 632 us |
| 1 000 000 | 1418 us | 625 us | 1619 us | 800 us |
| 4 000 000 | 2235 us | 1085 us | 2480 us | 1290 us |

So the compiled step is worth about a factor two at every size, and the figures
quoted from this workstation were contention, not physics.
`Tests/DEVTESTS/LAT/lat_detector_scaling.py` reproduces the table; it needs an
idle GPU.

Frame k holds the solution at t = k ms (k = 0 ... 250). The diffusion step is
Crank-Nicolson (`parab_solve = 1`, `theta = 0.5`, the defaults). With implicit
Euler (`-theta 1`) the same file gives 45.852 um/ms, -0.73%. The likely reading
(not verified by refining this mesh): the first-order time error slows the front
and cancels most of the space error of the linear elements, which speeds it up;
Crank-Nicolson removes most of the time error and leaves the space error,
+2.19%. On the 1D cable and the 2D
sheet under `Tests/CICD`, where the time error dominates, Crank-Nicolson is the
closer of the two (1D: -1.3% against -3.6%; 2D: -2.0% against -4.2%). The
potential dips about 1 mV below the model's -80 mV floor at the front: a small
Crank-Nicolson undershoot, which implicit Euler damps.

The two routes solve the same problem and give the same bits. Before version
1.7.0 they differed by 2e-2 to 6e-2 mV, as did two runs of the same file: the
matrices were assembled on the GPU with atomic additions, whose order changed
from run to run. The assembly is now on the host and deterministic.

Both runs take about 36 s in the time loop on an RTX A2000. Before the cell
model was compiled with XLA and the zero forcing was kept on the device, the
same file took 43 s; the fields of the two versions differ by at most 5.6e-2 mV,
which was the run-to-run difference of the GPU assembly at the time. On a mesh of this size the CG loop is
faster still as one GPU graph (24 s for this file): call
`model.solver().set_use_graph_loop(True)`. It is off by default because it is
slower on meshes of 1M nodes and more.

## Files

| File | What it is |
|------|------------|
| `make_mesh.py` | writes and removes the micrometre mesh (`--remove` to clean up) |
| `parameters.par` | the simulation, as a parameter file |
| `equivalent_script.py` | the same simulation through the Python API |
| `check_result.py` | conduction velocity vs the analytic value, from the frames and from the LAT file, and the two runs compared |
