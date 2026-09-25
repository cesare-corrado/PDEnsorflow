# LAT monitoring example

`lat.py` runs a planar wave on the triangulated square with the modified
Mitchell-Schaeffer cell model, and attaches a `LatDetector` (the equivalent of
the reference simulator's LAT detection, user guide 22.2/22.8) to record local
activation times.

## What it does

- The left edge is raised above threshold at `t = 0` (initial condition), so a
  wave sweeps across the strip in `+x`.
- A threshold-crossing detector at `-10 mV` (upstroke, `all = 0`) records the
  first activation time of each node.
- Because the front is planar, the activation time grows monotonically with
  `x`; fitting LAT against `x` gives the apparent conduction velocity.

## Run

```bash
E=/home/cc14/Libraries/anaconda3/envs/Claude_testing
cd PDEnsorflow/Tests/FEM/LAT
$E/bin/python lat.py
```

It prints the device banner, the number of activated nodes, the activation-time
range and the apparent conduction velocity, and writes `init_acts_vm_act.dat`
(the first-activation nodal vector, one value per line, `-1` where a node did
not activate within `Tend`). With `all = 1` the detector instead writes
`vm_act.dat`, a table of `node<TAB>tact` rows.

The `.dat` output is a run artefact: do not commit it.

## Parameters

`Tend = 60 ms` at `dt = 0.1 ms`. The wave does not reach the far edge in that
window, so only the activated part of the strip appears in the LAT map; raise
`Tend` to activate the whole strip.
