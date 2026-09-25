# Prepacing example

`prepacing.py` conditions the cell models of a tissue simulation before the run
starts, the equivalent of the reference simulator's `prepacing_*` parameters
(user guide 22.3&ndash;22.7), and shows it composing with the LAT detector of
`Tests/FEM/LAT`.

## What prepacing is

Not a few beats of the tissue. A cell started from its published initial
conditions is not at a limit cycle, and pacing the *tissue* to steady state
costs a linear solve per step. Prepacing instead paces **single cells**, which
need no linear solve at all, and hands their state to every node with a per-node
time offset taken from a map of local activation times:

```
offset  = floor(min(LAT) / bcl) * bcl
last_tm = bcl * beats
save[i] = last_tm - (LAT[i] - offset)
```

A node that activates early takes the paced cell's state from late in the train,
one that activates late takes it from earlier, so every cell enters the run at
the phase of the cycle consistent with when the wavefront is about to reach it.
That offset is what a constant initial condition cannot express.

## What the script does

1. runs the unconditioned problem once and records a first-activation map with a
   `LatDetector`;
2. rebuilds the same problem, prepaces it from that map, and runs it again;
3. compares the two activation maps.

The wave is launched by a **stimulus**, not by an initial condition, in both
runs. Prepacing rewrites the potential at every node, so a depolarised block
written into the initial condition would simply be erased by it. The reference
behaves the same way and paces with a stimulus for the same reason.

## Run

```bash
E=/home/cc14/Libraries/anaconda3/envs/Claude_testing
cd PDEnsorflow/Tests/FEM/Prepacing
$E/bin/python prepacing.py
```

## Parameters

`Tend = 60 ms` at `dt = 0.1 ms` on the triangulated square (63001 nodes, four
regions); prepacing is 5 beats at `bcl = 250 ms`, `stimdur = 1 ms`,
`stimstr = 60 uA/uF`.

## Result (RTX A2000, verified 2026-09-25)

```
reference: activated 20582 / 63001 nodes, LAT in [0.823, 59.412] ms
paced 4 cell(s) over 12492 steps in 7.85 s (one per region)
save times span [1180.000, 1249.177] ms of the prepacing train
resting  Vm: 1 distinct value(s), spread 0.000000 mV
prepaced Vm: 1 distinct value(s), spread 0.000000 mV
prepaced H_state: 127 distinct value(s), spread 1.856321e-01
prepaced: activated 17319 / 63001 nodes, LAT in [0.893, 59.943] ms
activation time shift: mean +4.9558 ms, max |shift| 11.5219 ms
```

Four regions, so four cells are paced and the whole 12492-step train costs about
8 s, the same order as one 600-step tissue run.

**The potential is the wrong thing to look at here.** The activation spread is
60 ms against a 250 ms cycle length, so every save time lands in diastole, where
this model's potential is flat: `Vm` comes out uniform. The phase offset is
carried by the gating variable, which takes 127 distinct values across the mesh.
This is exactly why prepacing distributes the whole state and not only the
potential. Its effect is visible in the run: the partially recovered gate slows
conduction, and activation arrives about 5 ms later on average.

## From a parameter file

The same thing, driven by the `.par` front end:

```
prepacing_lats    = init_acts_vm_act.dat
prepacing_beats   = 5
prepacing_bcl     = 250.0
prepacing_stimdur = 1.0
prepacing_stimstr = 60.0
```

`prepacing_bcl` is the switch: prepacing is off unless it is positive. The
activation-time file is the one `LatDetector.write()` produces with `all = 0`
(one value per node, in mesh order, `-1` where a node did not activate).
Prepacing is skipped when `start_statef` names a checkpoint, since that already
carries a conditioned state. It is **not** skipped when
`imp_region[].im_sv_init` names a single-cell state file (`../StateInit`): the
train then starts from that state instead of from the model's rest state, which
is what the reference does, since it paces its cell models in place.

The `.dat` output is a run artefact: do not commit it.
