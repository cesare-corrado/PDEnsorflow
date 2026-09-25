# Tests/FEM/StateInit

`imp_region[].im_sv_init`: start a tissue simulation from a **single-cell state
file**, the equivalent of the key of the same name in the reference simulator
(user guide 14.2). The file is one cell's state, and it is copied onto every
node of the ionic region that names it: "read in single cell state vector and
spread it out over the entire region" is how the reference describes its own
implementation.

## Why it is useful

A cell started from its published initial conditions is not at a limit cycle,
and a tissue run that begins there spends its first beats drifting. A state file
lets the conditioning be done once, in a single-cell run that costs no linear
solve, and reused by every tissue run of that model. It is also the way a state
computed by the other simulator enters a gpuSolve run: `singlecell` and `bench`
write the same format, so either program's file can start either program's run.

Prepacing (`../Prepacing`) solves the same problem differently: it paces the
cells itself and distributes the state with a per-node phase offset. The two
compose, and in the same order as in the reference: the file sets the state, and
the prepacing train then departs from it.

## What the script does

`state_init.py` is built so that the answer is known in advance:

1. `singlecell` paces one `MitchellSchaeffer` cell and saves its state 30 ms
   into the action potential (`cell.sv`);
2. `singlecell` continues that same cell from the file for 170 ms: this is the
   reference trajectory;
3. a 21-node cable is run for those 170 ms **with no stimulus at all** and
   `imp_region[0].im_sv_init = cell.sv`;
4. the cable's `Vm(t)` at the middle node is compared with the cell's.

The initial state is the same at every node, so the diffusion term is exactly
zero (a stiffness matrix annihilates a constant field) and the cable must
reproduce the single cell. A fifth run without the key is the control: it stays
at rest, which is what a key that silently did nothing would give.

## Run

```bash
E=/home/cc14/Libraries/anaconda3/envs/Claude_testing
cd PDEnsorflow/Tests/FEM/StateInit
$E/bin/python state_init.py
```

Everything it reads it also writes: `cable.pts/.elem/.lon`, `cell.sv`, the two
parameter files `state_init.par` and `rest.par`, the output directories
`OUT_sv_init` and `OUT_rest`, and the `singlecell` tables `cell_paced.txt` and
`cell_tail.txt`. Only the mesh and the `.igb` files match the ignore patterns:
these are run artefacts, none of them belongs in a commit.

## Result (CPU, verified 2026-09-25)

```
state file cell.sv written at t = 30 ms
cable      : 21 nodes, 171 frames, no stimulus
t = 0      : Vm = 13.542982 mV at every node (spread 0.00e+00 mV)
             the control run starts at -80.000000 mV (the model rest state)
cable vs the single cell continued from the same file, over 170 ms:
             max |dV| = 1.240e-05 mV, mean |dV| = 2.840e-06 mV
cable vs the control run (no im_sv_init), same window:
             max |dV| = 9.354e+01 mV
Vm at the probe: 13.543 -> -4.453 mV (cell: 13.543 -> -4.453 mV)
```

## What to look at

* the cable starts at the file's potential, identical at every node (the spread
  is zero), where the control run starts at the model's resting value;
* the cable and the single cell stay together for the whole 170 ms, to within
  the round-off of a float32 model plus the conjugate-gradient tolerance. A
  state file that had been read partially (the potential but not the gate, say)
  would show up at once: the gate is what drives the repolarisation that
  follows;
* the difference between the conditioned and the control run is the size of an
  action potential, not of a rounding error.
