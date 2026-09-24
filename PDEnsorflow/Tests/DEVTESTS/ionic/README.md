# Tests/DEVTESTS/ionic

## Description
This test runs a single-cell 100-beat pacing protocol for each ionic model: Fenton4v, ModifiedMS2v, CourtemancheRamirezNattel, and TenTusscherPanfilov.

For each model, the transmembrane potential of the last beat (last 1000 ms) is saved as a numpy array with two columns [time, U], recorded every 1 ms.

## Config

```
    config = {
        'dt': 0.02,
        'BCL': 1000.0,
        'n_beats': 100,
        'stim_intensity': 60.0,
        'stim_duration': 1.0,
        'record_interval': 1.0,
    }
```

- **Fenton4v**: dimensional (vmin=-80, vmax=20), V_init=-80.0
- **ModifiedMS2v**: dimensional (vmin=-80, vmax=20), V_init=-80.0
- **CourtemancheRamirezNattel**: V_init=-81.2
- **TenTusscherPanfilov**: V_init=-86.2, cell_type='EPI'

## Output

One `.npy` file per model:
- `fenton4v.npy`
- `mms2v.npy`
- `courtemanche_ramirez_nattel.npy`
- `ten_tusscher_panfilov.npy`

## Run

```
conda run -n PDEnsorflow python ionic.py
```

# tomek.py

Single-cell pacing of the Tomek (ToR-ORd) model for ENDO, EPI and MCELL at once
(three nodes of one model). The step follows the order of the reference
single-cell tool, `bench`: the stimulus is added to V, the model is advanced with
that V, then `V -= dt*Iion`. The model default uses Rush-Larsen for the gates
and the matrix exponential for the IKr Markov chain; `--forward_euler` selects
the scheme of the reference for both. The 1 ms stepping loop is compiled with XLA (about
12 s per beat on the RTX A2000 instead of about 19 minutes); `--no_xla` turns
that off.

```
python tomek.py --dt 0.01 --beats 100 [--forward_euler] [--reference DIR]
```

For each cell type it saves `tomek_<TYPE>_<rl|fe>_dt<dt>.npy`, columns
`[time (ms), V (mV), Cai (uM)]` of the last beat at 1 ms, and prints peak Vm,
APD90 and the `Cai` peak. With `--reference DIR` it prints the same numbers for
`bench` dumps in `DIR/ct0`, `DIR/ct1` and `DIR/ct2`, produced with

```
bench --imp Tomek --imp-par celltype=<0|1|2> --numstim <beats> --bcl 1000 \
      --duration <beats*1000> --dt <dt> --dt-out 1 -v
```

(`bench` reports `Cai` in uM; the model holds it in mM.)

# electroporation_debruin_krassowska98.py

Single cell with the electroporation plugin (`ElectroporationDeBruinKrassowska98`)
attached through `IonicModelWithPlugins`, compared step by step with the
reference single-cell tool, `bench`. The step follows `bench`'s order: the state
is recorded, the stimulus (from 1 to 2 ms) is added to V, the model and then the
plugin are advanced with that V, then `V -= dt*(Iion + I_ep)`.

```
python electroporation_debruin_krassowska98.py [--parent tomek|passive] [--vrest -80] \
       [--stim 0] [--duration 50] [--dt 0.01] [--reference DIR]
```

`--parent tomek` uses Tomek with forward-Euler gates (the reference's scheme).
`--parent passive` uses a passive membrane (the reference's `Plonsey` model,
written out in the script), which stays stable under a shock and so tests the
plugin alone. Tomek is comparable under a shock too, but above +222 mV (at
`dt = 0.01` ms) forward Euler on its IKr Markov chain is unstable in both codes,
so that part of its trajectory is a numerical artifact that both codes share.
It saves `electroporation_<parent>_stim<stim>_dt<dt>.npy`, columns
`[time (ms), V (mV), n (cm^-2)]` at every step. With `--reference DIR` it prints
the largest differences in V and n against `bench -v` dumps in `DIR`, produced with

```
bench --imp Tomek --plug-in Electroporation_DeBruinKrassowska98 \
      --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v
bench --imp Plonsey --imp-par "Vrest=<vrest>" --plug-in Electroporation_DeBruinKrassowska98 \
      --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v
```

Results at `dt = 0.01` ms (RTX A2000): passive parent at rest, 20 ms: 2.7e-10 mV
and 5.3e-11 (relative, n); passive parent with `--stim 1000`, 20 ms, V up to
+470.9 mV: 6.9e-7 mV and 1.8e-8; Tomek at rest, 50 ms, firing at about 48 ms:
2.3e-5 mV and 2.6e-10; Tomek with `--stim 1000`, 50 ms, V up to +470.2 mV:
7.5e-7 mV and 2.0e-8. With `--parent passive --vrest 0` the
plugin starts at exactly 0 mV, where its pore conductance is 0/0: `bench` turns
NaN from the first step, this code stays at the finite limit.

# defib_ashihara_trayanova.py

Single cell with the outward-current plugin (`DefibAshiharaTrayanova`) attached
through `IonicModelWithPlugins`, compared step by step with `bench`, in `bench`'s
step order (stimulus from 1 to 2 ms, then `V -= dt*(Iion + Ia)`).

```
python defib_ashihara_trayanova.py [--parent passive|tomek] [--vrest -80] \
       [--stim 400] [--duration 5] [--dt 0.01] [--reference-form] [--reference DIR]
```

`--reference-form` selects the lower branch of the reference model description,
`exp(0.09 (V - VtakeOff))`, the one `bench` computes. Without it the plugin uses
Cheng et al.'s `exp(0.09 (V - 100))` and differs from `bench` by design. It saves
`defib_<parent>_stim<stim>_dt<dt>[_refform].npy`, columns `[time (ms), V (mV)]`.
With `--reference DIR` it prints the largest difference in V against the
`bench -v` dumps in `DIR`, produced with

```
bench --imp Plonsey --imp-par "Vrest=<vrest>" --plug-in Defib_AshiharaTrayanova \
      --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v
bench --imp Tomek --plug-in Defib_AshiharaTrayanova \
      --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v
```

Results at `dt = 0.01` ms (RTX A2000), `--reference-form`: passive parent, 5 ms,
`--stim 0`, 400 and 2000 (V up to +163.7 and +227.6 mV): 0 mV (bit-identical);
Tomek, 50 ms, `--stim 0`: 2.0e-9 mV; `--stim 1000` (V up to +187.9 mV):
3.0e-7 mV. With Cheng et al.'s branch (the default) the peaks are the same
(the upper branch is shared) but the passive cell repolarises to 77.0 mV instead
of 96.3 mV at 5 ms (`--stim 400`), because Ia below `VtakeOff` is 221 times larger.

# singlecell_vs_bench.py

Runs the reference single-cell tool (`bench`, which must be on the PATH) and the
`singlecell` executable with the same command lines on every cell model the two
share, and prints the largest |Vm| difference, the APD90 and peak of the last
beat, and the wall time of each. It then swaps state files between the two
(each continues for one beat from the other's saved state, and from its own).
Everything is written to a scratch folder, never to the repository.

```
python singlecell_vs_bench.py [--beats 2] [--bcl 1000] [--workdir DIR]
```

Results with the defaults (2 beats, BCL 1000 ms, dt 0.01 ms, CPU, one thread):

| model | max\|dVm\| | APD90 bench / singlecell | bench s | singlecell s |
|---|---:|---:|---:|---:|
| Courtemanche | 30.9 mV | 294 / 392 ms | 0.17 | 4.67 |
| tenTusscherPanfilov | 0.108 mV | 302 / 302 ms | 0.20 | 4.83 |
| Tomek (`--reference-scheme`) | 5e-6 mV | 271 / 271 ms | 0.38 | 10.02 |
| MitchellSchaeffer (`V_min=0,V_max=1,tau_out=5`) | 5.8e-5 | 254 / 254 ms | 0.15 | 2.81 |

The singlecell times include the start of TensorFlow and the compilation of
the model (a few seconds). The state files swap to 1.3e-3 mV over one Tomek beat,
which is the 6-digit rounding of bench's file. `singlecell` itself matches a
hand-written loop over the same model to 5e-8 mV: the Courtemanche difference
is in the gpuSolve model, not in the front end. Its first diverging state is the
IKs gate `xs`, from t = 4 ms, on the plateau near 19.9 mV where the rate
expressions of `xs` are 0/0.

