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
that V, then `V -= dt*Iion`. The gates use Rush-Larsen, the model default;
`--forward_euler` selects the scheme of the reference. The 1 ms stepping loop is compiled with XLA (about
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
