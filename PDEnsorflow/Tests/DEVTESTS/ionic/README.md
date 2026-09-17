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
