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
