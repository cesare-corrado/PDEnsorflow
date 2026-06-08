# Tests/DEVTESTS/IO

## Description
This folder contains the development tests for the `gpuSolve.IO.writers`
submodule, in particular the `BaseWriter` variable-size GPU container that
buffers the solution and flushes it to disk in chunks.

`test_basewriter.py` implements two scenarios with Python's built-in
`unittest` framework:

* **Test 1 — iteration-based chunking (`every_N`)**
  A loop feeds `n` random TensorFlow tensors into the writer one at a time and
  checks that:
  - the solutions are kept in (GPU) memory between flushes;
  - a chunk is dumped to disk **only** on the exact multiples of `every_N`;
  - the final aggregated `.npy` file has the right shape and matches the
    original `n` solutions exactly.

* **Test 2 — memory-based chunking (`max_chunk_mb`)**
  The writer is initialised with a tiny `1 MB` threshold (`every_N=None`) and
  fed large tensors one at a time. The test checks that a disk dump is
  triggered as soon as the accumulated tensors exceed the threshold, and that
  the final file preserves the data across the memory chunks.

The test is **autonomous**: it creates an isolated temporary directory in
`setUp()` and removes every temporary `.npy`/`.igb` file and folder in
`tearDown()`.

## Usage
From the `Tests/DEVTESTS/IO` directory, run:
```bash
conda activate PDEnsorflow
source ~/TestCase/init.sh
python test_basewriter.py
```

To run a single scenario:
```bash
python -m unittest test_basewriter.TestBaseWriter.test_every_N_chunking
python -m unittest test_basewriter.TestBaseWriter.test_max_chunk_mb_chunking
```

The test prepends the local package checkout to `sys.path`, so it always
exercises the edited `gpuSolve` in this repository (not a globally installed
copy).

## Output
The script reports, for each scenario, the standard `unittest` PASS/FAIL line.
On success it prints:
```
Ran 2 tests in ...s

OK
```
