# Audit: M1, CPHD reader and PNG writer

Auditor: planning/auditing agent. Date: 2026-11-08.
Input: `briefs/M1-cphd-reader.md`, `briefs/M1-cphd-reader-report.md`, the working tree.

## Verdict

**Accepted**, with two follow-ups carried into the M2 brief (neither was in the M1
"done when"). Every "done when" item passes when run by the auditor. The report's
pasted outputs match what the auditor observed. The report also found two errors in the
brief and the plan, both confirmed and corrected below.

## What the auditor ran

```
/home/feildaw/mypyenv/bin/pip install -e ".[test,viz]"      -> Successfully installed rdr-0.1.0
/home/feildaw/mypyenv/bin/python -m pytest -q                -> 26 passed in 17.79s (0 skipped)
benchmarks/bench_cphd_read.py <2 GB file> --device cuda      run 1: raw_read 0.399 s, io 0.529 s, ratio 1.325x, peak CUDA 2,258,633,216 B (1.098x)
benchmarks/bench_cphd_read.py <2 GB file> --device cuda      run 2: raw_read 0.296 s, io 0.501 s, ratio 1.695x, peak CUDA same
examples/01_read_cphd_dump_png.py                            -> two PNGs written, exit 0
grep -rn "sarkit\|sarpy" src/rdr/                            -> no matches
check_tree(Path("src"))                                      -> violations [], sibling imports []
```

Targets: io within 2x of raw_read (1.3x to 1.7x, pass); peak CUDA under 2.2x signal
bytes (1.098x, pass).

## Auditor's own checks beyond the brief

| check | result |
| --- | --- |
| channel dimensions of all three files via lxml path query vs `SIGNAL_BLOCK_SIZE` | all three match; 1 GB file is 13152 x 9999 |
| `SignalNormal` type in the XSD | `xs:boolean`, `minOccurs="0"`; the brief was wrong to list it with signal formats |
| `src/rdr.egg-info/` and `examples/output/*.png` gitignored | yes, by `*.egg-info/` and `output/` |
| unknown channel name | `KeyError` with available channels listed |
| `iter_signal` with chunk larger than the file | one chunk, bit-equal to `signal()` on CUDA |
| `signal_memmap` | read-only, `>c8`, correct shape, no copy |
| `SIGNAL == 0` vector counts | 1, 431, 0 for the 2023-09, 2023-07, 2025-10 files; matches the report |
| PVP arrays native and C-contiguous | yes (`TxPos` float64 `(N, 3)`) |
| open file descriptors after `CphdFile` + `pvp()` + `signal()` | 0 pointing at the file; the object holds no handle |
| 2 GB `signal(device="cpu")` vs plain `memmap.astype(complex64)`, same cache state, alternated | 1.57 s / 1.18 s vs 0.80 s / 0.41 s |

## Files reviewed line by line

`src/rdr/io/cphd.py` (811 lines), `src/rdr/io/png.py`, `tests/test_cphd_reader.py`,
`tests/test_png.py`, `benchmarks/bench_cphd_read.py`, `examples/01_read_cphd_dump_png.py`,
`pyproject.toml` diff (adds the `slow` marker only).

Notes from the read:

- The CUDA path is sound. Two pinned staging buffers, two streams, the host write into a
  pinned buffer waits on that slot's stream, so the DMA cannot read a buffer being
  overwritten. The output tensor is allocated on the default stream and written from side
  streams, which is safe here because every stream is synchronized before return. The
  byte swap is a `uint8` view flipped along the last axis of width 4, verified
  bit-identical to sarkit on all three files by the tests.
- `iter_signal` on CUDA uses a synchronous copy from the single pinned buffer, and the
  yielded tensor is the output of `flip`, so it does not alias the staging buffer. Correct.
- The PVP dtype is built from the XML layout, offsets in 8-byte words, with the standard
  cited by section and page. Nested `TxAntenna` and `RcvAntenna` and `AddedPVP` handled.
  Byte size of each field is checked against its `Size`.
- No try/except hides a failure. Unsupported formats raise `NotImplementedError` naming
  the format. Compressed signal arrays raise.
- `png.py` imports Pillow inside `write_png` and the test confirms it. The remap dict
  reproduces the output exactly, including on a sub-chip, tested.
- The example was changed at the human's request, per the report, to center the range
  profiles and document the shift conventions. That is outside the brief but was a human
  decision. The auditor did not witness the request.

## Errors in the brief and plan, found by the implementer, confirmed by the auditor

1. The 1 GB file has 9999 samples, not 9082. The plan's number came from a regex grab of
   the XML, which is exactly the mistake the plan warns against. Corrected in `PLAN.md`
   and annotated in the M1 brief.
2. `SignalNormal` is a boolean flag meaning every vector has `SIGNAL == 1`, not a storage
   format. The implementer typed it `bool | None` and documented it. Correct.
3. The brief's grep over `src/` hits the generated `egg-info`. Future briefs grep
   `src/rdr/`.

## Follow-ups carried into the M2 brief

1. **CPU `signal()` path is about 3x slower than the simplest alternative.** The
   `np.copyto` from a `>f4` memmap into the torch view runs at 1.2 to 1.6 s on the 2 GB
   file; `np.ascontiguousarray(memmap).astype(complex64)` runs at 0.4 to 0.8 s in the
   same cache state. Not in M1's "done when", which targeted CUDA. Fix in M2 with the
   bit-equality tests unchanged, and record both numbers.
2. **An unverified numeric claim in the example docstring.** It says the Doppler
   micro-parameters shift range peaks by under 0.25 sample. The auditor's rough estimate
   agrees in magnitude but neither of us computed it from the file. Per `rules4robots.md`,
   compute it in the script from the PVP and print it, or remove the number.

## Decisions recorded

- Density remap accepts `clip_value` so a returned parameter dict reproduces the output.
  Keep.
- `CphdMeta`, `ChannelMeta`, `PvpField` will move to `rdr.core` in M2 so `sim` and
  `imaging` can share them without importing `rdr.io`. `rdr.io.cphd` will re-export them.
  This is the plan's rule that imaging never imports io, applied now that there is a type
  to share.
