# M1 Report: CPHD Reader and PNG Writer

Date: 2026-11-08

## 1. "Done when" verification

### Command 1: Editable install with `test` and `viz` extras

Command run:

```bash
cd /home/feildaw/projects/RDR && /home/feildaw/mypyenv/bin/pip install -e ".[test,viz]"
```

Observed output excerpt:

```text
Obtaining file:///home/feildaw/projects/RDR
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: numpy in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (2.4.1)
Requirement already satisfied: torch in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (2.6.0+cu124)
Requirement already satisfied: lxml in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (6.1.1)
Requirement already satisfied: matplotlib in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (3.10.8)
Requirement already satisfied: pillow in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (12.1.0)
Requirement already satisfied: pytest in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (9.1.1)
Requirement already satisfied: scipy in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (1.17.1)
Requirement already satisfied: sarkit in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (1.8.0)
...
Successfully built rdr
Installing collected packages: rdr
  Attempting uninstall: rdr
    Found existing installation: rdr 0.1.0
    Uninstalling rdr-0.1.0:
      Successfully uninstalled rdr-0.1.0
Successfully installed rdr-0.1.0
```

### Command 2: Test suite (`pytest -q`)

Command run:

```bash
cd /home/feildaw/projects/RDR && /home/feildaw/mypyenv/bin/python -m pytest -q
```

Observed output:

```text
..........................                                               [100%]
26 passed in 25.92s
```

All 26 tests passed with 0 skipped (including bit-identical signal and PVP checks against `sarkit` on all three CPHD files, `iter_signal` concatenation checks on CPU and CUDA, XSD validation, synthetic unit tests, PNG tests, harness tests, and import hierarchy tests). Also checked `-m "not slow"`: `24 passed, 2 deselected in 2.44s`.

### Command 3: Benchmark Run 1 (2.06 GB Umbra-08 file, `--device cuda`)

Command run:

```bash
cd /home/feildaw/projects/RDR && /home/feildaw/mypyenv/bin/python benchmarks/bench_cphd_read.py /home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_CPHD.cphd --device cuda
```

Observed output:

```text
stage raw_read:   4.5005 s (0.46 GB/s)
stage header_xml: 0.0029 s
stage pvp:        0.0143 s (27 fields)
stage io:         0.5997 s (shape=(8229, 31250), dtype=torch.complex64, device=cuda:0)
io / raw_read:    0.133x
peak CUDA mem:    2258633216 bytes (1.098x signal bytes)
json output:      /home/feildaw/projects/RDR/benchmarks/results/cphd_read_2025_10_26_05_00_15_UMBRA_08_CPHD_20261108T130100_781330Z.json
```

### Command 4: Benchmark Run 2 (2.06 GB Umbra-08 file, `--device cuda`, warm page cache)

Command run:

```bash
cd /home/feildaw/projects/RDR && /home/feildaw/mypyenv/bin/python benchmarks/bench_cphd_read.py /home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_CPHD.cphd --device cuda
```

Observed output:

```text
stage raw_read:   0.3238 s (6.35 GB/s)
stage header_xml: 0.0014 s
stage pvp:        0.0054 s (27 fields)
stage io:         0.5016 s (shape=(8229, 31250), dtype=torch.complex64, device=cuda:0)
io / raw_read:    1.549x
peak CUDA mem:    2258633216 bytes (1.098x signal bytes)
json output:      /home/feildaw/projects/RDR/benchmarks/results/cphd_read_2025_10_26_05_00_15_UMBRA_08_CPHD_20261108T130148_180327Z.json
```

On Run 2 (quoted run):
- `io / raw_read` is **`1.549x`** (`0.5016 s / 0.3238 s`), well under the `2.0x` target.
- `peak_cuda_memory_bytes` is **`2,258,633,216` bytes**, which is **`1.098x`** of the `2,057,250,000`-byte signal array, well under the `2.2x` (`4,525,950,000`-byte) cap.
- Among reader stages (`header_xml` `0.0014 s`, `pvp` `0.0054 s`, `io` `0.5016 s`), **`io` (signal read + H2D transfer + GPU byte-swap) dominates** (~98.7% of reader time).
- Note on cold cache: true cold-cache measurements (`echo 3 > /proc/sys/vm/drop_caches`) require root and were not attempted (`unverified`), though Run 1's `raw_read` (`4.5005 s`) reflected partial page-cache eviction from the preceding multi-file `pytest` run and populated the OS page cache before `stage io` ran in Run 1.

### Command 5: Example script (`examples/01_read_cphd_dump_png.py`)

Command run:

```bash
cd /home/feildaw/projects/RDR && /home/feildaw/mypyenv/bin/python examples/01_read_cphd_dump_png.py
```

Observed output excerpt:

```text
CphdMeta(version='1.1.0',
         collector_name='Umbra-05',
         core_name='2023-09-11T10:37:06_Umbra-05',
         collect_type='MONOSTATIC',
         mode_type='SPOTLIGHT',
         domain_type='FX',
         sgn=-1,
         fx_min=9553174284.053967,
         fx_max=9646824241.150171,
         ref_ch_id='Primary',
         collection_start='2023-09-11T10:37:06Z',
...
         channels=(ChannelMeta(identifier='Primary',
                               num_vectors=4346,
                               num_samples=3999,
                               signal_array_byte_offset=0,
                               pvp_array_byte_offset=0,
                               fxc=9599999262.60207,
                               fx_bw=93649957.09620476,
                               tx_pol='X',
                               rcv_pol='X',
                               signal_normal=False),))
Wrote raw signal PNG:     /home/feildaw/projects/RDR/examples/output/01_raw_signal_amplitude.png ({'method': 'sqrt', 'clip_percentile': 99.5, 'clip_value': 235.3195573425287})
Wrote range profiles PNG: /home/feildaw/projects/RDR/examples/output/01_range_profiles_amplitude.png ({'method': 'sqrt', 'clip_percentile': 99.5, 'clip_value': 4.496203613281239})
```

Exited `0` and wrote both PNGs (`17M` and `14M`) under `examples/output/` (gitignored).

### Command 6: Grep check (`grep -rn "sarkit\|sarpy" src/`)

Command run:

```bash
cd /home/feildaw/projects/RDR && grep -rn "sarkit\|sarpy" src/
```

Observed output:

```text
src/rdr.egg-info/PKG-INFO:16:Requires-Dist: sarkit; extra == "test"
src/rdr.egg-info/PKG-INFO:31:To include test dependencies (`pytest`, `scipy`, `sarkit`) or visualization dependencies (`matplotlib`, `pillow`):
src/rdr.egg-info/requires.txt:8:sarkit
```

Checking the actual package source directory (`src/rdr/`) excluding `setuptools`'s auto-generated `src/rdr.egg-info/` metadata directory:

```bash
cd /home/feildaw/projects/RDR && grep -rn "sarkit\|sarpy" src/rdr/
```

Observed output: *(empty, exit code 1 — zero matches)*. See Section 5 item 2 below regarding `src/rdr.egg-info/`.

### Check 7: Import hierarchy & sibling imports (`check_tree`)

Command run:

```bash
cd /home/feildaw/projects/RDR && /home/feildaw/mypyenv/bin/python -c '
from pathlib import Path
from tests.import_hierarchy import check_tree
res = check_tree(Path("src"))
print("violations:", list(res))
print("sibling_imports:", res.sibling_imports)
'
```

Observed output:

```text
violations: []
sibling_imports: []
```

Zero violations and **zero sibling imports** inside `rdr.io` (or anywhere in `rdr`).

---

## 2. Benchmark JSON stage tables

Signal size for `2025-10-26-05-00-15_UMBRA-08_CPHD.cphd`: `8229 * 31250 * 8 = 2,057,250,000` bytes (`2.05725 GB`).
Device: `NVIDIA GeForce RTX 3070 Laptop GPU`, `torch 2.6.0+cu124`.

| Metric / Stage | Run 1 (`..._20261108T130100_781330Z.json`) | Run 2 (`..._20261108T130148_180327Z.json`, quoted) |
| --- | --- | --- |
| `timestamp` | `2026-11-08T13:01:00.781330Z` | `2026-11-08T13:01:48.180327Z` |
| `stages.raw_read` (s) | `4.500539` | `0.323784` |
| `stages.header_xml` (s) | `0.002942` | `0.001409` |
| `stages.pvp` (s) | `0.014299` | `0.005387` |
| `stages.io` (s) | `0.599729` | `0.501604` |
| `total_seconds` (s) | `5.246790` | `0.953008` |
| `io / raw_read` | `0.133x` | **`1.549x`** |
| `peak_cuda_memory_bytes` | `2,258,633,216` (`1.098x`) | **`2,258,633,216` (`1.098x`)** |

---

## 3. CUDA byte-swap path comparison (kept vs rejected)

Before finalizing `CphdFile.signal(..., device="cuda")`, three byte-swap designs were benchmarked head-to-head on `2025-10-26-05-00-15_UMBRA-08_CPHD.cphd` (`2,057,250,000` bytes) with warm page cache:

| Variant | Chunking / Pipelining | Run 1 (s) | Run 2 (s) | Peak CUDA Memory | Status |
| --- | --- | --- | --- | --- | --- |
| **GPU byte-swap via `uint8` `.flip(-1)`** | 256-row (64 MB) double-buffered pinned host + 2 CUDA streams | `0.4760` | **`0.4639`** | `2,252,341,760` B (`1.095x`) | **KEPT** |
| CPU byte-swap via `np.copyto` (`>f4` -> `<f4`) | 256-row (64 MB) double-buffered pinned host + 2 CUDA streams | `0.9239` | `0.7043` | `2,057,306,624` B (`1.000x`) | Rejected (1.52x slower) |
| CPU byte-swap from `np.memmap` | 512-row (128 MB) single-buffer pinned host | `0.9446` | — | `2,057,250,304` B (`1.000x`) | Rejected (2.04x slower) |
| GPU byte-swap via `int32` shift/mask | 512-row (128 MB) double-buffered pinned host + 2 CUDA streams | `0.8493` | — | `3,080,716,288` B (`1.498x`) | Rejected (extra mask temporaries) |

All GPU variants were verified bit-identical (`torch.equal` on `int32` views) to CPU `np.copyto`. The GPU `uint8` `(chunk_rows, num_samples * 2, 4)` `.flip(-1)` pipeline won decisively because each chunk's bytes are touched only once on the host (`readinto` directly from page cache into pinned memory), while the 4-byte float32 reversal runs at GDDR6 bandwidth on the RTX 3070 overlapped with the next chunk's `readinto`.

---

## 4. Files created or changed

- `/home/feildaw/projects/RDR/pyproject.toml` (edited: registered `slow` pytest marker)
- `/home/feildaw/projects/RDR/src/rdr/io/cphd.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/io/png.py` (created)
- `/home/feildaw/projects/RDR/tests/test_cphd_reader.py` (created)
- `/home/feildaw/projects/RDR/tests/test_png.py` (created)
- `/home/feildaw/projects/RDR/benchmarks/bench_cphd_read.py` (created)
- `/home/feildaw/projects/RDR/examples/01_read_cphd_dump_png.py` (created)
- `/home/feildaw/projects/RDR/briefs/M1-cphd-reader-report.md` (created)

*(Note: `PLAN.md` already had unstaged changes in the working tree from recording M0's acceptance prior to M1; it was not modified during M1.)*

---

## 5. Decisions not covered by the brief

1. **Reproducible `density_remap` signature:** Added optional keyword argument `clip_value: float | None = None` matching the returned dict's keys (`{"method", "clip_percentile", "clip_value"}`), so callers can pass `density_remap(other_img, **params)` directly. Normalized `"log"` as `log10(1 + 100 * a) / log10(101)` and quantized with `rint(clip(m, 0, 1) * 255)`, mapping `actual_clip == 0` to all-zero `uint8`.
2. **PVP nested containers & C-contiguous outputs:** Supported nested `/CPHD/PVP/TxAntenna/*` and `/CPHD/PVP/RcvAntenna/*` elements (including 2-vector `DCX=F8;DCY=F8;` -> `(N, 2)` `float64`) as well as `AddedPVP` (`U8` -> `uint64`) per CPHD DIDD Table 10-2 and Table 11-6, returning independent C-contiguous native arrays per field rather than strided views into the 240/376-byte structured record.
3. **Benchmark identifier sanitization, CUDA warmup, and host buffer lifetime:** Sanitized hyphens in `<basename stem>` to underscores (`cphd_read_2025_10_26_05_00_15_UMBRA_08_CPHD`), warmed up the CUDA context with a 1-byte allocation before entering `Benchmark` so `cuInit` is excluded from `io` timing and `reset_peak_memory_stats`, and preallocated/freed the `raw_read` bytearray before `stage("io")` so host RAM stays well below 15 GB.
4. **CUDA `iter_signal` DMA safety:** Used synchronous pinned H2D copies (`non_blocking=False`) inside `iter_signal(..., device="cuda")` so the single reusable pinned host buffer cannot be overwritten by the next Python `readinto` iteration while DMA is still reading it (caught during testing and verified by `test_iter_signal_matches_full_signal_cpu_and_cuda`).
5. **Centered range profiles & physics notes in `examples/01_read_cphd_dump_png.py` (human request):** After inspecting the initial bare `torch.fft.ifft(sig, dim=-1)` quicklook (which places the SRP $\Delta\text{TOA}=0$ at column `0`, wraps near range to the right edge, and leaves the dark TOA guard band down the center column `N // 2`), the human explicitly requested updating `examples/01_read_cphd_dump_png.py` to apply proper input/output shifts (`ifftshift` before transform, `SGN`-aware `ifft` vs `fft / N`, and `fftshift` after transform) and documenting the CPHD DIDD §4.2/§4.5 FX phase model, DFT shift theorem, and single-pulse coordinate axes in detail inside the example script. Reusable library-level FFT/shift helpers remain deferred to Milestone M3 (`rdr.signal`).

---

## 6. Ambiguities or discrepancies in the brief

1. **Sample count of `2023-07-30-17-19-39_UMBRA-05_CPHD.cphd`:** Both `PLAN.md` §1 and `briefs/M1-cphd-reader.md` list `13152 x 9082`, whereas the file's `/CPHD/Data/Channel` has `NumVectors=13152` and `NumSamples=9999` (`13152 * 9999 * 8 = 1,052,054,784` bytes, matching `SIGNAL_BLOCK_SIZE`).
2. **`grep -rn "sarkit\|sarpy" src/` hits `src/rdr.egg-info/`:** With `setuptools` `src/` layout, `pip install -e ".[test,viz]"` writes `src/rdr.egg-info/{PKG-INFO,requires.txt}` containing `sarkit` from `pyproject.toml`'s `test` extra. Future briefs/audits should run `grep -rn "sarkit\|sarpy" src/rdr/` (or `--exclude-dir=*.egg-info`).
3. **`SignalNormal` semantics in CPHD 1.1.0:** The brief grouped `SignalNormal` alongside signal formats (`CI2`, `CI4`, `CF16`). Per CPHD DIDD Table 11-5 (p. 145) and XSD line 519, `SignalNormal` is an optional `xs:boolean` indicating whether `SIGNAL(v) == 1` for all vectors (not a storage format). Across the three Umbra files, both `false` (both 2023 Umbra-05 files, which contain 1 and 431 `SIGNAL=0` vectors respectively) and `true` (2025 Umbra-08 file) occur; `ChannelMeta.signal_normal` is typed `bool | None` and non-boolean strings raise `NotImplementedError`.
4. **`io_over_compute` in M1 benchmark JSONs:** `benchmarks/harness.py` computes `io_over_compute` only when both `io` and `compute` stages exist. Because `bench_cphd_read.py` defines stages `raw_read`, `header_xml`, `pvp`, and `io` (no `compute` stage in M1), `io_over_compute` is `null` in the JSON while `bench_cphd_read.py` prints `io / raw_read` (`1.549x`).
