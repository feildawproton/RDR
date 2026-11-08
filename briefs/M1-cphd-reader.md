# Brief: M1, CPHD reader and PNG writer

Written 2026-11-07 by the planning/auditing agent. This is the only context you get.
Read, in order:

1. `/home/feildaw/projects/rules4robots.md` (binding)
2. `/home/feildaw/projects/RDR/GOALS.md`
3. `/home/feildaw/projects/RDR/PLAN.md` sections 1, 2, 3/M1, 4, 6
4. `/home/feildaw/projects/RDR/briefs/M0-skeleton-audit.md` (what already exists and the
   benchmark JSON keys)
5. The CPHD standard: `/home/feildaw/projects/RDR/references/NGA.STND.0068-1_1.1.0_CPHD_DIDD_FINAL.pdf`,
   the file-format chapter (file header, XML block, support block, PVP block, signal
   block, byte order, signal array formats) and the PVP chapter. The schema
   `references/CPHD_schema_V1.1.0_2021_11_30_FINAL.xsd` is the authority on element
   names and paths. Cite section numbers in docstrings where you rely on them.

## Roles

You implement. The auditor runs the "done when" commands and reads every file. The
human commits. You do not commit.

## Scope

Milestone M1 only: read CPHD 1.1.0 files fast, and write density-remapped PNGs. No
geometry, no signal processing, no image formation. If something here needs a geometry
or signal function, stop and report; it belongs to a later milestone.

## Data

Three files in `/home/feildaw/data`, all CPHD 1.1.0, one channel, FX domain, CF8:

| file | vectors x samples | signal bytes |
| --- | --- | --- |
| `2023-09-11-10-37-05_UMBRA-05_CPHD.cphd` | 4346 x 3999 | 139 MB |
| `2023-07-30-17-19-39_UMBRA-05_CPHD.cphd` | 13152 x 9999 (brief originally said 9082; corrected after the implementer's report) | 1.05 GB |
| `2025-10-26-05-00-15_UMBRA-08_CPHD.cphd` | 8229 x 31250 | 2.06 GB |

Read-only. Never write under `/home/feildaw/data`. Machine: 15 GB RAM, RTX 3070 with
8 GB VRAM. Both limits matter for the 2 GB file.

## Deliverables

### 1. `src/rdr/io/cphd.py`

Public API (names are fixed; signatures may grow keyword arguments):

```python
class CphdFile:
    def __init__(self, path: str | os.PathLike): ...
    # Parses the file header and the XML block. Reads nothing else. Keeps the file
    # closed between calls (reopen per call, or memmap lazily); no open handle held
    # for the object's lifetime.
    header: dict[str, str]            # raw KEY := VALUE pairs from the file header
    xml: lxml.etree._Element          # root of the XML block
    meta: CphdMeta                    # see below
    channels: list[str]               # channel identifiers in Data/Channel order
    def pvp(self, channel: str) -> dict[str, np.ndarray]: ...
    def signal_memmap(self, channel: str) -> np.memmap: ...
    def signal(self, channel: str, device: str | torch.device = "cpu") -> torch.Tensor: ...
    def iter_signal(self, channel: str, vectors_per_chunk: int,
                    device: str | torch.device = "cpu") -> Iterator[tuple[int, torch.Tensor]]: ...
```

- `pvp()` returns native-byte-order arrays, one per PVP parameter including any
  `AddedPVP`, keyed by the XML name. Scalar `F8` becomes `float64` shape `(N,)`; `I8`
  becomes `int64`; `X=F8;Y=F8;Z=F8;` becomes `float64` shape `(N, 3)`. Build the
  structured dtype from the XML `PVP` block (Offset, Size, Format), not from a hardcoded
  list. Offsets are in 8-byte words; confirm against the standard and say where.
- `signal_memmap()` returns a read-only big-endian `complex64` view shaped
  `(vectors, samples)` with no copy.
- `signal()` returns a contiguous native `torch.complex64` tensor shaped
  `(vectors, samples)` on `device`. For `device="cuda"` the byte swap may happen on
  either side; measure both and keep the faster. Hard limits: never hold two full-size
  copies on the GPU at once, and host peak memory above the memmap itself must stay
  under one extra copy of the signal.
- `iter_signal()` yields `(first_vector_index, tensor)` chunks of `vectors_per_chunk`
  vectors (last chunk shorter), each native `complex64` on `device`. This is what the
  image formers will use for the 2 GB file.
- Signal formats: `CF8` is required. For `CI2`, `CI4`, `CF16`, compressed signal arrays,
  and `SignalNormal` other than what you find in these files, raise `NotImplementedError`
  with the format name in the message. Do not silently guess.
- No XSD validation on the read path. Provide a free function
  `validate_xml(xml_root, xsd_path) -> list[str]` that returns schema error strings,
  taking the XSD path as an argument because `references/` is not committed.
- Parse the XML with lxml using `{*}`-namespaced element paths. Never regex the XML.

`CphdMeta` is a frozen dataclass in the same module holding only what later
milestones need. Required fields, with the XML path each comes from documented in the
field's comment:

- `version: str` (from the file header's first line)
- `collector_name`, `core_name`, `collect_type`, `mode_type`: `str`
- `domain_type: str`, `sgn: int`, `fx_min: float`, `fx_max: float`
- `ref_ch_id: str`
- `collection_start: str`
- `srp_ecf: np.ndarray (3,)`, `iarp_ecf: np.ndarray (3,)`
- `uiax: np.ndarray (3,)`, `uiay: np.ndarray (3,)` (reference surface planar basis)
- `image_area_x1y1`, `image_area_x2y2`: `np.ndarray (2,)` or `None`
- `arp_pos: np.ndarray (3,)`, `arp_vel: np.ndarray (3,)`, `side_of_track: str`
  (ReferenceGeometry, monostatic). If the file is bistatic, set these to `None` and
  record `collect_type`.
- `num_bytes_pvp: int`, `signal_array_format: str`
- `pvp_layout: tuple[PvpField, ...]` where `PvpField(name, offset_words, size_words, format)`
- `channels: tuple[ChannelMeta, ...]` with `ChannelMeta(identifier, num_vectors,
  num_samples, signal_array_byte_offset, pvp_array_byte_offset, fxc, fx_bw, tx_pol,
  rcv_pol, signal_normal)`.

The full `xml` root stays available on `CphdFile` for anything not in `CphdMeta`.

### 2. `src/rdr/io/png.py`

```python
def density_remap(image, *, method: str = "sqrt", clip_percentile: float = 99.5) -> tuple[np.ndarray, dict]
def write_png(path, image_uint8: np.ndarray) -> None
```

- `image` is a 2D complex numpy array or torch tensor (any device). Amplitude is taken,
  clipped at `clip_percentile`, scaled to `[0, 1]`, then `sqrt` or `log` (`log` means
  `log10(1 + 100 a)` normalized, or state your own exact formula in the docstring),
  then quantized to `uint8`. The returned dict holds every parameter and the clip value
  actually used, so a PNG can be reproduced.
- `write_png` uses pillow, imported inside the function, with a clear `ImportError`
  message naming the `viz` extra if it is missing. `rdr.io` must not import pillow at
  module import time.

### 3. Tests, `tests/test_cphd_reader.py` and `tests/test_png.py`

- Data-independent unit tests: the file-header parser on a hand-written header string;
  the PVP dtype builder on a hand-written PVP XML fragment including a 3-vector field
  and an `AddedPVP`; `density_remap` on a small synthetic complex array checking the
  dtype, the shape, the clip, and that the returned dict reproduces the output.
- Data-dependent tests, skipped with a clear reason if `/home/feildaw/data` (or the
  directory in env var `RDR_DATA_DIR`) is absent: for each of the three files, the
  signal from `signal()` and every PVP array from `pvp()` must be bit-identical to
  sarkit (`sarkit.cphd.Reader.read_signal` / `read_pvps`). Mark the two large files
  with `@pytest.mark.slow` and register the marker in `pyproject.toml`; they still run
  by default, the marker is for selection only. Also check that concatenating
  `iter_signal()` chunks equals `signal()` for the small file on both cpu and cuda.
- sarkit is a test dependency only. No sarkit or sarpy import anywhere under `src/`.

### 4. Benchmark, `benchmarks/bench_cphd_read.py`

Command-line script: `python benchmarks/bench_cphd_read.py <cphd path> [--device cuda]`.
Uses `benchmarks.harness.Benchmark` with name `cphd_read_<basename stem>` (keep names
as simple identifiers; the harness does not sanitize them). Stages:

- `raw_read`: read the signal block bytes from disk into a preallocated buffer with
  plain file reads, nothing else. This is the floor to compare against.
- `header_xml`: construct `CphdFile`.
- `pvp`: `pvp(channel)`.
- `io`: `signal(channel, device)` including the byte swap and any transfer, ending on
  the tensor being resident and native on `device`.

Print each stage time as it completes (see M0 audit note 1), then the JSON path. The
page cache state matters: run it twice and record both; the second run is the number
quoted. The first run after `echo 3 > /proc/sys/vm/drop_caches` would be the cold
number, but that needs root; do not attempt it, just note that cold numbers were not
measured if that is the case.

### 5. Example, `examples/01_read_cphd_dump_png.py`

Reads the small file, prints the `CphdMeta`, dumps PNGs of (a) the raw signal
amplitude and (b) the per-vector inverse FFT along samples (range profiles,
`torch.fft.ifft` along the last axis, amplitude). Nothing more. Output goes to
`examples/output/`, which is gitignored via the `output/` rule. This is a smoke test for
the human's eyes, not a validation.

## Done when (the auditor will run exactly these)

```
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/pip install -e ".[test,viz]"
/home/feildaw/mypyenv/bin/python -m pytest -q
/home/feildaw/mypyenv/bin/python benchmarks/bench_cphd_read.py /home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_CPHD.cphd --device cuda
/home/feildaw/mypyenv/bin/python benchmarks/bench_cphd_read.py /home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_CPHD.cphd --device cuda
/home/feildaw/mypyenv/bin/python examples/01_read_cphd_dump_png.py
grep -rn "sarkit\|sarpy" src/
```

- All tests pass, none skipped when the data directory is present.
- On the second benchmark run, the `io` stage is within 2x of `raw_read`, and
  `peak_cuda_memory_bytes` is under 2.2 times the signal size in bytes.
- The example writes two PNGs and exits 0.
- The grep returns nothing.
- `check_tree` still reports zero violations; sibling imports inside `rdr.io` are
  allowed but list them in your report.

## Constraints

- Do not read code from `/home/feildaw/diffpfa` or `/home/feildaw/CLEAN_SAR`.
- Do not look at how sarkit or sarpy parse CPHD to decide behavior. Use the standard.
  Using sarkit as a test oracle is fine.
- Do not commit.
- No new runtime dependencies. pillow stays in the `viz` extra.
- No try/except that hides a failure. No out-of-memory fallbacks.
- Use `/home/feildaw/mypyenv/bin/python` explicitly.
- Byte order: the standard specifies it. Find the sentence, cite it, and build the
  dtype from that, not from "it worked on these files".

## How to report

Write `briefs/M1-cphd-reader-report.md`. For each "done when" line: the exact command,
the pasted output excerpt, or `NOT RUN`. Include both benchmark JSONs' stage tables.
Include which byte-swap path you kept for cuda and the numbers for the one you rejected.
Label anything you believe but did not check as `unverified`. List every file created
or changed, every decision the brief did not cover, and anything in the brief that was
wrong or ambiguous.
