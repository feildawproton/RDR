# Brief: M2, point-target phase history simulator

Written 2026-11-08 by the planning/auditing agent. This is the only context you get.
Read, in order:

1. `/home/feildaw/projects/rules4robots.md` (binding)
2. `/home/feildaw/projects/RDR/GOALS.md`
3. `/home/feildaw/projects/RDR/PLAN.md` sections 2, 2a, 3/M2, 4, 6
4. `/home/feildaw/projects/RDR/briefs/M1-cphd-reader-audit.md` (what exists, what to fix)
5. `src/rdr/io/cphd.py` and `src/rdr/core/conventions.py` as they are now
6. The CPHD standard, `references/NGA.STND.0068-1_1.1.0_CPHD_DIDD_FINAL.pdf`: the
   chapter that defines the FX-domain signal phase model and the per-vector parameters
   (the M1 implementer cited section 4.5 around page 55 for the phase model and
   Table 11-6 for PVPs; verify those citations yourself, do not inherit them).

## Roles

You implement. The auditor runs the "done when" commands and reads every file. The
human commits. You do not commit.

## Why this milestone matters

The simulator is the oracle for every image former that follows. If its phase model is
not exactly the CPHD model, every later validation is against the wrong answer. Be slow
here. Quote the standard's equation in the docstring with its section number, state
every simplification as a named assumption, and test the two consequences the plan
requires numerically.

## Scope

- A container type for in-memory phase history shared by reader, simulator, and later
  the image formers.
- The two-way delay geometry function.
- Synthesis of FX-domain signal for point targets given any PVP, synthetic or real.
- A synthetic straight-line spotlight trajectory generator that emits CPHD-shaped PVP.
- Multi-channel stepped-chirp parameterization of that trajectory.
- Two M1 follow-ups.

Not in scope: image formation of any kind, windows, FFT helpers (M3), antenna patterns,
range falloff, noise models beyond optional additive complex Gaussian, dwell/CPI
structure (M10), stripmap trajectories (can be added when RMA needs them).

## Deliverables

### 1. Move shared types to `rdr.core`

Create `src/rdr/core/phase_history.py`. Move `PvpField`, `ChannelMeta`, `CphdMeta`
there unchanged. `rdr.io.cphd` imports them from there and re-exports the same names
so `tests/test_cphd_reader.py` passes without edits. Add:

```python
@dataclass(frozen=True)
class PhaseHistory:
    meta: CphdMeta
    channel: str                       # identifier, must be in meta.channels
    pvp: dict[str, np.ndarray]         # native, C-contiguous, as CphdFile.pvp returns
    signal: torch.Tensor               # (num_vectors, num_samples) complex64 or complex128
    source: str                        # "cphd:<path>" or "sim:<description>"
```

`__post_init__` checks: channel exists in `meta.channels`; `signal.shape` equals that
channel's `(num_vectors, num_samples)`; every PVP array's first dimension equals
`num_vectors`; `signal` is complex. Raise `ValueError` naming the mismatch. No methods
beyond that. This is a container, not a reader; large files keep using
`CphdFile.iter_signal` directly.

Add `CphdFile.phase_history(channel, device="cpu") -> PhaseHistory` as a convenience
that calls `pvp()` and `signal()`.

### 2. `src/rdr/geometry/delay.py`

```python
def two_way_delay(tx_pos, rcv_pos, target_pos, ref_pos) -> torch.Tensor
```

All inputs float64 tensors; `tx_pos`, `rcv_pos`, `ref_pos` shaped `(V, 3)`,
`target_pos` shaped `(K, 3)`. Returns `(V, K)`: for each vector and target,
`(|tx - p| + |rcv - p| - |tx - ref| - |rcv - ref|) / c`, with `c` from
`rdr.core.conventions`. Vector math only (norms of differences). Document that this is
the quantity the CPHD model calls the time-of-arrival difference relative to the
reference point (find the standard's symbol and section). Tests: zero when the target
equals the reference; symmetric in `tx` and `rcv`; matches an independent numpy
computation written in the test on random inputs to 1e-12 relative.

`rdr.geometry` imports only `rdr.core`.

### 3. `src/rdr/sim/phase_history.py`

```python
def synthesize(
    meta: CphdMeta, channel: str, pvp: dict[str, np.ndarray], scene: Scene, *,
    device="cpu", out_dtype=torch.complex64, vectors_per_chunk: int = 256,
    noise_std: float | None = None, seed: int | None = None,
) -> PhaseHistory
```

- Frequency grid per vector: `f(v, s) = SC0[v] + s * SCSS[v]` from the PVP. Confirm in
  the standard that this is the FX-domain sample frequency definition and cite it.
- Phase model: for each target `k` with complex amplitude `A_k`, the contribution is
  `A_k * exp(j * 2 * pi * SGN * f(v, s) * dTOA_k(v))` with `dTOA_k` from
  `two_way_delay` using the PVP's `TxPos`, `RcvPos`, `SRPPos`, and `SGN` from
  `meta.sgn`. Quote the standard's equation and state these named assumptions in the
  docstring: the Doppler micro-parameters (`aFDOP`, `aFRR1`, `aFRR2`) are treated as
  zero; no antenna pattern; no range falloff; no tropospheric delay. If the standard's
  equation has any other term, name it and say it is dropped.
- Phases accumulate in float64, summed over targets in complex128, cast to `out_dtype`
  at the end. Chunk over vectors.
- Optional noise: circular complex Gaussian with the given standard deviation per
  sample, seeded. Off by default.
- Output `PhaseHistory` with `source="sim:..."`.

`rdr.sim.scene`:

```python
@dataclass(frozen=True)
class PointTarget: position_ecf: np.ndarray; amplitude: complex = 1.0 + 0.0j
@dataclass(frozen=True)
class Scene: targets: tuple[PointTarget, ...]
def scene_from_plane_offsets(meta: CphdMeta, offsets_xy: np.ndarray, amplitudes=None) -> Scene
```

`scene_from_plane_offsets` places targets at `IARP + x * uIAX + y * uIAY` using the
reference surface in `meta`. This works identically for synthetic and real metadata.

### 4. `src/rdr/sim/trajectory.py`

```python
def straight_line_spotlight(
    *, srp_ecf, up_unit, velocity_ecf, standoff_range_m, grazing_deg,
    num_vectors, dwell_s, fxc_hz, fx_bw_hz, num_samples,
    collector_name="RDR-SIM", t0_s=0.0, sgn=-1,
) -> tuple[CphdMeta, dict[str, np.ndarray]]
```

- Builds a constant-velocity monostatic spotlight collection steered at `srp_ecf`.
  The platform position at the aperture center is at `standoff_range_m` from the SRP
  with the given grazing angle relative to the plane normal to `up_unit`, on the side
  such that `velocity_ecf` is broadside at aperture center. This is the one place trig
  on a user-facing angle is allowed; everything downstream is vectors.
- PVP arrays, all `(V,)` or `(V, 3)` float64 except `SIGNAL` int64: `TxTime`, `TxPos`,
  `TxVel`, `RcvTime`, `RcvPos`, `RcvVel`, `SRPPos`, `aFDOP = 0`, `aFRR1 = 0`,
  `aFRR2 = 0`, `FX1`, `FX2`, `TOA1`, `TOA2`, `TOAE1`, `TOAE2`, `TDTropoSRP = 0`, `SC0`,
  `SCSS`, `SIGNAL = 1`. Receive time is transmit time plus the two-way delay to the SRP;
  receive position is the platform position at receive time (constant velocity). Say
  so in the docstring; stop-and-hop is `RcvPos = TxPos`, and we are not doing that.
- `SC0` and `SCSS` constant across vectors, centered so sample `num_samples // 2` is at
  `fxc_hz`. `FX1`, `FX2` are the band edges. `TOA1`, `TOA2`, `TOAE1`, `TOAE2` are set
  from `1 / SCSS` symmetric about zero; document the formula you use.
- `CphdMeta`: one channel named `"ch0"`, `domain_type="FX"`, `collect_type="MONOSTATIC"`,
  `mode_type="SPOTLIGHT"`, `uiax`/`uiay` an orthonormal pair perpendicular to `up_unit`
  with `uiax` along the ground projection of velocity, `iarp_ecf = srp_ecf`,
  `arp_pos`/`arp_vel` at aperture center, `side_of_track` derived from the geometry by a
  cross product (not from an input flag), `pvp_layout` describing the arrays above with
  offsets in 8-byte words, `num_bytes_pvp` consistent with it.

```python
def stepped_chirp_channels(
    meta: CphdMeta, pvp: dict, *, num_subbands: int, subband_bw_hz: float,
    step_delay_s: float, samples_per_subband: int,
) -> tuple[CphdMeta, dict[str, dict[str, np.ndarray]]]
```

Takes a single-channel synthetic collection and returns a `CphdMeta` with
`num_subbands` channels `"sb0"`, `"sb1"`, ... each with its own `FxC`, `FxBW`, `SC0`,
`SCSS`, `FX1`, `FX2`, and `TxTime` offset by `i * step_delay_s` with positions moved
accordingly along the velocity, plus a dict of per-channel PVP. Adjacent subbands abut
with no overlap. `synthesize` is then called per channel. Keep this modest; M9 is
where it gets exercised.

### 5. Tests, `tests/test_sim.py` and `tests/test_geometry_delay.py`

All with `out_dtype=torch.complex128` so phase is measured at float64 precision.

- **Flat phase at the reference.** One target at `SRPPos`, synthetic trajectory: the
  phase of every sample is zero to `1e-9` rad.
- **Linear phase for a range offset.** One target at `SRP + d * u` where `u` is the
  unit vector from `TxPos[v0]` to `SRP` and `d = 7.3 m`. For vector `v0`, the unwrapped
  phase across samples is linear with slope `2 * pi * SGN * dTOA * SCSS[v0]` per sample,
  where `dTOA` is computed in the test with its own numpy from the positions, not from
  `rdr.geometry`. Relative slope error under `1e-9`. Also check the sign: with
  `SGN = -1` and the target farther than the SRP, the slope is negative.
- **Linearity.** Two targets equals the sum of the two single-target signals to
  float64 roundoff.
- **Range-profile peak location.** One target at `d = 7.3 m`: after `ifftshift` along
  samples, `ifft` (for `SGN = -1`), `fftshift`, the peak column is at
  `num_samples // 2 + round(dTOA * num_samples * SCSS)` within 0.5 sample. State in the
  test that this convention is local to the test until M3 pins it.
- **Hybrid on real geometry.** Using `CphdFile` on the small Umbra file, `meta` and
  `pvp` from the file, one target at `meta.srp_ecf`: flat phase to `1e-9` rad, and the
  output shape equals the file's channel shape. Skip with a reason if the data directory
  is absent.
- **Trajectory sanity.** `|TxPos - SRP|` at the center vector equals `standoff_range_m`
  within `1e-6` m; `TxVel` equals `velocity_ecf`; `RcvTime - TxTime` equals the two-way
  SRP delay; `SIGNAL` all ones; `pvp_layout` round-trips through `build_pvp_dtype` with
  `num_bytes_pvp`; `side_of_track` flips when `velocity_ecf` is negated.
- **Stepped chirp.** Subband frequency grids abut with no gap and no overlap; `TxTime`
  of subband `i` is offset by `i * step_delay_s`; synthesizing all subbands for a target
  at the SRP gives flat phase in each.
- **Container.** `PhaseHistory` rejects a wrong-shaped signal and a PVP of the wrong
  length with `ValueError`.
- **Delay function.** The three tests listed under deliverable 2.
- Import hierarchy: zero violations; list any sibling imports in your report.

### 6. Example, `examples/02_simulate_point_targets.py`

Synthetic spotlight, three targets (one at the SRP, two offset in the plane), dump the
raw signal amplitude PNG and the centered range-profile PNG into `examples/output/`.
Then the hybrid case on the small Umbra file with the same three plane offsets, same
two PNGs. Print the parameters used. Nothing is formed into an image.

### 7. M1 follow-ups

- **CPU `signal()` path.** Measured by the auditor on the 2 GB file, same cache state,
  alternated: `signal(device="cpu")` 1.57 s and 1.18 s; `np.ascontiguousarray(memmap).astype(complex64)`
  0.80 s and 0.41 s. Replace the CPU path with whatever you measure fastest, keep the
  sarkit bit-equality tests unchanged, and report your before and after numbers.
- **Unverified number in `examples/01_read_cphd_dump_png.py`.** The docstring says the
  Doppler micro-parameters shift range peaks by under 0.25 sample. Either compute that
  bound in the script from the file's `aFDOP`, `aFRR1`, `aFRR2`, `TOAE1`, `TOAE2`,
  `SCSS` and print it, or delete the number. No unverified figures in docs.

## Done when (the auditor will run exactly these)

```
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/pip install -e ".[test,viz]"
/home/feildaw/mypyenv/bin/python -m pytest -q
/home/feildaw/mypyenv/bin/python examples/02_simulate_point_targets.py
/home/feildaw/mypyenv/bin/python examples/01_read_cphd_dump_png.py
grep -rn "sarkit\|sarpy" src/rdr/
/home/feildaw/mypyenv/bin/python -c "from rdr.io.cphd import CphdMeta, ChannelMeta, PvpField; from rdr.core.phase_history import CphdMeta as C; print(CphdMeta is C)"
```

- All tests pass, none skipped when the data directory is present. The M1 tests are
  unchanged and still pass.
- Example 02 writes four PNGs and exits 0. Example 01 still exits 0 and no longer
  contains an uncomputed numeric claim.
- The grep returns nothing. The last command prints `True`.
- `check_tree` reports zero violations.

## Constraints

- Do not read code from `/home/feildaw/diffpfa` or `/home/feildaw/CLEAN_SAR`.
- Do not derive the phase model from sarkit or sarpy source. Use the standard. Using
  sarkit to read the Umbra file in tests is fine but `CphdFile` is what the hybrid test
  should use.
- Do not commit. Do not write under `/home/feildaw/data`.
- No new runtime dependencies. No try/except that hides failure.
- Use `/home/feildaw/mypyenv/bin/python` explicitly.
- Trig is allowed only inside `straight_line_spotlight` to turn the grazing angle input
  into vectors. Nowhere else in this milestone.
- Every simplification of the standard's model is a named assumption in a docstring.

## How to report

Write `briefs/M2-simulator-report.md`. For each "done when" line: the exact command, the
pasted output excerpt, or `NOT RUN`. Quote the standard's phase-model equation as you
implemented it, with section and page. Give the before and after CPU-path numbers.
Label anything believed but not checked as `unverified`. List every file created or
changed, every decision the brief did not cover, and anything in the brief that was
wrong or ambiguous.
