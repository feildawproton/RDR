# RDR plan

Status: draft v1, 2026-11-07. Written by the planning/auditing agent after reading
`rules4robots.md`, `GOALS.md`, the NGA standards in `references/`, the data in
`/home/feildaw/data`, and the two older repos (`diffpfa`, `CLEAN_SAR`) as clues only.

Decisions confirmed with the human (2026-11-07):

- Differentiability: gradients must flow back to the **signal tensor** and, after the
  autofocus discussion below, to an optional **per-vector phase correction**. Geometry
  and PVP are constants.
- PFA is to be built from **reusable pieces** so a demo can show why uneven cross-range
  sample spacing may force a NUFFT.
- "Fast" CPHD reading: the **2 GB Umbra-08 file is the benchmark**. Loading and writing
  must not dominate processing.
- No SICD writing for now. **Density-remapped PNG** is enough to eyeball results.
- CLEAN in **PyTorch**.
- No GUI yet. Examples dump files.
- `references/` (standards, schemas, SIX source) is **never committed**.
- **Backprojection** is in scope (added to GOALS.md); it is the easiest to understand and
  serves as the cross-check for PFA and RMA.
- Vendor processing is opaque, so validation leans on **simulation with known answers**,
  not on matching Umbra's SICD numerically.

Further decisions confirmed later the same day:

- Package name **`rdr`**.
- **Full-resolution** images from the 2 GB file are feasible on this machine. The pattern
  the human used on larger multi-channel data and liked: hold one **results tensor** on
  the GPU, move **one channel at a time** to the GPU, process it, accumulate into the
  results tensor, delete it. Within a channel, **batch** the interpolation and FFT work
  over pulses or range bins because those steps need large intermediates.
- **IFP / IFA split.** The image formation *processor* (pipeline layer) decides the image
  plane and iterates channels. The image formation *algorithm* (imaging layer) is handed
  two basis vectors and a results tensor and does not know or care whether the plane is
  slant or ground. Slant plane is almost always what we want, but that choice lives in
  the IFP.
- Geometry is done with **vector math**, projections and cross products, **not trig**.
  This is a preference of the human's and should be followed throughout.
- No SICD reader until CLEAN (M8 after the autofocus insertion).
- The AWS CLI is installed in the venv (`/home/feildaw/mypyenv/bin/aws`, aws-cli 1.46.0,
  verified 2026-11-07). An earlier statement in this plan that it was missing was wrong;
  it had only been checked on the system PATH. See the new rule in `rules4robots.md`.

Decisions from the autofocus discussion (2026-11-07, evening):

- **Gradient-based autofocus is in scope** and is the main reason to keep the formers
  differentiable. It is an **optional stage** the user adds to their processing, never
  on by default.
- The human's workflow names unfocused products **SICDU** and focused ones **SICDF**.
  The pipeline must be able to produce SICDU for architectures that expect it, SICDF,
  or both from one run. The standard itself records focusing in
  `ImageFormation/AzAutofocus` and `RgAutofocus` with values NO, GLOBAL, SV; the names
  SICDU and SICDF were not found in any of the three SICD 1.5 documents (text search,
  2026-11-07) and are treated as filename suffixes only.
- The original motivation for differentiability was gradient-based work against SAR
  ATR models with the goal of carrying the result back to the signal domain. That is
  not a milestone here; it is context for why gradients to the signal tensor matter.
- **Learning is the primary purpose.** The autofocus milestone's example is a
  walkthrough that builds intuition before it shows a result. Whether a product comes
  out is secondary.

## 1. What we are working with

Data in `/home/feildaw/data` (not in the repo):

| file | channels | domain | vectors x samples | signal bytes |
| --- | --- | --- | --- | --- |
| 2023-09-11 UMBRA-05 | 1 | FX | 4346 x 3999 | 139 MB |
| 2023-07-30 UMBRA-05 | 1 | FX | 13152 x 9999 | 1.05 GB |
| 2025-10-26 UMBRA-08 | 1 | FX | 8229 x 31250 | 2.06 GB |

All are CPHD 1.1.0, spotlight, monostatic, CF8 (big-endian complex float32), with a
matching vendor SICD. None is stepped-chirp or stripmap; those must be simulated.
(Dimensions corrected 2026-11-08: the 1 GB file was listed as 9082 samples from a
regex grab of the XML; the M1 implementer caught it, and an lxml path query confirms
9999. Every row above now matches `NumVectors * NumSamples * 8 == SIGNAL_BLOCK_SIZE`.)
The two 2023 files have 1 and 431 vectors with `SIGNAL == 0`; the 2025 file has none.
More Umbra open data (stripmap, for example) can be listed and downloaded with
`/home/feildaw/CLEAN_SAR/tools/list_latest_umbra.py` (a copy also exists under
`diffpfa/tools`). It shells out to `aws`, which is in the venv. The human ran it
successfully on 2026-11-07.

Machine: RTX 3070 Laptop, 8 GB VRAM, 15 GB RAM, torch 2.6 + CUDA 12.4, nvcc 12.0,
gcc 13.3, no cmake. Python venv `/home/feildaw/mypyenv`.

Baseline I/O measurements taken today on the 2 GB file (page cache warm unless noted):

| path | time |
| --- | --- |
| raw disk read, cold | 2.8 s (746 MB/s) |
| raw disk read, cached | 0.25 s |
| numpy memmap + CPU byteswap to native complex64 | 0.67 s |
| pageable host to GPU copy of 2 GB complex64 | 0.42 s |
| sarkit open + read_signal + read_pvps | 2.65 s |
| sarpy open + full read | 4.45 s |

So sarkit is roughly 4x the memmap path and sarpy roughly 7x. A pinned-memory plus
GPU-byteswap variant measured 5.6 s, but that run competed with the sarpy run for page
cache, so treat it as unmeasured. The benchmark harness in M1 settles it properly.

Implication for design: the 2 GB signal alone is a quarter of VRAM, so nothing can
materialize a second full-size intermediate next to it. The design that handles this,
confirmed by the human from experience on larger data, is:

1. The IFP allocates the results tensor once (the combined k-space grid for PFA, the
   image accumulator for backprojection) sized from the union of all channels' support.
2. Each channel is moved to the GPU, processed, accumulated into the results tensor, and
   freed before the next.
3. Inside a channel, interpolation and FFT steps run in batches (over pulses for the
   range step, over range bins for the cross-range step) because Bluestein padding and
   gridding intermediates are several times the batch size.
4. Only the final inverse FFT touches a full-size grid.

Full resolution from the 2 GB file is the target, not a stretch goal. A sub-aperture
mode is still useful for quick looks and for the examples, but it is not the fallback
for memory.

## 2. Package shape

Package name `rdr`, src layout, installable with `pip install -e .`.

```
RDR/
  pyproject.toml
  src/rdr/
    core/        L0  constants, dtypes, device selection, small tensor utilities
    io/          L1  cphd reader, sicd reader (minimal), png writer
    geometry/    L1  ECF/WGS84, SRP, image plane basis, PVP -> k-space coordinates
    signal/      L1  windows, sinc interp, CZT, NUFFT, fft helpers, resampling
    sim/         L2  point-target phase history synthesis, stepped chirp, dwells/CPIs
    imaging/     L2  backprojection, pfa, rma   (composed from geometry + signal)
    deconv/      L3  IPR model, CLEAN
    autofocus/   L3  phase-error models, sharpness metrics, gradient-based autofocus
    pipeline/    L4  file -> image -> png glue, optional autofocus stage, CLI entry points
  examples/      scripts (notebooks later) that dump PNGs and plots
  benchmarks/    timing scripts that write JSON results
  tests/
  references/    gitignored
```

Import rule, enforced by a test: a module may import from `rdr.core` and from any
package at a **strictly lower** layer number. Never from the same layer, never upward.
`tests/test_import_hierarchy.py` walks the AST of every module and fails on violations.
This is the auditor's main structural tool, so it goes in at M0.

Notes on the layering:

- `imaging` does not import `io`. Image formers take tensors and a small metadata
  dataclass, never a file. That is what lets the simulator and the real reader feed the
  same code.
- `imaging` is the IFA. Its functions take: a channel's signal tensor, that channel's
  PVP arrays, two image-plane basis vectors in ECF, the image extent along each basis
  vector, and the shared results tensor to accumulate into. They return nothing of
  consequence beyond what they wrote into the results tensor. The IFP in `pipeline`
  picks the basis vectors (slant plane from LOS and velocity, or the CPHD reference
  surface for ground), iterates the channels, and runs the final inverse FFT.
- `geometry` is vector math only: dot products, cross products, projections, norms.
  No `atan2`, no `sin`/`cos` of an angle that was itself computed from vectors. The
  polar "angle" of a pulse is represented by its projections onto the basis vectors.
- `sim` and `imaging` sit at the same layer and must not import each other. The
  shared piece they both need, the forward model of range to a scatterer given PVP, lives
  in `geometry`.
- `deconv` needs the IPR, which depends on k-space support. The IPR model is computed
  from `geometry` plus `signal`, so `deconv` may import `imaging` for the k-space grid
  description but not for the image formers themselves. Revisit if that gets awkward.
- `autofocus` sits at layer 3 beside `deconv` and must not import it. It imports
  `imaging` because it forms images under a candidate phase correction and
  differentiates through them. It owns the phase-error parameterizations and the
  sharpness metrics. The IFP in `pipeline` decides whether to call it.

Runtime dependencies: numpy, torch, lxml. Optional: matplotlib, pillow. Test and
benchmark only: pytest, scipy, sarkit (used purely as an independent reader to compare
against). No sarpy dependency.

## 2a. Partial formation: terms and rules

A previous attempt at chipping broke image formation, caused confusion, and degraded
performance. So the terms are pinned here, and every partial mode is opt-in, tested
against a reference, and benchmarked. None of them is ever the default.

Terms:

- **Full-scene formation.** The whole image from the whole phase history, referenced to
  the CPHD SRP. Always built first. Everything below is compared against it.
- **Range gate.** After the range resample and a range IFFT, keep a band of range cells
  and every pulse. It is a data block, not an image product. It still spans the full
  aperture, so the whole slow-time phase error is observable in it. Used for autofocus
  estimation only.
- **Image chip.** A rectangular pixel region of the full-scene image, same reference
  point, same geometry. Two ways to get it: slice the full-scene image (the reference),
  or evaluate the final transform over only that extent using the CZT. The second must
  match the first to a stated tolerance, or it is wrong.
- **Subpatch.** A chip formed after re-referencing the phase history to the chip's own
  center. It differs from the full-scene chip on purpose, because it has less PFA
  wavefront-curvature error. Validated against backprojection and the simulator, never
  against the full-scene PFA chip.
- **Sub-aperture.** A subset of pulses. Lower cross-range resolution. Second choice for
  autofocus estimation because it sees only part of the slow-time error.

Rules:

1. Full-scene formation is accepted before any partial mode is started.
2. Each partial mode is its own deliverable with its own reference, test, and benchmark.
   The benchmark reports the partial mode's time next to the full-scene time on the
   same file, so a slowdown is a number, not a feeling.
3. Partial modes are functions a caller chooses, never a code path the IFP takes on its
   own because the data is big.
4. Every partial function's docstring says which of the five terms it implements and
   which reference it is tested against.

## 3. Milestones, in build order

Each milestone has a definition of done. Nothing moves to the next milestone with a
failing or skipped test. Results that were not actually run are not reported.

### M0. Skeleton

- pyproject, src layout, empty packages with docstrings stating their layer.
- `.gitignore` excluding `references/`, outputs, caches.
- Import hierarchy test.
- `benchmarks/` with a tiny harness that records wall time, peak GPU memory, and
  io_time / compute_time ratio to JSON.

Done when `pytest` passes on an empty package and the hierarchy test fails on a planted
violation.

Status: **accepted 2026-11-07**, see `briefs/M0-skeleton-audit.md`.

### M1. CPHD reader and PNG writer

Reader design:

- Parse the text file header (key := value lines) ourselves.
- Parse the XML block with lxml using proper element paths, not regex. (A regex grab
  today returned NumSamples=2810 for a file whose channel has 3999; the tag appeared
  earlier in a different context. Paths only.)
- Expose a small `CphdMeta` dataclass with only what imaging needs: channel ids, shapes,
  FX/TOA domain, FxC, FxBW, SCP/SRP, reference geometry, PVP layout. Keep the full lxml
  tree available for anything else.
- PVP block: numpy structured dtype built from the XML PVP layout, memmapped, byteswapped
  once, returned as a dict of native arrays.
- Signal block: three access modes. (a) numpy memmap view in big-endian, no copy.
  (b) chunked iterator over vector ranges yielding native complex64 torch tensors on a
  chosen device. (c) whole-channel load for files that fit.
- No XSD validation on the read path. Offer `validate(meta, xsd_path)` separately.

Writer: density remap (amplitude, log or sqrt, percentile clip) to 8-bit PNG, with the
remap parameters returned so examples can be reproduced.

Done when: signal and PVP arrays are bit-identical to sarkit on all three files; the
benchmark shows the 2 GB file resident on GPU as native complex64 within 2x of the raw
disk read time, and reports which stage dominates.

Status: **accepted 2026-11-08**, see `briefs/M1-cphd-reader-audit.md`. Measured: 2 GB
file GPU-resident native complex64 in 0.50 s against a 0.30 s warm raw read (1.3x to
1.7x), peak CUDA memory 1.098x the signal. Two follow-ups (slow CPU path, one unverified
number in the example) carried into M2.

### M2. Simulator

Built before any image former, because it is the oracle.

- Point-target scene in a local tangent plane around an SRP, arbitrary RCS and phase.
- Platform trajectory given as positions and times, or synthesized straight-line.
- Produces an in-memory object with the same fields as `CphdMeta` plus PVP and a signal
  tensor, in FX domain, monostatic, phase referenced to the SRP exactly as CPHD defines it
  (DIDD section on compensated phase history). Spotlight and stripmap modes.
- Stepped chirp: N subbands as N channels with their own FxC and FxBW, with configurable
  inter-step delay, so the coherent-combination problem is real.
- Dwell, sub-dwell, CPI parameterization comes later but the data model should not
  preclude it: a collection is a list of channels, a channel is vectors over time.

Done when: a single point target at the SRP produces a signal whose phase versus
frequency is flat to float64 roundoff, and a target offset in range produces the
predicted linear phase ramp. Both checked numerically, not by eye.

### M3. Signal pieces

Each piece is a pure torch function, differentiable with respect to its data input,
with a numpy or scipy reference test and a float64 `gradcheck`.

- Windows: uniform, Taylor, Hamming, Hann, as continuous functions evaluated on a grid.
- 1D sinc (and windowed-sinc) interpolation from arbitrary input positions to a uniform
  grid and back, batched over the other dimension.
- CZT via Bluestein. Test against direct DFT evaluation on the same contour. The
  interface takes an arbitrary output start and step, so later a chip's pixels can be
  evaluated directly from k-space. Test that evaluating a sub-extent equals slicing the
  full transform.
- NUFFT type 1 and type 2 with Kaiser-Bessel gridding, written here rather than pulled
  from a library, because showing the gridding step is part of the educational goal.
  Test against a direct nonuniform DFT sum on small sizes.
- FFT helpers, zero padding, centering conventions, all in one place so sign and shift
  conventions are decided once.

Done when every function has a reference test, a gradcheck, and a documented convention.

### M4. Backprojection

Time-domain backprojection from FX-domain data: inverse FFT each pulse to range
profiles, upsample, interpolate each pixel's range, apply phase, accumulate. Chunked over
pulses so memory is bounded. This is the slow but trustworthy former.

The chunk loop takes a `checkpoint` flag. When set, each chunk's forward pass is
wrapped in `torch.utils.checkpoint` so backward recomputes it instead of storing it,
and the gradient to the signal costs one chunk of memory instead of all of them. Off by
default. Test: the signal gradient with the flag on equals the gradient with it off on
a small problem. This is what the ATR-style use needs; autofocus does not.

Done when: on a simulated multi-target scene, peak positions land within a fraction of a
resolution cell of truth and measured IPR widths match the closed-form values for the
window used. Then run on the 139 MB Umbra file at reduced resolution and dump a PNG next
to the vendor PNG for a sanity look.

### M5. PFA from pieces

Stages, each a separate function so examples can swap them:

0. Re-referencing, in `geometry`. Given the signal, PVP, and a new reference point,
   multiply each (pulse, sample) by the phase of the range difference between the new
   and old reference points for that pulse, and return the new per-pulse reference
   positions for stage 1. Exact, cheap, differentiable, and a constant with respect to
   the signal. Tests: re-referencing to the current SRP is the identity; a simulated
   point target at the new reference comes out with flat phase across frequency. Used
   by subpatches (M5b), autofocus range gates (M6), and RMA (M7). Not used in
   full-scene formation.
1. PVP to polar k-space coordinates. For each pulse, the vector from the phase center
   (midpoint of TxPos and RcvPos for monostatic) to the SRP is projected onto the two
   basis vectors and divided by its norm, giving the direction cosines without trig.
   Each sample's spatial frequency is 2 f / c, with f from SC0 and SCSS. Lives in
   `geometry`.
2. Range (radial) resampling onto a uniform keystone grid: sinc interpolation, and a CZT
   variant that does the same resampling as a scaled transform. Both must agree.
3. Cross-range (angular) resampling: sinc on the assumption of uniform polar angle
   spacing, and NUFFT type 1 from the true nonuniform angles. The demo plots the actual
   angle spacing from the Umbra PVP, shows where the uniform assumption breaks, and shows
   the resulting image difference.
4. 2D inverse FFT with window, in the plane defined by the basis vectors handed in.

Stages 2 and 3 write into the shared k-space results tensor; stage 4 is run once by the
IFP after all channels are accumulated. Stages 2 and 3 are batched internally.

Between stages 2 and 3 there is a **per-vector weight hook**: an optional complex
tensor shaped `(vectors,)` multiplied into each pulse before the cross-range step.
Autofocus (M6) supplies `exp(-j phi[n])` through it. The hook must be differentiable
with respect to the weight. Backprojection (M4) gets the same hook applied per pulse
before accumulation. Both are designed in now so M6 does not have to reopen M4 or M5.

Done when: PFA and backprojection agree on simulated scenes to within the expected
PFA wavefront-curvature error for the scene size, and on the Umbra files the PNG looks
like the vendor's. Also compare our derived values against the vendor SICD XML where
the meaning is unambiguous: ImpRespBW, ImpRespWid, sample spacings, image plane basis
vectors. That is a cheap check that does not require understanding their pipeline.

The PFA chunk loop gets the same `checkpoint` flag and test as backprojection.

### M5b. Partial formation, after M5 is accepted

Each item is separate, opt-in, and lands only after full-scene PFA is accepted. Terms
are the ones in section 2a.

- **Image chip by CZT extent.** Evaluate the final transform over a given pixel extent
  from the full k-space. Reference: slice of the full-scene image. Test: equal to a
  stated tolerance. Benchmark: chip time versus full-scene time on the 139 MB and 2 GB
  files.
- **Range gate.** Range resample, range IFFT, keep a band of cells, all pulses.
  Reference: the same band sliced from the full range-compressed data. Test: equal.
- **Subpatch.** Re-reference to the patch center, form the patch with PFA. Reference:
  backprojection of the same patch, and a simulated point target placed at the patch
  center and at the patch edge. Test: peak location and IPR width within tolerance,
  and the subpatch error at the patch edge is smaller than the full-scene PFA error at
  the same pixel. Benchmark: N subpatches versus one full scene.
- A mosaic of subpatches is **not** in this milestone. It is listed so nobody builds it
  by accident.

Done when each item above has its test and its benchmark JSON, and an example writes
side-by-side PNGs of the full-scene chip, the CZT chip, and the subpatch for the same
region with the difference images.

### M6. Gradient-based autofocus

Why: the formers are differentiable, so autofocus becomes "find the slow-time phase
that makes the image sharpest," solved by autograd. This is the milestone the human
most wants to learn from, so the example is a walkthrough, not a demo.

- Phase error model in `rdr.autofocus`: a per-vector phase `phi[n]` applied as
  `exp(-j phi[n])` through the hook in M4/M5. Two parameterizations: a low-order
  polynomial in slow time (quadratic, cubic; few parameters, well conditioned) and a
  free per-pulse vector (many parameters, needs a good initialization). Do polynomial
  first, then free per-pulse initialized from the polynomial result.
- Unobservable terms: a constant phase and a linear phase (a pure image shift) do not
  change sharpness. Remove them from `phi` after every step and from the truth before
  comparing, and explain why in the example.
- Metrics: image entropy and an intensity-sharpness metric (normalized sum of
  intensity squared), both differentiable in torch. The example first plots the metric
  as a 1D function of a quadratic coefficient so the landscape is seen before any
  gradient step is taken.
- Optimizer: L-BFGS for the polynomial parameters, Adam for the free per-pulse vector.
  Record the metric and the residual phase per iteration.
- Memory: autograd must hold the graph for every chunk the image depends on, which
  conflicts with channel-at-a-time accumulation on the 2 GB file. Approach: estimate
  `phi` on a **range gate** (section 2a) that forms in one shot, then apply `phi` to
  the full-resolution formation with no grad. For the error classes we start with,
  `phi` is common across range, so a range-gate estimate is valid globally. The
  `checkpoint` flag from M4/M5 is the fallback if a full-resolution estimate is ever
  needed.
- Reference point inside the gate: PFA's residual wavefront-curvature error grows with
  distance from the reference point and looks like a deterministic, mostly quadratic
  phase error. If the gate holds bright scatterers far from the SRP, the optimizer
  absorbs that residual into `phi`, and applying `phi` globally then misfocuses the
  scene center. So the gate is **re-referenced to its own center** (M5 stage 0) before
  estimation, and the estimate is transported back. Test: inject a known `phi`, put the
  gate well away from the SRP, recover `phi` within tolerance; then repeat without
  re-referencing and show the bias, so the reason is visible in the example.
- Pipeline integration: autofocus is an optional IFP stage. One run can produce the
  unfocused image (SICDU), the focused image (SICDF), or both; for PFA the correction
  applies after the range resample, so both outputs share that work and differ by one
  cross-range transform. When the SICD writer exists, SICDF files set
  `AzAutofocus` to GLOBAL (or SV if the correction becomes range-variant) and SICDU
  files set it to NO.
- The argument for focusing here rather than downstream: the correction is a vector
  along slow time, it lives naturally with the phase history, applying it costs one
  extra cross-range transform, and the estimate can use the phase history rather than
  an already-formed image. Downstream tools that expect SICDU still get it. This
  argument gets tested rather than asserted: once the SICD writer exists, compare an
  image-domain correction applied to the SICDU against the phase-history-domain
  correction on the same data.
- Optional, for learning only: phase gradient autofocus as the classic baseline to
  compare against. Not required for "done."

Done when: on simulated scenes with injected quadratic, cubic, sinusoidal, and
random-walk `phi`, the recovered `phi` matches truth after removing constant and linear
terms to a tolerance stated in the test, with metric-versus-iteration and residual
plots written by the example; on the small Umbra file with an injected `phi` (the vendor
data is already focused, so a known error has to be added to have a known answer), the
same recovery holds; the gradient with respect to the signal tensor still passes its
check with the hook active.

### M7. RMA (omega-k)

Validated primarily on simulated stripmap, since that is where RMA is natural. The Stolt
interpolation reuses the sinc interpolator from M3. Applying RMA to the Umbra spotlight
data requires re-referencing the SRP-compensated phase history to a line (undo the SRP
motion compensation, then reference to a straight track). That re-referencing step is a
risk and may be a milestone of its own.

### M8. CLEAN in torch

- IPR model from k-space support and window, space-variant for PFA output. Validate the
  model by forming a simulated single point target and comparing with the model's
  prediction, not by assuming the model is right.
- Hogbom loop with complex gain, stopping on threshold or iteration count, restoring beam
  matched to the IPR 3 dB width.
- Validate on two simulated targets closer than a resolution cell. Then run on a chip of
  the Umbra SICD read by the minimal SICD reader, which is built in this milestone and
  not before.

### M9. Stepped chirp

Coherent combination of simulated subband channels before PFA, compared against the
monolithic full-band reference from the same simulator. Inter-step delay and LO phase
behavior are the interesting knobs.

### M10. Dwells, sub-dwells, CPIs

Later, per GOALS.md.

## 4. Validation philosophy

- The simulator is the oracle. Closed-form expectations (peak position, IPR width,
  sidelobe level for a known window) are the tests.
- Backprojection is the cross-check for PFA and RMA.
- Vendor SICDs are for eyeballing PNGs and for comparing metadata values whose meaning is
  defined by the SICD standard, not for numerical image matching.
- Every image former gets a gradient test: loss on the image, backward to the signal
  tensor and to the per-vector weight hook, finite-difference check on a tiny problem
  in float64.
- Autofocus is validated by injecting a known slow-time phase error in the simulator,
  and by injecting one into already-focused real data, then measuring recovery after
  removing the unobservable constant and linear terms.
- Benchmarks are run, recorded as JSON, and quoted from the file. Numbers in docs come
  from a run, with the date.

## 5. Risks and open questions

Risks:

- 8 GB VRAM versus the 2 GB file. Chunking is mandatory and will shape every former.
- NUFFT accuracy and speed written from scratch. Mitigation: small, tested, compared
  against a direct sum; speed only matters after correctness.
- RMA on SRP-compensated spotlight data (see M7).
- Autograd through chunked, channel-at-a-time formation holds every chunk's graph.
  Mitigation in M6: estimate the correction on a range gate that forms in one shot,
  apply it globally without grad; the `checkpoint` flag if that proves insufficient.
- Partial formation broke a previous attempt. Mitigation: section 2a. Full scene first,
  one partial mode at a time, each with a reference, a test, and a benchmark, none the
  default. If a partial mode cannot be made to match its reference, it is dropped, not
  tolerated.
- Convention drift (FFT sign, k-space handedness, row/column order). Mitigation: one
  conventions module, one document, tests that pin them.

Open questions for the human: none at the moment. The four from draft v1 were answered
on 2026-11-07 and folded into the decisions list above.

## 5a. Clues taken from diffpfa (read 2026-11-07, with the human's caution in mind)

Worth keeping as design clues, not as code:

- The channel loop: compute per-channel k-space extents first, take the union to size
  the grid, allocate the combined grid once, then per channel do range resample, then
  cross-range gridding, then add into the grid, then free. This is the pattern the human
  wants.
- Batching the range CZT over pulses (it used batches of 256) and the cross-range
  gridding over range bins. The CZT pads to the next power of two above N plus M and
  runs in complex128, so a batch is many times the input size.
- k-space from projections onto the basis vectors with no trig. The slant basis was
  LOS as one axis and velocity with its LOS component removed as the other, sign set by
  side of track.
- CZT phases evaluated in float64. The code's own note says float32 phases at these
  magnitudes give tens of milliradians of error per sample. Worth a test of our own
  rather than taking on faith.
- Keeping the k-space bounds and centers from the former so the writer could fill
  SICD Grid fields later. Cheap to return, so return them.

Things that looked suspicious or confusing and should be tested rather than copied:

- The cross-range gridding ran with no oversampling of the grid by default. Kaiser-Bessel
  gridding normally oversamples by 1.25 to 2 so the kernel's spatial taper does not alias
  into the image. The post-correction divided by the taper anyway. Test the gridding
  error against a direct nonuniform DFT at several oversampling factors.
- The range resample went k-space to spatial domain by CZT, conjugated, and back to a
  uniform k-space by a second CZT. It works but it is two transforms and a conjugation
  for what is one resampling. We will have a direct sinc resampler to compare against.
- A transpose at the end with the comment "either SICD wants different x-y than natural
  from cphd or I'm confused as usual." This is exactly the convention risk in section 5.
  Conventions get pinned by tests, not by trial.
- An out-of-memory retry path referenced an undefined variable, so it had never run.
  We should not have OOM fallbacks; we should size batches so they are not needed.
- Hardcoded Kaiser-Bessel constants in two places that had to agree with each other.
  Kernel parameters belong in one object shared by gridding and its correction.

## 6. Working agreement

- Planning and auditing agent: this file, reviews of each milestone, hierarchy test,
  benchmark review. Implementation by whoever is assigned, possibly a different agent.
- Each milestone gets a self-contained brief in `briefs/<milestone>.md`. The brief is
  the implementer's only context, so it names the files to read, the exact scope, the
  exact "done when" commands the auditor will run, and the reporting format. The
  implementer's report and the audit go next to it as `briefs/<milestone>-report.md`
  and `briefs/<milestone>-audit.md`.
- A milestone is reviewed against its "done when" line, with tests run, before the plan
  is updated. The plan is a living file; edit it rather than letting it rot.
- Every claim in a report, review, or doc says what was checked. A negative claim ("X
  does not exist", "Y is not installed", "Z passes") requires a check that actually
  tests it. Anything not checked is labeled unverified. This is written into
  `rules4robots.md` and applies to every agent working here.
