# Brief: M0, package skeleton

Written 2026-11-07 by the planning/auditing agent. This is the only context you get, so
read the three files below before doing anything else, in this order:

1. `/home/feildaw/projects/rules4robots.md` (standing rules, short, binding)
2. `/home/feildaw/projects/RDR/GOALS.md`
3. `/home/feildaw/projects/RDR/PLAN.md` (sections 2, 3/M0, 4, 6)

## Roles

- You: implementer for this milestone.
- Planning/auditing agent: wrote the plan and this brief, will audit your result against
  the "done when" list below.
- The human: owns decisions, commits to git. You do not commit.

## Scope

Milestone M0 from PLAN.md, nothing more. No CPHD reader, no signal code, no geometry.
If you finish early, stop and report. M1 gets its own brief after M0 is audited.

## Deliverables

1. **`pyproject.toml`** at `/home/feildaw/projects/RDR/`. Package name `rdr`, src
   layout (`src/rdr/`), Python >= 3.12, setuptools build backend. Runtime dependencies:
   numpy, torch, lxml. Optional extra `viz`: matplotlib, pillow. Optional extra `test`:
   pytest, scipy, sarkit. Pytest config pointing at `tests/`. Must install with
   `/home/feildaw/mypyenv/bin/pip install -e .` into the existing venv.

2. **Package tree** with one `__init__.py` per package. Each package `__init__.py`
   contains a module docstring stating its purpose in one or two sentences and a
   module-level constant `__layer__ = <int>`:

   | package | layer |
   | --- | --- |
   | `rdr.core` | 0 |
   | `rdr.io` | 1 |
   | `rdr.geometry` | 1 |
   | `rdr.signal` | 1 |
   | `rdr.sim` | 2 |
   | `rdr.imaging` | 2 |
   | `rdr.deconv` | 3 |
   | `rdr.pipeline` | 4 |

   Packages are otherwise empty. No placeholder functions.

3. **`src/rdr/core/conventions.py`**: a module with no code beyond constants and a
   docstring that pins these conventions. Future modules import constants from here
   instead of restating them.
   - `SPEED_OF_LIGHT = 299792458.0` (m/s).
   - Spatial frequency `k` is in cycles per meter, `k = 2 f / c` for monostatic.
   - Phase history arrays are shaped `(vectors, samples)`, vectors first.
   - Images are shaped `(rows, cols)` with rows along the first basis vector handed to
     the former and cols along the second. Which physical axis that is (range or
     cross-range) is the caller's choice and is not assumed by lower layers.
   - Signal data is `complex64`. Phases, geometry, and k-space coordinates are computed
     in `float64` and cast at the last step.
   - FFT sign and shift conventions are NOT decided here. Leave a line saying they are
     pinned in M3 in `rdr.signal`.

4. **Import hierarchy checker and test.**
   - `tests/import_hierarchy.py`: a function `check_tree(src_root) -> list[Violation]`
     that walks every `.py` file under `src_root`, parses it with `ast` (do not import
     the modules), reads each package's `__layer__` from its `__init__.py` via `ast` as
     well, and reports every `import` / `from ... import` that targets an `rdr.*` package
     whose layer is greater than or equal to the importing package's layer, except
     imports of `rdr.core`, which are always allowed, and imports within the importing
     package itself.
   - Within-package sibling imports (for example `rdr.signal.czt` importing
     `rdr.signal.windows`) are allowed but must be **counted and listed** in a second
     return value or attribute, because the goal is to keep them rare.
   - Relative imports must be resolved to absolute names before checking.
   - `tests/test_import_hierarchy.py`:
     - test that the real tree has zero violations;
     - test that a planted violation is caught: build a tiny fake tree in `tmp_path`
       with two packages at layers 1 and 1 where one imports the other, assert one
       violation; also plant an upward import (layer 1 importing layer 2) and assert it
       is caught; also plant a `rdr.core` import and assert it is NOT a violation.

5. **Benchmark harness**: `benchmarks/harness.py`. A small context-manager based timer:

   ```python
   with Benchmark("cphd_read_2gb") as b:
       with b.stage("io"):
           ...
       with b.stage("compute"):
           ...
   ```

   On exit it writes `benchmarks/results/<name>_<UTC timestamp>.json` containing: name,
   timestamp, per-stage wall seconds, total wall seconds, `io_over_compute` ratio when
   both stages exist, peak CUDA memory in bytes if CUDA is available (use
   `torch.cuda.max_memory_allocated`, reset at start), torch version, GPU name or
   "cpu". `benchmarks/results/` is already gitignored. Include `tests/test_harness.py`
   that runs a trivial benchmark with two stages and checks the JSON has the fields.

6. **`README.md`** at repo root, short: what RDR is (two sentences from GOALS.md),
   how to install into the venv, how to run tests, where the plan is. No claims about
   features that do not exist yet.

## Done when (the auditor will run exactly these)

```
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/pip install -e ".[test]"
/home/feildaw/mypyenv/bin/python -c "import rdr, rdr.core.conventions; print(rdr.core.__layer__)"
/home/feildaw/mypyenv/bin/python -m pytest -q
```

- Install succeeds. The import prints `0`.
- All tests pass, including the planted-violation tests and the harness test.
- `git status` shows no files under `references/` or `benchmarks/results/` as untracked.
- No file under `src/rdr/` imports anything from `rdr` except as the checker allows.

## Constraints

- Do not read code from `/home/feildaw/diffpfa` or `/home/feildaw/CLEAN_SAR`. The
  human has said that code misleads. The plan already extracted the clues.
- Do not read or write anything under `/home/feildaw/data`. M0 needs no data.
- Do not `git commit`. The human commits after audit.
- Do not add dependencies beyond those listed. If something listed is not installable,
  stop and report rather than substituting.
- No try/except fallbacks that hide failures. In particular no out-of-memory retry
  paths, ever, in this project.
- Use the venv interpreter explicitly: `/home/feildaw/mypyenv/bin/python`. Do not
  assume `python` on PATH is it.

## How to report

Your final report must contain, for each line in "Done when", the exact command you ran
and the relevant output excerpt, pasted, not paraphrased. If you did not run one, write
`NOT RUN` next to it. Anything you believe but did not check is labeled `unverified`.
Do not describe results you did not observe. Per `rules4robots.md`, a wrong confident
claim costs more here than an admitted gap.

Also list: every file you created or changed, any decision you had to make that this
brief did not cover, and anything in this brief that turned out to be wrong or
ambiguous. Those last two feed the next brief.
