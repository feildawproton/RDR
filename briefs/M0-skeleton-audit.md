# Audit: M0, package skeleton

Auditor: planning/auditing agent. Date: 2026-11-07 (evening, local).
Input: `briefs/M0-skeleton.md`, `briefs/M0-skeleton-report.md`, the working tree.

## Verdict

**Accepted.** All "done when" items pass when run by the auditor, not just as reported.
The checker also catches eight adversarial cases the implementer's tests did not plant.
Minor notes below; none block M1.

## What the auditor ran

```
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/pip install -e ".[test]"        -> "Successfully installed rdr-0.1.0"
/home/feildaw/mypyenv/bin/python -c "import rdr, rdr.core.conventions; print(rdr.core.__layer__)"   -> 0
/home/feildaw/mypyenv/bin/python -m pytest -q             -> "8 passed in 2.54s"
git status --short -uall | grep -E 'references/|benchmarks/results/'   -> no matches
grep -H "__layer__" src/rdr/*/__init__.py                 -> core 0, io 1, geometry 1, signal 1, sim 2, imaging 2, deconv 3, pipeline 4
```

Adversarial checker cases, each in a fresh temp tree, result = number of violations:

| case | expected | got |
| --- | --- | --- |
| upward import inside a function body | 1 | 1 |
| `from rdr import pipeline` from layer 2 | 1 | 1 |
| nested subpackage, `from ...imaging import pfa` from layer 1 | 1 | 1 |
| `rdr.core` importing `rdr.io` | 1 | 1 |
| same-layer sim and imaging importing each other | 2 | 2 |
| downward imports from layer 2 to layer 1 | 0 | 0 |
| `from rdr.imaging import *` from layer 1 | 1 | 1 |
| subpackage with no `__layer__` | raises | raises ValueError |

Untouched-file claim: modification times of `.gitignore`, `GOALS.md`, `PLAN.md`,
`briefs/M0-skeleton.md` are 12:09 to 12:55 local (auditor's session); implementer's
files are 20:01 onward. `.gitignore` content matches what the auditor wrote. Claim holds.

## Files reviewed line by line

`pyproject.toml`, `src/rdr/core/conventions.py`, all eight `__init__.py`,
`tests/import_hierarchy.py`, `tests/test_import_hierarchy.py`, `benchmarks/harness.py`,
`tests/test_harness.py`, `README.md`.

All match the brief. Conventions text is faithful to the brief, including the explicit
"FFT conventions not decided here" line. README makes no claims about features that do
not exist.

## Notes (non-blocking)

1. `benchmarks/harness.py`: if an exception escapes the `with` block, no JSON is written.
   That is a reasonable choice (no record of a run that did not finish) and should stay,
   but M1's benchmark script should print stage times as it goes so a crash late in a
   run still leaves evidence on the terminal.
2. `benchmarks/harness.py`: `io_over_compute` divides by the compute stage time with no
   guard. Unreachable in practice. Leave it.
3. `benchmarks/harness.py`: the benchmark name is used raw in the filename. Fine as long
   as names stay simple identifiers. M1 brief will say so.
4. `tests/test_import_hierarchy.py::test_real_src_tree_has_zero_violations` is trivially
   true at M0 because no module imports anything yet. It becomes meaningful at M1. Not a
   defect.
5. `benchmarks/results/m0_harness_smoke_*.json` shows peak CUDA memory of 0, meaning the
   smoke run did no GPU work. Harmless, gitignored.
6. The report's decisions section was useful. Keeping the `HierarchyResult` list
   subclass with `.sibling_imports`, and `check_tree` accepting either `src/` or
   `src/rdr/`.

## Carried into the M1 brief

- Benchmark JSON key names: `name`, `timestamp`, `stages`, `total_seconds`,
  `io_over_compute`, `peak_cuda_memory_bytes`, `torch_version`, `device`.
- Stage names `io` and `compute` are the ones the ratio looks for.
- `check_tree(REPO_ROOT / "src")` is the call the real-tree test uses.
