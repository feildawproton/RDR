# M0 Report: Package Skeleton

Date: 2026-11-07

## 1. "Done when" verification

### Command 1: Editable install with `test` extras

Command run:

```bash
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/pip install -e ".[test]"
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
Requirement already satisfied: pytest in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (9.1.1)
Requirement already satisfied: scipy in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (1.17.1)
Requirement already satisfied: sarkit in /home/feildaw/mypyenv/lib/python3.12/site-packages (from rdr==0.1.0) (1.8.0)
...
Building wheels for collected packages: rdr
  Building editable for rdr (pyproject.toml) ... done
  Created wheel for rdr: filename=rdr-0.1.0-0.editable-py3-none-any.whl size=1815 sha256=51cab240adf0514655fbff3cbd71628e3a59892b89be0dd4ffafdf1cf44666dc
Successfully built rdr
Installing collected packages: rdr
  Attempting uninstall: rdr
    Found existing installation: rdr 0.1.0
    Uninstalling rdr-0.1.0:
      Successfully uninstalled rdr-0.1.0
Successfully installed rdr-0.1.0
```

Also verified `/home/feildaw/mypyenv/bin/pip install -e .` and `/home/feildaw/mypyenv/bin/pip install -e ".[test,viz]"` (both exited 0 and installed `rdr-0.1.0`).

### Command 2: Import check

Command run:

```bash
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/python -c "import rdr, rdr.core.conventions; print(rdr.core.__layer__)"
```

Observed output:

```text
0
```

### Command 3: Pytest suite (`-q` and `-v`)

Command run:

```bash
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/python -m pytest -q
```

Observed output:

```text
........                                                                 [100%]
8 passed in 2.62s
```

Verbose run (`/home/feildaw/mypyenv/bin/python -m pytest -v`):

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/feildaw/mypyenv/bin/python
cachedir: .pytest_cache
rootdir: /home/feildaw/projects/RDR
configfile: pyproject.toml
testpaths: tests
collecting ... collected 8 items

tests/test_harness.py::test_default_results_dir_location PASSED          [ 12%]
tests/test_harness.py::test_benchmark_writes_json_with_expected_fields PASSED [ 25%]
tests/test_import_hierarchy.py::test_real_src_tree_has_zero_violations PASSED [ 37%]
tests/test_import_hierarchy.py::test_planted_same_layer_violation_is_caught PASSED [ 50%]
tests/test_import_hierarchy.py::test_planted_upward_import_is_caught PASSED [ 62%]
tests/test_import_hierarchy.py::test_rdr_core_import_is_allowed PASSED   [ 75%]
tests/test_import_hierarchy.py::test_downward_and_relative_and_sibling_imports PASSED [ 87%]
tests/test_import_hierarchy.py::test_relative_upward_violation_is_caught PASSED [100%]

============================== 8 passed in 2.92s ===============================
```

### Check 4: `git status` (`references/` and `benchmarks/results/` not untracked)

Command run (after also verifying default `Benchmark` output into `benchmarks/results/`):

```bash
cd /home/feildaw/projects/RDR
git status -uall
```

Observed output:

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	GOALS.md
	PLAN.md
	benchmarks/harness.py
	briefs/M0-skeleton-report.md
	briefs/M0-skeleton.md
	pyproject.toml
	src/rdr/__init__.py
	src/rdr/core/__init__.py
	src/rdr/core/conventions.py
	src/rdr/deconv/__init__.py
	src/rdr/geometry/__init__.py
	src/rdr/imaging/__init__.py
	src/rdr/io/__init__.py
	src/rdr/pipeline/__init__.py
	src/rdr/signal/__init__.py
	src/rdr/sim/__init__.py
	tests/import_hierarchy.py
	tests/test_harness.py
	tests/test_import_hierarchy.py

no changes added to commit (use "git add" and/or "git commit -a")
```

### Check 5: No disallowed imports under `src/rdr/`

Verified via `tests/test_import_hierarchy.py::test_real_src_tree_has_zero_violations` (`check_tree(REPO_ROOT / "src")` returned `[]` with `sibling_count == 0`).

## 2. Files created or changed

- `/home/feildaw/projects/RDR/pyproject.toml` (created)
- `/home/feildaw/projects/RDR/README.md` (created in working tree; `git status` shows `modified: README.md` because `HEAD` previously tracked `README.md` which had been deleted in the working tree before M0)
- `/home/feildaw/projects/RDR/src/rdr/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/core/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/core/conventions.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/io/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/geometry/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/signal/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/sim/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/imaging/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/deconv/__init__.py` (created)
- `/home/feildaw/projects/RDR/src/rdr/pipeline/__init__.py` (created)
- `/home/feildaw/projects/RDR/tests/import_hierarchy.py` (created)
- `/home/feildaw/projects/RDR/tests/test_import_hierarchy.py` (created)
- `/home/feildaw/projects/RDR/benchmarks/harness.py` (created)
- `/home/feildaw/projects/RDR/tests/test_harness.py` (created)
- `/home/feildaw/projects/RDR/briefs/M0-skeleton-report.md` (created per `PLAN.md` section 6)

Files pre-existing and left untouched: `.gitignore`, `GOALS.md`, `PLAN.md`, `briefs/M0-skeleton.md`, `references/`.

## 3. Decisions made beyond the brief

1. **Return type of `check_tree(src_root)` vs sibling import tracking:** The brief specifies `check_tree(src_root) -> list[Violation]` and states that within-package sibling imports must be counted and listed "in a second return value or attribute". Used the attribute option by returning `HierarchyResult(list[Violation])` (a `list` subclass with `.sibling_imports: list[SiblingImport]` and `.sibling_count: int`), preserving full `list[Violation]` compatibility.
2. **`src_root` path handling in `check_tree`:** Accepted either the directory containing `rdr/` (e.g. `src/` or `tmp_path` containing `rdr/`) or the `rdr` directory directly (e.g. `src/rdr/` or `tmp_path` containing subpackages directly).
3. **Top-level `src/rdr/__init__.py`:** Created `src/rdr/__init__.py` with a docstring only (no `__layer__`, since the brief defines `__layer__` for the 8 subpackages `rdr.core`..`rdr.pipeline`). In `check_tree`, `rdr/__init__.py` is treated at effective layer 0 so it cannot import higher-layer subpackages.
4. **Benchmark JSON field names and GPU synchronization:** Only `io_over_compute` was given as a literal key name in the brief. Standardized the JSON keys as `name`, `timestamp`, `stages`, `total_seconds`, `io_over_compute`, `peak_cuda_memory_bytes`, `torch_version`, and `device`. Added `torch.cuda.synchronize()` before/after timed stages when CUDA is available so asynchronous GPU operations are timed accurately, accumulated repeated entries to the same stage name for chunked loops, and added an optional `results_dir` parameter (defaulting to `benchmarks/results`) so tests can write to `tmp_path`.
5. **Pytest `pythonpath`:** Set `pythonpath = ["."]` in `[tool.pytest.ini_options]` so `tests.*` and `benchmarks.*` imports work identically under both `python -m pytest` and `pytest`.

## 4. Ambiguities or discrepancies noted in the brief

1. **Report destination (`PLAN.md` vs `M0-skeleton.md`):** `PLAN.md` section 6 specifies saving the implementer's report at `briefs/<milestone>-report.md`, whereas `briefs/M0-skeleton.md` ("How to report") describes the required report contents without naming `briefs/M0-skeleton-report.md`. Wrote `briefs/M0-skeleton-report.md` and included the same report in the final response.
2. **Exact JSON key names in `benchmarks/harness.py`:** Future briefs that read benchmark JSONs programmatically in M1+ may want to reference the exact key names pinned here (`name`, `timestamp`, `stages`, `total_seconds`, `io_over_compute`, `peak_cuda_memory_bytes`, `torch_version`, `device`).
3. **`check_tree(src_root)` argument convention:** Future briefs/audits may want to state explicitly whether `check_tree` is called with `Path("src")` or `Path("src/rdr")` (both are supported here).
