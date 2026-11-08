"""Tests for the AST-based ``rdr`` layered import hierarchy checker."""

from __future__ import annotations

from pathlib import Path

from tests.import_hierarchy import check_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pkg(rdr_root: Path, name: str, layer: int) -> Path:
    pkg_dir = rdr_root / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text(
        f'"""Fake {name} package."""\n\n__layer__ = {layer}\n',
        encoding="utf-8",
    )
    return pkg_dir


def test_real_src_tree_has_zero_violations() -> None:
    result = check_tree(REPO_ROOT / "src")
    assert result == []
    assert result.sibling_count == 0
    assert result.sibling_imports == []


def test_planted_same_layer_violation_is_caught(tmp_path: Path) -> None:
    rdr_root = tmp_path / "rdr"
    pkg_a = _write_pkg(rdr_root, "alpha", 1)
    _write_pkg(rdr_root, "beta", 1)
    (pkg_a / "mod.py").write_text("import rdr.beta\n", encoding="utf-8")

    result = check_tree(tmp_path)
    assert len(result) == 1
    assert result[0].importer == "rdr.alpha.mod"
    assert result[0].importer_layer == 1
    assert result[0].target_package == "rdr.beta"
    assert result[0].target_layer == 1


def test_planted_upward_import_is_caught(tmp_path: Path) -> None:
    rdr_root = tmp_path / "rdr"
    pkg_low = _write_pkg(rdr_root, "low", 1)
    _write_pkg(rdr_root, "high", 2)
    (pkg_low / "mod.py").write_text("from rdr.high import something\n", encoding="utf-8")

    result = check_tree(tmp_path)
    assert len(result) == 1
    assert result[0].importer_layer == 1
    assert result[0].target_package == "rdr.high"
    assert result[0].target_layer == 2


def test_rdr_core_import_is_allowed(tmp_path: Path) -> None:
    rdr_root = tmp_path / "rdr"
    core = _write_pkg(rdr_root, "core", 0)
    (core / "conventions.py").write_text("SPEED_OF_LIGHT = 299792458.0\n", encoding="utf-8")
    pkg_low = _write_pkg(rdr_root, "low", 1)
    (pkg_low / "mod.py").write_text(
        "from rdr.core.conventions import SPEED_OF_LIGHT\n",
        encoding="utf-8",
    )

    result = check_tree(tmp_path)
    assert result == []
    assert result.sibling_count == 0


def test_downward_and_relative_and_sibling_imports(tmp_path: Path) -> None:
    rdr_root = tmp_path / "rdr"
    core = _write_pkg(rdr_root, "core", 0)
    (core / "conventions.py").write_text("C = 1\n", encoding="utf-8")
    sig = _write_pkg(rdr_root, "signal", 1)
    (sig / "windows.py").write_text("def hann():\n    return None\n", encoding="utf-8")
    (sig / "czt.py").write_text(
        "from . import windows\nfrom ..core import conventions\n",
        encoding="utf-8",
    )
    img = _write_pkg(rdr_root, "imaging", 2)
    (img / "pfa.py").write_text(
        "from ..signal import czt\n",
        encoding="utf-8",
    )

    result = check_tree(tmp_path)
    assert result == []
    assert result.sibling_count == 1
    assert result.sibling_imports[0].importer == "rdr.signal.czt"
    assert result.sibling_imports[0].target == "rdr.signal.windows"
    assert result.sibling_imports[0].package == "rdr.signal"


def test_relative_upward_violation_is_caught(tmp_path: Path) -> None:
    rdr_root = tmp_path / "rdr"
    sig = _write_pkg(rdr_root, "signal", 1)
    _write_pkg(rdr_root, "imaging", 2)
    (sig / "czt.py").write_text("from .. import imaging\n", encoding="utf-8")

    result = check_tree(tmp_path)
    assert len(result) == 1
    assert result[0].target_package == "rdr.imaging"
