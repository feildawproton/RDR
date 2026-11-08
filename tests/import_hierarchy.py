"""AST-based layered import hierarchy checker for the ``rdr`` package tree."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
import importlib.util
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    """A disallowed cross-package import under the ``rdr`` layer rules."""

    file: Path
    line: int
    importer: str
    importer_package: str
    importer_layer: int
    target: str
    target_package: str
    target_layer: int | None
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.file}:{self.line}: {self.importer} (L{self.importer_layer}) -> "
            f"{self.target} ({self.reason})"
        )


@dataclass(frozen=True)
class SiblingImport:
    """An allowed within-package sibling import, tracked so they stay rare."""

    file: Path
    line: int
    importer: str
    target: str
    package: str


class HierarchyResult(list[Violation]):
    """List of :class:`Violation` instances with tracked sibling imports attached."""

    def __init__(
        self,
        violations: Iterable[Violation] = (),
        sibling_imports: Iterable[SiblingImport] = (),
    ) -> None:
        super().__init__(violations)
        self.sibling_imports: list[SiblingImport] = list(sibling_imports)

    @property
    def sibling_count(self) -> int:
        """Total number of within-package sibling imports observed."""
        return len(self.sibling_imports)


def _extract_layer(tree: ast.Module, init_path: Path) -> int:
    """Read integer ``__layer__`` from a package ``__init__.py`` AST."""
    for node in tree.body:
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__layer__":
                    value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__layer__":
                value_node = node.value

        if value_node is not None:
            value = ast.literal_eval(value_node)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"__layer__ in {init_path} must be an int, got {value!r}"
                )
            return value

    raise ValueError(f"Package {init_path} does not define __layer__")


def _module_and_package(rel_path: Path) -> tuple[str, str]:
    """Map a path relative to the ``rdr`` package directory to (module, __package__)."""
    if rel_path == Path("__init__.py"):
        return "rdr", "rdr"
    if rel_path.name == "__init__.py":
        mod = ".".join(("rdr", *rel_path.parent.parts))
        return mod, mod
    mod = ".".join(("rdr", *rel_path.with_suffix("").parts))
    pkg = ".".join(("rdr", *rel_path.parent.parts)) if rel_path.parent.parts else "rdr"
    return mod, pkg


def _resolve_import_targets(
    node: ast.Import | ast.ImportFrom,
    current_package: str,
    known_modules: set[str],
) -> list[str]:
    """Resolve an AST import statement to absolute target module/package names."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    if node.level == 0:
        base = node.module or ""
    else:
        rel_spec = "." * node.level + (node.module or "")
        base = importlib.util.resolve_name(rel_spec, current_package)

    if base == "rdr":
        targets: list[str] = []
        for alias in node.names:
            if alias.name == "*":
                targets.append("rdr.*")
            else:
                targets.append(f"rdr.{alias.name}")
        return targets

    if not base.startswith("rdr."):
        return []

    resolved: list[str] = []
    for alias in node.names:
        candidate = f"{base}.{alias.name}" if alias.name != "*" else base
        target = candidate if candidate in known_modules else base
        if target not in resolved:
            resolved.append(target)
    return resolved


def check_tree(src_root: str | Path) -> HierarchyResult:
    """Walk ``src_root`` with ``ast`` and return all layer hierarchy violations.

    ``src_root`` may point either to the directory containing ``rdr/`` (e.g. ``src/``)
    or directly to the ``rdr`` package directory itself.

    Returns a :class:`HierarchyResult` (a ``list[Violation]`` subclass) whose
    ``sibling_imports`` and ``sibling_count`` attributes list and count allowed
    within-package sibling imports.
    """
    root = Path(src_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"src_root is not a directory: {root}")

    if (root / "rdr").is_dir():
        rdr_dir = root / "rdr"
        extra_py = [
            p for p in sorted(root.rglob("*.py")) if not p.is_relative_to(rdr_dir)
        ]
        if extra_py:
            raise ValueError(
                f"Unexpected Python files outside {rdr_dir}: {extra_py}"
            )
    else:
        rdr_dir = root

    py_files = sorted(rdr_dir.rglob("*.py"))
    parsed_files: list[tuple[Path, str, str, ast.Module]] = []
    known_modules: set[str] = set()
    package_layers: dict[str, int] = {}

    for py_file in py_files:
        rel = py_file.relative_to(rdr_dir)
        mod_name, cur_pkg = _module_and_package(rel)
        known_modules.add(mod_name)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        parsed_files.append((py_file, mod_name, cur_pkg, tree))

        if len(rel.parts) == 2 and rel.name == "__init__.py":
            pkg_name = f"rdr.{rel.parts[0]}"
            package_layers[pkg_name] = _extract_layer(tree, py_file)

    for py_file, mod_name, _, _ in parsed_files:
        parts = mod_name.split(".")
        if len(parts) >= 2:
            top_pkg = f"rdr.{parts[1]}"
            if top_pkg not in package_layers and (rdr_dir / parts[1]).is_dir():
                raise ValueError(
                    f"Subpackage {top_pkg} (containing {py_file}) is missing "
                    f"__init__.py with integer __layer__"
                )

    violations: list[Violation] = []
    sibling_imports: list[SiblingImport] = []

    for py_file, mod_name, cur_pkg, tree in parsed_files:
        parts = mod_name.split(".")
        if len(parts) >= 2 and f"rdr.{parts[1]}" in package_layers:
            importer_pkg = f"rdr.{parts[1]}"
            importer_layer = package_layers[importer_pkg]
        else:
            importer_pkg = "rdr"
            importer_layer = 0

        import_nodes = sorted(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom))
            ),
            key=lambda n: (n.lineno, n.col_offset),
        )

        for node in import_nodes:
            for target in _resolve_import_targets(node, cur_pkg, known_modules):
                if not target.startswith("rdr."):
                    continue

                target_parts = target.split(".")
                target_pkg = f"rdr.{target_parts[1]}"

                if target_pkg == importer_pkg:
                    if target != mod_name:
                        sibling_imports.append(
                            SiblingImport(
                                file=py_file,
                                line=node.lineno,
                                importer=mod_name,
                                target=target,
                                package=importer_pkg,
                            )
                        )
                    continue

                if target_pkg == "rdr.core":
                    continue

                target_layer = package_layers.get(target_pkg)
                if target_layer is None:
                    violations.append(
                        Violation(
                            file=py_file,
                            line=node.lineno,
                            importer=mod_name,
                            importer_package=importer_pkg,
                            importer_layer=importer_layer,
                            target=target,
                            target_package=target_pkg,
                            target_layer=None,
                            reason=f"Target package {target_pkg} is unknown",
                        )
                    )
                elif target_layer >= importer_layer:
                    violations.append(
                        Violation(
                            file=py_file,
                            line=node.lineno,
                            importer=mod_name,
                            importer_package=importer_pkg,
                            importer_layer=importer_layer,
                            target=target,
                            target_package=target_pkg,
                            target_layer=target_layer,
                            reason=(
                                f"Layer {importer_layer} ({importer_pkg}) cannot "
                                f"import layer {target_layer} ({target_pkg})"
                            ),
                        )
                    )

    return HierarchyResult(violations, sibling_imports)
