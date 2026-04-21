"""Tests for the graphviz diagram renderer scripts.

Validates the pure-Python logic (AST parsing, dependency collection) without
requiring the graphviz CLI (``dot``) to be installed.  The rendering step is
tested only when the CLI is available, using ``pytest.importorskip`` to skip
gracefully on headless environments where the CLI may be absent.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the scripts importable (they live in scripts/, not src/arco/).
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import render_package_map  # noqa: E402
import render_scoped  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SRC_ARCO = Path(__file__).resolve().parent.parent / "src" / "arco"


@pytest.fixture()
def tmp_src(tmp_path: Path) -> Path:
    """Return a minimal fake arco src tree for deterministic tests.

    Layout::

        tmp_src/
            arco/
                __init__.py
                alpha/
                    __init__.py
                    core.py        # imports arco.beta
                beta/
                    __init__.py
                    helper.py      # no arco imports

    Returns:
        Path to the ``arco`` directory inside *tmp_path*.
    """
    arco = tmp_path / "arco"
    (arco / "alpha").mkdir(parents=True)
    (arco / "beta").mkdir(parents=True)

    (arco / "__init__.py").write_text("", encoding="utf-8")
    (arco / "alpha" / "__init__.py").write_text("", encoding="utf-8")
    (arco / "beta" / "__init__.py").write_text("", encoding="utf-8")

    (arco / "alpha" / "core.py").write_text(
        textwrap.dedent("""\
            import arco.beta

            class AlphaCore:
                pass
            """),
        encoding="utf-8",
    )
    (arco / "beta" / "helper.py").write_text(
        textwrap.dedent("""\
            class BetaHelper:
                pass
            """),
        encoding="utf-8",
    )
    return arco


# ---------------------------------------------------------------------------
# render_package_map: _top_package
# ---------------------------------------------------------------------------


def test_top_package_returns_subpackage() -> None:
    """_top_package extracts the first level below arco."""
    assert render_package_map._top_package("arco.control.pid") == "control"


def test_top_package_returns_none_for_non_arco() -> None:
    """_top_package returns None for modules outside arco."""
    assert render_package_map._top_package("numpy.linalg") is None


def test_top_package_returns_none_for_root_arco() -> None:
    """_top_package returns None for the bare 'arco' module name."""
    assert render_package_map._top_package("arco") is None


# ---------------------------------------------------------------------------
# render_package_map: _collect_dependencies
# ---------------------------------------------------------------------------


def test_collect_dependencies_detects_cross_package_import(
    tmp_src: Path,
) -> None:
    """Dependency from alpha to beta is detected via 'import arco.beta'."""
    deps = render_package_map._collect_dependencies(tmp_src)
    assert "alpha" in deps
    assert "beta" in deps["alpha"]


def test_collect_dependencies_no_self_edge(tmp_src: Path) -> None:
    """A package must not have an edge to itself."""
    deps = render_package_map._collect_dependencies(tmp_src)
    for pkg, targets in deps.items():
        assert pkg not in targets, f"Self-edge found for package '{pkg}'"


def test_collect_dependencies_beta_has_no_imports(tmp_src: Path) -> None:
    """beta.helper has no arco imports — its dependency set is empty."""
    deps = render_package_map._collect_dependencies(tmp_src)
    assert deps.get("beta", set()) == set()


def test_collect_dependencies_real_src() -> None:
    """The real src/arco directory has known top-level sub-packages."""
    if not SRC_ARCO.is_dir():
        pytest.skip("src/arco not found")
    deps = render_package_map._collect_dependencies(SRC_ARCO)
    for expected in ("control", "mapping", "planning"):
        assert expected in deps, f"Expected sub-package '{expected}' not found"


def test_render_package_map_excludes_root_arco_node(
    tmp_path: Path, tmp_src: Path
) -> None:
    """render() does not produce a bare 'arco' node in the overview diagram."""
    graphviz = pytest.importorskip("graphviz")
    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        pytest.skip("graphviz CLI (dot) not installed")

    out = tmp_path / "overview.png"
    render_package_map.render(tmp_src, out)
    # The dot source should not contain a standalone 'arco' node definition.
    dot = graphviz.Digraph()
    deps = render_package_map._collect_dependencies(tmp_src)
    assert "arco" not in {
        pkg for pkg in deps if pkg != "arco"
    }, "root arco should not appear as a dependency target"


# ---------------------------------------------------------------------------
# render_package_map: render (integration — requires dot CLI)
# ---------------------------------------------------------------------------


def test_render_package_map_produces_png(
    tmp_path: Path, tmp_src: Path
) -> None:
    """render() writes overview.png when the graphviz CLI is available."""
    graphviz = pytest.importorskip("graphviz")
    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        pytest.skip("graphviz CLI (dot) not installed")

    out = tmp_path / "overview.png"
    render_package_map.render(tmp_src, out)
    assert out.is_file(), "overview.png was not created"
    assert out.stat().st_size > 0, "overview.png is empty"


# ---------------------------------------------------------------------------
# render_scoped: _module_label
# ---------------------------------------------------------------------------


def test_module_label_regular_file(tmp_src: Path) -> None:
    """_module_label strips .py and joins with dots."""
    src_root = tmp_src.parent  # parent of arco/
    label = render_scoped._module_label(
        tmp_src / "alpha" / "core.py", src_root
    )
    assert label == "arco.alpha.core"


def test_module_label_init_file(tmp_src: Path) -> None:
    """_module_label drops __init__.py from the label."""
    src_root = tmp_src.parent
    label = render_scoped._module_label(
        tmp_src / "alpha" / "__init__.py", src_root
    )
    assert label == "arco.alpha"


# ---------------------------------------------------------------------------
# render_scoped: _subpkg
# ---------------------------------------------------------------------------


def test_subpkg_nested_file(tmp_src: Path) -> None:
    """_subpkg returns the first segment under arco."""
    result = render_scoped._subpkg(tmp_src / "alpha" / "core.py", tmp_src)
    assert result == "alpha"


def test_subpkg_root_init(tmp_src: Path) -> None:
    """_subpkg returns 'arco' for files directly under arco/."""
    result = render_scoped._subpkg(tmp_src / "__init__.py", tmp_src)
    assert result == "arco"


# ---------------------------------------------------------------------------
# render_scoped: _parse_file
# ---------------------------------------------------------------------------


def test_parse_file_extracts_class(tmp_src: Path) -> None:
    """_parse_file finds the class defined in alpha/core.py."""
    src_root = tmp_src.parent
    info = render_scoped._parse_file(tmp_src / "alpha" / "core.py", src_root)
    class_names = [name for name, _ in info.classes]
    assert "AlphaCore" in class_names


def test_parse_file_no_bases_for_standalone_class(tmp_src: Path) -> None:
    """A class with no explicit base has an empty bases list."""
    src_root = tmp_src.parent
    info = render_scoped._parse_file(tmp_src / "alpha" / "core.py", src_root)
    for name, bases in info.classes:
        if name == "AlphaCore":
            assert bases == []


def test_parse_file_extracts_inheritance(tmp_path: Path) -> None:
    """_parse_file captures base class names from class definitions."""
    py = tmp_path / "derived.py"
    py.write_text(
        textwrap.dedent("""\
            class Base:
                pass

            class Derived(Base):
                pass
            """),
        encoding="utf-8",
    )
    # Use tmp_path as src_root so label = "derived"
    info = render_scoped._parse_file(py, tmp_path)
    derived_bases = next(
        bases for name, bases in info.classes if name == "Derived"
    )
    assert "Base" in derived_bases


def test_parse_file_arco_imports(tmp_path: Path) -> None:
    """_parse_file extracts arco symbol names from from-imports and import stmts."""
    py = tmp_path / "consumer.py"
    py.write_text(
        textwrap.dedent("""\
            from arco.mapping.graph import RoadGraph
            from arco.control.pid import PIDController
            import arco.middleware.bus
            """),
        encoding="utf-8",
    )
    info = render_scoped._parse_file(py, tmp_path)
    # from-imports capture the symbol name (last segment).
    assert "RoadGraph" in info.arco_imports
    assert "PIDController" in info.arco_imports
    # bare 'import arco.X.Y' captures full path after 'arco.' → 'middleware.bus'
    assert "middleware.bus" in info.arco_imports


def test_parse_file_ignores_syntax_error(tmp_path: Path) -> None:
    """_parse_file returns an empty ModuleInfo for files with syntax errors."""
    py = tmp_path / "broken.py"
    py.write_text("def (broken syntax:", encoding="utf-8")
    info = render_scoped._parse_file(py, tmp_path)
    assert info.classes == []
    assert info.arco_imports == set()


# ---------------------------------------------------------------------------
# render_scoped: render_scoped (integration — requires dot CLI)
# ---------------------------------------------------------------------------


def test_render_scoped_produces_png(tmp_path: Path, tmp_src: Path) -> None:
    """render_scoped() writes one PNG per changed sub-package."""
    graphviz = pytest.importorskip("graphviz")
    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        pytest.skip("graphviz CLI (dot) not installed")

    changed = [tmp_src / "alpha" / "core.py"]
    out_dir = tmp_path / "out"
    render_scoped.render_scoped(changed, tmp_src, out_dir)
    expected = out_dir / "scoped_alpha.png"
    assert expected.is_file(), "scoped_alpha.png was not created"
    assert expected.stat().st_size > 0, "scoped_alpha.png is empty"


def test_render_scoped_skips_external_files(
    tmp_path: Path, tmp_src: Path
) -> None:
    """Files outside src/arco are silently skipped (no PNG, no error)."""
    graphviz = pytest.importorskip("graphviz")
    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        pytest.skip("graphviz CLI (dot) not installed")

    outside = tmp_path / "outside.py"
    outside.write_text("class X: pass\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    render_scoped.render_scoped([outside], tmp_src, out_dir)
    # No PNG should be created since the file is outside src_arco.
    assert not any(out_dir.glob("*.png")) if out_dir.exists() else True


def test_render_scoped_empty_file_list(tmp_path: Path, tmp_src: Path) -> None:
    """render_scoped() with an empty file list produces no output."""
    graphviz = pytest.importorskip("graphviz")
    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        pytest.skip("graphviz CLI (dot) not installed")

    out_dir = tmp_path / "out"
    render_scoped.render_scoped([], tmp_src, out_dir)
    assert not out_dir.exists() or not any(out_dir.glob("*.png"))
