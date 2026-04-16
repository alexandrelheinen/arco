#!/usr/bin/env python3
"""Render a high-level package dependency map for the arco source tree.

Walks every Python module under ``src/arco/``, extracts intra-package import
relationships (``import arco.X`` and ``from arco.X import …``), and emits a
single overview PNG using :mod:`graphviz`.

Usage::

    python scripts/render_package_map.py --output diagrams/overview.png

The resulting diagram contains one node per top-level sub-package and one
directed edge for each import dependency between them.  Method signatures and
attributes are intentionally omitted to keep the diagram readable.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path


def _top_package(module: str) -> str | None:
    """Return the top-level arco sub-package name, or None.

    Args:
        module: A dotted module name, e.g. ``arco.control.pid``.

    Returns:
        The first sub-package after ``arco`` (e.g. ``control``), or ``None``
        if the module is not inside the ``arco`` namespace.
    """
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "arco":
        return None
    return parts[1]


def _collect_dependencies(src_root: Path) -> dict[str, set[str]]:
    """Walk *src_root* and collect inter-package import edges.

    Args:
        src_root: Path to the ``src/arco`` directory.

    Returns:
        Mapping from package name to the set of package names it imports.
    """
    deps: dict[str, set[str]] = {}

    for py_file in sorted(src_root.rglob("*.py")):
        # Determine which sub-package this file belongs to.
        rel = py_file.relative_to(src_root)
        if len(rel.parts) < 2:
            # Files directly under src/arco (e.g. __init__.py) → root pkg.
            owner = "arco"
        else:
            owner = rel.parts[0]

        if owner not in deps:
            deps[owner] = set()

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = _top_package(alias.name)
                    if pkg and pkg != owner:
                        deps[owner].add(pkg)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    pkg = _top_package(node.module)
                    if pkg and pkg != owner:
                        deps[owner].add(pkg)
                elif node.level and node.level > 0:
                    # Relative import — skip; handled within same package.
                    pass

    return deps


def render(src_root: Path, output: Path) -> None:
    """Render the package dependency map to *output*.

    Args:
        src_root: Path to ``src/arco``.
        output: Destination PNG path (parent directory must exist or will be
            created).
    """
    try:
        import graphviz  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: 'graphviz' Python package not found. "
            "Install it with: pip install graphviz",
            file=sys.stderr,
        )
        sys.exit(1)

    deps = _collect_dependencies(src_root)

    dot = graphviz.Digraph(
        name="arco_packages",
        comment="ARCO — high-level package dependency map",
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "nodesep": "0.6",
            "ranksep": "0.8",
            "fontname": "Helvetica",
            "bgcolor": "white",
        },
        node_attr={
            "shape": "box",
            "style": "filled",
            "fillcolor": "#ddeeff",
            "fontname": "Helvetica",
            "fontsize": "12",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "color": "#555555",
        },
    )

    # Add nodes for every known sub-package (skip the bare 'arco' root entry
    # which would appear alongside sub-packages and add visual clutter).
    for pkg in sorted(deps):
        if pkg == "arco":
            continue
        dot.node(pkg, pkg)

    # Add edges (also skip root 'arco' as source/target).
    for src_pkg, targets in sorted(deps.items()):
        if src_pkg == "arco":
            continue
        for tgt in sorted(targets):
            if tgt == "arco":
                continue
            dot.edge(src_pkg, tgt)

    output.parent.mkdir(parents=True, exist_ok=True)
    # graphviz renders to <stem>.<fmt>; remove the extension before passing.
    out_stem = str(output.with_suffix(""))
    fmt = output.suffix.lstrip(".") or "png"
    dot.render(out_stem, format=fmt, cleanup=True)
    print(f"✅  overview diagram → {output}")


def main() -> None:
    """Entry point for the package-map renderer."""
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "src" / "arco"

    parser = argparse.ArgumentParser(
        description="Render high-level ARCO package dependency map."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=default_src,
        help="Path to src/arco (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagrams/overview.png"),
        help="Output PNG path (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        print(
            f"ERROR: source directory not found: {args.src}", file=sys.stderr
        )
        sys.exit(1)

    render(args.src, args.output)


if __name__ == "__main__":
    main()
