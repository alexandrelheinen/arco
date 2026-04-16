#!/usr/bin/env python3
"""Render scoped class diagrams for modules touched by a PR.

For each changed Python file that lives inside ``src/arco/`` the script:

1. Parses the file with :mod:`ast` to extract class names, base classes, and
   imported arco symbols.
2. Groups classes by their containing module (sub-package).
3. Emits one PNG per module using :mod:`graphviz` with:

   - A subgraph cluster per module.
   - One node per class.
   - Inheritance edges (solid arrow).
   - Import / composition edges (dashed arrow).

Usage::

    python scripts/render_scoped.py --output-dir diagrams src/arco/control/pid.py …

Changed files that live outside ``src/arco/`` (tests, other scripts) are
silently skipped as specified in the issue.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _module_label(py_file: Path, src_root: Path) -> str:
    """Return a dotted module label relative to *src_root*.

    Args:
        py_file: Absolute path to a ``.py`` file.
        src_root: Absolute path to the ``src`` directory (parent of ``arco``).

    Returns:
        Dotted module name, e.g. ``arco.control.pid``.
    """
    rel = py_file.relative_to(src_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip .py
    return ".".join(parts)


def _subpkg(py_file: Path, src_arco: Path) -> str:
    """Return the top-level sub-package name (e.g. ``control``).

    Args:
        py_file: Path to a Python source file inside ``src/arco/``.
        src_arco: Path to ``src/arco``.

    Returns:
        First path segment under ``arco``, or ``arco`` if the file is at root.
    """
    rel = py_file.relative_to(src_arco)
    return rel.parts[0] if len(rel.parts) > 1 else "arco"


class _ModuleInfo:
    """Parsed information about a single Python module.

    Attributes:
        label: Dotted module name, e.g. ``arco.control.pid``.
        classes: List of ``(class_name, [base_names])`` tuples.
        arco_imports: Set of arco symbols imported by this module.
    """

    def __init__(self, label: str) -> None:
        """Initialise with an empty class list and import set.

        Args:
            label: Dotted module name.
        """
        self.label = label
        self.classes: list[tuple[str, list[str]]] = []
        self.arco_imports: set[str] = set()


def _parse_file(py_file: Path, src_root: Path) -> _ModuleInfo:
    """Parse *py_file* and return a :class:`_ModuleInfo`.

    Args:
        py_file: Path to the Python source file.
        src_root: Path to the ``src`` directory (parent of ``arco``).

    Returns:
        Populated :class:`_ModuleInfo` instance.
    """
    label = _module_label(py_file, src_root)
    info = _ModuleInfo(label)
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            info.classes.append((node.name, bases))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("arco."):
                    # Capture the full qualified name after 'arco.' so that
                    # 'import arco.mapping.graph' yields 'mapping.graph'
                    # rather than just 'graph', preserving package context.
                    info.arco_imports.add(alias.name[len("arco.") :])

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("arco."):
                for alias in node.names:
                    if alias.name != "*":
                        info.arco_imports.add(alias.name)

    return info


# ---------------------------------------------------------------------------
# Diagram renderer
# ---------------------------------------------------------------------------


def _safe_id(name: str) -> str:
    """Return a graphviz-safe node identifier.

    Args:
        name: Arbitrary string.

    Returns:
        String with dots replaced by underscores.
    """
    return name.replace(".", "_").replace("-", "_")


def render_scoped(
    changed_files: list[Path],
    src_arco: Path,
    output_dir: Path,
) -> None:
    """Render one PNG per changed module.

    Args:
        changed_files: List of changed ``.py`` file paths.
        src_arco: Path to ``src/arco``.
        output_dir: Directory where PNG files are written.
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

    src_root = src_arco.parent  # e.g. src/

    # Group changed files by top-level sub-package.
    groups: dict[str, list[Path]] = {}
    for f in changed_files:
        try:
            f.relative_to(src_arco)
        except ValueError:
            # File is outside src/arco — skip silently.
            continue
        pkg = _subpkg(f, src_arco)
        groups.setdefault(pkg, []).append(f)

    if not groups:
        print("No changed files inside src/arco — skipping scoped diagrams.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for pkg, files in sorted(groups.items()):
        infos = [_parse_file(f, src_root) for f in sorted(files)]

        dot = graphviz.Digraph(
            name=f"arco_{pkg}",
            comment=f"ARCO — scoped class diagram for {pkg}",
            graph_attr={
                "rankdir": "TB",
                "splines": "ortho",
                "nodesep": "0.5",
                "ranksep": "0.7",
                "fontname": "Helvetica",
                "bgcolor": "white",
            },
            node_attr={
                "shape": "box",
                "style": "filled",
                "fillcolor": "#eef5db",
                "fontname": "Helvetica",
                "fontsize": "11",
            },
            edge_attr={
                "fontname": "Helvetica",
                "fontsize": "9",
                "color": "#333333",
            },
        )

        # Collect all class names defined in *this* set of changed files so
        # that we can resolve intra-diagram inheritance edges.
        all_classes: set[str] = {
            cls for info in infos for cls, _ in info.classes
        }

        for info in infos:
            mod_id = _safe_id(info.label)
            # Subgraph cluster for this module.
            with dot.subgraph(name=f"cluster_{mod_id}") as sub:
                sub.attr(
                    label=info.label,
                    style="filled",
                    fillcolor="#f0f4ff",
                    color="#7799cc",
                    fontname="Helvetica",
                    fontsize="10",
                )
                for cls_name, _ in info.classes:
                    node_id = f"{mod_id}__{cls_name}"
                    sub.node(node_id, cls_name)

            # Inheritance edges (solid).
            for cls_name, bases in info.classes:
                node_id = f"{mod_id}__{cls_name}"
                for base in bases:
                    if base in all_classes:
                        # Resolve base to its defining module's node id.
                        base_mod_id = next(
                            (
                                _safe_id(i.label)
                                for i in infos
                                if any(c == base for c, _ in i.classes)
                            ),
                            mod_id,
                        )
                        base_node_id = f"{base_mod_id}__{base}"
                    else:
                        # External base class — add a standalone node.
                        base_node_id = _safe_id(f"ext__{base}")
                        dot.node(
                            base_node_id,
                            base,
                            shape="box",
                            style="filled,dashed",
                            fillcolor="#eeeeee",
                        )
                    dot.edge(
                        node_id,
                        base_node_id,
                        arrowhead="empty",
                        style="solid",
                        tooltip=f"{cls_name} inherits {base}",
                    )

            # Import edges (dashed) — arco symbols imported into this module.
            # Use a dedicated invisible anchor node for the module so that
            # import edges always originate from a stable, deterministic point
            # rather than an arbitrarily chosen first class.
            anchor_id = f"{mod_id}__anchor"
            dot.node(
                anchor_id,
                info.label,
                shape="plaintext",
                style="",
                fillcolor="transparent",
                fontsize="9",
                fontcolor="#555555",
            )
            for imported in sorted(info.arco_imports):
                if imported in all_classes:
                    # Already in the diagram — skip to avoid clutter.
                    continue
                imp_node_id = _safe_id(f"imp__{imported}")
                dot.node(
                    imp_node_id,
                    imported,
                    shape="box",
                    style="filled,dashed",
                    fillcolor="#fff3cc",
                )
                dot.edge(
                    anchor_id,
                    imp_node_id,
                    style="dashed",
                    arrowhead="open",
                    tooltip=f"{info.label} imports {imported}",
                )

        out_png = output_dir / f"scoped_{pkg}.png"
        out_stem = str(out_png.with_suffix(""))
        dot.render(out_stem, format="png", cleanup=True)
        print(f"✅  scoped diagram → {out_png}")


def main() -> None:
    """Entry point for the scoped class-diagram renderer."""
    repo_root = Path(__file__).resolve().parent.parent
    default_src_arco = repo_root / "src" / "arco"

    parser = argparse.ArgumentParser(
        description="Render scoped class diagrams for changed arco modules."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagrams"),
        help="Directory for output PNG files (default: %(default)s)",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=default_src_arco,
        help="Path to src/arco (default: %(default)s)",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Changed Python files to analyse",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        print(
            f"ERROR: source directory not found: {args.src}", file=sys.stderr
        )
        sys.exit(1)

    render_scoped([p.resolve() for p in args.files], args.src, args.output_dir)


if __name__ == "__main__":
    main()
