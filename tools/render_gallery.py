"""Render the ARCO illustration gallery.

Every plate is a 16:9 PNG drawn from a real run of the shipped planners
and controllers, in a purpose-built scene (see
``tools/illustration/world.py``).  Solver output is cached under
``tools/output/gallery_cache``, so the first render is slow and every
re-render after a styling change is immediate.

Output::

    docs/images/gallery/<theme>/<plate>.png

Usage::

    python3 tools/render_gallery.py
    python3 tools/render_gallery.py --theme nocturne --plate 01_field
    python3 tools/render_gallery.py --width 12 --dpi 160   # quick proof
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from illustration.plates import PLATES  # noqa: E402
from illustration.solution import Solution  # noqa: E402
from illustration.theme import THEMES  # noqa: E402
from illustration.world import build_world  # noqa: E402

DEFAULT_OUTPUT = ROOT / "docs" / "images" / "gallery"


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--theme",
        action="append",
        choices=sorted(THEMES),
        help="Theme to render; repeatable.  Defaults to every theme.",
    )
    parser.add_argument(
        "--plate",
        action="append",
        help="Plate key to render; repeatable.  Defaults to every plate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory receiving <theme>/<plate>.png.",
    )
    parser.add_argument(
        "--width", type=float, default=16.0, help="Figure width in inches."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output resolution; 16 x 120 gives a 1920 x 1080 PNG.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Render the requested plates.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    themes = args.theme or sorted(THEMES)
    wanted = set(args.plate) if args.plate else None

    world = build_world()
    solution = Solution(world)

    for theme_name in themes:
        theme = THEMES[theme_name]
        directory = args.output / theme_name
        directory.mkdir(parents=True, exist_ok=True)
        for key, module in PLATES:
            if wanted is not None and key not in wanted:
                continue
            target = directory / f"{key}.png"
            started = time.perf_counter()
            module.render(
                theme,
                world,
                solution,
                target,
                width_in=args.width,
                dpi=args.dpi,
            )
            elapsed = time.perf_counter() - started
            print(f"wrote {target}  ({elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
