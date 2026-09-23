"""Render the quiet web image set.

Every file is a PNG with no title, legend or metric. Solver output is
cached under ``tools/output/web_cache``. The release workflow runs this
and uploads the directory.

Output::

    <output>/<plate>-<ground>.png
    <output>/arc-og.png

Usage::

    python3 tools/render_web.py --list
    python3 tools/render_web.py --output /tmp/release_images
    python3 tools/render_web.py --plate mark --ground dark
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from illustration.web.names import (  # noqa: E402
    GROUNDS,
    PLATES,
    release_filenames,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the release filenames, one per line, and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "images" / "web",
        help="Directory receiving the PNG files.",
    )
    parser.add_argument(
        "--ground",
        action="append",
        choices=GROUNDS,
        help="Ground to render; repeatable. Defaults to both.",
    )
    parser.add_argument(
        "--plate",
        action="append",
        choices=PLATES,
        help="Plate to render; repeatable. Defaults to every plate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Sample budget for both RRT* and SST. Defaults to the release budget.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Raster resolution. 100 gives a 1600 x 900 master.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Render the requested plates, or list the release filenames.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    if args.list:
        for name in release_filenames():
            print(name)
        return 0
    from illustration.web.render import render_release

    render_release(
        args.output,
        grounds=args.ground,
        plates=args.plate,
        rrt_samples=args.samples,
        sst_samples=args.samples,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
