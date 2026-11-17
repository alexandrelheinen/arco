#!/usr/bin/env python3
"""Check an ARCO wheel filename for the linux aarch64 abi3 tags.

FR-BUILD-07 and FR-BUILD-08 require a single abi3 manylinux2014 aarch64
wheel. This module parses the filename and refuses anything else.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_PYTHON = "cp310"
EXPECTED_ABI = "abi3"
EXPECTED_PLATFORM = "manylinux_2_17_aarch64.manylinux2014_aarch64"
EXPECTED_NAME = "arco"


class WheelTagError(ValueError):
    """Raised when a wheel filename fails the aarch64 abi3 tag check."""


@dataclass(frozen=True)
class WheelInfo:
    """Parsed PEP 427 tags from an ARCO wheel filename."""

    filename: str
    name: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str


def check_wheel_filename(
    wheel: str | Path,
    *,
    expect_version: str | None = None,
) -> WheelInfo:
    """Parse ``wheel`` and assert the linux aarch64 abi3 manylinux tags.

    Args:
        wheel: Wheel path or basename.
        expect_version: When set, the filename version must match.

    Returns:
        Parsed tags from the filename.

    Raises:
        WheelTagError: If any required tag is wrong.
    """
    filename = Path(wheel).name
    if not filename.endswith(".whl"):
        raise WheelTagError(f"{filename!r} is not a .whl filename")

    stem = filename[: -len(".whl")]
    parts = stem.split("-")
    # name-version-python-abi-platform; platform may contain dots only.
    if len(parts) < 5:
        raise WheelTagError(
            f"{filename!r} does not look like a PEP 427 wheel name"
        )

    name = parts[0]
    version = parts[1]
    python_tag = parts[2]
    abi_tag = parts[3]
    platform_tag = "-".join(parts[4:])

    if name != EXPECTED_NAME:
        raise WheelTagError(
            f"{filename!r} name is {name!r}, expected {EXPECTED_NAME!r}"
        )
    if expect_version is not None and version != expect_version:
        raise WheelTagError(
            f"{filename!r} version is {version!r}, "
            f"expected {expect_version!r}"
        )
    if python_tag != EXPECTED_PYTHON or abi_tag != EXPECTED_ABI:
        raise WheelTagError(
            f"{filename!r} is not tagged {EXPECTED_PYTHON}-{EXPECTED_ABI}"
        )
    if platform_tag != EXPECTED_PLATFORM:
        raise WheelTagError(
            f"{filename!r} platform is {platform_tag!r}, "
            f"expected {EXPECTED_PLATFORM!r} "
            f"(must include manylinux2014_aarch64)"
        )

    return WheelInfo(
        filename=filename,
        name=name,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry: check one wheel path and print the parsed tags."""
    parser = argparse.ArgumentParser(
        description=(
            "Assert an ARCO wheel carries the "
            "cp310-abi3 manylinux2014 aarch64 tags (FR-BUILD-07/08)."
        )
    )
    parser.add_argument("wheel", help="Wheel path or filename")
    parser.add_argument(
        "--expect-version",
        default=None,
        help="Require this distribution version in the filename",
    )
    args = parser.parse_args(argv)
    try:
        info = check_wheel_filename(
            args.wheel, expect_version=args.expect_version
        )
    except WheelTagError as exc:
        print(f"wheel_check: {exc}", file=sys.stderr)
        return 1
    print(
        f"{info.filename}: {info.name} {info.version} "
        f"{info.python_tag}-{info.abi_tag}-{info.platform_tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
