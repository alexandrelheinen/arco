"""Wheel filename tag checks for FR-BUILD-07 and FR-BUILD-08."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from wheel_check import WheelTagError, check_wheel_filename  # noqa: E402

GOOD = (
    "arco-0.5.0-cp310-abi3-"
    "manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
)


def test_fr_build_07_accepts_abi3_manylinux2014_aarch64_tag() -> None:
    """FR-BUILD-07: the expected aarch64 abi3 manylinux2014 tag passes."""
    info = check_wheel_filename(GOOD, expect_version="0.5.0")
    assert info.name == "arco"
    assert info.version == "0.5.0"
    assert info.python_tag == "cp310"
    assert info.abi_tag == "abi3"
    assert info.platform_tag == (
        "manylinux_2_17_aarch64.manylinux2014_aarch64"
    )


def test_fr_build_08_rejects_cp312_cp312_tag() -> None:
    """FR-BUILD-08: a non-abi3 CPython tag fails the check."""
    bad = (
        "arco-0.5.0-cp312-cp312-"
        "manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
    )
    with pytest.raises(WheelTagError, match="cp310-abi3"):
        check_wheel_filename(bad, expect_version="0.5.0")


def test_fr_build_08_rejects_x86_64_platform_tag() -> None:
    """FR-BUILD-08: an x86_64 platform tag fails the aarch64 check."""
    bad = (
        "arco-0.5.0-cp310-abi3-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    with pytest.raises(WheelTagError, match="aarch64"):
        check_wheel_filename(bad, expect_version="0.5.0")


def test_fr_build_08_rejects_manylinux_2_39_only_tag() -> None:
    """FR-BUILD-08: a newer-only glibc tag fails the manylinux2014 check."""
    bad = "arco-0.5.0-cp310-abi3-manylinux_2_39_aarch64.whl"
    with pytest.raises(WheelTagError, match="manylinux2014_aarch64"):
        check_wheel_filename(bad, expect_version="0.5.0")


def test_fr_build_08_rejects_wrong_version() -> None:
    """FR-BUILD-08: --expect-version must match the wheel filename."""
    with pytest.raises(WheelTagError, match="0.5.0"):
        check_wheel_filename(GOOD, expect_version="0.6.0")


def test_fr_build_07_accepts_path_and_basename() -> None:
    """FR-BUILD-07: a path is checked by its basename only."""
    path = ROOT / "dist" / GOOD
    info = check_wheel_filename(path, expect_version="0.5.0")
    assert info.filename == GOOD
