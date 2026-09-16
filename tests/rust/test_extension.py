"""The compiled extension loads and reports a version matching the package.

A stale extension is the failure mode this guards.  An editable install
keeps the Python sources live while the compiled module is whatever the
last build produced, so a mismatch here means the caller is running Rust
code that does not correspond to the checked-out tree.
"""

from __future__ import annotations

import importlib.metadata

import arco._arco as extension


def test_extension_is_importable():
    """The compiled module loads from inside the arco package."""
    assert extension.__name__.endswith("_arco")


def test_extension_version_matches_the_installed_package():
    """A stale build is caught here rather than in a confusing test later."""
    assert extension.version() == importlib.metadata.version("arco")
