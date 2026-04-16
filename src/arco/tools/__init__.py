"""ARCO tool scripts: examples, simulators, and CLI entry points.

Optional dependencies:

- ``tools`` extra (matplotlib, pyyaml) — required by :mod:`arco.tools.examples`
  and the ``arcosim --image`` static image mode.
- ``pygame`` extra (pygame, PyOpenGL) — required by
  :mod:`arco.tools.arcosim` and :mod:`arco.tools.simulator`.

Both sub-packages are silently ignored when their dependencies are absent.

.. deprecated::
    :mod:`arco.tools.arcoex` is deprecated.  Use ``arcosim --image`` instead.
"""
