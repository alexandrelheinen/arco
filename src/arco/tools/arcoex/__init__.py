"""arcoex — ARCO examples platform (deprecated).

.. deprecated::
    ``arcoex`` is deprecated.  Use ``arcosim --image`` instead::

        # was: arcoex map/city.yml --save output.png
        arcosim map/city.yml --image --record output.png

Provides the ``arcoex`` CLI entry point for running ARCO example scenarios
from YAML scenario files as a thin wrapper over ``arcosim --image``.
Requires the ``tools`` optional dependency group::

    pip install arco[tools]
"""
