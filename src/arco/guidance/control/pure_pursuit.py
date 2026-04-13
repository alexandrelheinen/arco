"""PurePursuitController: pure pursuit path tracking controller.

.. deprecated::
    Import from :mod:`arco.control.pure_pursuit` instead.
"""

from arco.control.pure_pursuit import (
    PurePursuitController,
    _circle_segment_intersection,
    _find_lookahead,
    _wrap_angle,
)

__all__ = [
    "PurePursuitController",
    "_circle_segment_intersection",
    "_find_lookahead",
    "_wrap_angle",
]
