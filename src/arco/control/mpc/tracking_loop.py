"""MPCTrackingLoop, re-exported from the compiled extension.

The implementation is ``PyMpcTrackingLoop`` in the ``arco-py`` crate. It
drives a Python vehicle and a Python tracker, so it holds the interpreter
for the whole of a step rather than releasing it: every line of the step
is a call back into the interpreter.
"""

from arco._arco import MPCTrackingLoop

__all__ = ["MPCTrackingLoop"]
