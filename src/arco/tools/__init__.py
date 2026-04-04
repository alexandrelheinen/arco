"""ARCO tools package — optional utilities for examples and simulation.

This package contains utilities for working with ARCO:

- ``arco.tools.config``: Configuration file loading from ${ARCO_ROOT_DIR}/tools/config
- ``arco.tools.logging_config``: Shared logging setup for tool scripts
- ``arco.tools.graph``: Graph generation utilities
- ``arco.tools.viewer``: Visualization utilities (matplotlib-based)
- ``arco.tools.simulator``: Interactive simulators (pygame/OpenGL-based)

To use these tools, install the optional dependencies::

    pip install -e ".[tools]"

For the simulator, also install::

    pip install -e ".[pygame]"
"""
