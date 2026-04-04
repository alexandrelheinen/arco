# ARCO Tools Migration Guide

## Summary

The `tools` directory has been moved from the repository root to `src/arco/tools` and is now a proper Python package within the ARCO namespace.

## What Changed

### Directory Structure

- **Before**: `tools/` at repository root
- **After**: `src/arco/tools/` as part of the arco package

### Import Statements

All imports have been updated:

```python
# Before (with sys.path hacks)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import load_config
from logging_config import configure_logging
from viewer.graph import draw_graph

# After (clean imports)
from arco.tools.config import load_config
from arco.tools.logging_config import configure_logging
from arco.tools.viewer.graph import draw_graph
```

### Running Examples

```bash
# Before
python tools/examples/astar_graph.py
python tools/examples/rrt_planning.py

# After
python -m arco.tools.examples.astar_graph
python -m arco.tools.examples.rrt_planning
```

### Running Simulators

```bash
# Before
cd tools/simulator
python main/astar.py

# After
python -m arco.tools.simulator.main.astar
```

### Configuration Files

Configuration files are now loaded from `${ARCO_ROOT_DIR}/config/` where:
- Default `ARCO_ROOT_DIR` is `<repo>/src/arco/tools`
- Can be overridden via environment variable for custom file systems

```bash
export ARCO_ROOT_DIR=/path/to/custom/root
# Configs will be loaded from /path/to/custom/root/config/
```

### Installation

The tools are now installed as part of the arco package:

```bash
# Install core library only
pip install -e .

# Install with tools (examples, viewers, config)
pip install -e ".[tools]"

# Install with simulator support
pip install -e ".[simulator]"
```

## Benefits

1. **No more sys.path hacks**: All imports are clean and explicit
2. **Proper package structure**: Tools are now part of the arco namespace
3. **Optional installation**: Tools can be installed separately from core library
4. **Configurable**: ARCO_ROOT_DIR environment variable for custom deployments
5. **Better IDE support**: IDEs can now properly resolve imports and provide autocomplete

## Migration Checklist

If you have custom code using the old structure:

- [ ] Update all `from config import` to `from arco.tools.config import`
- [ ] Update all `from viewer.X import` to `from arco.tools.viewer.X import`
- [ ] Update all `from graph.generator import` to `from arco.tools.graph.generator import`
- [ ] Remove all `sys.path.insert()` calls
- [ ] Change script execution from `python tools/examples/X.py` to `python -m arco.tools.examples.X`
- [ ] Reinstall with `pip install -e ".[tools]"` to get the tools package
