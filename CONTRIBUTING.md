# Contributing to ARCO

Thank you for contributing.

## Prerequisites

- Python 3.10+
- Git

## Setup

```bash
git clone https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

## Development Workflow

1. Create a feature branch from main.
2. Implement the change with focused commits.
3. Add or update tests in the mirrored tests layout.
4. Run local validation before opening a pull request.

## Local Validation

```bash
pytest tests/ -v
python -m black --target-version py312 src/ tests/
python -m isort src/ tests/
```

## Coding Rules

The authoritative coding standard is [docs/guidelines.md](docs/guidelines.md).

All contributions, including AI-assisted changes, must follow this file.

## Pull Request Checklist

- Behavior changes are covered by tests
- Public APIs are typed and documented (Google-style docstrings)
- Formatting and imports are clean
- Diff is scoped to the requested task

## Documentation

Update documentation when adding or changing behavior:

- [README.md](README.md) for user-facing behavior
- [docs/PLANNING.md](docs/PLANNING.md) and algorithm notes for planning changes
- [docs/ROADMAP.md](docs/ROADMAP.md) for future work updates
