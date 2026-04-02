# AGENTS Instructions

This file defines repository-wide instructions for coding agents.

## Scope

Applies to all automated agents working in this repository.

## Source of Truth

Follow [docs/guidelines.md](docs/guidelines.md) as the authoritative standard for:

- Architecture and package structure
- Naming conventions
- Docstring style
- Formatting and typing
- Testing and quality gates

## Agent Policy

- Do not introduce patterns that violate [docs/guidelines.md](docs/guidelines.md).
- Keep edits narrow and relevant to the task.
- When changing behavior, update tests in the mirrored tests structure.
- When adding public APIs, include type annotations and Google-style docstrings.
- Prefer project-local conventions over generic defaults.

## Conflict Resolution

When instructions conflict, resolve in this order:

1. Direct maintainer request in the active task
2. [docs/guidelines.md](docs/guidelines.md)
3. This file
