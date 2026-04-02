# CLAUDE Instructions

This repository uses this file as persistent guidance for Claude-based coding agents.

## Mandatory Standard

Use [docs/guidelines.md](docs/guidelines.md) as the required coding and architecture reference for all changes.

## Required Practices

- Respect mapping/planning/guidance boundaries and naming rules.
- Keep one main class per file unless guidelines explicitly allow otherwise.
- Add tests with behavior changes and keep test layout mirrored with source layout.
- Write Google-style docstrings for public interfaces.
- Use explicit typing for public methods and APIs.
- Keep changes small, composable, and aligned with existing style.

## Delivery Quality

Before finishing a task, verify:

- Relevant tests pass.
- New public APIs are documented and typed.
- File organization and imports comply with [docs/guidelines.md](docs/guidelines.md).
