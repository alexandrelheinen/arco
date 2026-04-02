# Copilot Instructions for ARCO

These instructions apply to all GitHub Copilot chat and coding-agent interactions in this repository.

## Mandatory Rule

Always follow [docs/guidelines.md](../docs/guidelines.md) as the authoritative coding standard.

If there is any conflict between an ad hoc request and the guidelines, prioritize the guidelines unless the user explicitly asks to change the guidelines themselves.

## Required Behaviors

- Preserve project architecture boundaries: mapping, planning, guidance.
- Keep one primary class per file unless a tightly coupled nested helper is justified by the guidelines.
- Add or update tests alongside behavior changes.
- Use Google-style docstrings for public APIs.
- Keep formatting and import order compliant with Black and isort.
- Prefer minimal, focused diffs and avoid unrelated refactors.

## Validation Checklist Before Finalizing

- Tests relevant to the change pass locally.
- Public APIs have typing and docstrings.
- File/module naming follows the package conventions.
- Changes remain consistent with [docs/guidelines.md](../docs/guidelines.md).
