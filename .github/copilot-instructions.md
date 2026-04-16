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

**Run `bash scripts/pre_push.sh` as the first and last validation step.**
This single script runs all required CI gates locally:

| Script | Gate |
|--------|------|
| `scripts/check_formatting.sh` | black + isort (blocking), pydocstyle (warning) |
| `scripts/run_tests.sh` | pytest unit tests |
| `scripts/run_examples.sh` | arcosim --image headless image generation |
| `scripts/run_smoke_tests.sh` | arcosim headless recordings |
| `scripts/generate_videos.sh` | arcosim full-length simulation videos |

Additional checks:
- Tests relevant to the change pass locally.
- Public APIs have typing and docstrings.
- File/module naming follows the package conventions.
- Changes remain consistent with [docs/guidelines.md](../docs/guidelines.md).

### Respect the V-cycle

As indicated above, all work of an AI must include all the descending and ascending steps of the cycle. For each row numerated below, the two actions (descend and ascend) must be done **at the same time**:

1. Add a documentation of the work, feature, but... Use Github issues if possible, otherwise, go diretly into the Github PR and document every step in comments. It includes: goal/objectives and acceptable criteria
2. Implement the architecture of the code (classes, public interface, file organization, dependencies) and the (functional) unit tests at the same time. The testing must come first then coding: the performance of the algorithm is independent of its implementation. By reading the acceptance criteria above, you must already know which values to expect.
3. Do the coding. This is the 3-rd step: Fill the stubs lefted by the architecture definition. Implement algorithms, data structure and private/local utilities. Add unit testing for private functions as well (fine testing/non-functional tests).
4. Then, run the tests. Ideally proving 100% coverage (at least 90% would be great!). If this step fails, go back to step 2: Review your architecture, your functional tests, and go back to the cycle.
5. Implement high level simulations if all the testing are passing. Add visual inspection (either images or videos) in the `tools` folder. Add material for the presentation and documentation of the tool. Add the appropriate documentation of the newly implemented feature, of fix the lines affected by the changes. All github workflows must pass: both at push and release! If some is wrong in this step, go back to step number 1.

This complete the V-cycle. Once the acceptance criteria are met and all the Github workflows (autotests) are passing (both at push and release, test them all locally or add the tooling to test it), you can push your branch and trigger the review.