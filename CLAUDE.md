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

- Run `bash scripts/pre_push.sh` — this single command runs all required
  CI gates locally (formatting, tests, examples, smoke tests, videos).  Use
  `--no-examples --no-smoke --no-videos` **only** when the environment
  genuinely cannot run pygame (no virtual framebuffer or no ffmpeg). **Do not
  skip `--no-smoke` as a routine shortcut** — smoke tests are the only local
  gate that imports every simulator module at startup, catching import-time
  `KeyError` / `ImportError` regressions that unit tests miss (see §12 of
  docs/guidelines.md). State the reason in the PR when skipping.
- After restructuring any shared config file (`colors.yml`, etc.): audit every
  consumer and run a quick import check on all simulator entry points before
  pushing (see §12 of docs/guidelines.md).
- Relevant tests pass.
- New public APIs are documented and typed.
- File organization and imports comply with [docs/guidelines.md](docs/guidelines.md).

### Respect the V-cycle

As indicated above, all work of an AI must include all the descending and ascending steps of the cycle. For each row numerated below, the two actions (descend and ascend) must be done **at the same time**:

1. Add a documentation of the work, feature, but... Use Github issues if possible, otherwise, go diretly into the Github PR and document every step in comments. It includes: goal/objectives and acceptable criteria
2. Implement the architecture of the code (classes, public interface, file organization, dependencies) and the (functional) unit tests at the same time. The testing must come first then coding: the performance of the algorithm is independent of its implementation. By reading the acceptance criteria above, you must already know which values to expect.
3. Do the coding. This is the 3-rd step: Fill the stubs lefted by the architecture definition. Implement algorithms, data structure and private/local utilities. Add unit testing for private functions as well (fine testing/non-functional tests).
4. Then, run the tests. Ideally proving 100% coverage (at least 90% would be great!). If this step fails, go back to step 2: Review your architecture, your functional tests, and go back to the cycle.
5. Implement high level simulations if all the testing are passing. Add visual inspection (either images or videos) in the `tools` folder. Add material for the presentation and documentation of the tool. Add the appropriate documentation of the newly implemented feature, of fix the lines affected by the changes. All github workflows must pass: both at push and release! If some is wrong in this step, go back to step number 1.

This complete the V-cycle. Once the acceptance criteria are met and all the Github workflows (autotests) are passing (both at push and release, test them all locally or add the tooling to test it), you can push your branch and trigger the review.