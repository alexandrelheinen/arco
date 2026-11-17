# Building the linux aarch64 wheel

ARCO publishes a `cp310-abi3` manylinux2014 aarch64 wheel so a Raspberry
Pi 5 (Ubuntu 24.04, CPython 3.12) can `pip install` without a Rust
toolchain (`FR-BUILD-06` through `FR-BUILD-09`). GitHub Actions credit
expired on 2026-10-31, so the wheel is built and verified locally; the
`linux-aarch64` entry in `.github/workflows/wheels.yml` is dormant until
credit returns.

## Prerequisites (x86_64 Linux host)

```bash
rustup target add aarch64-unknown-linux-gnu
python3 -m pip install 'maturin>=1.7,<2' ziglang
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

`scripts/build_wheel_linux_aarch64.sh` exits 127 and prints the install
command if `maturin`, `ziglang`/`zig`, or the aarch64 Rust target is
missing.

## Build

From a checkout that has the scripts (usually `main`):

```bash
bash scripts/build_wheel_linux_aarch64.sh
```

To attach the wheel to the existing `v0.5.0` release (the FRET pin),
build from a worktree of that tag while using the scripts from `main`:

```bash
git fetch --tags
git worktree add /tmp/arco-v0.5.0 v0.5.0
bash scripts/build_wheel_linux_aarch64.sh /tmp/arco-v0.5.0
```

The script writes exactly one file:

```text
dist/arco-<version>-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

under the source tree (so under `/tmp/arco-v0.5.0/dist/` for the
worktree case). It prints the wall time and the SHA-256.

### Alternatives

1. **manylinux container under buildx** (slower, QEMU): if a crate fails
   under `zig cc`, run maturin inside a manylinux image with
   `docker buildx build --platform linux/arm64` (or `docker run
   --platform linux/arm64`). The wheel must still carry the
   `manylinux2014_aarch64` tag.
2. **Native build on the Pi**: fast, but do not use a bare host build
   that would tag the Pi's newer glibc. Use the manylinux container or
   `--zig` so the platform tag stays `manylinux_2_17_aarch64.manylinux2014_aarch64`.

## Verify

```bash
bash scripts/verify_wheel_aarch64.sh \
  /tmp/arco-v0.5.0/dist/arco-0.5.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
  --expect-version 0.5.0
```

The script:

1. Runs `tools/wheel_check.py` (name, version, `cp310-abi3`,
   `manylinux_2_17_aarch64.manylinux2014_aarch64`).
2. Starts fresh `linux/arm64` containers for `python:3.10-slim` and
   `ubuntu:24.04` (CPython 3.12), prints `uname -m` (must be
   `aarch64`), installs with `pip install --only-binary=:all:`, and runs
   `scripts/wheel_smoke.py`.
3. Prints the SHA-256 of the verified wheel.

If binfmt is missing, the script detects `exec format error` and prints
the `tonistiigi/binfmt --install arm64` command; it never reports a
false pass.

Record the hash:

```bash
sha256sum /tmp/arco-v0.5.0/dist/*.whl | tee /tmp/arco-v0.5.0/dist/SHA256SUMS
```

## Release attach (human only)

After the tooling PR is merged and Alexandre writes the explicit release
words, attach the verified wheel and `SHA256SUMS` to the chosen release
(recommended: existing `v0.5.0`):

```bash
gh release upload v0.5.0 \
  dist/arco-0.5.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
  dist/SHA256SUMS \
  -R alexandrelheinen/arco
curl -fsSL -o /tmp/check.whl \
  https://github.com/alexandrelheinen/arco/releases/download/v0.5.0/arco-0.5.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
sha256sum /tmp/check.whl
```

Add one line to the release notes with the platform tag and the SHA-256.
Nothing is removed from the release; only assets are added.

## Handoff: FRET OPS-05

FRET does not reimplement this build. OPS-05 invokes the scripts in this
document (or downloads the published wheel from the ARCO release) and
checks the SHA-256 against `SHA256SUMS` before the Pi image pins the
asset. The marker line lands in FRET PR5.
