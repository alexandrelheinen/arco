#!/usr/bin/env bash
# scripts/install_hooks.sh
#
# Configures git to use the committed hooks/ directory.
# Run once after cloning:
#
#   bash scripts/install_hooks.sh
#
# This points git's hook search path at hooks/ so that hooks/pre-push
# runs automatically before every `git push`, blocking pushes that fail
# formatting or unit tests.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$REPO_ROOT" config core.hooksPath hooks
chmod +x "$REPO_ROOT/hooks/pre-push"

echo "✅  Git hooks installed. hooks/pre-push will now run before every push."
