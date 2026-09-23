#!/usr/bin/env bash
# scripts/publish_release_images.sh
#
# Uploads the quiet web image set to a GitHub release.
#
# Usage: bash scripts/publish_release_images.sh <tag> <images_dir> [--dry-run]
#
# Required environment variables (unless --dry-run):
#   GH_TOKEN  GitHub token with contents:write permission
#
# The filename list comes from tools/render_web.py --list, so a new plate
# is published without editing this script.
#
# Exit code: 0 = all uploaded, 1 = at least one failure.

set -euo pipefail

TAG="${1:?Usage: $0 <tag> <images_dir> [--dry-run]}"
IMAGES_DIR="${2:?Usage: $0 <tag> <images_dir> [--dry-run]}"
DRY_RUN=0
if [ "${3:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "=== Publishing images to release ${TAG} ==="
echo "Source directory: $IMAGES_DIR"

REQUIRED=()
while IFS= read -r name; do
    [ -z "$name" ] && continue
    REQUIRED+=("$name")
done < <("$PYTHON" "$REPO_ROOT/tools/render_web.py" --list)

FAILED=0
for NAME in "${REQUIRED[@]}"; do
    FILE="${IMAGES_DIR}/${NAME}"
    echo "--- $NAME ---"
    if [ ! -f "$FILE" ]; then
        echo "❌  File not found: $FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would upload $FILE"
        continue
    fi
    if gh release upload "${TAG}" "$FILE" --clobber; then
        echo "✅  $NAME uploaded"
    else
        echo "❌  $NAME upload failed"
        FAILED=$((FAILED + 1))
    fi
done

echo "======================================"
if [ "$FAILED" -eq 0 ]; then
    echo "✅  All images published"
    exit 0
else
    echo "❌  $FAILED image(s) failed to publish"
    exit 1
fi
