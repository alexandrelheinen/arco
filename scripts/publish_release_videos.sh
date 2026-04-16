#!/usr/bin/env bash
# scripts/publish_release_videos.sh
#
# Uploads all scenario MP4 videos from a directory to a GitHub release.
#
# Usage: bash scripts/publish_release_videos.sh <tag> <videos_dir>
#
# Required environment variables:
#   GH_TOKEN  GitHub token with contents:write permission
#
# Exit code: 0 = all uploaded, 1 = at least one failure.

set -euo pipefail

TAG="${1:?Usage: $0 <tag> <videos_dir>}"
VIDEOS_DIR="${2:?Usage: $0 <tag> <videos_dir>}"

SCENARIOS=(astar city ppp rr vehicle rrp occ)

echo "=== Publishing videos to release ${TAG} ==="
echo "Source directory: $VIDEOS_DIR"

FAILED=0
for SCENARIO in "${SCENARIOS[@]}"; do
    MP4="${VIDEOS_DIR}/arcosim_${SCENARIO}.mp4"
    echo "--- $SCENARIO ---"
    if [ ! -f "$MP4" ]; then
        echo "❌  File not found: $MP4"
        FAILED=$((FAILED + 1))
        continue
    fi
    if gh release upload "${TAG}" "$MP4" --clobber; then
        echo "✅  $SCENARIO uploaded"
    else
        echo "❌  $SCENARIO upload failed"
        FAILED=$((FAILED + 1))
    fi
done

echo "======================================"
if [ $FAILED -eq 0 ]; then
    echo "✅  All videos published"
    exit 0
else
    echo "❌  $FAILED video(s) failed to publish"
    exit 1
fi
