#!/usr/bin/env bash
# scripts/zip_images.sh
#
# Zips all files in an images directory into a single archive.
#
# Usage: bash scripts/zip_images.sh <images_dir> <output_zip>
#
# Exit code: 0 = success, 1 = failure.

set -euo pipefail

IMAGES_DIR="${1:?Usage: $0 <images_dir> <output_zip>}"
OUTPUT_ZIP="${2:?Usage: $0 <images_dir> <output_zip>}"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌  Images directory not found: $IMAGES_DIR"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_ZIP")"

echo "=== Zipping generated images ==="
echo "Source : $IMAGES_DIR"
echo "Output : $OUTPUT_ZIP"

(cd "$IMAGES_DIR" && zip -r "$OUTPUT_ZIP" .)

echo "✅  Archive created: $OUTPUT_ZIP"
