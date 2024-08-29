#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset


src_dir="$1"
dst_dir="$2"

echo "moving .mp4 files"
find "$src_dir" -type f -name "*.pose"|parallel --bar --eta mv "{.}.mp4" "$dst_dir" 

echo "moving .pose files"
find "$src_dir" -type f -name "*.pose"|parallel --bar --eta mv "{}" "$dst_dir"
