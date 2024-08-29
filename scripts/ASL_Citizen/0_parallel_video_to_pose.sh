#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset


dir_to_search="$1"
#batch_size=100
#find ./videos/ -type f -name "*.mp4"|head -n "$batch_size" |parallel --progress --results "/tmp/" video_to_pose --format mediapipe -i "{}" -o "{.}.pose"
find "$dir_to_search" -type f -name "*.mp4"|parallel -j6 --progress --eta --results "/tmp/" video_to_pose --format mediapipe -i "{}" -o "{.}.pose"
