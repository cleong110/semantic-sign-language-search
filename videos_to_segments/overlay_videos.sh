#!/bin/bash
set -o errexit 
set -o pipefail
set -o nounset

dir_to_search=$1
# TODO: activate correct env, e.g. signsegment



find "$dir_to_search"/*.pose | parallel -j1 visualize_pose -i "{}" -o "{.}".pose.mp4
find "$dir_to_search"/*.pose | parallel -j1 visualize_pose --normalize -i "{}" -o "{.}".pose.normalized.mp4
find "$dir_to_search"/*.pose | parallel -j1 python overlay_pose_on_video.py -v "{.}".mp4 -p "{}"
find "$dir_to_search"/*.pose | parallel -j1 python overlay_pose_on_video.py -v "{.}".mp4 -p "{}" --normalize -o "{.}".overlay.normalized.mp4
