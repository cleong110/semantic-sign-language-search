#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# usage: ./2_find_lexicon_from_filenames.sh posed_videos > posed_videos_lexicon_list.txt
dir_to_search="$1"


# search for files ending in .mp4 | take everything after the first dash | everything before the .|take out the word "seed"|sort| unique values
find "$dir_to_search" -type f -name "*.mp4"|cut -d "-" -f 1 --complement |cut -d "." -f 1|sed 's/seed//g'|sort|uniq
