#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# usage: ./5_make_subset_from_lexicon.sh 400_most_frequent_from_500_random_words.txt 10 ./posed_videos ./asl_citizen_400_words_10_examples_each

# given a wordlist...
wordlist_file="$1"

# look for at most this many...
examples_per_word="$2"

# in here...
search_dir="$3"

# and copy them here
output_dir="$4"

paths_file="$output_dir/source_video_paths.txt"

prompt="OK, we will take words from $wordlist_file, and look for videos in $search_dir, then copy up to $examples_per_word to $output_dir, creating it if it doesn't exist. Continue? (Y/N):"
read -r -p "$prompt " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

mkdir -p "$output_dir/videos"
cp -v "$wordlist_file" "$output_dir"

parallel -a "$wordlist_file" "find \"$search_dir\" -type f -iname \"*-{}.mp4*\"|grep -v \"overlay\"|head -n $examples_per_word" > "$paths_file"


echo "copying .mp4 files"
parallel --bar --progress --eta -a "$paths_file" cp "{}" "$output_dir/videos/"

read -r -p "Look in the same dir for matching .pose files, and copy? (Y/N)" confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
echo "copying any matching .pose files"
parallel --bar --progress --eta -a "$paths_file" cp "{.}.pose" "$output_dir/videos/"

read -r -p "Look in the same dir for matching .npy files, and copy? (Y/N)" confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
echo "copying any matching .npy files"
parallel --bar --progress --eta -a "$paths_file" cp "{.}.npy" "$output_dir/videos/"
