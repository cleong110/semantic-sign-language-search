#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset


#usage: ./1c_remove_the_word_seed_from_filenames.sh ./posed_videos/
dir_to_search="$1"

remove_space() {
	filename="$1"
	#part_before_dash=$(echo "$filename"|cut -d "-" -f 1)
	#part_after_dot=$(echo "$filename"|cut -d "." -f 2)
	#gloss=$(echo "$filename"|cut -d "-" -f 2|cut -d "." -f 1)
	#filename_without_spaces=$(echo "$filename"| sed 's/ /_/g')
	filename_without_spaces=$(echo "${filename// /_}")
	
	#echo "$filename"
	#echo "$filename_without_spaces"
	#echo "	  $part_before_dash"
	#echo "    $gloss"
	mv -i "$filename" "$filename_without_spaces" 
	
}
export -f remove_space

# search for files ending in .mp4 | take everything after the dash | everything before the .|take out the word "seed"|sort| unique values
find "$dir_to_search" -type f -name "* *.*" |parallel --bar --eta remove_space
