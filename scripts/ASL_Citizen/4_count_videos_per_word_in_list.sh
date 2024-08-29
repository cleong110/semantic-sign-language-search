#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# usage: ./4_count_videos_per_word_in_list.sh 500_random_words.txt posed_videos > 500_random_words_counts.txt

wordlist_file="$1"
dir_to_search="$2"
#cat "$wordlist_file"|parallel "echo {} && find \"$dir_to_search\" -type f -iname \"*-{}.mp4*\"|grep -v \"overlay\"|wc -l"
search_dir() {
	dir_to_search="$1"
	word_to_find="$2"
	#echo "*-$word_to_find.mp4"
	echo -n "$word_to_find: "
	find "$dir_to_search" -type f -iname "*-$word_to_find.mp4"|grep -v "overlay" -c
}
export -f search_dir

# in parallel, count each one, pipe the result to sort, sort by second column by numerical value
#parallel -a "$wordlist_file" search_dir "$dir_to_search" "{}" 
parallel -a "$wordlist_file" search_dir "$dir_to_search" "{}"| sort -k2 -n 
