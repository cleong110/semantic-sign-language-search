#!/bin/bash
set -o nounset
set -o pipefail
set -o errexit

# ./3_pick_n_random_words_from_lexicon.sh posed_videos_lexicon_list.txt 500 > 500_random_words.txt

# input word list lexicon_list.txt
input_list_file="$1"

# how many words should we grab?
desired_word_count="$2"

# where should we put the wordlist?
#output_wordlist="$3"

# unneeded
#prompt="OK, we will take $desired_word_count words from $input_list_file. Continue? (Y/N):"
#read -r -p "$prompt " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

shuf "$input_list_file" |head -n "$desired_word_count" #> "$output_wordlist"
