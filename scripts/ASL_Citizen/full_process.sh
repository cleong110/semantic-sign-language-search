#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset


desired_lexicon_size=500
top_n=400
count_per_word=15
src_dir="./videos"
out_dir="./posed_videos"

mkdir -p "$out_dir"/
./0_parallel_video_to_pose.sh "$src_dir"
./1a_move_finished_poses.sh "$src_dir"/ "$out_dir"/
./1b_remove_the_word_seed_from_filenames.sh "$out_dir"/ 
./1c_remove_spaces_with_underscores_in_filenames.sh "$out_dir"/
./2_find_lexicon_from_filenames.sh "$out_dir" > full_lexicon_from_filenames.txt
./3_pick_n_random_words_from_lexicon.sh full_lexicon_from_filenames.txt "$desired_lexicon_size" > "$desired_lexicon_size"_random_words.txt
./4_count_videos_per_word_in_list.sh "$desired_lexicon_size"_random_words.txt "$out_dir" > "$desired_lexicon_size"_random_words_sorted_by_counts.txt

# take the top N with the most examples. optional, makes it neater because I had at least "$count_per_word" per class
tail -n "$top_n" "$desired_lexicon_size"_random_words_sorted_by_counts.txt |cut -d ":" -f1 > "$top_n"_most_frequent_from_"$desired_lexicon_size"_random_words.txt
./5_make_subset_from_lexicon.sh "$top_n"_most_frequent_from_"$desired_lexicon_size"_random_words.txt "$count_per_word" "$out_dir" ./asl_citizen_"$top_n"_words_"$count_per_word"_examples_each
