#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset


./0_parallel_video_to_pose.sh ./videos
./1a_move_finished_poses.sh ./videos/ ./posed_videos/
./1b_remove_the_word_seed_from_filenames.sh ./posed_videos/ 
./1c_remove_the_word_seed_from_filenames.sh ./posed_videos/
./2_find_lexicon_from_filenames.sh posed_videos > posed_videos_lexicon_list.txt
./3_pick_n_random_words_from_lexicon.sh posed_videos_lexicon_list.txt 500 > 500_random_words.txt
./4_count_videos_per_word_in_list.sh 500_random_words.txt posed_videos > 500_random_words_sorted_by_counts.txt

# optional, makes it neater because I had at least 10 per class
tail -n 400 500_random_words_sorted_by_counts.txt |cut -d ":" -f1 > 400_most_frequent_from_500_random_words.txt
./5_make_subset_from_lexicon.sh 400_most_frequent_from_500_random_words.txt 10 ./posed_videos ./asl_citizen_400_words_10_examples_each
