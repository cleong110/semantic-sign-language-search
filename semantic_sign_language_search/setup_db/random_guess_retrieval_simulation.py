import random
import numpy as np
import argparse # TODO:
from tqdm import tqdm

def retrieve_n_randomly(item:int, pool_without_item:list, n:int):   
    random.shuffle(pool_without_item) 

    random_n_results = pool_without_item[:n]
    # print(f"{n} random retrieved values from {len(pool_without_item)}: {random_n_results}")
    matches = [retrieved for retrieved in random_n_results if retrieved == item]
    # print(f"matches for {item}: {len(matches)}")
    return len(matches)


if __name__ == "__main__":
    item_classes = 400
    examples_per_class = 10
    count_of_items_to_retrieve = 10
    trials_count = 10000

    mean_match_counts_for_each_trial = []
    print(f"Simulating {item_classes} classes with {examples_per_class} examples each: Running {trials_count} trials")
    for trial_index in tqdm(range(trials_count)):
        retrieval_pool_for_trial = []
        for i in range(item_classes):
            items_for_this_class = [i] * examples_per_class
            retrieval_pool_for_trial.extend(items_for_this_class)

        # print(f"Retrieval pool is now of length {len(retrieval_pool)}")
        # print(retrieval_pool)

        match_counts_for_this_trial = []
        for i, item in enumerate(retrieval_pool_for_trial[:10]):
            pool_without_item = retrieval_pool_for_trial.copy()
            pool_without_item.pop(i)
            # print(f"pool with {item} at index {i} removed, now length {len(pool_without_item)}")
            # print(pool_without_item)
            match_count = retrieve_n_randomly(item, pool_without_item, count_of_items_to_retrieve)
            match_counts_for_this_trial.append(match_count)

        mean_match_count_for_this_trial = np.mean(match_counts_for_this_trial)
        # print(f"Mean match count for trial {trial_index}: {mean_match_count_for_this_trial}")
        mean_match_counts_for_each_trial.append(mean_match_count_for_this_trial)

    mean_mean_match_count_for_all_trials = np.mean(mean_match_counts_for_each_trial)
    print(f"After running {trials_count} trials, the mean of the mean match counts was: {mean_mean_match_count_for_all_trials}")
