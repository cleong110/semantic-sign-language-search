import argparse
from embedding_db import db, db_name, SignVideo, Embedding
import numpy as np
from pathlib import Path

# import inquirer # TODO: search options


import random_guess_expected_correct_results

def search_vid_against_all(vid_item:SignVideo, retrieve_n:int, same_pose_embedding_model=True)->int:
    # TODO: different datasets, 
    # TODO: same
    results_limit = retrieve_n +1 # later we discard the top result, aka the video itself
    vid_path = Path(vid_item.video_path)
    vid_name = vid_path.name
    #vid_gloss = vid_path.stem.split("-")[-1] # videos are of the form <alphanumeric ID>-<gloss>.mp4
    print(f"{vid_name}, gloss: {vid_item.vid_gloss}")

    # load and put in array so we get (1,768) shape, same as when originally embedded
    #db_pose_embedding = np.array([vid_item.pose_embedding.embedding])

    # top 5 closest vectors to this one
    print("\tAND THE CLOSEST VECTORS ARE...")
    pose_embedding_model_for_this = vid_item.pose_embedding.embedding_model

    if same_pose_embedding_model:
        print(f"restricting to embeddings using the same model")
        embedding_population = Embedding.select().where(Embedding.embedding_model == pose_embedding_model_for_this)

    else:
        print(f"Not restricting to embeddings using the same model")
        embedding_population = Embedding.select()
        

    embedding_neighbors = (
        embedding_population
        .order_by(Embedding.embedding.l2_distance(vid_item.pose_embedding.embedding))
        .limit(results_limit)
    )

    match_count = 0
    for i, embedding_neighbor in enumerate(embedding_neighbors):

        # TODO: refactor as in https://github.com/coleifer/peewee/issues/1667
        # .videos gives you a ModelSelect, something like this:  
        #       SELECT "t1"."id", "t1"."pose_embedding_id", "t1"."pose_id", "t1"."vid_gloss", "t1"."video_path" FROM "signvideo" AS "t1" WHERE ("t1"."pose_embedding_id"     = 1)
        # we just want the first one, there should only be one
        neighbor=embedding_neighbor.videos[0]
        neighbor_path = Path(neighbor.video_path)
        if neighbor_path == vid_path:
            continue
        neighbor_name = neighbor_path.name
        #neighbor_gloss = neighbor_path.stem.split("-")[-1]
        result_output = (
            f"\t\t{i:<2} {neighbor_name:<40}, gloss: {neighbor.vid_gloss:<30}"
        )
        if neighbor.vid_gloss == vid_item.vid_gloss:
            result_output = result_output + "  MATCH!"
            match_count = match_count + 1
        print(result_output)
    
    return match_count



def search_all_against_all(retrieve_n=10, K=None, same_pose_embedding_model=True):
    results_limit = retrieve_n +1 # later we discard the top result, aka the video itself


    # TODO FIXME make the pose embedding model settable. "search all against all that use this specific model"
    if same_pose_embedding_model:
        print(f"restricting to embeddings using the same model")
        embedding_population = Embedding.select().where(Embedding.embedding_model == pose_embedding_model_for_this)

    else:
        print(f"Not restricting to embeddings using the same model")
        embedding_population = Embedding.select()
    
    
    print("testing! Let's look at what's in the SignVideo table:")
    population_size = SignVideo.select().count()
    match_counts = []
    for vid_item in SignVideo.select():
        match_count = search_vid_against_all(vid_item=vid_item, retrieve_n=retrieve_n)
        match_counts.append(match_count)
        print(f"{match_count}/{results_limit-1} with the same gloss")
    print(
        f"Did {len(match_counts)} searches. Mean match count (out of {results_limit-1} search results retrieved each time): {np.mean(match_counts):.3f}"
    )
    if K is not None:
        expected_mean_if_random = random_guess_expected_correct_results.expected_correct_results(N=population_size, n=retrieve_n, K=K)
        print(f"Expected mean match count if randomly retrieving {retrieve_n}, given {K} possible correct results to retrieve: {expected_mean_if_random:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TODO", description="setup embedding search", epilog="TODO"
    )
    parser.add_argument('-n', "--retrieve_n", default=10, type=int, help="Number of search results retrieved")
    #parser.add_argument('-N', "--population_size", type=int, help="Total population size") # calculable.
    parser.add_argument('-K', "--number_correct_per_class", type=int, help="Number of correct search results in the population")    
    args = parser.parse_args()

    # questions = [
    # inquirer.List('size',
    #                 message="What size do you need?",
    #                 choices=['Jumbo', 'Large', 'Standard', 'Medium', 'Small', 'Micro'],
    #             ),
    # ]
    # answers = inquirer.prompt(questions)
    search_all_against_all(args.retrieve_n, args.number_correct_per_class)    
