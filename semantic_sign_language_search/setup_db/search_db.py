import argparse
from .embedding_db import db, db_name, SignVideo, Embedding, load_pose_embedding
# from embedding_db import db, db_name, SignVideo, Embedding, load_pose_embedding
from peewee import ModelSelect
import numpy as np
import pandas as pd
from pathlib import Path

# TODO: New search features:
# -[ ] Search files in dataset A against dataset B

# TODO: outputs and metrics:
# -[ ] dataframe output instead of print statements
# -[ ] json/csv output
# -[ ] most-confused
# See https://github.com/J22Melody/fairseq/blob/main/examples/MMPT/mmpt/evaluators/metric.py

# DONE:
# -[x] Given a specific embedded video, search against a dataset


# import inquirer # TODO: search options
from . import random_guess_expected_correct_results
# import random_guess_expected_correct_results

def calculate_mean_embedding_vector_norm(joined_embedding_and_signvideo_popularion:ModelSelect):
    pass


def get_subpopulation_matching_filename(vid_name, joined_embedding_and_signvideo_population:ModelSelect):
    # NOTE: do not do .select() on a ModelSelect! Just chain .where() calls. 
    # adding a .select makes a subquery, which means it will only return IDs
    subpopulation = joined_embedding_and_signvideo_population.where(SignVideo.video_path.contains(vid_name))
    return subpopulation
    
def search_population_given_numpy_file(numpy_file, retrieve_n:int, joined_embedding_and_signvideo_population:ModelSelect, gloss_if_known = None):    
    loaded_embedding = load_pose_embedding(numpy_file)[0]
    print(f"Loaded embedding: {loaded_embedding.shape}")
    print(f"Searching population with count {joined_embedding_and_signvideo_population.count()}")

    # results_limit = retrieve_n +1 # later we discard the top result, aka the video itself
    embedding_neighbors = retrieve_from_population_with_embedding(loaded_embedding, retrieve_n, joined_embedding_and_signvideo_population)

    print(f"Embedding Neighbors count: {embedding_neighbors.count()}")
    result_dict = {
        "filename": [],
        "dataset": [],
        "gloss": [],
        "embedding_model": [],
        "match": []
    }

    for i, embedding_neighbor in enumerate(embedding_neighbors):
        neighbor_name = Path(embedding_neighbor.video.video_path).name
        result_dict["filename"].append(neighbor_name)
        result_dict["dataset"].append(embedding_neighbor.video.dataset)
        result_dict["gloss"].append(embedding_neighbor.video.vid_gloss)
        result_dict["embedding_model"].append(embedding_neighbor.embedding_model)
        if gloss_if_known is not None:
            result_dict["match"].append(embedding_neighbor.video.vid_gloss == gloss_if_known)

        else:
            result_dict["match"] = "?"
    
    if gloss_if_known is not None:
        correct_answer_population = joined_embedding_and_signvideo_population.where(SignVideo.vid_gloss == gloss_if_known)
        possible_correct_answer_count = correct_answer_population.count()
    else:
        possible_correct_answer_count = None
    return possible_correct_answer_count, pd.DataFrame(data=result_dict)





def search_vid_in_db_against_population(embedded_vid, retrieve_n:int, joined_embedding_and_signvideo_population:ModelSelect)->tuple:
    result_dict = {
        "filename": [],
        "dataset": [],
        "gloss": [],
        "embedding_model": [],
        "match": []
    }
    # Expects embedded_vid and signvideo joined table
    results_limit = retrieve_n +1 # later we discard the top result, aka the video itself
    vid_path = Path(embedded_vid.video.video_path)
    vid_name = vid_path.name
    #vid_gloss = vid_path.stem.split("-")[-1] # videos are of the form <alphanumeric ID>-<gloss>.mp4
    print(f"{vid_name}, \n\t* gloss: {embedded_vid.video.vid_gloss}, \n\t* dataset:{embedded_vid.video.dataset}, \n\t* embedded with {embedded_vid.embedding_model}")

    # load and put in array so we get (1,768) shape, same as when originally embedded
    #db_pose_embedding = np.array([vid_item.pose_embedding.embedding])

    


    # top 5 closest vectors to this one
    print("\tAND THE CLOSEST VECTORS ARE...")
    embedding_population = joined_embedding_and_signvideo_population

    # pose_embedding_model_for_this = vid_item.pose_embedding.embedding_model

    # if same_pose_embedding_model:        
    #     embedding_population = Embedding.select().where(Embedding.embedding_model == pose_embedding_model_for_this)
    #     print(f"\trestricting to embeddings using the same model: {embedding_population.count()} found")

    # else:
        
    #     embedding_population = Embedding.select()
    #     print(f"\tNot restricting to embeddings using the same model: {embedding_population.count()} found")
        

    correct_answer_population = embedding_population.where(SignVideo.vid_gloss == embedded_vid.video.vid_gloss)
    possible_correct_answer_count = correct_answer_population.count() -1
    # correct_answer_count = embedding_population_videos.where(SignVideo.vid_gloss == vid_item.vid_gloss)
    print(f"\tThere are {possible_correct_answer_count} correct items to retrieve in this population, not counting the video itself")

    embedding_neighbors = retrieve_from_population_with_embedding(embedded_vid.embedding, results_limit, embedding_population)

    # The number of correct answers is: 
    # correct_neighbors = embedding_neighbors.where(Embedding.videos[0].vid_gloss == vid_item.vid_gloss)
    # print(f"There are {correct_neighbors.count()} items with the correct gloss, which we are looking for")

    match_count = 0
    # TODO: try tabulate here?
    output_lines = []
    column_names_and_widths = [
        ("i", 2),    
        ("filename",35), 
        ("dataset",60), 
        ("gloss",30),
        ("embedding_model",30),
        ]
    output_line = "\t\t" + ", ".join([f"{spec[0]:{spec[1]}}" for spec in column_names_and_widths])
    output_lines.append(output_line)

    for i, embedding_neighbor in enumerate(embedding_neighbors):

        neighbor_path = Path(embedding_neighbor.video.video_path)
        if neighbor_path == vid_path:
            continue # it's the same one
        neighbor_name = neighbor_path.name

        widths = [spec_tuple[1] for spec_tuple in column_names_and_widths]
        result_output = (
            f"\t\t{i:<{widths[0]}}, {neighbor_name:{widths[1]}}, {embedding_neighbor.video.dataset:<{widths[2]}}, {embedding_neighbor.video.vid_gloss:<{widths[3]}}, {embedding_neighbor.embedding_model:<{widths[4]}}"
        )
        # outputs = [i, neighbor_name, embedding_neighbor.dataset, embedding_neighbor.vid_gloss]
        # result_output = "\t\t" + ", ".join([f"{output[0]:<{output[1][1]}}" for output in zip(outputs, column_names_and_widths)])

        if embedding_neighbor.video.vid_gloss == embedded_vid.video.vid_gloss:
            result_output = result_output + "\tMATCH!"
            match_count = match_count + 1
        
        # print(result_output)
        result_dict["filename"].append(neighbor_name)
        result_dict["dataset"].append(embedding_neighbor.video.dataset)
        result_dict["gloss"].append(embedding_neighbor.video.vid_gloss)
        result_dict["embedding_model"].append(embedding_neighbor.embedding_model)
        result_dict["match"].append(embedding_neighbor.video.vid_gloss == embedded_vid.video.vid_gloss)
        output_lines.append(result_output)
    for output_line in output_lines:
        print(output_line)
    
    result_df =pd.DataFrame(data=result_dict)
    return match_count, possible_correct_answer_count, result_df

def retrieve_from_population_with_embedding(embedding, results_limit, embedding_population):
    embedding_neighbors = (
        embedding_population
        .order_by(Embedding.embedding.l2_distance(embedding))
        .limit(results_limit)
    )
    
    return embedding_neighbors


def get_embedding_models():
#     mydb=# SELECT DISTINCT dataset FROM signvideo;
#           dataset           
# ----------------------------
#  ASL_Citizen_curated_sample
# (1 row)

# mydb=# SELECT DISTINCT embedding_model FROM embedding;
#           embedding_model          
# -----------------------------------
#  signclip_finetuned_on_asl_citizen
#  temporal
    distinct_rows = Embedding.select(Embedding.embedding_model).distinct()
    return [row.embedding_model for row in distinct_rows]


def get_glosses_in_population(population:ModelSelect):
    distinct_rows = population.select(Embedding.video.vid_gloss).distinct()
    return [row.video.vid_gloss for row in distinct_rows]

def get_gloss_counts_in_population(population:ModelSelect, gloss_values=None):
    if gloss_values is None:
        gloss_values = get_glosses_in_population(population)
    print(f"calculating gloss counts in {population.count()} items")
    
    result_dict = {
        "gloss": [],
        "count": [],
    }
    
    # print(gloss_values)
    for gloss_value in get_glosses_in_population(population):
        # print(gloss_value)
        gloss_count = population.select().where(Embedding.video.vid_gloss==gloss_value).count()
        result_dict["gloss"].append(gloss_value)
        result_dict["count"].append(gloss_count)
    return pd.DataFrame(data=result_dict)

def get_datasets():
    distinct_rows = SignVideo.select(SignVideo.dataset).distinct()
    return [row.dataset for row in distinct_rows]


def search_all_against_all(retrieve_n=10, K=None, population:ModelSelect=None):
    results_limit = retrieve_n +1 # later we discard the top result, aka the video itself
    
    
    print("testing! Let's look at what's in the SignVideo table:")
    if population is None:
        print(f"no population provided, using the whole SignVideo table")
        population = SignVideo.select().join(Embedding)

    
        
    

    population_size = population.count() 
    print(f"Population count for testing is : {population_size}")
    # print(population)
    
    # exit()
    match_counts = []
    possible_correct_answer_counts = []
    for vid_and_embedding_item in population:
        match_count, possible_correct_answer_count, result_df = search_vid_in_db_against_population(embedded_vid=vid_and_embedding_item, 
                                                                                   retrieve_n=retrieve_n, 
                                                                                   joined_embedding_and_signvideo_population=population)
        match_counts.append(match_count)
        possible_correct_answer_counts.append(possible_correct_answer_count)
        expected_if_guessing_for_this_item = random_guess_expected_correct_results.expected_correct_results(N=population_size, n=retrieve_n, K=possible_correct_answer_count)
        print(f"\t{match_count}/{retrieve_n} were matches (search results with the same gloss, not counting the video itself). Random guessing should give on avg: {expected_if_guessing_for_this_item:.3f}")
    print(
        f"Did {len(match_counts)} searches. Mean match count (out of {retrieve_n} search results retrieved each time): {np.mean(match_counts):.3f}"
    )
    if K is not None:
        expected_mean_if_random = random_guess_expected_correct_results.expected_correct_results(N=population_size, n=retrieve_n, K=K)
        print(f"Expected mean match count of randomly retrieving {retrieve_n}, given {K} possible correct results to retrieve: {expected_mean_if_random:.3f}")
    else: 
        K = round(np.mean(possible_correct_answer_counts))
        print(f"STATS! ")
        print(f"* On average each video had {np.mean(possible_correct_answer_counts)} others which were valid matches, which rounds to about {K}")
        print(f"* The lowest number was {np.min(possible_correct_answer_counts)} valid matches, the max number was {np.max(possible_correct_answer_counts)}.")
        print(f"* The most common number was {np.argmax(np.bincount(possible_correct_answer_counts))}")
        expected_mean_if_random = random_guess_expected_correct_results.expected_correct_results(N=population_size, n=retrieve_n, K=K)
        print(f"* Expected mean match count of randomly retrieving {retrieve_n}, given about {K} possible correct results to retrieve: {expected_mean_if_random:.3f}")


def get_population_of_signvideo_and_embedding(pose_embedding_model, dataset):
    # population = SignVideo.select().join(Embedding)
    population = Embedding.select().join(SignVideo)
    if pose_embedding_model is not None:        
        population = population.select().where(Embedding.embedding_model==pose_embedding_model)
        print(f"Selecting items with embedding model {pose_embedding_model}. Population is now {population.count()} ")

    if dataset is not None:
        
        population = population.select().where(SignVideo.dataset==dataset)        
        print(f"Selecting items with dataset {dataset}. Population is now {population.count()} ")
    return population


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SearchSignDB", 
        description="Sign Embedding Search", 
        #epilog="TODO"
    )
    parser.add_argument("--list_datasets", action="store_true", help="List distinct datasets")
    parser.add_argument("--list_embedding_models", action="store_true", help="List distinct embedding models")

    parser.add_argument("--search_model", type=str, default=None, help="Restrict search population to this embedding model")
    parser.add_argument("--search_dataset", type=str, default=None, help="Restrict search population to this dataset")    
    parser.add_argument("--count_pop", action="store_true", help="output the size of the population after Restrictions")
 
    parser.add_argument("--search_all_against_all", action="store_true", help="Test by searching for every video against every other")
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

    if args.list_datasets:
        datasets = get_datasets()
        print("\nDatasets: ")
        for dataset in datasets:
            print(f"* {dataset}:")

    if args.list_embedding_models:
        embedding_models = get_embedding_models()
        print("\nEmbedding Models: ")
        for model in embedding_models:
            print(f"* {model}")
    
    population = get_population_of_signvideo_and_embedding(pose_embedding_model=args.search_model, dataset=args.search_dataset)
    if args.count_pop:
        print(f"After embeddng model and dataset restrictions, population count for testing is: {population.count()}")


    if args.search_all_against_all:
        search_all_against_all(args.retrieve_n, args.number_correct_per_class, population)
        
