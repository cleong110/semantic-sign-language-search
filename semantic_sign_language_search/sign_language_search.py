import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from setup_db.search_db import search_vid_in_db_against_population, get_embedding_models, get_population_of_signvideo_and_embedding, get_datasets, get_subpopulation_matching_filename, search_population_given_numpy_file, get_glosses_in_population, get_gloss_counts_in_population
from setup_db import embedding_db
test_data = Path("/home/vlab/projects/semantic-sign-language-search/test_data")
test_mp4s = list(test_data.rglob("*.mp4"))

def display_retrieval_results(test_mp4s, possible_correct_answer_count, result_df):
    
    st.dataframe(result_df,use_container_width=False, column_order=["match", "filename","gloss"])

    match_count = (result_df.match == True).sum()
    if possible_correct_answer_count is not None:
        st.write(f"Out of {possible_correct_answer_count} correct videos to retrieve within the population, retrieved {match_count}")
    else: 
        st.write(f"Correct gloss not know, cannot calculate how many possible correct answers")
            
    match_filenames = list(result_df["filename"])
    # st.write(match_filenames)
    match_mp4s = []


    st.subheader("Retrieved Videos:")
    for i, filename in enumerate(match_filenames):
        # st.write(i, len(test_mp4s))
        found_example = False
        
        for test_mp4 in test_mp4s:
            # st.write(f"Checking if {filename} in {test_mp4}")
            if filename in str(Path(test_mp4)):
                st.write(f"{filename} found at {test_mp4}")
                match_mp4s.append(test_mp4)
                # match_filenames.pop(i)
                st.video(test_mp4.open("rb"))
                # st.write(test_mp4)
        if found_example:
            print(f"Found {filename}, so let's skip")
            continue
                        
    # st.write(match_mp4s)

# from ..setup_db.search_db import search_vid_against_population
# from setup_db.search_db import search_vid_against_population
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py
st.title("Sign Language Video Search")

st.write("querying database...")

embedding_models=get_embedding_models()
embedding_models.append(None)


datasets_in_db = get_datasets()
datasets_in_db.append(None)


st.write("Loading database:")
search_dataset = st.selectbox(f"Do you want to restrict to a specific dataset?", datasets_in_db)
search_model = st.selectbox("Do you want to restrict to a specific embedding model?", embedding_models)
# print(search_dataset)
# print(search_model)
# print(type(search_model))

if search_dataset is not None:
    population = get_population_of_signvideo_and_embedding(pose_embedding_model=search_model, dataset=search_dataset)
    glosses_in_population = get_glosses_in_population(population)
    st.write(f"After embedding model and dataset restrictions, population count for testing is: {population.count()}")
    st.write(f"There are {len(glosses_in_population)} glosses in the population")
    

    if st.button("Show detailed counts:"):
        st.dataframe(get_gloss_counts_in_population(population))
    # else:
    #     get_gloss_counts_in_population(population, glosses_in_population[:5])

uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    st.write(uploaded_file.name)
    st.write(Path(uploaded_file.name))
    
    st.header("Search Results")
    # for thing in population:
    #     print(thing.embedding_model)
    population_with_this_name = get_subpopulation_matching_filename(Path(uploaded_file.name).stem, population)
    # for thing in population_with_this_name:
    #     print(thing.embedding_model)
    st.write(f"{population_with_this_name.count()} files with this name in population")
    print(population_with_this_name)

    
    if population_with_this_name:

        st.write("Item exists in database")
        # print("**********************")
        # print(population_with_this_name.select().dicts())
        # print(population_with_this_name._row_type )
        # print(population_with_this_name._joins)
        # print(population_with_this_name.model)
        # print(type(population_with_this_name))
        # print(population_with_this_name.count())
        # print("&&&&&&&&&&&&&&&77")
        # items_with_this_name = [i.embedding_model for i in population_with_this_name]
        # for foo in population_with_this_name:
            
        #     # current join context is Embedding
        #     # dict_keys(['id', 'video', 'input_modality', 'embedding_model', 'embedding'])
            
        #     print(type(foo))
        #     print(foo.embedding_model)
        #     print(foo.video_id)
        #     print(dir(foo))
            
            # print(i.video)
            # print(f"embedding model: {i.embedding.embedding_model}")

            # print(i.video.video_path)
            # print(i.embedding_model)
        # selected_model = st.selectbox(f"{population_with_this_name.count()} embeddings in restricted population with name: {uploaded_file.name}", items_with_this_name)

        st.write(f"{population_with_this_name.count()} embeddings in restricted population with name: {uploaded_file.name}")
        # with open("/home/vlab/projects/semantic-sign-language-search/test_data/asl_citizen_400_words_10_examples_each/videos/6761850646856167-THIEF_2.mp4", "rb") as video_file:
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)

        
        
        for item in population_with_this_name:
            match_count, possible_correct_answer_count, result_df = search_vid_in_db_against_population(item, 10, population)
            display_retrieval_results(test_mp4s, possible_correct_answer_count, result_df)
            

    else:
        st.write("Item is not in database, attempting to load numpy embedding")
        if uploaded_file.name.endswith(".npy"):
            possible_gloss = uploaded_file.name.split("-")[1]
            st.write(f"Parsed filename, gloss is possibly {possible_gloss}, is this correct?")
            if st.checkbox("Is this correct?",value=True):
                gloss = possible_gloss
                st.write("")
            else:
                gloss = st.text_input(label="Do you know the gloss?")
                if gloss:
                    st.write(f"Gloss: {gloss}")
                else:
                    gloss = None
            if st.button("search"):
                possible_correct_answer_count, result_df = search_population_given_numpy_file(uploaded_file, 10, population,gloss_if_known=gloss)
                display_retrieval_results(test_mp4s, possible_correct_answer_count, result_df)
            # match_count = (result_df.match == True).sum()