import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from embedding_db import db, db_name, SignVideo, Embedding, Pose



def parse_filename_asl_citizen_format(file_path:Path):
    search_string_for_model = "-using-model-"
    path_str = str(file_path.stem)

    model_name=None

    gloss = path_str.split("-")[1] 
    vid_id = str(file_path.stem.split("-")[0])
    
    if search_string_for_model in path_str:
        split_name = path_str.split(search_string_for_model)
        model_name =split_name[-1]
        
    return vid_id, gloss, model_name

def parse_files_in_dir_to_dict(video_dir:Path):
    dictionary_of_filenames = {}
    video_paths = list(video_dir.rglob("*.mp4")) 
    npy_paths = list(video_dir.rglob("*.npy")) 
    for video_path in video_paths:   
        vid_id, gloss, model_name = parse_filename_asl_citizen_format(video_path)
        dictionary_of_filenames[vid_id] = {}
        dictionary_of_filenames[vid_id]["path"] = str(video_path)
        dictionary_of_filenames[vid_id]["gloss"] = gloss
        dictionary_of_filenames[vid_id]["embeddings"] = []

    for npy_path in npy_paths:
        vid_id, gloss, model_name = parse_filename_asl_citizen_format(npy_path)
        # print(vid_id, gloss, model_name)
        if vid_id in dictionary_of_filenames:
            dictionary_of_filenames[vid_id]["embeddings"].append((str(npy_path), model_name))
        else: 
            dictionary_of_filenames[vid_id]= {}
            # TODO: handle this case better. Should not happen

    return dictionary_of_filenames


def add_dir_to_db(video_dir:Path, dataset_name:str):
    parsed_dict = parse_files_in_dir_to_dict(video_dir=video_dir)
    for vid_id, vid_dict in parsed_dict.items():
        # print(vid_id, vid_dict)

        video_path = Path(vid_dict["path"])
        vid_item = SignVideo.create(
            vid_gloss=vid_dict["gloss"],
            video_path=video_path,
            dataset=dataset_name,
        )

        # Pose
        pose_file = video_path.with_suffix(".pose")
        pose_item=Pose.create(video=vid_item,
                              path=pose_file, 
                              pose_format="mediapipe")


        # Embeddings
        for embedding_tuple in vid_dict["embeddings"]:
            embedding_path, model_name = embedding_tuple
            pose_embedding_vector = load_pose_embedding(embedding_path=embedding_path)[0] # shape is (1,768)

            embedding_item = Embedding.create(
                video=vid_item,
                input_modality="pose",
                embedding_model= model_name,
                embedding=pose_embedding_vector,
            )



def populate_with_video_paths(video_dir:Path, dataset_name:str, pose_embedding_model:str = None):

    video_paths = list(video_dir.rglob("*.mp4"))    
    
    print(f"searching {video_dir} for vids, found {len(video_paths)}.")

    # find the ones with embeddings
    print(f"Searching for accompanying embeddings:")
    # TODO: search for .npy files that match the names, elsewhere? e.g. rglob for .npy, video_path.name in 
    if pose_embedding_model is not None:
        suffix = ".npy"
    
    else:
        # search to find 
        suffix = "-using-model-"+ model_name+".npy" 
    video_paths = [video_path for video_path in video_paths if video_path.with_suffix(suffix).is_file()]



    print(f"Found {len(video_paths)} with accompanying .npy file. Adding to db with embedding model {pose_embedding_model}")
    # TODO: bulk inserts using bulk_create https://docs.peewee-orm.com/en/latest/peewee/querying.html#bulk-inserts
    # pose_data_source = []
    # sign_data_source = []
    # embedding_data_source = []
    # signvideo_data_source =[]

    for video_path in tqdm(video_paths):

        vid_gloss = video_path.stem.split("-")[-1] # videos are of the form <alphanumeric ID>-<gloss>.mp4
        pose_file = video_path.with_suffix(".pose")
        pose_embedding = load_pose_embedding(video_path.with_suffix(".npy")) # shape is (1,768)

        pose_item=Pose.create(path=pose_file, pose_format="mediapipe")
        # pose_item = {
        #     "path": pose_file,
        #     "pose_format":"mediapipe"
        #     }

        #sign_item = Sign.get_or_create(gloss=vid_gloss, language_code=language_code)
        pose_embedding_item = Embedding.create(input_modality="pose", 
                                               embedding_model=pose_embedding_model, 
                                               embedding=pose_embedding[0])
        # pose_embedding_item= {
        #     "input_modality":"pose",
        #     "embedding_model":pose_embedding_model,
        #     "embedding":pose_embedding[0],
        # }
        # embedding_data_source.append(pose_embedding_item)

        video_item = SignVideo.create(
            pose_embedding=pose_embedding_item,
            pose=pose_item,
            vid_gloss=vid_gloss, # TODO: figure out Sign table, many to one
            video_path=video_path,                        
            dataset=dataset_name,
        )

        # pose_data_source.append(pose_item) 
    signvideo_count = SignVideo.select().count()
    print(f"Now the SignVideo table has {signvideo_count} rows")


def load_pose_embedding(embedding_path):
    embeddings = np.load(embedding_path)
    # print(f"loaded embeddings with shape {embeddings.shape}")  # (1, 768)
    #   print(f"{embeddings[0]}=") # big long bunch of numbers
    # print(f"loaded embeddings with shape {embeddings}")
    return embeddings

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="TODO", description="setup embedding search", epilog="TODO"
    )
    parser.add_argument("video_dir", type=Path, default=Path.cwd())
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--pose_embedding_model", default=None)

    language_code="ase"
    
    args = parser.parse_args()

    pose_embedding_model = args.pose_embedding_model or "unknown"
    dataset_name = args.dataset_name or args.video_dir.stem

    print(f"dataset_name: {dataset_name}")
    print(f"Pose embedding model: {pose_embedding_model}")
    add_dir_to_db(args.video_dir, dataset_name=dataset_name)
    signvideo_count = SignVideo.select().count()
    print(f"Now the SignVideo table has {signvideo_count} rows, and the Embedding table has {Embedding.select().count()}")
    # populate_with_video_paths(video_dir=args.video_dir, dataset_name=dataset_name, pose_embedding_model=pose_embedding_model)