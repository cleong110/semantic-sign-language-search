# requires you to setup postgresql first, see
# https://www.digitalocean.com/community/tutorials/how-to-backup-postgresql-databases-on-an-ubuntu-vps
# TODO: Figure out how to store/search Pose format
# TODO: find a way to specify db_name in args
# TODO: table of input modalities
# TODO: can I also store Text + embeddings
import argparse
from peewee import Model, PostgresqlDatabase, CharField, ForeignKeyField
from pgvector.peewee import VectorField
import numpy as np
from pathlib import Path
from tqdm import tqdm
from embedding_db import db, db_name, SignVideo, Sign, Embedding, Pose





def walk_dir_find_vids(embeddings_dir: Path):
    return embeddings_dir.rglob("*.mp4")


def populate_with_video_paths(video_paths, dataset_name):
    print(f"searching {args.video_dir} for vids, found {len(video_paths)}.")

    # find the ones with embeddings
    print(f"Searching for accompanying embeddings:")
    video_paths = [video_path for video_path in video_paths if video_path.with_suffix(".npy").is_file()]

    print(f"Found {len(video_paths)} with accompanying .npy file. Adding to db")
    for video_path in tqdm(video_paths):
        vid_gloss = video_path.stem.split("-")[-1] # videos are of the form <alphanumeric ID>-<gloss>.mp4
        pose_file = video_path.with_suffix(".pose")
        pose_embedding = load_pose_embedding(video_path.with_suffix(".npy")) # shape is (1,768)

        pose_item=Pose.create(path=pose_file, pose_format="mediapipe")

        sign_item = Sign.get_or_create(gloss=vid_gloss, language_code=language_code)
        pose_embedding_item = Embedding.create(input_modality="pose", embedding_model=pose_embedding_model, embedding=pose_embedding[0])

        video_item = SignVideo.create(
            video_path=video_path,
            vid_gloss=vid_gloss, # TODO: figure out Sign table, many to one
            pose_embedding=pose_embedding_item,
            sign=sign_item,
            pose=pose_item,
            dataset=dataset_name,
        )


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
    parser.add_argument("dataset_name")
    #parser.add_argument("--pose_dir", type=Path, default=Path.cwd()) # TODO: specify a different dir for poses?
    parser.add_argument("--pose_embedding_model", default=None)
    parser.add_argument("--retrieve_k", default=10)
    #parser.add_argument("--db_name", default=None)
    args = parser.parse_args()

    # TODO: how to read in language code?
    language_code="ase"
    pose_embedding_model = args.pose_embedding_model or "unknown"
    results_limit = args.retrieve_k+1 # later we discard the top result, aka the video itself
    #if args.db_name is not None:
    #    db = PostgresqlDatabase(args.db_name)

    print(f"connecting to database {db_name}")
    db.connect()

    video_paths = list(walk_dir_find_vids(args.video_dir))
    populate_with_video_paths(video_paths, args.dataset_name)
