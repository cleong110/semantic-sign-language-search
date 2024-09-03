# requires you to setup postgresql first, see
# https://www.digitalocean.com/community/tutorials/how-to-backup-postgresql-databases-on-an-ubuntu-vps
# TODO: Figure out how to store/search Pose format
import argparse
from peewee import Model, PostgresqlDatabase, CharField
import numpy as np
from pathlib import Path

from pgvector.peewee import VectorField

db_name = "mydb"

# Create an object that handles connections and queries
db = PostgresqlDatabase(db_name)


def walk_dir_find_vids(embeddings_dir: Path):
    return embeddings_dir.rglob("*.mp4")


def load_pose_embedding(embedding_path):
    embeddings = np.load(embedding_path)
    #print(f"loaded embeddings with shape {embeddings.shape}")  # (1, 768)
    #   print(f"{embeddings[0]}=") # big long bunch of numbers
    # print(f"loaded embeddings with shape {embeddings}")
    return embeddings 

class BaseModel(Model):
    class Meta:
        database = db  # this model uses the database specified with "db_name" above


class VideoItem(BaseModel):
    # primary key is automatically created actually https://docs.peewee-orm.com/en/latest/peewee/models.html#primary-keys-composite-keys-and-other-tricks
    # id = AutoField()
    video_path = CharField()
    pose_embedding = VectorField(
        dimensions=768
    )  # SignCLIP embedding for example. TODO: sep. table?
    # pose_frames = VectorField(dimensions=768) # TODO: what is the shape of Pose fields?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TODO", description="setup embedding search", epilog="TODO"
    )
    parser.add_argument("video_dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    print(f"connecting to database {db_name}")
    db.connect()

    print(f"dropping and recreating VideoItem table")
    db.drop_tables([VideoItem])
    db.create_tables([VideoItem])

    # TODO: can I also store Text + embeddings
    # TODO: can I even load a numpy array in?
    video_paths = walk_dir_find_vids(args.video_dir)
    print(f"searching {args.video_dir} for vids")
    for video_path in video_paths:
        # video_path = Path("ASL_Citizen_curated_sample_embedded_with_signCLIP/videos/17131909542806167-FRANCE.mp4")
        pose_embedding = load_pose_embedding(video_path.with_suffix(".npy"))
        print(video_path)
        video_item = VideoItem.create(
            video_path=video_path, pose_embedding=pose_embedding[0]
        )

    save_result = video_item.save()
    print(f"saving... result: {save_result}")

    results_limit = 11

    print("testing! Let's look at what's in the VideoItem table:")
    match_counts =[]
    for vid_item in VideoItem.select():
        vid_path = Path(vid_item.video_path)
        vid_name = vid_path.name
        vid_gloss = vid_path.stem.split("-")[-1]
        print(f"{vid_name}, gloss: {vid_gloss}")

        # load and put in array so we get (1,768) shape
        db_pose_embedding = np.array([vid_item.pose_embedding])


        # is it the same as if we load it from the numpy?
        loaded_pose_embedding = load_pose_embedding(Path(vid_item.video_path).with_suffix(".npy"))
        print(
            f"\tIs the loaded the same as the original?\t{np.array_equal(db_pose_embedding,
        loaded_pose_embedding)}"
        )

        # top 5 closest vectors to this one
        print("\tAND THE CLOSEST VECTORS ARE...")

        neighbors = VideoItem.select().order_by(VideoItem.pose_embedding.l2_distance(db_pose_embedding[0])).limit(results_limit)
        match_count = 0
        for i, neighbor in enumerate(neighbors):
            neighbor_path = Path(neighbor.video_path)
            if neighbor_path == vid_path: 
                continue
            neighbor_name = neighbor_path.name
            neighbor_gloss = neighbor_path.stem.split("-")[-1]
            result_output = f"\t\t{i:<2} {neighbor_name:<40}, gloss: {neighbor_gloss:<20}"
            if neighbor_gloss == vid_gloss:
                result_output = result_output + "  MATCH!"
                match_count = match_count + 1
            print(result_output)
        print(f"{match_count}/{results_limit-1} with the same gloss")
        match_counts.append(match_count)
    print(f"Mean match count (up to 4 correct out of {results_limit-1} retrieved) {np.mean(match_counts)}")
