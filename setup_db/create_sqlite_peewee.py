# requires you to setup postgresql first, see
# https://www.digitalocean.com/community/tutorials/how-to-backup-postgresql-databases-on-an-ubuntu-vps
# TODO: Figure out how to store/search Pose format
# TODO: find a way to specify db_name in args
# TODO: table of input modalities
# TODO: can I also store Text + embeddings
import argparse
from peewee import Model, PostgresqlDatabase, CharField, ForeignKeyField
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pgvector.peewee import VectorField

db_name = "mydb"

# Create an object that handles connections and queries
db = PostgresqlDatabase(db_name)


def walk_dir_find_vids(embeddings_dir: Path):
    return embeddings_dir.rglob("*.mp4")

def drop_and_recreate(table_names):
    db.drop_tables(model_names, safe=True)
    db.create_tables(model_names)
  
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

def search_all_against_all():
    print("testing! Let's look at what's in the SignVideo table:")
    match_counts = []
    for vid_item in SignVideo.select():
        vid_path = Path(vid_item.video_path)
        vid_name = vid_path.name
        #vid_gloss = vid_path.stem.split("-")[-1] # videos are of the form <alphanumeric ID>-<gloss>.mp4
        print(f"{vid_name}, gloss: {vid_item.vid_gloss}")

        # load and put in array so we get (1,768) shape, same as when originally embedded
        #db_pose_embedding = np.array([vid_item.pose_embedding.embedding])

        # top 5 closest vectors to this one
        print("\tAND THE CLOSEST VECTORS ARE...")

        embedding_neighbors = (
            Embedding.select()
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
        print(f"{match_count}/{results_limit-1} with the same gloss")
        match_counts.append(match_count)
    print(
        f"Mean match count (out of {results_limit-1} search results retrieved each time) {np.mean(match_counts)}"
    )


class BaseModel(Model):
    class Meta:
        database = db  # this model uses the database specified with "db_name" above

class Sign(BaseModel):
    gloss=CharField(unique=True)
    language_code=CharField()

class Embedding(BaseModel):
    input_modality=CharField()
    embedding_model=CharField()
    embedding = VectorField(
        dimensions=768
    )  # SignCLIP embedding for example. 

class Pose(BaseModel):
    path=CharField()
    pose_format=CharField()


#class SignVideoToEmbedding(BaseModel): # one to many relationships
#    # We could have multiple embeddings per video. TODO: fill this out

class SignVideo(BaseModel): # clip of a single Sign
    # primary key is automatically created actually https://docs.peewee-orm.com/en/latest/peewee/models.html#primary-keys-composite-keys-and-other-tricks
    # TODO: each file does have a unique name/ID. Use that?
    pose_embedding = ForeignKeyField(Embedding, backref="videos") 
    pose = ForeignKeyField(Pose, backref="videos") # TODO: allow null? we might not have pose data for this
    #sign = ForeignKeyField(Sign, backref="videos", null=True) # maybe we dunno yet
    vid_gloss = CharField() # maybe we dunno yet
    # id = AutoField()
    video_path = CharField()
    dataset=CharField()
    # pose_frames = VectorField(dimensions=768) # TODO: what is the shape of Pose fields?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TODO", description="setup embedding search", epilog="TODO"
    )
    parser.add_argument("video_dir", type=Path, default=Path.cwd())
    parser.add_argument("dataset_name")
    #parser.add_argument("--pose_dir", type=Path, default=Path.cwd()) # TODO: specify a different dir for poses?
    parser.add_argument("--pose_embedding_model", default=None)
    parser.add_argument("--retrieve_k", default=10)
    parser.add_argument("--recreate", help="drop and recreate database", action="store_true")
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

    model_names=[SignVideo, Sign, Pose, Embedding]

    if args.recreate:
        print("dropping and recreating database")
        drop_and_recreate(model_names)


    video_paths = list(walk_dir_find_vids(args.video_dir))
    populate_with_video_paths(video_paths, args.dataset_name)

    search_all_against_all()
