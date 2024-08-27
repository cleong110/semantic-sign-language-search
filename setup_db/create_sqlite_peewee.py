# requires you to setup postgresql first, see
# https://www.digitalocean.com/community/tutorials/how-to-backup-postgresql-databases-on-an-ubuntu-vps
from peewee import *
import numpy as np
from pathlib import Path

from pgvector.peewee import VectorField


db = PostgresqlDatabase("mydb")



def load_pose_embedding(embedding_path):
    embeddings = np.load(embedding_path)
    print(f"loaded embeddings with shape {embeddings.shape}") # (1, 768)
 #   print(f"{embeddings[0]}=") # big long bunch of numbers
    # print(f"loaded embeddings with shape {embeddings}")
    return embeddings




class VideoItem(Model):
    # primary key is automatically created actually https://docs.peewee-orm.com/en/latest/peewee/models.html#primary-keys-composite-keys-and-other-tricks
    # id = AutoField()
    video_path = CharField() 
    pose_embedding = VectorField(dimensions=768) # SignCLIP embedding for example
    #pose_vector = VectorField(dimensions=768) # TODO: what is the shape of these

    class Meta:
        database = db # this model uses the "people.db" database

if __name__ == "__main__":
    db.connect()
    #db.create_tables([Person, Pet, ITEM])
    db.create_tables([VideoItem])

    # TODO: can I also store Text + embeddings
    # TODO: can I even load a numpy array in?
    video_path = Path("ASL_Citizen_curated_sample_embedded_with_signCLIP/videos/17131909542806167-FRANCE.mp4")
    pose_embedding = load_pose_embedding(video_path.with_suffix(".npy"))
    print(video_path)
    video_item = VideoItem.create(video_path=video_path, pose_embedding=pose_embedding[0])
    print(f"saving... result: {video_item.save()}")

    for vid_item in VideoItem.select():
        print(vid_item.video_path)
        loaded_pose_embedding = vid_item.pose_embedding
        print(loaded_pose_embedding.shape) # (768,)
        loaded_pose_embedding_np = np.array([loaded_pose_embedding])
        print(loaded_pose_embedding_np.shape) # (1, 768)
