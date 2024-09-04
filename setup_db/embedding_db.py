import argparse
from peewee import Model, PostgresqlDatabase, CharField, ForeignKeyField
from pgvector.peewee import VectorField
from pathlib import Path

db_name = "mydb"

# Create an object that handles connections and queries
db = PostgresqlDatabase(db_name)
table_model_names = []

class BaseModel(Model):
    class Meta:
        database = db  # this model uses the database specified with "db_name" above

# class Sign(BaseModel):
#     gloss=CharField(unique=True)
#     language_code=CharField()

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

def drop_and_recreate(model_names):
    db.drop_tables(model_names, safe=True)
    db.create_tables(model_names)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TODO", description="setup embedding search", epilog="TODO"
    )
    parser.add_argument("--recreate", help="drop and recreate database", action="store_true")
    # parser.add_argument("--db_name", help="database to connect to", default=None)
    args = parser.parse_args()

    table_model_names=[SignVideo, 
                #  Sign, 
                 Pose, 
                 Embedding]


    if args.recreate:
        print("dropping and recreating database")
        drop_and_recreate(table_model_names)