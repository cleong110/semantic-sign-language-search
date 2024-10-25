import argparse
from peewee import Model, PostgresqlDatabase, CharField, ForeignKeyField, ModelSelect, RawQuery
from pgvector.peewee import VectorField
from pathlib import Path

db = PostgresqlDatabase(None)
# Function to set up the database connection
#def get_database(db_name):
#    return PostgresqlDatabase(db_name)

# Base model class for all tables
class BaseModel(Model):
    class Meta:
        database = db  # Will be set in main() to avoid module-level execution


# Define the Dataset model
class Dataset(BaseModel):  
    name = CharField(unique=True)

# a video clip with a single sign
class SignVideo(BaseModel): 
    # primary key is automatically created actually https://docs.peewee-orm.com/en/latest/peewee/models.html#primary-keys-composite-keys-and-other-tricks
    # TODO: each file does have a unique name/ID. Use that?
    # pose_embeddings = ForeignKeyField(Embedding, backref="videos") 
    vid_gloss = CharField(null=True)  # Optional gloss annotations
    video_path = CharField()  # Path to the video file
    # pose_frames = VectorField(dimensions=768) # TODO: what is the shape of Pose fields?

# Define the Pose model
class Pose(BaseModel):
    path = CharField()
    pose_format = CharField()
    video = ForeignKeyField(SignVideo, backref="poses")

# Define the Embedding model
class Embedding(BaseModel):
    video = ForeignKeyField(SignVideo, backref="embeddings")
    input_modality = CharField()
    embedding_model = CharField()
    embedding = VectorField(dimensions=768)  # SignCLIP embedding

# Define the many-to-many relationship table between SignVideo and Dataset
class SignVideoDataset(BaseModel):
    sign_video = ForeignKeyField(SignVideo, backref='datasets')
    dataset = ForeignKeyField(Dataset, backref='videos')
    split = CharField()  # Train/test split

# Function to drop and recreate tables
def drop_and_recreate(model_names, db):
    with db:
        for model_name in model_names:
            model_class = globals()[model_name]
            try:
                model_class.drop_table(safe=True)
                model_class.create_table(safe=True)
            except Exception as e:
                print(f"Error with {model_name}: {e}")

def summarize_database():
    # Table counts
    dataset_count = Dataset.select().count()
    video_count = SignVideo.select().count()
    pose_count = Pose.select().count()
    embedding_count = Embedding.select().count()
    video_dataset_count = SignVideoDataset.select().count()

    print("Database Summary:")
    print(f" - Total datasets: {dataset_count}")
    print(f" - Total videos: {video_count}")
    print(f" - Total poses: {pose_count}")
    print(f" - Total embeddings: {embedding_count}")
    print(f" - Total video-dataset associations: {video_dataset_count}")
    print()

    # Distinct dataset names and splits
    dataset_names = [d.name for d in Dataset.select(Dataset.name).distinct()]
    splits = [sd.split for sd in SignVideoDataset.select(SignVideoDataset.split).distinct()]

    print("Datasets and Splits:")
    print(" - Datasets:", ", ".join(dataset_names))
    print(" - Splits:", ", ".join(splits))
    print()

    # Videos and their details
    print("Videos Summary:")
    videos = SignVideo.select()
    for video in videos:
        num_poses = Pose.select().where(Pose.video == video).count()
        num_embeddings = Embedding.select().where(Embedding.video == video).count()
        associated_datasets = (Dataset
                               .select(Dataset.name)
                               .join(SignVideoDataset)
                               .where(SignVideoDataset.sign_video == video))

        print(f" - Video ID {video.id}:")
        print(f"     Gloss: {video.vid_gloss if video.vid_gloss else 'None'}")
        print(f"     Video Path: {video.video_path}")
        print(f"     Number of Poses: {num_poses}")
        print(f"     Number of Embeddings: {num_embeddings}")
        print("     Datasets:", ", ".join([dataset.name for dataset in associated_datasets]))
        print()

def summarize_tables():
    print("Database Tables and Columns:\n")

    # Query to get table names
    tables_query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """

    # Execute the query to get a list of tables
    tables = db.execute_sql(tables_query).fetchall()

    for (table_name,) in tables:
        print(f"Table: {table_name}")

        # Query to get column details for each table
        columns_query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """

        # Execute the query and print column details
        columns = db.execute_sql(columns_query).fetchall()
        for column_name, data_type, is_nullable in columns:
            nullable_text = "YES" if is_nullable == "YES" else "NO"
            print(f"   Column: {column_name}")
            print(f"     - Data Type: {data_type}")
            print(f"     - Nullable: {nullable_text}")
        print()

    print("Database summary complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
       description="Setup semantic sign embedding search database"
    )
    parser.add_argument("--recreate", help="drop and recreate database", action="store_true")
    parser.add_argument('--db_name', type=str, default='mydb_v2', help='Name of the database to connect to')
    # parser.add_argument("--db_name", help="database to connect to", default=None)
    args = parser.parse_args()
    #https://docs.peewee-orm.com/en/latest/peewee/database.html#dynamic-db
    db.init(args.db_name) # connect dynamically

    # Connect to the database
#    db = get_database(args.db_name)

    # Attach the database to the models
#    BaseModel._meta.database = db
    model_names = ["Dataset", "SignVideo", "Pose", "Embedding", "SignVideoDataset"]

    if args.recreate:
        print("dropping and recreating database")
        drop_and_recreate(model_names, db)
    summarize_database()
    summarize_tables()

    
