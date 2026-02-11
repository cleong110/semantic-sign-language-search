import argparse
import csv
import re
from itertools import chain
from pathlib import Path
import shutil
from typing import Iterator, Set, Optional, Union
from pose_format import Pose
import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm import tqdm
import random
from pyzstd import decompress

# conda activate /opt/home/cleong/envs/papc


def read_csvs_to_dfs(directory: Path) -> pd.DataFrame:
    """Finds and reads train.csv, test.csv, etc"""
    dfs_list = []
    csv_files = list(directory.rglob("*.csv"))
    for split_csv in csv_files:
        df = pd.read_csv(split_csv)

        print(f"Class count: {len(df['Gloss'].unique())}")
        # Extract the first part of "Video file" before "-"
        df["Video ID"] = df["Video file"].str.split("-").str[0]
        df["Split"] = split_csv.stem

        print(f"{split_csv.stem} read to df")
        # print(df.head())
        dfs_list.append(df)

    if dfs_list:
        df_combined = pd.concat(dfs_list, ignore_index=True)
    else:
        df_combined = pd.DataFrame()
    return df_combined


def get_pose_files(directory: Path) -> Iterator[Path]:
    """
    Lazily finds all '.pose' files in the given directory.
    """
    return chain(directory.rglob("*.pose"), directory.rglob("*.pose.zst"))


def get_pose_file_stem(file_path: Path):
    """Gets you the actual stem even for .pose.zst files"""

    if file_path.name.endswith(".pose.zst"):
        return Path(file_path.stem).stem
    else:
        return file_path.stem


def get_pose_data(file_path: Union[Path, str]) -> Pose:
    """Loads a .pose or .pose.zst, returns a Pose object"""
    file_path = Path(file_path)
    if file_path.name.endswith(".pose.zst"):
        return Pose.read(decompress(file_path.read_bytes()))
    else:
        return Pose.read(file_path.read_bytes())

def get_pose_stats(pose:Pose):
    stats_dict = {
        "fps": pose.body.fps,
        "total_points": np.prod(pose.body.data.shape),
        "masked_points": np.sum(ma.getmaskarray(pose.body.data))

    }

    return stats_dict


def gather_asl_citizen_df(directory: Path):
    base_path = Path(directory)
    pose_files = list(get_pose_files(base_path))
    video_files = list(directory.rglob("*.mp4"))
    video_files_df = pd.DataFrame(
        {
            "Video ID": [path.name.split("-")[0] for path in video_files],
            "Video File Name": [path.name for path in video_files],
            "Video File Path": [path for path in video_files],
        }
    )
    print(pose_files[:5])

    # Convert file paths to a DataFrame
    df_paths = pd.DataFrame(
        {
            "Pose File Path": [str(f) for f in pose_files],
            "Pose File Name": [f.name for f in pose_files],
        }
    )
    df_paths["Video ID"] = df_paths["Pose File Name"].str.extract(r"/?([^/-]+)-")
    print("PATHS DF")
    print(df_paths.head())

    # Extract Video ID from the filenames in the paths
    # df_paths["Video ID"] = df_paths["File Path"].name.split("-").str[0]
    # print(df_paths.head())

    splits_df = read_csvs_to_dfs(base_path)
    print("SPLITS DF")
    print(splits_df.head())

    # merge the pose paths
    df_merged = splits_df.merge(df_paths, on="Video ID", how="left")
    # merge the mp4 paths
    df_merged = df_merged.merge(video_files_df, on="Video ID", how="left")
    df_merged = df_merged.dropna(subset=["Pose File Path"]).reset_index(drop=True)
    print(df_merged.info())
    print(df_merged.head())
    return df_merged


def gather_semlex_df(directory: Path):

    metadata_file = list(directory.rglob("semlex_metadata.csv"))
    print(metadata_file)

    metadata_df = pd.read_csv(str(metadata_file[0]))
    print(metadata_df)

    pose_files = get_pose_files(directory)

    pose_files_dict = {
        "Pose File Path": [],
        "Pose File Name": [],
        "video_id": [],
        "fps": [],
    }

    for pose_file in pose_files:
        pose_files_dict["Pose File Name"].append(pose_file.name)
        pose_files_dict["Pose File Path"].append(str(pose_file))
        video_id = get_pose_file_stem(pose_file)
        pose_files_dict["video_id"].append(video_id)

        pose = get_pose_data(pose_file)
        pose_stats = get_pose_stats(pose)
        for key, value in pose_stats.items():
            # key = f"Pose {key}"
            if key not in pose_files_dict:
                pose_files_dict[key] = []
            
            pose_files_dict[key].append(value)


    pose_df = pd.DataFrame(pose_files_dict)
    print(pose_df.head())
    

    embedding_files_dict = {
        "Embedding File Path": [],
        "Embedding File Name": [],
        "Embedding Model":[],
        "video_id": [],
    }
    embedding_files = list(directory.rglob("*.npy"))
    for embedding_file in embedding_files:
        video_id = embedding_file.stem.split("-")[0]
        model = embedding_file.stem.split("-model-")[-1]
        embedding_files_dict["video_id"].append(video_id)
        embedding_files_dict["Embedding File Name"].append(embedding_file.name)
        embedding_files_dict["Embedding File Path"].append(embedding_file)
        embedding_files_dict["Embedding Model"].append(model)

    embeddings_df = pd.DataFrame(embedding_files_dict)
    print(embeddings_df)


    df = metadata_df.merge(pose_df, on="video_id", how="left")
    df = df.merge(embeddings_df, on="video_id", how="left")
    filtered_df = df[df['Pose File Path'].notna()]
    filtered_df = filtered_df[filtered_df['Embedding File Path'].notna()]
    print(filtered_df.head())
    print(filtered_df.info())


def main():
    parser = argparse.ArgumentParser(
        description="Process pose files and merge metadata into a DataFrame. Expects a dir with split csvs and .pose or .pose.zst files"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to the root directory containing pose files and videos.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="asl_citizen",
        choices=["asl_citizen", "sem-lex"],
        help="Path to save the resulting CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="output.csv",
        help="Path to save the resulting CSV file.",
    )

    args = parser.parse_args()

    if args.dataset == "asl_citizen":
        print(f"Processing directory: {args.directory}")
        df = gather_asl_citizen_df(args.directory)

        df = df.rename(
            columns={
                "File Path": "Pose File Path",
                "File Name": "Pose File Name",
            }
        )

        # Load pose files and extract metadata with a progress bar
        metadata = []
        for file_path in tqdm(df["Pose File Path"], desc="Loading Pose Data"):
            pose = get_pose_data(file_path)
            frame_count = pose.body.data.shape[0]
            person_count = pose.body.data.shape[1]
            keypoint_count = pose.body.data.shape[2]
            fps = pose.body.fps
            missing_keypoints_per_frame = np.any(pose.body.data.mask, axis=-1).sum(
                axis=(1, 2)
            )
            total_points = np.prod(pose.body.data.shape)
            total_missing = np.sum(missing_keypoints_per_frame)

            metadata.append(
                [
                    frame_count,
                    person_count,
                    keypoint_count,
                    fps,
                    total_points,
                    total_missing,
                ]
            )

        # Convert to DataFrame and merge
        df_metadata = pd.DataFrame(
            metadata,
            columns=[
                "Frame Count",
                "Person Count",
                "Keypoint Count",
                "FPS",
                "Total Points",
                "Total Missing",
            ],
        )

        df_merged = pd.concat([df, df_metadata], axis=1)

        print(f"Saving DataFrame to {args.output}")
        df_merged.to_csv(args.output, index=False)
        print("Processing complete.")
    elif args.dataset == "sem-lex":
        gather_semlex_df(directory=args.directory)


if __name__ == "__main__":
    main()
