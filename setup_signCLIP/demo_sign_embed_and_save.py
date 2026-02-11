import argparse
from pathlib import Path
from typing import Union, List
import mediapipe as mp
import numpy as np
import torch
from mmpt.models import MMPTModel
from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs
from pyzstd import decompress
from tqdm import tqdm

# TODO: don't load every model at once
# TODO: get things into main

SIGNCLIP_MAX_FRAMES = 256  # or just edit the limit in the .yaml file

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [
    str(p)
    for p in sorted(
        set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup])
    )
]

# paths to custom config files in
# setup_signCLIP/projects/retri/semantic-search
# e.g. semantic-sign-language-search/setup_signCLIP/projects/retri/semantic-search/embed_with_sem_lex.yaml
model_configs = [
    (
        "baseline_temporal_checkpoint_best",
        "semantic-search/embed_with_baseline_temporal",
    ),  # made my own -Colin
    ("sem_lex_finetune_checkpoint_best", "semantic-search/embed_with_sem_lex"),
    ("asl_signs_finetune_checkpoint_best", "semantic-search/embed_with_asl_signs"),
    ("asl_citizen_finetune_checkpoint_best", "semantic-search/embed_with_asl_citizen"),
    ("pop_sign_finetune_checkpoint_best", "semantic-search/embed_with_pop_sign"),
]
models = {}


def load_models(model_names_to_load):
    print("*" * 40)
    for model_name, config_path in model_configs:
        if model_name in model_names_to_load:

            print(f"loading Model {model_name}")

            # Go get the config file, the config file tells you where to get the checkpoint
            model, tokenizer, aligner = MMPTModel.from_pretrained(
                f"projects/retri/{config_path}.yaml",
                video_encoder=None,
            )
            model.eval()

            if torch.cuda.is_available():
                model.cuda()

            models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "aligner": aligner,
            }
            print(f"loaded Model {model_name}")
            print("*" * 40)


def preprocess_pose(pose: Pose, max_frames=None):
    pose = pose.get_components(
        [
            "POSE_LANDMARKS",
            "FACE_LANDMARKS",
            "LEFT_HAND_LANDMARKS",
            "RIGHT_HAND_LANDMARKS",
        ],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )

    pose = pose.normalize()
    pose = pose_hide_legs(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    if max_frames is not None:
        feat = feat[:max_frames]  # Truncate to max_frames if provided

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()

    return pose_frames


def find_pose_files(directory: Path) -> List[Path]:
    """Recursively find all .pose and .pose.zst files in the given directory."""
    return sorted(directory.rglob("*.pose")) + sorted(directory.rglob("*.pose.zst"))


def load_pose_file(file_path: Union[Path, str]):
    file_path = Path(file_path)
    if file_path.name.endswith(".pose.zst"):
        return Pose.read(decompress(file_path.read_bytes()))
    else:
        return Pose.read(file_path.read_bytes())


def get_pose_file_stem(file_path: Path):
    """Gets you the actual stem even for .pose.zst files"""

    if file_path.name.endswith(".pose.zst"):
        return Path(file_path.stem).stem
    else:
        return file_path.stem


def preprocess_text(text, model_name="default"):
    aligner = models[model_name]["aligner"]
    tokenizer = models[model_name]["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_pose(pose, model_name="default", max_frames=None):
    model = models[model_name]["model"]

    caps, cmasks = preprocess_text("", model_name)
    poses = pose if isinstance(pose, List) else [pose]
    embeddings = []

    for pose in poses:
        pose_frames = preprocess_pose(pose, max_frames=max_frames)

        with torch.no_grad():
            output = model(pose_frames, caps, cmasks, return_score=False)
            embeddings.append(output["pooled_video"].cpu().numpy())

    return np.concatenate(embeddings)


def save_pose_embedding(embeddings, out_path, model_name="default"):
    np.save(out_path, embeddings, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pose_embedder", description="Embed a folder of poses"
    )
    parser.add_argument("pose_dir", type=Path)
    parser.add_argument(
        "--model_names", type=str, help="Comma-separated list of model names"
    )
    parser.add_argument(
        "--out_folder",
        type=Path,
        help="Where to save the resulting embeddings. Default: same dir as the pose file",
    )

    parser.add_argument(
        "--truncate_long_files",
        "-t",
        action="store_true",
        help=f"Whether to truncate files longer than max frames {SIGNCLIP_MAX_FRAMES}",
    )

    parser.add_argument(
        "--overwrite_embeddings",
        "-o",
        action="store_true",
        help="Whether to write over embedding .npy files if they already exist (default false)",
    )

    parser.add_argument(
        "--skip_existing",
        "-s",
        action="store_true",
        help="Whether to skip over embedding .npy files if they already exist (default false)",
    )

    args = parser.parse_args()

    print("%" * 20)

    print(f"Searching {args.pose_dir}")
    pose_paths = find_pose_files(args.pose_dir)
    print(f"{len(pose_paths)} pose files found")

    max_frames = None
    if args.truncate_long_files:
        max_frames = SIGNCLIP_MAX_FRAMES

    out_folder = args.out_folder
    if out_folder is not None:
        out_folder = Path(out_folder)
        if not out_folder.is_dir():
            out_folder.mkdir(parents=True, exist_ok=True)

    if args.model_names is None:
        model_names = [
            str(model_config_tuple[0]) for model_config_tuple in model_configs
        ]
    else:
        model_names = args.model_names.split(",")
    load_models(model_names)
    print(f"Embedding with models {model_names}")

    for pose_path in tqdm(pose_paths, desc="Embedding"):
        for model_name in model_names:
            pose = load_pose_file(pose_path)
            if args.out_folder is None:
                out_folder = Path(pose_path).parent
            embed_out_name = (
                str(out_folder / get_pose_file_stem(pose_path))
                + "-using-model-"
                + model_name
                + ".npy"
            )

            if Path(embed_out_name).is_file():
                if args.overwrite_embeddings:
                    pass  # no issue, keep going and overwrite it.
                if args.skip_existing:
                    continue  # skip!
                else:
                    raise FileExistsError(
                        f"{embed_out_name} Exists! Rerun with -o (overwrite) or -s (skip) if you are sure"
                    )
            try:
                embeddings = embed_pose(pose, model_name, max_frames=max_frames)
                save_pose_embedding(embeddings, out_path=Path(embed_out_name))
            except AssertionError as e:
                if "Video too long. Received frame count" in str(e):
                    # print warning

                    error_file = Path(embed_out_name).with_suffix(".txt")

                    print(
                        f"Could not embed {pose_path}, video too long. Writing exception to {error_file}"
                    )
                    with error_file.open("w", encoding="utf-8") as ef:
                        error_to_write = f"{pose_path} \n {e}"
                        ef.writelines(error_to_write)

                else:
                    raise
