import sys

import torch
import numpy as np
from pose_format import Pose

from mmpt.models import MMPTModel

import mediapipe as mp

from pathlib import Path

import os
import typing

# TODO: don't load every model at once
# TODO: get this into proper version control

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [
    str(p)
    for p in sorted(
        set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup])
    )
]


model_configs = [
    # ("baseline_temporal", "semantic-search/embed_with_baseline_temporal"),  # made my own -Colin
    # ("sem-lex", "semantic-search/embed_with_sem_lex"),
    # ("asl-signs", "semantic-search/embed_with_asl_signs"),
    ("asl-citizen", "semantic-search/embed_with_asl_citizen"),
    # ('default', 'signclip_v1_1/baseline_temporal'),
    # ('asl_citizen', 'signclip_asl/asl_citizen_finetune'),
]
models = {}

for model_name, config_path in model_configs:
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


def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"), p2=("pose_keypoints_2d", "LShoulder")
        )


def pose_hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    else:
        raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(pose):
    pose = pose.get_components(
        [
            "POSE_LANDMARKS",
            "FACE_LANDMARKS",
            "LEFT_HAND_LANDMARKS",
            "RIGHT_HAND_LANDMARKS",
        ],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )

    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)

    # from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
    # from pose_anonymization.appearance import remove_appearance

    # pose = remove_appearance(pose)
    # pose = pre_process_mediapipe(pose)
    # pose = normalize_mean_std(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()

    return pose_frames


def preprocess_text(text, model_name="default"):
    aligner = models[model_name]["aligner"]
    tokenizer = models[model_name]["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_pose(pose, model_name="default"):
    model = models[model_name]["model"]

    caps, cmasks = preprocess_text("", model_name)
    poses = pose if type(pose) == list else [pose]
    embeddings = []

    for pose in poses:
        pose_frames = preprocess_pose(pose)

        with torch.no_grad():
            output = model(pose_frames, caps, cmasks, return_score=False)
            embeddings.append(output["pooled_video"].cpu().numpy())

    return np.concatenate(embeddings)


def embed_text(text, model_name="default"):
    model = models[model_name]["model"]

    # pose_frames = torch.randn(1, 1, 534)
    pose_frames = torch.randn(1, 1, 609)
    texts = text if type(text) == list else [text]
    embeddings = []

    for text in texts:
        caps, cmasks = preprocess_text(text, model_name)

        with torch.no_grad():
            output = model(pose_frames, caps, cmasks, return_score=False)
            embeddings.append(output["pooled_text"].cpu().numpy())

    return np.concatenate(embeddings)


def score_pose_and_text(pose, text, model_name="default"):
    model = models[model_name]["model"]

    pose_frames = preprocess_pose(pose)
    caps, cmasks = preprocess_text(text)

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=True)
        # print(model) # see baseline_temporal_architecture.md
        # print(type(model)) # <class 'mmpt.models.mmfusion.MMPTModel'>
        # print(dir(model)) # lots of things expected of a torch module
        # help(model) # Help on MMPTModel in module mmpt.models.mmfusion object: class MMPTModel(torch.nn.modules.module.Module)

        # print(output) # things like {'score': tensor([84.7832])}

    return text, float(output["score"])  # dot-product


def score_pose_and_text_batch(pose, text, model_name="default"):
    pose_embedding = embed_pose(pose, model_name)
    text_embedding = embed_text(text, model_name)

    scores = np.matmul(pose_embedding, text_embedding.T)
    return scores

def score_pose_embedding_and_text_batch(pose_embedding, text, model_name="default"):
    # pose_embedding = embed_pose(pose, model_name)
    text_embedding = embed_text(text, model_name)

    scores = np.matmul(pose_embedding, text_embedding.T)
    return scores


def save_pose_embedding(embeddings, out_path, model_name="default"):
    
    np.save(out_path, embeddings, allow_pickle=False)
    print(f"Saving pose embeddings with shape {embeddings.shape}")
    print(f"saved embedding to {out_path}")

def load_pose_embedding(embedding_path):
    embeddings = np.load(embedding_path)
    print(f"loaded embeddings with shape {embeddings.shape}")
    # print(f"loaded embeddings with shape {embeddings}")
    return embeddings



if __name__ == "__main__":
    pose_path = (
        "/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/pose/1_1_1_cam2.pose"
        if len(sys.argv) < 2
        else sys.argv[1]
    )

    with open(pose_path, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

        # https://relatedwords.io/house
        related_words = [
            "family",
            "home",
            "apartment",
            "residence",
            "bedroom",
            "bathroom",
            "room",
            "building",
            "cottage",
            "mansion",
            "kitchen",
            "hall",
            "door",
            "household",
            "car",
            "villa",
            "property",
            "bungalow",
        ]

        unrelated_words = [
            "moon",
            "soon",
            "kenya",
            "sign of the zodiac",
            "homeoplasty",
            "homeopathy",
            "porterhouse",
            "chouse",

        ]

        # print(score_pose_and_text(pose, "random text"))
        # print(score_pose_and_text(pose, "houses"))
        # print(score_pose_and_text(pose, "home"))
        # print(score_pose_and_text(pose, "house"))
        # print(score_pose_and_text(pose, "<en> <ase> house"))
        # print(score_pose_and_text(pose, "<en> <ase> home"))
        # print(score_pose_and_text(pose, "<en> <ase> houses"))

        # print(score_pose_and_text_batch(pose,  "<en> <ase> house")) # same result. The underlying functions support lists or just single items


        embeddings = embed_pose(pose, model_name)
        embed_out_name = Path(pose_path).stem + "-using-model-"+ model_name+".npy"      
        print(embed_out_name)  
        save_pose_embedding(embeddings, out_path=Path(embed_out_name))
        


        # loaded_embeddings = load_pose_embedding(embedding_path=Path(pose_path).with_suffix(".npy"))
        # print(f"Are loaded embeddings the same as original?")

        # print(np.array_equal(embeddings, loaded_embeddings))

        # print(score_pose_and_text(pose, "<en> <gsg> house"))
        # print(score_pose_and_text(pose, "<en> <fsl> house"))
        # print(score_pose_and_text(pose, "<en> <ase> sun"))
        # print(score_pose_and_text(pose, "<en> <ase> police"))
        # print(score_pose_and_text(pose, "<en> <ase> how are you?"))

        # print("various related words")
        # for related_word in related_words:
        #     print(score_pose_and_text(pose, f"<en> <ase> {related_word}"))

        # print("some unrelated words")
        # for unrelated_word in unrelated_words:
        #     print(score_pose_and_text(pose, f"<en> <ase> {unrelated_word}"))


