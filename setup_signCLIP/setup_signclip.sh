#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

######################
# Script to setup SignCLIP on a new workstation.
# TODO: create my own version of /J22Melody/fairseq/examples/MMPT/projects/retri/signclip_v1_1/baseline_temporal.yaml and commit it here


# https://saturncloud.io/blog/activating-conda-environments-from-scripts-a-guide-for-data-scientists/#activating-a-conda-environment-from-a-script
eval "$(conda shell.bash hook)"
conda create -n gdown pip
conda activate gdown
pip install gdown 

# Edit this if you want it setup elsewhere
signclip_parent="$HOME/projects/semantic-sign-language-search/setup_signCLIP"
mkdir -p "$signclip_parent"
cd "$signclip_parent"


# https://github.com/J22Melody/fairseq/tree/5f9ab7ebd1fe7e000f282da1bce9f212ba9871c2/examples/MMPT
git clone https://github.com/J22Melody/fairseq.git || echo "already cloned"

##################################################
# Download checkpoints and models
# wget -nc is "no clobber" aka don't redownload.

pretrained_models_folder="fairseq/examples/MMPT/pretrained_models"
demo_model_folder="runs/signclip_embed/"
mkdir -p "$pretrained_models_folder"
mkdir -p "$demo_model_folder"


# VideoCLIP README at https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT
# it says "We use pre-trained S3D for video feature extraction. Please place the models as pretrained_models/s3d_dict.npy and pretrained_models/s3d_howto100m.pth."
# S3D links to https://github.com/antoine77340/S3D_HowTo100M
#wget -nc https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth -O "$pretrained_models_folder/s3d_howto100m.pth" || echo "$pretrained_models_folder/s3d_dict.npy already retrieved"
#wget -nc https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy -O "$pretrained_models_folder/s3d_dict.npy" || echo "$pretrained_models_folder/s3d_dict.npy already retrieved"



# VideoCLIP README says 
# Download VideoCLIP checkpoint https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt to runs/retri/videoclip 
# or VLM checkpoint https://dl.fbaipublicfiles.com/MMPT/mtm/vlm/checkpoint_best.pt to runs/mtm/vlm.
#runs_folder="fairseq/examples/MMPT/runs"
#mkdir -p "$runs_folder/retri/videoclip"
#wget -nc https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt -O "$runs_folder/retri/videoclip/checkpoint_best.pt" || echo "$runs_folder/retri/videoclip/checkpoint_best.pt already retrieved"

#mkdir -p "$runs_folder/mtm/vlm"
#wget -nc https://dl.fbaipublicfiles.com/MMPT/mtm/vlm/checkpoint_best.pt -O "$runs_folder/mtm/vlm/checkpoint_best.pt" || echo "$runs_folder/mtm/vlm/checkpoint_best.pt already retrieved"


# The actual model for SignCLIP, which needs to be named checkpoint_best.pt for SignCLIP to work
# https://drive.google.com/file/d/1_B_VZMaLqY1nV6z9AokWU_G6LvOQLZFu/view?usp=drive_link
# Use --continue flag to skip fully downloaded files
gdown --continue "https://drive.google.com/uc?id=1_B_VZMaLqY1nV6z9AokWU_G6LvOQLZFu" # baseline_temporal_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/1Xun_2MQpyR6Ze2LuV1N_xMnTzRcts9Jv/view?usp=drive_link" # sem_lex_finetune_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/1qst_2vt8zeNnmEEiONfkqa1ApMSgwU1t/view?usp=drive_link" # asl_signs_finetune_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/166aUSU5HkrMlpCkMNQF_rBLymX56P3fn/view?usp=drive_link" # asl_citizen_finetune_checkpoint_best.pt

# the model HAS to be named "checkpoint_best.pt"
mkdir -p "$demo_model_folder/baseline_temporal_checkpoint_best/"
mkdir -p "$demo_model_folder/asl_citizen_finetune_checkpoint_best/"
mkdir -p "$demo_model_folder/asl_signs_finetune_checkpoint_best/"
mkdir -p "$demo_model_folder/sem_lex_finetune_checkpoint_best/"
cp -v "baseline_temporal_checkpoint_best.pt" "$demo_model_folder/baseline_temporal_checkpoint_best/checkpoint_best.pt"
cp -v "asl_citizen_finetune_checkpoint_best.pt" "$demo_model_folder/asl_citizen_finetune_checkpoint_best/checkpoint_best.pt"
cp -v "asl_signs_finetune_checkpoint_best.pt" "$demo_model_folder/asl_signs_finetune_checkpoint_best/checkpoint_best.pt"
cp -v "sem_lex_finetune_checkpoint_best.pt" "$demo_model_folder/sem_lex_finetune_checkpoint_best/checkpoint_best.pt"

###################
# Conda env
# python spec: VideoCLIP says it was developed with 3.8.8, but Zifan's demo at
# https://colab.research.google.com/drive/1r8GtyZOJoy_tSu62tvi7Zi2ogxcqlcsz?usp=sharing#scrollTo=zXOTOpOluavd uses Python 3.10.12
# pip spec is to avoid weird errors with fairseq and omegaconf: 
# "WARNING: Ignoring version 2.0.5 of omegaconf since it has invalid metadata:
# Please use pip<24.1 if you need to use this version."

#conda create -y -n signclip python=3.10.14 "pip<24.1"
conda create -y -n signclip python=3.8.8 "pip<24.1"
conda activate signclip

echo "signclip env created:"
conda list

cd fairseq
pip install -e .
echo "installed fairseq! pip list:"
pip list

cd examples/MMPT 
pip install -e .
echo "installed MMPT! pip list:"
pip list

##############################################
# requirements.txt
# setup a requirements.txt with mediapipe, vidgear, pose-format,git+https://github.com/sign-language-processing/transcription.git@1f2cef8 and that installed.
# added in git+https://github.com/sign-language-processing/pose-anonymization, git+https://github.com/sign-language-processing/sign-vq and that installed too
# Got an issue with NameError: name 'BertEmbeddings' is not defined, perhaps the wrong version of transformers? tried adding transformers==3.4, as VideoCLIP mentions it
# Then I got an error with "error: failed to parse manifest at `/home/cleong/.cargo/registry/src/github.com-1ecc6299db9ec823/byteorder-1.5.0/Cargo.toml`"
#      
#      Caused by:
#        failed to parse the `edition` key
#      
#      Caused by:
#        this version of Cargo is older than the `2021` edition, and only supports `2015` and `2018` editions.
#      error: `cargo metadata --manifest-path Cargo.toml --format-version 1` failed with code 101
# so I tried installing Rust... already installed, version 1.8
# https://discuss.streamlit.io/t/new-trouble-could-not-build-wheels-for-tokenizers-which-is-required-to-install-pyproject-toml-based-projects/54273
# I tried downgrading pip a few times, couldn't go lower than 20 with Python 3.10, so I started from the top, asking Conda to install Python 3.8.8, which helped. 
# then I got an error with wanting webvtt, so added webvtt-py to the list. 

echo "after installing fairseq and MMPT, pip list is: "
pip list


# assumes you've got the signclip_requirements.txt in the same folder
pip install -r signclip_requirements.txt



echo "installed requirements.txt. pip list is now:"
pip list
