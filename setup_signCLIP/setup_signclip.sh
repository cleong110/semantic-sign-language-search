#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset


############################################################################################################
# Script to setup SignCLIP on a new workstation for embedding files. Run from project root
# Steps include: 
# 1. download pretrained models
# 2. clone SignCLIP source
# 3. Download checkpoints and models, and put them in expected places
# 4. Setup Conda env

# Edit this if you want it setup elsewhere
# signclip_parent="$HOME/projects/semantic-sign-language-search/setup_signCLIP"
signclip_parent="${PWD}/setup_signCLIP"


####################################
# 1. Download Pretrained Models

# https://saturncloud.io/blog/activating-conda-environments-from-scripts-a-guide-for-data-scientists/#activating-a-conda-environment-from-a-script
eval "$(conda shell.bash hook)"
conda create -n gdown pip
conda activate gdown
pip install gdown 


mkdir -p "$signclip_parent"
cd "$signclip_parent"


############################################################################################################
# 2. clone SignCLIP source
# https://github.com/J22Melody/fairseq/tree/5f9ab7ebd1fe7e000f282da1bce9f212ba9871c2/examples/MMPT
git clone https://github.com/J22Melody/fairseq.git || echo "already cloned"

############################################################################################################
# 3. Download checkpoints and models, and put them in expected places
# If you want to add one, you need a corresponding .yaml folder with save_path edited.
# See setup_signCLIP/projects/retri/semantic-search

pretrained_models_folder="fairseq/examples/MMPT/pretrained_models"
demo_model_folder="runs/signclip_embed/"
mkdir -p "$signclip_parent/$pretrained_models_folder"
mkdir -p "$signclip_parent/$demo_model_folder"


# The actual model for SignCLIP, which needs to be named checkpoint_best.pt for SignCLIP to work
# https://drive.google.com/file/d/1_B_VZMaLqY1nV6z9AokWU_G6LvOQLZFu/view?usp=drive_link
# https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc is the full folder
# Use --continue flag to skip fully downloaded files
gdown --continue "https://drive.google.com/uc?id=1_B_VZMaLqY1nV6z9AokWU_G6LvOQLZFu" # baseline_temporal_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/1Xun_2MQpyR6Ze2LuV1N_xMnTzRcts9Jv/view?usp=drive_link" # sem_lex_finetune_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/1qst_2vt8zeNnmEEiONfkqa1ApMSgwU1t/view?usp=drive_link" # asl_signs_finetune_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/166aUSU5HkrMlpCkMNQF_rBLymX56P3fn/view?usp=drive_link" # asl_citizen_finetune_checkpoint_best.pt
gdown --continue --fuzzy "https://drive.google.com/file/d/1kyneTEzmsMyOZvw7-O0b7PEU9Wnn7I_Q/view?usp=drive_link" # pop_sign_finetune_checkpoint_best.pt

# the model HAS to be named "checkpoint_best.pt"
# also it has to be in the place the .yaml file says to look for it. 
# setup_signCLIP/projects/retri/semantic-search has the yaml files
mkdir -p "$demo_model_folder/baseline_temporal_checkpoint_best/"
mkdir -p "$demo_model_folder/asl_citizen_finetune_checkpoint_best/"
mkdir -p "$demo_model_folder/asl_signs_finetune_checkpoint_best/"
mkdir -p "$demo_model_folder/sem_lex_finetune_checkpoint_best/"
mkdir -p "$demo_model_folder/pop_sign_finetune_checkpoint_best/"
cp -v "baseline_temporal_checkpoint_best.pt" "$demo_model_folder/baseline_temporal_checkpoint_best/checkpoint_best.pt"
cp -v "asl_citizen_finetune_checkpoint_best.pt" "$demo_model_folder/asl_citizen_finetune_checkpoint_best/checkpoint_best.pt"
cp -v "asl_signs_finetune_checkpoint_best.pt" "$demo_model_folder/asl_signs_finetune_checkpoint_best/checkpoint_best.pt"
cp -v "sem_lex_finetune_checkpoint_best.pt" "$demo_model_folder/sem_lex_finetune_checkpoint_best/checkpoint_best.pt"
cp -v "pop_sign_finetune_checkpoint_best.pt" "$demo_model_folder/pop_sign_finetune_checkpoint_best/checkpoint_best.pt"

############################################################################################################
# 4. Setup Conda env
# python spec: VideoCLIP says it was developed with 3.8.8, but Zifan's demo at
# https://colab.research.google.com/drive/1r8GtyZOJoy_tSu62tvi7Zi2ogxcqlcsz?usp=sharing#scrollTo=zXOTOpOluavd uses Python 3.10.12
# however that calls a signclip server. We want to run it directly.
# pip spec is to avoid weird errors with fairseq and omegaconf: 
# "WARNING: Ignoring version 2.0.5 of omegaconf since it has invalid metadata:
# Please use pip<24.1 if you need to use this version."


##############################################
# How I made this signclip_requirements.txt
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
# Also: on another machine I got an error about cargo not being installed, so I added THAT step

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

# NOTE: Some requirements require Rust/Cargo to be installed First!
curl https://sh.rustup.rs -sSf | sh



echo "after installing fairseq and MMPT, pip list is: "
pip list

# assumes you've got the signclip_requirements.txt in the same folder
pip install -r signclip_requirements.txt

echo "installed requirements.txt. pip list is now:"
pip list
