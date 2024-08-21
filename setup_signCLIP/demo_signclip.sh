#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# FileNotFoundError: [Errno 2] No such file or directory: 'runs/retri_v1_1/baseline_temporal/checkpoint_best.pt'
# that comes from baseline_temporal.yaml
# edited demo_sign.py to use my config file, which points it to MMPT/runs/signclip_demo/baseline_temporal_checkpoint_best/
# where it expects to find a "checkpoint_best.pt", that exact name. 

# https://saturncloud.io/blog/activating-conda-environments-from-scripts-a-guide-for-data-scientists/#activating-a-conda-environment-from-a-script
eval "$(conda shell.bash hook)"
conda activate signclip

SignCLIP_folder="/home/cleong/code/J22Melody/fairseq/examples/MMPT/"
wget -nc https://media.spreadthesign.com/video/mp4/13/58463.mp4 -O house.mp4 || echo "already downloaded house.mp4"
video_to_pose -i house.mp4 --format mediapipe -o house.pose

cp house.pose "$SignCLIP_folder"
cd "$SignCLIP_folder"
python demo_sign.py house.pose
