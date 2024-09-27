import argparse
from pathlib import Path
from typing import List
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info
from pose_format.utils.generic import pose_hide_legs
import numpy as np
# see also:
# https://github.com/sign-language-processing/pose/blob/master/src/python/pose_format/bin/pose_visualizer.py


def remove_world_landmarks(pose: Pose):
    remove_specified_landmarks(pose, "POSE_WORLD_LANDMARKS")
    
def remove_specified_landmarks(pose: Pose, landmark_names:List[str]):
    if isinstance(landmark_names, str):
        landmark_names = [landmark_names]
    components_without_specified_names = [c.name for c in pose.header.components if c.name not in landmark_names]
    new_pose = pose.get_components(components_without_specified_names)
    pose.body=new_pose.body
    pose.header=new_pose.header

# python overlay_pose_on_video.py -p "/home/vlab/Downloads/ASl_Citizen/ASL_Citizen/visualized_poses/002012924350435652-PAIR.pose" -v "/home/vlab/Downloads/ASl_Citizen/ASL_Citizen/visualized_poses/002012924350435652-PAIR.mp4"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlays .pose on a video, outputs a .pose.overlay.mp4"
    )
    parser.add_argument(
        "-p", "--pose_input", type=Path, required=True, help="Path to input pose file"
    )
    parser.add_argument(
        "-v", "--video_input", type=Path, required=True, help="Path to input video file"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output video path")

    parser.add_argument("-n", "--normalize", default=False, action="store_true")
    
    parser.add_argument("--keep_world_landmarks", action="store_true", help="Keep the world landmarks, even though it makes for weird visual bugs as described in https://github.com/sign-language-processing/pose/issues/114") # store_true is false by default
    

    args = parser.parse_args()

    with open(str(args.pose_input), "rb") as f:
        pose = Pose.read(f.read())

    pose_hide_legs(pose)
    
    if args.keep_world_landmarks:
        print("Not removing world landmarks")
    else:
        print("Removing world landmarks")
        print([c.name for c in pose.header.components])        
        remove_world_landmarks(pose)
        print("removed world landmarks")
        print([c.name for c in pose.header.components])        

    if args.normalize:
        print("normalizing pose")

        pose = pose.normalize(pose_normalization_info(pose.header))
        new_width = 500
        shift = 1.25
        shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
        pose.body.data = (pose.body.data + shift_vec) * new_width
        pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    v = PoseVisualizer(pose)

    # Draws pose on top of video.
    # by default, put it in the same directory as the .mp4 
    output_path = args.video_input.parent / Path(args.video_input.stem + ".pose.overlay.mp4")
    if args.output is not None:
        output_path = args.output


    print(f"outputting overlay video to {output_path.absolute()}")
    v.save_video(str(output_path), v.draw_on_video(str(args.video_input)))
