import argparse
from pathlib import Path
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
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
    parser.add_argument("-o", "--output", type=Path, help="output video path")

    args = parser.parse_args()

    with open(str(args.pose_input), "rb") as f:
        pose = Pose.read(f.read())

    v = PoseVisualizer(pose)

    # Draws pose on top of video.
    # by default, put it in the same directory as the .mp4 
    output_path = args.video_input.parent / Path(args.video_input.stem + ".pose.overlay.mp4")
    if args.output is not None:
        output_path = args.output


    print(f"outputting overlay video to {output_path.absolute()}")
    v.save_video(str(output_path), v.draw_on_video(str(args.video_input)))
