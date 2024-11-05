
# Based on code fromf https://colab.research.google.com/drive/1-6Wj-1dbY_0ywx-qMC9zdqsFbzZOSgw2#scrollTo=-A3BFHsl02mB
import argparse
from PIL import Image
import os
from pathlib import Path
from uuid import uuid4
import cv2
from typing import List

def stitch_pngs(png_files, output_file, step=1):
  """Stitches a list of PNG files together into one large image.

  Args:
    png_files: A list of paths to PNG files.
    output_file: The path to save the stitched image.
    step: whether to skip by this many
  """
  if not png_files:

    return

  images = [Image.open(x) for x in png_files]
  print(f"Found {len(images)} images. Taking every {step}")
  images = images[::step]
  print(f"Result: {len(images)}")

  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save(output_file)

def downsample_image(image:cv2.typing.MatLike, scale: float = None, target_width: int = 64, target_height: int = 64):
    """
    Downsamples a .png image and saves it to the specified output path.
    
    Parameters:
        image_path (Path): Path to the input .png image file.
        output_path (Path): Path to save the downsampled image.
        scale (float, optional): Scaling factor for downsampling. 
                                 E.g., 0.5 will halve the dimensions.
        target_width (int, optional): Target width for resizing.
        target_height (int, optional): Target height for resizing.
        
    Note:
        If both `scale` and `target_width/target_height` are provided, `scale` will be used.
    """

    # Determine new dimensions
    if scale:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
    elif target_width and target_height:
        new_width = target_width
        new_height = target_height
    elif target_width:
        aspect_ratio = image.shape[0] / image.shape[1]
        new_width = target_width
        new_height = int(target_width * aspect_ratio)
    elif target_height:
        aspect_ratio = image.shape[1] / image.shape[0]
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        raise ValueError("Either scale or target_width/target_height must be provided.")

    # Resize the image
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return downsampled_image

def save_video_frames_as_png(video_path: Path, output_folder: Path, downsample=True):
    """
    Extract frames from an MP4 video and save each as a PNG file.
    
    Parameters:
        video_path (Path): Path to the input .mp4 video file.
        output_folder (Path): Path to the directory where frames will be saved.
    """
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_index = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames are left

        # Define the path to save each frame as a PNG
        frame_path = output_folder / f"frame_{frame_index:04d}.png"
        if downsample:
           frame = downsample_image(frame)
        cv2.imwrite(str(frame_path), frame)

        frame_index += 1

    # Release the video capture object
    cap.release()
    # print(f"Frames saved in {output_folder}")

def stitch_pngs_from_folder(folder_path, output_file):
  """Stitches PNG files from a folder together into one large image.

  Args:
    folder_path: The path to the folder containing PNG files.
    output_file: The path to save the stitched image.
  """
  png_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
  if not png_files:
    print(f"No PNG files found in the specified folder: {folder_path}")
    return
  stitch_pngs(png_files, output_file)

def stitch_images_vertically(image_paths: List[str], output_path: str) -> None:
  """Stitches a list of images together vertically.

  Args:
    image_paths: A list of paths to the images.
    output_path: The path to save the stitched image.
  """
  images = [Image.open(image_path) for image_path in image_paths]
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  total_height = sum(heights)

  new_im = Image.new('RGB', (total_width, total_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

  new_im.save(output_path)

def make_grid(video_paths, output_file, downsample=True, step=1):
    frames_folder = Path.cwd() / "frames"

    for video in video_paths:
        video_frames_folder = frames_folder / video.stem
        save_video_frames_as_png(video, video_frames_folder, downsample=downsample)
        png_files = video_frames_folder.glob("*.png")
        stitch_pngs(png_files=png_files, output_file=frames_folder/f"{video.stem}_stitched.png", step=step)
    stitched_pngs = sorted(frames_folder.glob("*_stitched.png"))
    print(stitched_pngs)
    stitch_images_vertically(stitched_pngs, output_file)
    
    # video_frames_folders = [item for item in frames_folder.iterdir() if item.is_dir()]
    # for video_frames_folder in video_frames_folders:
       

  

def main():
    parser = argparse.ArgumentParser(description="Stitch video_frames together into a big graph")
    parser.add_argument("input_videos_folder", type=Path, help="Paths to the input .mp4 video file.")
    parser.add_argument("output_file", type=Path, help="path to the output image")
    parser.add_argument("--step", type=int, default=10, help="rate at which to take frames. Default is 1, meaning every frame. 4 would mean only every 4th frame")
    

    args = parser.parse_args()

    videos = list(args.input_videos_folder.glob("*.mp4"))
    # print(videos)
    make_grid(video_paths=videos, output_file=args.output_file, step=args.step)

if __name__ == "__main__":
  main()
