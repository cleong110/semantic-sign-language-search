# Usage: python downsampled_and_hands_and_face.py '/home/cleong/data/Colin_signed/ASL_Citizen_signs_performed_by_Colin/ASL_Citizen_words_signed_by_Colin_embedded_with_4_signCLIP_models/ASL_Citizen_words/WIN_20240904_12_20_56_Pro-SAD.mp4' ./WIN_20240904_12_20_56_Pro-SAD_frames --crop_hands --to_gifs
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import cv2
import numpy as np
import requests
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mpHands = mp.solutions.hands 
import imageio
from collections import defaultdict
import json
from itertools import cycle
import shutil

# URL to download the face detection model

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = "detector.tflite"


import cv2
import mediapipe as mp
from pathlib import Path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_and_crop_hands(image_path: Path, output_folder: Path):
    """
    Detects hands in an image, crops each detected hand, and saves it to the output folder.
    
    Parameters:
        image_path (Path): Path to the input image file.
        output_folder (Path): Path to save the cropped hand images.
    """
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    crops = []

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        
        # Process the image and detect hands
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print("No hands detected.")
            return

        # Iterate through detected hands
        for idx, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # Determine if left or right hand
            hand_label = hand_info.classification[0].label.lower()  # "left" or "right"
            
            # Get the bounding box coordinates
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * image_width
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * image_width
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * image_height
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * image_height

            # Crop the hand region
            x_min, x_max = int(x_min), int(x_max)
            y_min, y_max = int(y_min), int(y_max)
            cropped_hand = image[y_min:y_max, x_min:x_max]

            # Define output filename
            output_filename = f"{image_path.stem}_{hand_label}_hand_{idx + 1}.png"
            output_path = output_folder / output_filename

            # Save the cropped hand image
            cv2.imwrite(str(output_path), cropped_hand)
            crops.append(output_path)
            print(f"Saved {output_path}")
    return crops




def download_model_if_needed(model_path: Path, model_url: str):
    """
    Download the model file if it does not exist locally.
    
    Parameters:
        model_path (Path): Path where the model file should be stored.
        model_url (str): URL from where to download the model file.
    """
    if not model_path.exists():
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url)
        model_path.write_bytes(response.content)
        print(f"Model downloaded and saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

def load_face_detector(model_path: Path = Path(MODEL_PATH)):
    """
    Load the MediaPipe FaceDetector, downloading the model if necessary.
    
    Parameters:
        model_path (Path): Path to the model file (detector.tflite).
        
    Returns:
        detector: Initialized MediaPipe FaceDetector object.
    """
    # Ensure model is downloaded
    download_model_if_needed(model_path, MODEL_URL)

    # Load the FaceDetector model
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceDetectorOptions(base_options=base_options)
    return vision.FaceDetector.create_from_options(options)

# Function to detect and crop faces using MediaPipe
def detect_and_crop_faces_mediapipe(image_path: Path, output_folder: Path, detector):
    """
    Detect faces in an image, crop them, and save each cropped face to an output folder.
    
    Parameters:
        image_path (Path): Path to the input .png image file.
        output_folder (Path): Path to the directory where cropped faces will be saved.
        detector: Initialized MediaPipe FaceDetector object.
    """
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the image using MediaPipe's Image class
    image = mp.Image.create_from_file(str(image_path))

    # Perform face detection
    detection_result = detector.detect(image)

    # list of crops made
    crops = []

    # Process each detected face
    face_index = 1
    for face in detection_result.detections:
        # Extract bounding box coordinates from the face detection result
        bbox = face.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        # Convert MediaPipe Image to numpy array (RGB format)
        image_np = image.numpy_view()

        # Crop the face region
        cropped_face = image_np[y:y+h, x:x+w]

        # Convert RGB to BGR for OpenCV compatibility
        cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # Define the output path with the new filename
        output_filename = f"{image_path.stem}_face_{face_index:02d}.png"
        face_output_path = output_folder / output_filename
        
        # Save the cropped face
        cv2.imwrite(str(face_output_path), cropped_face_bgr)

        crops.append(face_output_path)

        face_index += 1

    print(f"{face_index - 1} faces detected and saved in {output_folder}")
    return crops




def save_video_frames_as_png(video_path: Path, output_folder: Path):
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
        cv2.imwrite(str(frame_path), frame)

        frame_index += 1

    # Release the video capture object
    cap.release()
    print(f"Frames saved in {output_folder}")

def detect_and_crop_faces_haar(image_path: Path, output_folder: Path):
    """
    Detect faces in an image, crop them, and save each cropped face to an output folder.
    
    Parameters:
        image_path (Path): Path to the input .png image file.
        output_folder (Path): Path to the directory where cropped faces will be saved.
    """
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to grayscale for the face detection process
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    face_index = 1
    for (x, y, w, h) in faces:
        # Crop the face from the image
        face = image[y:y+h, x:x+w]
        
        # Define the output path with the new filename
        output_filename = f"{image_path.stem}_face_{face_index:02d}.png"
        face_output_path = output_folder / output_filename
        
        # Save the cropped face
        cv2.imwrite(str(face_output_path), face)
        
        face_index += 1
    
    print(f"{face_index - 1} faces detected and saved in {output_folder}")

def detect_and_crop_faces(input_folder, output_folder, detector="blaze_face_short_range"):
    # TODO: let people select haar, mediapipe models, etc.
    print(f"Detecting faces...")
    # Initialize the MediaPipe FaceDetector, automatically downloading the model if needed
    detector_model = load_face_detector()
    frame_crops = defaultdict(dict)
    for frame_path in tqdm(input_folder.glob("frame*.png")):
        # detect_and_crop_faces(frame_path, faces_folder)
        # Detect and save faces from an example image
        crops = detect_and_crop_faces_mediapipe(frame_path, output_folder, detector_model)
        frame_crops[frame_path]["face_crops"] = crops
    return frame_crops


def downsample_image(image_path: Path, output_path: Path, scale: float = None, target_width: int = 64, target_height: int = 64):
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
    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

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

    # Save the downsampled image
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    cv2.imwrite(str(output_path), downsampled_image)
    
    print(f"Downsampled image saved to {output_path}")

def pngs_to_gif(png_paths, output_path):
    print(f"Creating {output_path}")
    
    with imageio.get_writer(output_path, mode='I', fps=1) as writer:
        
        for png_path in png_paths:
            # print(png_path)
            image = imageio.imread(png_path)
            writer.append_data(image)

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them as PNG files.")
    parser.add_argument("input_video_path", type=Path, help="Path to the input .mp4 video file.")
    parser.add_argument("output_frames_folder_path", type=Path, help="Path to the folder where frames will be saved.")
    parser.add_argument("--crop_faces", action="store_true", help="Should we make face crops?")
    parser.add_argument("--crop_hands", action="store_true", help="Should we make hand crops?")
    parser.add_argument("--downsample", action="store_true", help="Should we make downsampled?")
    parser.add_argument("--alternating", action="store_true", help="Switch between face, hands")
    parser.add_argument("--to_gifs", action="store_true", help="Should we make gifs?")
    

    args = parser.parse_args()
    frames_folder = args.output_frames_folder_path
    frames_dict = defaultdict(dict)
    
    save_video_frames_as_png(args.input_video_path, frames_folder)
    for frame_path in frames_folder.glob("*.png"):
        frames_dict[frame_path]["full_res"] = frame_path
    

    if args.crop_faces:
        faces_folder = args.output_frames_folder_path / "faces"

        frame_crops_dict = detect_and_crop_faces(frames_folder, faces_folder)
        for frame_path, frame_crops in frame_crops_dict.items():
            frames_dict[frame_path]["face_crops"]=frame_crops
        # Example usage

    if args.downsample:
    #     # Example usage
    #     downsample_image(
    #         image_path=Path("input.png"),
    #         output_path=Path("output/downsampled.png"),
    #         scale=0.5  # or use target_width=640, target_height=480
    #     )
        downsampled_folder = frames_folder / "downsampled"
        downsampled_folder.mkdir(parents=True, exist_ok=True)
    
        for frame_path in tqdm(frames_folder.glob("*.png"), desc=f"downsampling images to {downsampled_folder}"):
            output_path = downsampled_folder / f"{frame_path.stem}_downsampled.png"
            downsample_image(frame_path, output_path)
            frames_dict[frame_path]["downsampled"]=output_path

    if args.crop_hands:
        # Example usage
        # detect_and_crop_hands(Path("frame_0000.png"), Path("output_hands"))
        hands_folder = frames_folder/ "hands"
        hands_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in tqdm(frames_folder.glob("*.png"), desc=f"cropping hand images to {hands_folder}"):
            hand_crops = detect_and_crop_hands(frame_path, hands_folder)
            frames_dict[frame_path]["hand_crops"] = hand_crops

    if args.to_gifs:
        folder_and_subfolders = [x for x in frames_folder.iterdir() if x.is_dir()]
        folder_and_subfolders.append(frames_folder)
        for png_folder in folder_and_subfolders:
            output_path = png_folder /f"{png_folder.stem}.gif"
            print(f"Saving gif for {png_folder} to {output_path}")
            png_paths = [png_path for png_path in png_folder.glob("*.png")]
            png_paths = sorted(png_paths)
            if "hands" in str(png_folder):
                left_hand_pngs = sorted([png_path for png_path in png_paths if "left" in str(png_path)])
                pngs_to_gif(png_paths=left_hand_pngs, output_path=png_folder/f"{output_path.stem}_left.gif")
                right_hand_pngs = sorted([png_path for png_path in png_paths if "right" in str(png_path)])
                pngs_to_gif(png_paths=right_hand_pngs, output_path=png_folder/f"{output_path.stem}_right.gif")

            else:
                pngs_to_gif(png_paths=png_paths, output_path=output_path)




    # print(frames_dict)

    


    if args.alternating:
        alternating_dir = frames_folder/"alternating"
        alternating_dir.mkdir(parents=True, exist_ok=True)
        frames = frames_folder.glob("*.png")
        things_to_look_at = ["downsampled", "faces", "left", "right"]
        
        where_to_look = cycle(things_to_look_at)
        frames = sorted(frames)
        print(frames)
        for frame in frames:
            key = frame
            full_res = frames_dict[key]["full_res"]
            
            # print(f"For frame with full res at {full_res}, we look at {looking_at}")
            


            for place in ["downsampled", "hands", "faces"]:
                place_path = frames_folder / place
                # print(place_path)
                # print(full_res.stem)
                if place == "hands":
                    frames_dict[key]["left"] = list(place_path.glob(f"{full_res.stem}*left*"))
                    frames_dict[key]["right"] = list(place_path.glob(f"{full_res.stem}*right*"))
                else:
                    frames_dict[key][place] = list(place_path.glob(f"{full_res.stem}*"))

            # print(frames_dict[key])

            #     frames_dict[]

            
            
            # print(f"For frame with full res at {full_res}, we look at {looking_at}, giving us {frames_dict[key][looking_at]} ")
            for i in range(len(things_to_look_at)):
                looking_at = next(where_to_look)
                if frames_dict[key][looking_at]:
                    file_to_copy = frames_dict[key][looking_at][0]
                    where_to_copy_it = alternating_dir/ file_to_copy.name
                    print(f"For frame {key} we look at {looking_at}, giving us {file_to_copy}")
                    
                    shutil.copy(str(file_to_copy), str(where_to_copy_it))
                    break
                else:
                    print(f"For frame {key} we have NO RESULTS for {looking_at}, moving on!!!")

                # look for 
            

            
            
        

    #     for frame_path in frames_folder:
    #         print



        

        

if __name__ == "__main__":
    main()
