import cv2
import os
import glob
import argparse

def create_video_from_images(input_dir, output_file, fps):
    # Supported image extensions (modify as needed)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # Read the first image to get the frame dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print("Failed to read the first image.")
        return
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"Creating video {output_file} from {len(image_files)} images at {fps} fps...")
    for img_file in image_files:
        frame = cv2.imread(img_file)
        if frame is None:
            print(f"Warning: Skipping file {img_file} (could not be read).")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of images with detections into a video."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing the detection images."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.mp4", 
        help="Output video file name (default: output.mp4)."
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=10.0, 
        help="Frame rate for the output video (default: 10 fps)."
    )

    args = parser.parse_args()
    create_video_from_images(args.input_dir, args.output, args.fps)
