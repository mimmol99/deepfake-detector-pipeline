# src/main.py
"""
Main entry point for the deepfake detection pipeline.
Parses arguments, sets up environment, runs processing, and prints results.
"""

import argparse
import time
import sys
import os
import torch
import dlib
from pathlib import Path

# Use relative imports for modules within the 'src' package
from . import config
from . import utils
from . import detection
from . import pipeline

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake Detection Pipeline")
    parser.add_argument(
        "video_folder",
        type=str,
        help="Path to the folder containing video files to process."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(config.MODEL_BASE_DIR),
        help=f"Directory to store/load downloaded models (dlib, detector). Default: {config.MODEL_BASE_DIR}"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=config.LOG_FILE,
        help=f"Path to the log file. Default: {config.LOG_FILE}"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=config.NUM_FRAMES_PER_VIDEO,
        help=f"Number of frames to sample per video (for linspace/consecutive modes). Default: {config.NUM_FRAMES_PER_VIDEO}"
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default=config.FRAME_SAMPLING_MODE,
        choices=['linspace', 'consecutive', 'stride'],
        help=f"Frame sampling mode. Default: {config.FRAME_SAMPLING_MODE}"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=config.FRAME_STRIDE,
        help=f"Frame stride (for stride mode). Default: {config.FRAME_STRIDE}"
    )
    # Add more arguments if needed (e.g., device selection, batch size if applicable)

    return parser.parse_args()

# --- Main Execution ---
def main():
    args = parse_args()

    # --- Setup Logger ---
    logger = utils.create_logger(args.log_file, logger_name="pipeline_logger")
    logger.info("Starting Deepfake Detection Pipeline")
    logger.info(f"Arguments: {args}")

    # --- Resolve Paths ---
    # Make model paths relative to the specified model_dir
    model_base_dir = Path(args.model_dir)
    dlib_predictor_path = model_base_dir / config.DLIB_PREDICTOR_DIR.name / config.DLIB_PREDICTOR_NAME
    detection_model_path = model_base_dir / config.DETECTION_MODEL_DIR.name / config.DETECTION_MODEL_NAME

    # --- Setup Environment (Download Models) ---
    utils.setup_environment(
        dlib_url=config.DLIB_DOWNLOAD_URL,
        dlib_path=dlib_predictor_path,
        model_url=config.DETECTION_MODEL_URL,
        model_path=detection_model_path,
        logger=logger
    )

    # --- Load Models and Preprocessors ---
    logger.info("--- Loading models ---")

    # Dlib face detector and predictor
    try:
        face_detector = dlib.get_frontal_face_detector()
        logger.info("Loaded dlib frontal face detector.")
        face_predictor = dlib.shape_predictor(str(dlib_predictor_path))
        logger.info(f"Loaded dlib shape predictor from {dlib_predictor_path}.")
    except Exception as e:
        logger.error(f"Failed to load dlib models. Error: {e}", exc_info=True)
        sys.exit(1)

    # Deepfake detection model
    detection_model = detection.load_detection_model(detection_model_path)
    if detection_model is None:
        logger.error("Failed to load the detection model. Exiting.")
        sys.exit(1)

    # Detection model preprocessor
    detection_preprocessor = detection.get_detection_preprocessor(
        input_size=config.DETECTION_MODEL_INPUT_SIZE
    )
    logger.info(f"Using detection input size: {config.DETECTION_MODEL_INPUT_SIZE}")

    # Set device (use GPU if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
    detection_model.to(device)


    # --- Check if Video Folder Exists ---
    video_folder = Path(args.video_folder)
    if not video_folder.is_dir():
        logger.error(f"Video folder not found: {video_folder}")
        logger.error("Please provide a valid path to a folder containing videos.")
        sys.exit(1)

    # --- Run the Pipeline ---
    logger.info(f"--- Starting video processing pipeline for folder: {video_folder} ---")
    start_time = time.monotonic()

    video_results = pipeline.process_video_folder(
        folder_path=video_folder,
        face_detector=face_detector,
        face_predictor=face_predictor,
        detection_model=detection_model,
        detection_preprocessor=detection_preprocessor,
        device=device,
        sampling_mode=args.sampling_mode,
        num_frames=args.num_frames,
        stride=args.stride,
        face_res=config.FACE_CROP_RESOLUTION,
        face_alignment_scale=config.FACE_ALIGNMENT_SCALE
    )

    end_time = time.monotonic()
    duration_seconds = end_time - start_time
    logger.info(f"--- Pipeline finished in {duration_seconds:.2f} seconds ---")

    # --- Display Results ---
    print("\n" + "="*30 + " Detection Results " + "="*30) # More visible separator
    successful_predictions = 0
    failed_videos = 0
    if not video_results:
        print("No videos were processed or found in the specified folder.")
    else:
        for video_path_str, result in video_results.items():
            video_name = os.path.basename(video_path_str)
            if result:
                print(f"Video: {video_name}")
                print(f"  Avg Real Prob: {result['real_prob']:.4f}")
                print(f"  Avg Fake Prob: {result['fake_prob']:.4f}")
                # Basic thresholding for label
                label = "FAKE" if result['fake_prob'] > result['real_prob'] else "REAL"
                confidence = max(result['real_prob'], result['fake_prob'])
                print(f"  Predicted Label: {label} (Confidence: {confidence:.4f})")
                print("-" * 20)
                successful_predictions += 1
            else:
                print(f"Video: {video_name}")
                print("  Processing failed or no faces detected/processed.")
                print("-" * 20)
                failed_videos += 1

        print("\n" + "="*30 + " Summary " + "="*30)
        print(f"Total videos found: {len(video_results)}")
        print(f"Successfully processed: {successful_predictions}")
        if failed_videos > 0:
             print(f"Failed to process: {failed_videos}")
        print("="*79) # Footer separator

    logger.info("Pipeline execution finished.")

if __name__ == "__main__":
