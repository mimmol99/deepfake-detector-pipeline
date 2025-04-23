# src/pipeline.py
"""
Core pipeline functions for processing videos.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import torch
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Tuple

# Use relative imports within the 'src' package
from .preprocessing import extract_face_frame
from .detection import predict_frames

logger = logging.getLogger("pipeline_logger." + __name__) # Child logger


def select_frame_indices(total_frames: int, sampling_mode: str, num_frames: int, stride: int) -> np.ndarray:
    """Selects frame indices based on the chosen sampling strategy."""
    indices = []
    if total_frames <= 0:
        logger.warning(f"Video has {total_frames} frames. Cannot select indices.")
        return np.array([], dtype=int)

    effective_num_frames = min(num_frames, total_frames)

    if sampling_mode == 'linspace':
        if effective_num_frames > 0:
            indices = np.linspace(0, total_frames - 1, effective_num_frames, endpoint=True, dtype=int)
        else: # Handle case where num_frames is 0 or less
             logger.warning("num_frames is non-positive for linspace sampling.")
             indices = np.array([], dtype=int)
    elif sampling_mode == 'consecutive':
        if effective_num_frames > 0:
            start_idx = 0
            if total_frames > effective_num_frames:
                start_idx = max(0, (total_frames - effective_num_frames) // 2) # Center crop frames
            # Ensure end index doesn't exceed total_frames
            end_idx = min(start_idx + effective_num_frames, total_frames)
            indices = np.arange(start_idx, end_idx, dtype=int)
        else:
            logger.warning("num_frames is non-positive for consecutive sampling.")
            indices = np.array([], dtype=int)
    elif sampling_mode == 'stride':
        if stride > 0:
            indices = np.arange(0, total_frames, stride, dtype=int)
        else:
            logger.warning("Stride must be positive for stride sampling.")
            indices = np.array([], dtype=int)
            # Fallback? Maybe sample first frame?
            # if total_frames > 0: indices = np.array([0], dtype=int)
    else:
        logger.warning(f"Invalid sampling mode: {sampling_mode}. Defaulting to linspace.")
        # Default to linspace
        if effective_num_frames > 0:
             indices = np.linspace(0, total_frames - 1, effective_num_frames, endpoint=True, dtype=int)
        else:
             indices = np.array([], dtype=int)

    # Remove duplicates and sort, although sampling methods should generally avoid this
    indices = np.unique(indices)

    if len(indices) == 0 and total_frames > 0:
        logger.warning(f"No frame indices were selected for video (Total: {total_frames}). "
                       f"Mode: {sampling_mode}, NumFrames: {num_frames}, Stride: {stride}. "
                       "Sampling first frame as fallback.")
        indices = np.array([0], dtype=int) # Sample first frame if others failed
    elif len(indices) == 0 and total_frames == 0:
         pass # Already logged earlier

    return indices


def process_video(
    video_path: Path,
    face_detector, # dlib.fhog_object_detector
    face_predictor, # dlib.shape_predictor
    detection_model, # torch.jit.ScriptModule
    detection_preprocessor, # transforms.Compose
    device: torch.device,
    sampling_mode: str,
    num_frames: int,
    stride: int,
    face_res: int,
    face_alignment_scale: float
) -> Optional[Tuple[float, float]]:
    """
    Processes a single video: extracts faces, runs detection, returns avg (real, fake) probability.
    """
    logger.info(f"Processing video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video Info - Total Frames: {total_frames}, FPS: {fps:.2f}")

    if total_frames <= 0: # Check again after opening
        logger.warning(f"Video has {total_frames} frames after opening: {video_path}")
        cap.release()
        return None

    # --- Select Frame Indices ---
    indices = select_frame_indices(total_frames, sampling_mode, num_frames, stride)
    if len(indices) == 0:
         logger.warning(f"Could not select any frames for {video_path.name}. Skipping.")
         cap.release()
         return None

    logger.info(f"Selected {len(indices)} frames to process: {indices[:5]}...{indices[-5:]}") # Log first/last few

    # --- Process Selected Frames ---
    processed_faces = []
    processed_frame_indices = [] # Keep track of which frames were successfully processed
    current_frame_idx = -1
    indices_set = set(indices.tolist()) # Use set for faster lookup
    max_index_to_process = max(indices)

    pbar_frames = tqdm(total=len(indices), desc=f"Extracting faces from {video_path.name}", leave=False)

    try: # Wrap frame reading loop
        while True:
            if current_frame_idx > max_index_to_process: # Optimization
                 logger.debug(f"Reached max index {max_index_to_process}, stopping read.")
                 break

            ret, frame = cap.read()
            current_frame_idx += 1

            if not ret:
                logger.warning(f"Read error or end of video reached at frame {current_frame_idx} (before max index {max_index_to_process}) for {video_path.name}")
                break # End of video or error

            if current_frame_idx in indices_set:
                face_pil = extract_face_frame(frame, face_detector, face_predictor, face_res, face_alignment_scale)
                if face_pil:
                    processed_faces.append(face_pil)
                    processed_frame_indices.append(current_frame_idx)
                # else: # Already logged in extract_face_frame
                #     logger.warning(f"No face detected/processed frame {current_frame_idx}")
                pbar_frames.update(1) # Update progress bar for each targeted frame

    except Exception as e:
         logger.error(f"Error during frame reading loop for {video_path.name}: {e}", exc_info=True)
    finally:
        pbar_frames.close()
        cap.release()
        logger.debug(f"Released video capture for {video_path.name}")


    # --- Run Detection ---
    if not processed_faces:
        logger.warning(f"No faces could be extracted from video: {video_path}")
        return None # Indicate failure or no faces

    logger.info(f"Running detection on {len(processed_faces)} extracted faces...")
    frame_probs = predict_frames(detection_model, processed_faces, detection_preprocessor, device)

    if frame_probs is None or len(frame_probs) == 0:
         logger.error(f"Prediction failed or returned empty for video {video_path}")
         return None

    # Average probabilities across successfully processed frames
    avg_probs = frame_probs.mean(dim=0)
    num_predicted = len(frame_probs) # Should match len(processed_faces)
    logger.info(f"Processed {num_predicted} frames ({processed_frame_indices}) for {video_path.name}. "
                f"Avg Real Prob: {avg_probs[0]:.4f}, Avg Fake Prob: {avg_probs[1]:.4f}")

    # Return tuple of (real_prob, fake_prob)
    return avg_probs[0].item(), avg_probs[1].item()


def process_video_folder(
    folder_path: Path,
    face_detector, face_predictor, detection_model, detection_preprocessor,
    device: torch.device,
    sampling_mode: str, num_frames: int, stride: int, face_res: int, face_alignment_scale: float
) -> Dict[str, Optional[Dict[str, float]]]:
    """Processes all videos in a folder and returns results."""
    # Find video files (adjust extensions if needed)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(folder_path.rglob(ext))) # Recursive search

    if not video_files:
        logger.error(f"No video files ({', '.join(video_extensions)}) found in {folder_path} or subdirectories.")
        return {}

    # Remove duplicates if rglob finds same file via different paths (e.g. symlinks)
    video_files = sorted(list(set(video_files)))

    logger.info(f"Found {len(video_files)} videos to process in {folder_path}.")

    results: Dict[str, Optional[Dict[str, float]]] = {}
    # Using tqdm for overall progress bar
    for video_path in tqdm(video_files, desc="Processing Videos", unit="video"):
        avg_probs_tuple = process_video(
            video_path, face_detector, face_predictor, detection_model,
            detection_preprocessor, device, sampling_mode, num_frames, stride, face_res, face_alignment_scale
        )
        if avg_probs_tuple is not None:
            results[str(video_path)] = {
                'real_prob': avg_probs_tuple[0],
                'fake_prob': avg_probs_tuple[1]
            }
        else:
             results[str(video_path)] = None # Indicate processing failure for this video

