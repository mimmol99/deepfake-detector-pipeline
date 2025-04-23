# src/config.py
"""
Configuration settings for the deepfake detection pipeline.
"""

from pathlib import Path

# --- Paths ---
# Base directory (assuming models are stored relative to the project root or a specific data dir)
# We might want to make these configurable via command-line or environment variables later.
MODEL_BASE_DIR = Path("./models") # Or choose a persistent location

DLIB_PREDICTOR_DIR = MODEL_BASE_DIR / "dlib_tools"
DLIB_PREDICTOR_NAME = "shape_predictor_81_face_landmarks.dat"
DLIB_PREDICTOR_PATH = DLIB_PREDICTOR_DIR / DLIB_PREDICTOR_NAME
DLIB_DOWNLOAD_URL = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat"

DETECTION_MODEL_DIR = MODEL_BASE_DIR / "detection_model"
DETECTION_MODEL_NAME = "deepfake_detector.torchscript"
DETECTION_MODEL_PATH = DETECTION_MODEL_DIR / DETECTION_MODEL_NAME
DETECTION_MODEL_URL = "https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.torchscript"

# --- Logging ---
LOG_FILE = "pipeline.log"

# --- Processing Parameters ---
FRAME_SAMPLING_MODE = 'linspace' # Options: 'linspace', 'consecutive', 'stride'
NUM_FRAMES_PER_VIDEO = 16       # Number of frames to sample if mode is 'linspace' or 'consecutive'
FRAME_STRIDE = 10               # Stride if mode is 'stride'
FACE_CROP_RESOLUTION = 256      # Resolution for aligned face cropping (dlib part)
FACE_ALIGNMENT_SCALE = 1.3      # Scale factor for face bounding box in alignment
DETECTION_MODEL_INPUT_SIZE = (224, 224) # Input size expected by the TorchScript model

# --- Device ---
# Device selection logic can be placed here or in main.py/utils.py
