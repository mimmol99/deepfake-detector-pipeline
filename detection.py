# src/detection.py
"""
Functions related to the deepfake detection model: loading, preprocessing, inference.
"""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("pipeline_logger." + __name__) # Child logger


def load_detection_model(model_path: Path) -> Optional[torch.jit.ScriptModule]:
    """Loads the TorchScript detection model."""
    if not model_path.exists():
        logger.error(f"Detection model file not found: {model_path}")
        return None
    try:
        model = torch.jit.load(str(model_path)) # Use string representation of Path
        model.eval() # Set to evaluation mode
        logger.info(f"Successfully loaded detection model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load detection model from {model_path}. Error: {e}", exc_info=True)
        return None


def get_detection_preprocessor(input_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns the preprocessing transform for the detection model."""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])


def predict_frames(model: torch.jit.ScriptModule, frames: List[Image.Image],
                   preprocessor: transforms.Compose, device: torch.device) -> Optional[torch.Tensor]:
    """Runs inference on a list of PIL Image frames and returns probabilities on CPU."""
    if not frames:
        logger.warning("predict_frames called with an empty list of frames.")
        return None

    try:
        # Preprocess frames and create batch
        # Ensure frames are valid PIL Images before preprocessing
        valid_frames = [frame for frame in frames if isinstance(frame, Image.Image)]
        if len(valid_frames) != len(frames):
             logger.warning(f"Some invalid items found in frame list. Processing {len(valid_frames)} valid frames.")
        if not valid_frames:
            logger.error("No valid PIL frames to process in predict_frames.")
            return None

        batch_tensors = torch.stack([preprocessor(frame) for frame in valid_frames]).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensors)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu() # Return probabilities on CPU

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
