# src/preprocessing.py
"""
Face preprocessing functions: detection, landmark extraction, alignment, cropping.
"""

import dlib
import cv2
import numpy as np
from skimage import transform as trans
from PIL import Image
import logging

logger = logging.getLogger("pipeline_logger." + __name__) # Child logger


def get_keypts_dlib(image: np.ndarray, face_rect: dlib.rectangle, predictor: dlib.shape_predictor) -> np.ndarray | None:
    """Detects 5 key facial landmarks using dlib predictor."""
    try:
        shape = predictor(image, face_rect)
        h, w = image.shape[:2]

        # Indices from original DeepfakeBench code (relative to 81-point model)
        key_indices = [37, 44, 30, 49, 55]
        pts = []
        for i in key_indices:
            if i >= shape.num_parts:
                 logger.warning(f"Landmark index {i} out of bounds for {shape.num_parts} parts.")
                 return None
            x = min(max(0, shape.part(i).x), w - 1)
            y = min(max(0, shape.part(i).y), h - 1)
            pts.append([x, y])

        if len(pts) != 5:
            logger.warning("Failed to extract exactly 5 keypoints.")
            return None

        return np.array(pts, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error during landmark detection: {e}", exc_info=True)
        return None


def align_and_crop_face(img: np.ndarray, landmark: np.ndarray, outsize: tuple[int, int]=(256, 256), scale: float=1.3) -> np.ndarray | None:
    """
    Aligns and crops the face using 5 key landmarks.
    Based on DeepfakeBench's img_align_crop. Returns RGB NumPy array.
    """
    try:
        target_size = [112, 112] # Reference size for standard alignment points
        dst_ref = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
            [33.5493, 92.3655], [62.7299, 92.2041]
        ], dtype=np.float32)

        dst = dst_ref.copy()
        if target_size[1] == 112:
            dst[:, 0] += 8.0 # Adjust for insightface alignment standard if needed

        # Scale reference points based on the output size
        scale_x = outsize[1] / target_size[1]
        scale_y = outsize[0] / target_size[0]
        dst[:, 0] *= scale_x
        dst[:, 1] *= scale_y

        # Calculate margin and apply translation (based on output size and scale factor)
        margin_rate = scale - 1.0
        x_margin = outsize[1] * margin_rate / 2.0
        y_margin = outsize[0] * margin_rate / 2.0
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # Rescale destination points to fit within the scaled final image size (outsize * scale)
        # This step seems complex and might be simplified depending on the exact goal.
        # Let's stick to the original logic for now.
        final_w = outsize[1] * scale
        final_h = outsize[0] * scale
        dst[:, 0] *= outsize[1] / final_w
        dst[:, 1] *= outsize[0] / final_h

        # Estimate similarity transform
        src = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        if not tform.estimate(src, dst):
             logger.warning("Failed to estimate similarity transform.")
             return None
        M = tform.params[0:2, :]

        # Apply affine transformation
        warped_img = cv2.warpAffine(img, M, (outsize[1], outsize[0]), borderValue=0.0) # Use black border

        return warped_img

    except Exception as e:
        logger.error(f"Error during face alignment/cropping: {e}", exc_info=True)
        return None


def extract_face_frame(image: np.ndarray, face_detector: dlib.fhog_object_detector,
                         face_predictor: dlib.shape_predictor, target_resolution: int,
                         face_alignment_scale: float) -> Image.Image | None:
    """
    Detects the largest face, extracts landmarks, aligns and crops it.
    Takes BGR NumPy array input (from cv2.read).
    Returns the cropped face image (RGB PIL Image) or None if no face found/processed.
    """
    if image is None:
        logger.warning("Received None image for face extraction.")
        return None

    height, width = image.shape[:2]
    if height == 0 or width == 0:
         logger.warning("Received empty frame (height or width is 0).")
         return None

    try:
        # Convert BGR (from OpenCV) to RGB for dlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
         logger.error(f"OpenCV error during color conversion: {e}")
         return None


    try:
        # Detect faces
        faces = face_detector(rgb_image, 1) # Upsample 1x
        if not faces:
            # logger.debug("No face detected in frame.") # Debug level might be too verbose
            return None # No face detected

        # Find the largest face based on area
        face_rect = max(faces, key=lambda rect: rect.width() * rect.height())

        # Get 5 key landmarks for alignment
        landmarks = get_keypts_dlib(rgb_image, face_rect, face_predictor)
        if landmarks is None:
            # logger.warning("Could not get 5 key landmarks for the largest face.") # Already logged in get_keypts_dlib
            return None # Landmarks not detected properly

        # Align and crop using RGB image
        cropped_face_np = align_and_crop_face(
            rgb_image,
            landmarks,
            outsize=(target_resolution, target_resolution),
            scale=face_alignment_scale
        )

        if cropped_face_np is None:
             # logger.warning("Failed to align and crop face.") # Already logged in align_and_crop_face
             return None

        # Convert aligned RGB NumPy array to PIL Image
        pil_face = Image.fromarray(cropped_face_np)
        return pil_face

    except Exception as e:
        logger.error(f"Unexpected error during face extraction: {e}", exc_info=True)
