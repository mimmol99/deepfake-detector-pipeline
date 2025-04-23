# src/utils.py
"""
Utility functions for logging, file downloads, etc.
"""

import logging
import requests
import sys
from pathlib import Path
from tqdm import tqdm

def create_logger(log_path, logger_name="pipeline_logger"):
    """Creates a logger object."""
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): # Avoid adding multiple handlers
        logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler(sys.stdout) # Use stdout for console
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def download_file(url: str, output_path: Path, logger: logging.Logger):
    """Downloads a file from a URL to the specified path with progress."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading {url} to {output_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=output_path.name)
        with open(output_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
             logger.error(f"Download error: Size mismatch for {output_path.name}")
             return False

        logger.info(f"Successfully downloaded {output_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}. Error: {e}")
        # Attempt to clean up partially downloaded file
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                 logger.error(f"Could not remove partial file {output_path}")
        return False
    except Exception as e:
         logger.error(f"An unexpected error occurred during download {url}. Error: {e}")
         if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                 logger.error(f"Could not remove partial file {output_path}")
         return False


def setup_environment(dlib_url, dlib_path, model_url, model_path, logger):
    """Downloads necessary models if they don't exist."""
    logger.info("--- Setting up environment ---")
    all_files_present = True

    # Download dlib predictor
    if not dlib_path.exists():
        logger.info(f"Dlib predictor not found at {dlib_path}.")
        if not download_file(dlib_url, dlib_path, logger):
            logger.error("Exiting due to dlib predictor download failure.")
            all_files_present = False
    else:
        logger.info(f"Dlib predictor already exists at {dlib_path}.")

    # Download detection model
    if not model_path.exists():
        logger.info(f"Detection model not found at {model_path}.")
        if not download_file(model_url, model_path, logger):
             logger.error("Exiting due to detection model download failure.")
             all_files_present = False
    else:
        logger.info(f"Detection model already exists at {model_path}.")

    if all_files_present:
        logger.info("--- Environment setup complete ---")
    else:
        logger.error("--- Environment setup failed ---")
