# Deepfake Detection Pipeline

This project implements a pipeline to detect deepfakes in videos. It uses face detection and alignment based on the [DeepfakeBench preprocessing](https://github.com/SCLBD/DeepfakeBench) scripts and a pre-trained deepfake detection model (originally from [yermandy/deepfake-detection](https://huggingface.co/yermandy/deepfake-detection) on Hugging Face, likely CLIP-based).

## Features

* Processes a folder of videos (`.mp4`, `.avi`, `.mov`, `.mkv`).
* Detects the largest face in sampled frames using `dlib`.
* Aligns and crops faces based on 5 facial landmarks.
* Applies a pre-trained TorchScript model for real/fake classification.
* Aggregates frame-level predictions to a video-level prediction.
* Provides configurable frame sampling (linspace, consecutive, stride).
* Logs processing details.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd deepfake_detector_pipeline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    * **`dlib` prerequisites:** `dlib` requires `CMake` and a C++ compiler. Install them first.
        * On Ubuntu/Debian: `sudo apt-get update && sudo apt-get install build-essential cmake`
        * On macOS (with Homebrew): `brew install cmake`
        * On Windows: Install CMake ([cmake.org](https://cmake.org/download/)) and a C++ compiler like Visual Studio Build Tools.
    * **Install Python packages:**
        ```bash
        pip install -r requirements.txt
        ```

## Usage

Run the main script from the project's root directory, providing the path to the folder containing your videos:

```bash
