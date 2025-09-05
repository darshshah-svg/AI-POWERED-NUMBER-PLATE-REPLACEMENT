# Video Text Replacement Pipeline

This project implements a comprehensive pipeline for detecting, editing, and propagating text replacements in video frames, specifically targeting applications like number plate replacement. The pipeline leverages advanced computer vision and deep learning techniques to achieve seamless and temporally consistent text replacement across video sequences.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Pipeline Components](#pipeline-components)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The **Video Text Replacement Pipeline** automates the process of detecting text regions (e.g., number plates) in video frames, replacing them with new text from a CSV file, and ensuring temporal consistency across the video.  

The pipeline is designed for scalability and integrates with AWS-hosted services for automated submission and retrieval.

### Key Objectives
- Detect and extract text regions using **YOLOv8**.  
- Normalize and stabilize text regions with **Spatio-Temporal Transformer Networks (STTN)**.  
- Select the highest-quality reference frame for text replacement.  
- Perform text erasure and replacement using **LaMa** and **SRNet**, matching the original style.  
- Propagate edits across frames using **optical flow and motion tracking**.  
- Ensure temporal consistency and assemble the final video with **FFmpeg**.  

---

## Pipeline Components

### 1. ROI Extraction (Text Detection)
- Uses **YOLOv8** to detect bounding boxes of text regions (e.g., number plates).  
- Fetches replacement text from a **CSV file**.  

### 2. STTN (Frontalization & Alignment)
- Applies **Spatio-Temporal Transformer Network** to correct distortions due to rotation, perspective, or motion.  
- Stabilizes text regions for consistent editing.  

### 3. Reference Frame Selection
- Evaluates frame quality using **variance of Laplacian** for sharpness.  
- Selects the clearest, least distorted frame as the reference for text replacement.  

### 4. SRNet (Text Replacement in Reference Frame)
- Uses **LaMa** for inpainting to remove original text while preserving background texture.  
- Inserts new text from CSV, matching font, color, lighting, and perspective.  

### 5. TPM (Text Propagation Across Frames)
- Tracks motion using **optical flow models** (e.g., RAFT, Lucas-Kanade).  
- Propagates edited text across frames, adapting to environmental changes.  

### 6. Temporal Consistency & Final Video Assembly
- Applies temporal smoothing (e.g., **TecoGAN, RAFT**) to eliminate flicker or jitter.  
- Reconstructs and encodes the final video using **FFmpeg**, ensuring audio-video sync.  

---

## Requirements

- Python **3.8+**  
- Libraries:  
  - `torch` (PyTorch for YOLOv8, STTN, SRNet, and RAFT)  
  - `opencv-python` (video processing and optical flow)  
  - `numpy` (numerical operations)  
  - `pandas` (CSV handling)  
  - `ffmpeg-python` (video assembly)  
  - `lama-inpainting` (text erasure)  
  - `tecoGAN` (optional, for temporal smoothing)  
- **FFmpeg** installed on the system  
- **AWS CLI** (optional, for AWS-hosted integration)  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/video-text-replacement.git
cd video-text-replacement

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg
# Ubuntu
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
# Windows: Download from https://ffmpeg.org and add to PATH
