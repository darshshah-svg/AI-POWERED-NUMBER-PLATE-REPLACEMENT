# YOLO Number Plate Detection and Video Processing Suite

A comprehensive Python suite for detecting, tracking, and replacing vehicle number plates in video streams using YOLOv8. The project offers multiple processing modes optimized for different use cases, from simple frame-by-frame processing to advanced temporal stability and seamless video output generation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Processing Modes](#processing-modes)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Advanced Usage](#advanced-usage)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project provides a complete solution for number plate detection and replacement in video content using a custom-trained YOLOv8 model (16 epochs). The suite includes multiple processing variants designed for different requirements:

- **Frame-based processing** for high-quality individual frame analysis
- **Seamless video processing** for smooth video output generation
- **Advanced temporal tracking** for consistent plate replacement across frames
- **Stable video processing** with enhanced smoothing and tracking algorithms

The system uses advanced computer vision techniques including perspective correction, dynamic text sizing, temporal consistency, and optional logo overlays to produce professional-quality results.

---

## Features

### Core Capabilities
- ✅ **YOLOv8-based Detection**: Custom trained model with 16 epochs for accurate number plate detection
- ✅ **Multiple Processing Modes**: Choose from 4 different processing approaches based on your needs
- ✅ **Perspective Correction**: Automatic correction for angled and distorted plates
- ✅ **Dynamic Text Sizing**: Intelligent font scaling to fit replacement text perfectly
- ✅ **Temporal Consistency**: Advanced tracking for stable replacements across video frames
- ✅ **Logo Integration**: Optional logo overlay with proportional scaling
- ✅ **Quality Assessment**: Sharpness-based frame selection for optimal visual results

### Advanced Features
- 🔄 **Temporal Smoothing**: Gaussian filtering for stable bounding box tracking
- 📊 **Comprehensive Reporting**: JSON metadata and detailed processing statistics
- 🎯 **Multi-format Support**: Works with various video formats and resolutions
- ⚡ **GPU Acceleration**: CUDA support for faster inference
- 🔧 **Configurable Parameters**: Extensive customization options for different scenarios

---

## Project Structure

```
yolo_model_16epochs/
├── yolo model testing/
│   ├── Models/
│   │   └── best .pt              # Trained YOLOv8 model (16 epochs)
│   ├── Assets/                   # Fonts and logo files
│   ├── fonts/                    # Custom font files
│   ├── output/                   # Frame-based processing output
│   ├── output_seamless/          # Seamless video processing output
│   ├── output_advanced/          # Advanced processing output
│   ├── output_stable/            # Stable processing output
│   ├── final3.py                 # Main frame-based processor
│   ├── seamless_video_processor.py    # Video output generator
│   ├── advanced_video_processor.py    # Advanced tracking processor
│   ├── stable_video_processor.py      # Stable processing with smoothing
│   ├── Dynamic Values.csv        # Replacement text data
│   └── Readme.md                 # This file
```

---

## Processing Modes

### 1. Frame-based Processing (`final3.py`)
**Best for**: High-quality individual frame analysis, detailed inspection
- Processes each frame independently
- Highest quality output for individual frames
- Comprehensive metadata generation
- Ideal for analysis and quality assessment

### 2. Seamless Video Processing (`seamless_video_processor.py`)
**Best for**: Creating smooth video outputs, basic video editing
- Generates complete video files with replaced plates
- Basic temporal consistency
- Optimized for video playback
- Good balance of quality and performance

### 3. Advanced Video Processing (`advanced_video_processor.py`)
**Best for**: Professional video production, complex tracking scenarios
- Enhanced tracking algorithms
- Improved temporal consistency
- Advanced perspective correction
- Professional-grade output quality

### 4. Stable Video Processing (`stable_video_processor.py`)
**Best for**: Long videos, challenging lighting conditions, maximum stability
- Gaussian smoothing for bounding box stability
- Enhanced tracking with temporal filtering
- Robust handling of detection gaps
- Maximum temporal consistency

---

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM (16GB recommended for large videos)
- **GPU**: CUDA-compatible GPU recommended for faster processing

### Python Dependencies
```bash
pip install opencv-python numpy ultralytics pytesseract pandas Pillow scipy
```

### Core Libraries
- `opencv-python` (4.5+): Video and image processing
- `numpy` (1.21+): Numerical operations
- `ultralytics` (8.0+): YOLOv8 model inference
- `pytesseract` (0.3+): OCR capabilities (optional)
- `pandas` (1.3+): CSV data handling
- `Pillow` (8.0+): Advanced image and text rendering
- `scipy` (1.7+): Signal processing for temporal smoothing

### External Dependencies
- **Tesseract OCR**: Required for text recognition features
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  
  # macOS
  brew install tesseract
  
  # Windows: Download from GitHub releases
  ```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd yolo_model_16epochs/yolo\ model\ testing/
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install opencv-python numpy ultralytics pytesseract pandas Pillow scipy
   ```

4. **Install Tesseract OCR** (see requirements section above)

5. **Verify model file**:
   Ensure `Models/best .pt` exists (52MB YOLOv8 model file)

---

## Quick Start

### Basic Frame Processing
```python
python final3.py
```

### Video Processing with Output
```python
python seamless_video_processor.py
```

### Advanced Processing
```python
python advanced_video_processor.py
```

### Maximum Stability Processing
```python
python stable_video_processor.py
```

### Configuration
Edit the configuration section in your chosen processor:

```python
# Basic Configuration
MODEL_PATH = "Models/best .pt"
VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_DIR = "output/"
CSV_PATH = "Dynamic Values.csv"

# Optional Assets
FONT_PATH = "Assets/custom_font.otf"
LOGO_PATH = "Assets/logo.png"

# Processing Parameters
CONFIDENCE_THRESHOLD = 0.5
FRAME_INTERVAL = 1  # Process every nth frame
```

---

## Configuration

### Detection Parameters
- `confidence_threshold`: Minimum detection confidence (0.1-0.9)
- `frame_interval`: Frame sampling rate (1 = every frame)
- `max_detections`: Maximum plates per frame

### Replacement Parameters
- `text_color`: RGB color for replacement text
- `border_thickness`: Border width around plates
- `logo_width_ratio`: Logo size relative to plate width (0.2-0.4)

### Temporal Parameters (Advanced/Stable modes)
- `smoothing_window`: Frames for temporal smoothing
- `tracking_threshold`: Distance threshold for plate tracking
- `gaussian_sigma`: Smoothing strength for stable processing

### Quality Parameters
- `min_plate_area`: Minimum plate size for processing
- `sharpness_threshold`: Minimum frame sharpness for selection
- `aspect_ratio_range`: Valid plate aspect ratio range

---

## Output Formats

### Frame-based Output
- **Images**: Individual processed frames as JPEG files
- **Metadata**: JSON files with detection coordinates and confidence
- **Reports**: Text summaries with processing statistics

### Video Output
- **MP4 Files**: Complete processed videos with replaced plates
- **Frame Sequences**: Individual frames for further processing
- **Tracking Data**: JSON files with temporal tracking information

### Metadata Structure
```json
{
  "frame_index": 123,
  "detections": [
    {
      "confidence": 0.85,
      "bbox": [x1, y1, x2, y2],
      "replacement_text": "MH 18 EQ 0001",
      "sharpness_score": 245.6
    }
  ],
  "processing_time": 0.15
}
```

---

## Advanced Usage

### Custom Text Replacement
Modify `Dynamic Values.csv`:
```csv
Sr. No,Name
1,MH 18 EQ 0001
2,TN 57 CY 9006
3,Custom Text Here
```

### Batch Processing
```python
# Process multiple videos
video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in video_files:
    # Configure and process each video
    process_video(video)
```

### Custom Font Integration
```python
# Add custom fonts to Assets/ directory
FONT_PATH = "Assets/your_custom_font.ttf"
FALLBACK_FONT_PATH = "Assets/fallback_font.ttf"
```

### Logo Overlay
```python
# Enable logo overlay
ENABLE_LOGO = True
LOGO_PATH = "Assets/company_logo.png"
LOGO_WIDTH_RATIO = 0.3  # 30% of plate width
```

---

## Performance Optimization

### GPU Acceleration
```python
# Enable CUDA for faster processing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)
```

### Memory Management
- Process videos in chunks for large files
- Adjust frame interval for faster processing
- Use lower resolution for preview processing

### Processing Speed Tips
1. **Use GPU**: 5-10x faster inference
2. **Adjust frame interval**: Process every 2nd or 3rd frame
3. **Reduce resolution**: Scale down input for faster processing
4. **Batch processing**: Process multiple detections together

---

## Troubleshooting

### Common Issues

**Model not found**:
```
FileNotFoundError: Models/best .pt
```
- Ensure the model file exists in the Models/ directory
- Check file permissions and path separators

**CUDA out of memory**:
```
RuntimeError: CUDA out of memory
```
- Reduce batch size or use CPU processing
- Process smaller video chunks

**No detections found**:
- Check confidence threshold (try lowering to 0.3)
- Verify video quality and plate visibility
- Ensure model is compatible with your use case

**Font loading errors**:
- Verify font file paths and formats
- Use fallback fonts for compatibility
- Check font file permissions

### Performance Issues
- **Slow processing**: Enable GPU acceleration, adjust frame interval
- **Poor quality**: Increase confidence threshold, check input video quality
- **Inconsistent tracking**: Use stable processing mode, adjust smoothing parameters

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contributing

We welcome contributions to improve the project:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Submit a pull request** with detailed description

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Include unit tests for new features
- Update documentation for API changes

---

## License

This project is provided for educational and research purposes. Please ensure compliance with:

- YOLOv8 model licensing terms
- Video content usage rights
- Font and logo licensing requirements
- Local privacy and data protection regulations

---

## Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **OpenCV** community for computer vision tools
- **Contributors** who helped improve this project

For questions, issues, or contributions, please visit our GitHub repository or contact the development team.

---

**Last Updated**: September 2025  
**Version**: 2.0  
**Model Version**: YOLOv8 (16 epochs)