import cv2
import numpy as np
import os
import subprocess
import json
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import pytesseract
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
from collections import deque
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def order_points(pts):
    """Order coordinates in a quadrilateral (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Find center point
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    
    # Sort points by quadrant relative to center
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    
    for point in pts:
        x, y = point[0], point[1]
        if x < center_x and y < center_y:  # Top-left quadrant
            if top_left is None or (x + y) < (top_left[0] + top_left[1]):
                top_left = point
        elif x >= center_x and y < center_y:  # Top-right quadrant
            if top_right is None or (x - y) > (top_right[0] - top_right[1]):
                top_right = point
        elif x >= center_x and y >= center_y:  # Bottom-right quadrant
            if bottom_right is None or (x + y) > (bottom_right[0] + bottom_right[1]):
                bottom_right = point
        else:  # Bottom-left quadrant
            if bottom_left is None or (y - x) > (bottom_left[1] - bottom_left[0]):
                bottom_left = point
    
    # Fallback to original method if any point is None
    if any(p is None for p in [top_left, top_right, bottom_right, bottom_left]):
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
    else:
        rect[0] = top_left
        rect[1] = top_right
        rect[2] = bottom_right
        rect[3] = bottom_left
    
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to warp a quadrilateral into a rectangle."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate all four edge lengths for better accuracy
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    # Use average of opposite sides for more stable dimensions
    maxWidth = int((widthA + widthB) / 2)
    maxHeight = int((heightA + heightB) / 2)
    
    # Ensure minimum aspect ratio for number plates (typically 3:1 to 4:1)
    aspect_ratio = maxWidth / maxHeight if maxHeight > 0 else 3.5
    if aspect_ratio < 2.5:
        maxWidth = int(maxHeight * 3.5)
    elif aspect_ratio > 5.0:
        maxHeight = int(maxWidth / 3.5)
    
    # Create destination points with proper rectangular order
    dst = np.array([
        [0, 0],                    # top-left
        [maxWidth, 0],             # top-right  
        [maxWidth, maxHeight],     # bottom-right
        [0, maxHeight]             # bottom-left
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M, (maxWidth, maxHeight)

class DetectionStabilizer:
    """Stabilizes detection coordinates across frames to reduce jitter."""
    
    def __init__(self, buffer_size=10, smoothing_factor=0.3):
        self.buffer_size = buffer_size
        self.smoothing_factor = smoothing_factor
        self.detection_history = deque(maxlen=buffer_size)
        self.stable_detection = None
        
    def add_detection(self, detection_box, confidence):
        """Add a new detection and return stabilized coordinates."""
        self.detection_history.append({
            'box': detection_box,
            'confidence': confidence,
            'timestamp': len(self.detection_history)
        })
        
        return self._get_stabilized_detection()
    
    def _get_stabilized_detection(self):
        """Calculate stabilized detection from history."""
        if not self.detection_history:
            return None
            
        # Weight recent detections more heavily
        weights = []
        boxes = []
        
        for i, detection in enumerate(self.detection_history):
            # Exponential decay for older detections
            age_weight = np.exp(-0.1 * (len(self.detection_history) - i - 1))
            confidence_weight = detection['confidence']
            total_weight = age_weight * confidence_weight
            
            weights.append(total_weight)
            boxes.append(detection['box'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        boxes = np.array(boxes)
        stabilized_box = np.average(boxes, axis=0, weights=weights)
        
        # Apply smoothing if we have a previous stable detection
        if self.stable_detection is not None:
            stabilized_box = (self.smoothing_factor * stabilized_box + 
                            (1 - self.smoothing_factor) * self.stable_detection)
        
        self.stable_detection = stabilized_box
        return stabilized_box
    
    def get_last_stable_detection(self):
        """Get the last stable detection for frames with no new detections."""
        return self.stable_detection

class TemporalConsistencyManager:
    """Manages temporal consistency of plate replacements across frames."""
    
    def __init__(self, consistency_threshold=0.8):
        self.consistency_threshold = consistency_threshold
        self.reference_plate = None
        self.reference_transform = None
        self.frame_count = 0
        
    def should_update_reference(self, current_confidence):
        """Determine if we should update the reference plate."""
        if self.reference_plate is None:
            return True
        
        # Update reference every N frames or if confidence is significantly higher
        if (self.frame_count % 30 == 0 or 
            current_confidence > self.consistency_threshold):
            return True
            
        return False
    
    def set_reference_plate(self, plate_image, transform_info):
        """Set the reference plate for consistent replacement."""
        self.reference_plate = plate_image.copy()
        self.reference_transform = transform_info.copy()
        
    def get_consistent_replacement(self, target_size):
        """Get a consistently sized replacement plate."""
        if self.reference_plate is None:
            return None
            
        # Validate target size
        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            return None
            
        # Resize reference plate to target size
        try:
            consistent_plate = cv2.resize(self.reference_plate, (target_width, target_height))
            return consistent_plate
        except Exception:
            return None
    
    def increment_frame(self):
        """Increment frame counter."""
        self.frame_count += 1

def detect_plate_contours_stable(plate_region):
    """Detect plate contours with stability enhancements."""
    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for stability
    gray_plate = cv2.GaussianBlur(gray_plate, (3, 3), 0)
    
    # Multiple detection methods with stability focus
    methods = []
    
    # Method 1: Adaptive thresholding (most stable)
    thresh1 = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    # Morphological operations for stability
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    methods.append(('adaptive_stable', contours1))
    
    # Method 2: OTSU thresholding
    _, thresh2 = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    methods.append(('otsu_stable', contours2))
    
    # Find the most stable contour
    best_contour = None
    best_score = 0
    
    for method_name, contours in methods:
        if contours:
            for contour in contours:
                # Filter by area and aspect ratio for stability
                area = cv2.contourArea(contour)
                region_area = plate_region.shape[0] * plate_region.shape[1]
                area_ratio = area / region_area
                
                if area_ratio < 0.15 or area_ratio > 0.85:
                    continue
                
                # Check aspect ratio stability
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if aspect_ratio < 2.0 or aspect_ratio > 6.0:
                    continue
                
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Score based on stability factors
                stability_score = area_ratio
                if len(approx) == 4:
                    stability_score *= 2.0  # Prefer quadrilaterals
                elif len(approx) >= 4:
                    stability_score *= 1.5
                
                # Prefer contours closer to expected aspect ratio
                ideal_ratio = 3.5
                ratio_penalty = abs(aspect_ratio - ideal_ratio) / ideal_ratio
                stability_score *= (1.0 - ratio_penalty * 0.5)
                
                if stability_score > best_score:
                    best_score = stability_score
                    if len(approx) >= 4:
                        best_contour = approx[:4] if len(approx) > 4 else approx
                    else:
                        # Create stable rectangle
                        best_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    
    return best_contour

def create_replacement_plate(width, height, new_plate_number, font_path, logo_image, logo_image_rgb, use_custom_font=True):
    """Create a replacement number plate with white background, logo, and text."""
    # Create white plate
    white_plate = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw border with consistent thickness
    border_thickness = max(2, min(width, height) // 100)  # Adaptive border thickness
    cv2.rectangle(white_plate, (0, 0), (width-1, height-1), (0, 0, 0), border_thickness)
    
    # Add logo and text with consistent positioning
    image_width_ratio = 0.3
    image_max_width = int(width * image_width_ratio)
    image_max_height = height - 10
    
    if logo_image is not None:
        logo_height, logo_width = logo_image.shape[:2]
        scale = min(image_max_width / logo_width, image_max_height / logo_height)
        new_logo_width = int(logo_width * scale)
        new_logo_height = int(logo_height * scale)
        logo_resized = cv2.resize(logo_image, (new_logo_width, new_logo_height), interpolation=cv2.INTER_AREA)
        logo_y = (height - new_logo_height) // 2
        logo_x = 5
    else:
        new_logo_width = 0
        logo_y, logo_x = 0, 0
    
    text_max_width = width - new_logo_width - 15
    
    if use_custom_font and os.path.exists(font_path):
        pil_image = Image.fromarray(cv2.cvtColor(white_plate, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate font size with stability
        font_size = int(min(height, text_max_width) * 0.4)  # More conservative sizing
        min_size = 8
        while font_size > min_size:
            try:
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = font.getbbox(new_plate_number)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                if text_width <= text_max_width * 0.85 and text_height <= height * 0.7:
                    break
                font_size -= 1
            except Exception:
                font_size -= 1
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = font.getbbox(new_plate_number)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            start_x = logo_x + new_logo_width + 10
            start_y = (height - text_height) // 2
            draw.text((start_x, start_y), new_plate_number, fill=(0, 0, 0), font=font)
            
            if logo_image is not None:
                pil_logo = Image.fromarray(logo_image_rgb)
                pil_logo = pil_logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)
                mask = pil_logo.split()[3] if logo_image.shape[2] == 4 else None
                try:
                    pil_image.paste(pil_logo, (logo_x, logo_y), mask)
                except Exception:
                    pil_image.paste(pil_logo, (logo_x, logo_y))
            
            white_plate = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            pass  # Fall back to OpenCV text
    
    return white_plate

class StableVideoProcessor:
    def __init__(self, model_path, video_path, output_dir, new_plate_number, font_path, logo_path):
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = output_dir
        self.new_plate_number = new_plate_number
        self.font_path = font_path
        self.logo_path = logo_path
        
        # Stability components
        self.stabilizer = DetectionStabilizer(buffer_size=15, smoothing_factor=0.2)
        self.consistency_manager = TemporalConsistencyManager(consistency_threshold=0.85)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Load logo
        self.logo_image = None
        self.logo_image_rgb = None
        if logo_path and os.path.exists(logo_path):
            try:
                self.logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
                if self.logo_image is not None:
                    if self.logo_image.shape[2] == 4:
                        self.logo_image_rgb = cv2.cvtColor(self.logo_image, cv2.COLOR_BGRA2RGBA)
                    else:
                        self.logo_image_rgb = cv2.cvtColor(self.logo_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error loading logo: {e}")
    
    def stable_plate_replacement(self, frame, detection_box, confidence):
        """Replace number plate with stability enhancements."""
        x1, y1, x2, y2 = map(int, detection_box)
        plate_region = frame[y1:y2, x1:x2].copy()
        
        # Detect stable contours
        plate_contour = detect_plate_contours_stable(plate_region)
        
        if plate_contour is not None:
            try:
                pts = plate_contour.reshape(4, 2).astype(np.float32)
                warped, M, (maxWidth, maxHeight) = four_point_transform(plate_region, pts)
                
                # Validate dimensions before processing
                if maxWidth <= 0 or maxHeight <= 0:
                    raise ValueError(f"Invalid dimensions: {maxWidth}x{maxHeight}")
                
                # Create or get consistent replacement plate
                if self.consistency_manager.should_update_reference(confidence):
                    replacement_plate = create_replacement_plate(
                        maxWidth, maxHeight, self.new_plate_number, 
                        self.font_path, self.logo_image, self.logo_image_rgb,
                        use_custom_font=os.path.exists(self.font_path)
                    )
                    self.consistency_manager.set_reference_plate(replacement_plate, {
                        'width': maxWidth, 'height': maxHeight, 'points': pts
                    })
                else:
                    # Use consistent replacement
                    replacement_plate = self.consistency_manager.get_consistent_replacement((maxWidth, maxHeight))
                    if replacement_plate is None:
                        replacement_plate = create_replacement_plate(
                            maxWidth, maxHeight, self.new_plate_number, 
                            self.font_path, self.logo_image, self.logo_image_rgb,
                            use_custom_font=os.path.exists(self.font_path)
                        )
                
                # Transform replacement back with stability
                dst_pts = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype=np.float32)
                src_pts = order_points(pts)
                inverse_M = cv2.getPerspectiveTransform(dst_pts, src_pts)
                
                # Warp the replacement plate back
                warped_replacement = cv2.warpPerspective(replacement_plate, inverse_M, (x2 - x1, y2 - y1))
                
                # Create smooth mask for blending
                mask = np.ones((maxHeight, maxWidth), dtype=np.uint8) * 255
                warped_mask = cv2.warpPerspective(mask, inverse_M, (x2 - x1, y2 - y1))
                
                # Apply Gaussian blur to mask edges for smoother blending
                warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
                
                # Apply the replacement with smooth blending
                plate_roi = frame[y1:y2, x1:x2]
                mask_norm = warped_mask.astype(np.float32) / 255.0
                
                for c in range(3):
                    plate_roi[:, :, c] = (mask_norm * warped_replacement[:, :, c] + 
                                        (1 - mask_norm) * plate_roi[:, :, c]).astype(np.uint8)
                
                frame[y1:y2, x1:x2] = plate_roi
                
                # Draw stable border
                frame_pts = pts + np.array([x1, y1])
                self._draw_stable_border(frame, frame_pts, thickness=2)
                
                return True
                
            except Exception as e:
                print(f"Stable quadrilateral replacement failed: {e}")
        
        # Fallback to rectangular replacement
        plate_width = x2 - x1
        plate_height = y2 - y1
        
        replacement_plate = self.consistency_manager.get_consistent_replacement((plate_width, plate_height))
        if replacement_plate is None:
            replacement_plate = create_replacement_plate(
                plate_width, plate_height, self.new_plate_number,
                self.font_path, self.logo_image, self.logo_image_rgb,
                use_custom_font=os.path.exists(self.font_path)
            )
        
        # Smooth rectangular replacement
        frame[y1:y2, x1:x2] = replacement_plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        return True
    
    def _draw_stable_border(self, frame, pts, thickness=2, color=(0, 0, 0)):
        """Draw stable border with anti-aliasing."""
        ordered_pts = order_points(pts)
        pts_int = ordered_pts.astype(np.int32)
        
        # Draw with anti-aliasing for smoother appearance
        cv2.polylines(frame, [pts_int], True, color, thickness, cv2.LINE_AA)
    
    def process_frame_stable(self, frame):
        """Process frame with stability enhancements."""
        results = self.model(frame, conf=0.5)
        
        current_detection = None
        best_confidence = 0
        
        # Find best detection in current frame
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf[0].cpu().numpy()
                    if confidence > 0.5 and confidence > best_confidence:
                        best_confidence = confidence
                        current_detection = box.xyxy[0].cpu().numpy()
        
        # Use stabilized detection coordinates
        if current_detection is not None:
            stable_detection = self.stabilizer.add_detection(current_detection, best_confidence)
        else:
            # Use last stable detection if no current detection
            stable_detection = self.stabilizer.get_last_stable_detection()
            if stable_detection is not None:
                best_confidence = 0.7  # Assume reasonable confidence for interpolated detection
        
        # Apply stable replacement if we have a detection
        if stable_detection is not None:
            self.stable_plate_replacement(frame, stable_detection, best_confidence)
        
        # Increment frame counter for temporal consistency
        self.consistency_manager.increment_frame()
        
        return frame
    
    def process_video_stable(self):
        """Process video with comprehensive stability enhancements."""
        print("Processing video with stability enhancements...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output path
        video_name = Path(self.video_path).stem
        output_path = os.path.join(self.output_dir, f"{video_name}_stable.mp4")
        
        # Setup video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with stability
            processed_frame = self.process_frame_stable(frame)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        out.release()
        
        print(f"Stable processing complete! Output: {output_path}")
        return output_path

def main():
    """Main function implementing stable number plate replacement."""
    MODEL_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Models/best .pt"
    VIDEO_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Assets /videoplayback.mp4"
    OUTPUT_DIR = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/output_stable"
    CSV_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Dynamic Values.csv"
    FONT_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/fonts/dealerplate california.otf"
    FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.otf"
    IMAGE_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Assets /IND number plate.png"

    try:
        # Read plate number from CSV
        print(f"Reading plate number from CSV: {CSV_PATH}")
        try:
            df = pd.read_csv(CSV_PATH)
            if 'Name' not in df.columns or df.empty:
                print("Warning: CSV file missing 'Name' column or is empty. Falling back to 'ABC123'.")
                new_plate_number = "ABC123"
            else:
                new_plate_number = str(df['Name'].iloc[0]).strip()
                print(f"Using plate number from CSV: {new_plate_number}")
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}. Falling back to 'ABC123'.")
            new_plate_number = "ABC123"

        # Setup font path
        font_path = FONT_PATH if os.path.exists(FONT_PATH) else FALLBACK_FONT_PATH

        # Initialize stable processor
        processor = StableVideoProcessor(
            MODEL_PATH, VIDEO_PATH, OUTPUT_DIR, 
            new_plate_number, font_path, IMAGE_PATH
        )

        # Process video with stability enhancements
        output_path = processor.process_video_stable()
        
        print(f"\n=== Stable Processing Complete ===")
        print(f"Output: {output_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
