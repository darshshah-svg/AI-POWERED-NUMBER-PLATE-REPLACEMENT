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

def draw_quadrilateral_border(frame, pts, thickness=2, color=(0, 0, 0)):
    """Draw border lines along the quadrilateral edges in proper rectangular order."""
    ordered_pts = order_points(pts)
    pts_int = ordered_pts.astype(np.int32)
    
    # Draw lines connecting the points in rectangular order
    cv2.line(frame, tuple(pts_int[0]), tuple(pts_int[1]), color, thickness)
    cv2.line(frame, tuple(pts_int[1]), tuple(pts_int[2]), color, thickness)
    cv2.line(frame, tuple(pts_int[2]), tuple(pts_int[3]), color, thickness)
    cv2.line(frame, tuple(pts_int[3]), tuple(pts_int[0]), color, thickness)
    
    return frame

def detect_plate_contours(plate_region):
    """Detect the actual number plate contours within the detected region."""
    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing techniques to find the best contour
    methods = []
    
    # Method 1: Simple thresholding
    _, thresh1 = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    methods.append(('thresh', contours1))
    
    # Method 2: Adaptive thresholding
    thresh2 = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    methods.append(('adaptive', contours2))
    
    # Method 3: Edge detection + morphology
    edges = cv2.Canny(gray_plate, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours3, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    methods.append(('edges', contours3))
    
    # Find the best quadrilateral from all methods
    best_contour = None
    best_score = 0
    
    for method_name, contours in methods:
        if contours:
            for contour in contours:
                # Filter by area (should be significant portion of the region)
                area = cv2.contourArea(contour)
                region_area = plate_region.shape[0] * plate_region.shape[1]
                area_ratio = area / region_area
                
                if area_ratio < 0.1 or area_ratio > 0.9:  # Too small or too large
                    continue
                
                # Try to approximate to quadrilateral
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Score based on how close to quadrilateral and area ratio
                if len(approx) == 4:
                    score = area_ratio * 2  # Prefer quadrilaterals
                elif len(approx) >= 4:
                    score = area_ratio * 1.5  # Accept complex shapes
                else:
                    score = area_ratio  # Basic scoring
                
                if score > best_score:
                    best_score = score
                    if len(approx) >= 4:
                        best_contour = approx[:4] if len(approx) > 4 else approx
                    else:
                        # Create bounding rectangle as fallback
                        x, y, w, h = cv2.boundingRect(contour)
                        best_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    
    return best_contour

def create_replacement_plate(width, height, new_plate_number, font_path, logo_image, logo_image_rgb, use_custom_font=True):
    """Create a replacement number plate with white background, logo, and text."""
    # Create white plate
    white_plate = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw border
    border_thickness = 2
    cv2.line(white_plate, (0, 0), (width-1, 0), (0, 0, 0), border_thickness)
    cv2.line(white_plate, (0, height-1), (width-1, height-1), (0, 0, 0), border_thickness)
    cv2.line(white_plate, (0, 0), (0, height-1), (0, 0, 0), border_thickness)
    cv2.line(white_plate, (width-1, 0), (width-1, height-1), (0, 0, 0), border_thickness)
    
    # Add logo and text
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
        
        # Calculate font size
        font_size = int(min(height, width) * 0.5)
        min_size = 5
        while font_size > min_size:
            try:
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = font.getbbox(new_plate_number)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                if text_width <= text_max_width * 0.8 and text_height <= height * 0.6:
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

def seamless_plate_replacement(frame, detection_box, template_plate, new_plate_number, font_path, logo_image, logo_image_rgb):
    """Replace only the actual number plate region seamlessly."""
    x1, y1, x2, y2 = map(int, detection_box)
    plate_region = frame[y1:y2, x1:x2].copy()
    
    # Detect the actual plate contours within the region
    plate_contour = detect_plate_contours(plate_region)
    
    if plate_contour is not None:
        # Use quadrilateral transformation for precise replacement
        try:
            pts = plate_contour.reshape(4, 2).astype(np.float32)
            warped, M, (maxWidth, maxHeight) = four_point_transform(plate_region, pts)
            
            # Create replacement plate with exact dimensions
            replacement_plate = create_replacement_plate(
                maxWidth, maxHeight, new_plate_number, 
                font_path, logo_image, logo_image_rgb,
                use_custom_font=os.path.exists(font_path)
            )
            
            # Transform replacement back to original perspective
            dst_pts = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype=np.float32)
            src_pts = order_points(pts)
            inverse_M = cv2.getPerspectiveTransform(dst_pts, src_pts)
            
            # Warp the replacement plate back
            warped_replacement = cv2.warpPerspective(replacement_plate, inverse_M, (x2 - x1, y2 - y1))
            
            # Create mask for seamless blending
            mask = np.ones((maxHeight, maxWidth), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, inverse_M, (x2 - x1, y2 - y1))
            
            # Apply the replacement only where the mask is active
            plate_roi = frame[y1:y2, x1:x2]
            for c in range(3):
                plate_roi[:, :, c] = np.where(warped_mask > 50, warped_replacement[:, :, c], plate_roi[:, :, c])
            
            frame[y1:y2, x1:x2] = plate_roi
            
            # Draw precise border on the quadrilateral edges
            frame_pts = pts + np.array([x1, y1])
            draw_quadrilateral_border(frame, frame_pts, thickness=2, color=(0, 0, 0))
            
            return True
            
        except Exception as e:
            print(f"Quadrilateral replacement failed: {e}, falling back to rectangular")
    
    # Fallback to rectangular replacement if contour detection fails
    plate_width = x2 - x1
    plate_height = y2 - y1
    
    replacement_plate = create_replacement_plate(
        plate_width, plate_height, new_plate_number,
        font_path, logo_image, logo_image_rgb,
        use_custom_font=os.path.exists(font_path)
    )
    
    # Replace the entire detection region
    frame[y1:y2, x1:x2] = replacement_plate
    
    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    return True

class SeamlessVideoProcessor:
    def __init__(self, model_path, video_path, output_dir, new_plate_number, font_path, logo_path):
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = output_dir
        self.new_plate_number = new_plate_number
        self.font_path = font_path
        self.logo_path = logo_path
        
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
    
    def find_best_detection_frame(self, sample_interval=30):
        """Find the best frame with the clearest number plate detection."""
        print("Analyzing video to find best detection frame...")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        best_frame = None
        best_detection = None
        best_confidence = 0
        best_frame_idx = 0
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every N frames for efficiency
            if frame_idx % sample_interval == 0:
                results = self.model(frame, conf=0.5)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = box.conf[0].cpu().numpy()
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_detection = box
                                best_frame = frame.copy()
                                best_frame_idx = frame_idx
                                print(f"New best detection at frame {frame_idx}: confidence {confidence:.3f}")
            
            frame_idx += 1
            if frame_idx % 300 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Analysis progress: {progress:.1f}%")
        
        cap.release()
        
        if best_frame is not None:
            print(f"Best detection found at frame {best_frame_idx} with confidence {best_confidence:.3f}")
            return best_frame, best_detection, best_frame_idx
        else:
            print("No suitable detection found!")
            return None, None, None
    
    def process_frame_with_seamless_replacement(self, frame):
        """Process a frame and replace only detected number plates seamlessly."""
        results = self.model(frame, conf=0.5)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf[0].cpu().numpy()
                    if confidence > 0.5:
                        detection_box = box.xyxy[0].cpu().numpy()
                        
                        # Apply seamless replacement only to detected plates
                        seamless_plate_replacement(
                            frame, detection_box, None, self.new_plate_number,
                            self.font_path, self.logo_image, self.logo_image_rgb
                        )
        
        return frame
    
    def process_video_seamless(self):
        """Process video with seamless number plate replacement."""
        print("Processing video with seamless plate replacement...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output path
        video_name = Path(self.video_path).stem
        output_path = os.path.join(self.output_dir, f"{video_name}_seamless.mp4")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with seamless replacement
            processed_frame = self.process_frame_with_seamless_replacement(frame)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        out.release()
        
        print(f"Seamless processing complete! Output: {output_path}")
        return output_path

def main():
    """Main function implementing seamless number plate replacement."""
    MODEL_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Models/best .pt"
    VIDEO_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Assets /videoplayback.mp4"
    OUTPUT_DIR = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/output_seamless"
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

        # Initialize processor
        processor = SeamlessVideoProcessor(
            MODEL_PATH, VIDEO_PATH, OUTPUT_DIR, 
            new_plate_number, font_path, IMAGE_PATH
        )

        # Process video with seamless replacement
        output_path = processor.process_video_seamless()
        
        print(f"\n=== Seamless Processing Complete ===")
        print(f"Output: {output_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
