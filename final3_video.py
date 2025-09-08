import cv2
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import json
import pytesseract
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Set the path to Tesseract executable (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux example; update for your OS

def order_points(pts):
    """ Order coordinates in a quadrilateral with improved accuracy (top-left, top-right, bottom-right, bottom-left). """
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
    """ Apply perspective transform to warp a quadrilateral into a rectangle with improved accuracy. """
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
    """ Draw border lines along the quadrilateral edges in proper rectangular order. """
    # Ensure points are in correct order: top-left, top-right, bottom-right, bottom-left
    ordered_pts = order_points(pts)
    
    # Convert to integer coordinates
    pts_int = ordered_pts.astype(np.int32)
    
    # Draw lines connecting the points in rectangular order
    # Top edge: top-left to top-right
    cv2.line(frame, tuple(pts_int[0]), tuple(pts_int[1]), color, thickness)
    # Right edge: top-right to bottom-right  
    cv2.line(frame, tuple(pts_int[1]), tuple(pts_int[2]), color, thickness)
    # Bottom edge: bottom-right to bottom-left
    cv2.line(frame, tuple(pts_int[2]), tuple(pts_int[3]), color, thickness)
    # Left edge: bottom-left to top-left
    cv2.line(frame, tuple(pts_int[3]), tuple(pts_int[0]), color, thickness)
    
    return frame

def get_dynamic_font_size_pil(font_path, text, max_width, max_height, max_text_width_ratio=0.8, max_text_height_ratio=0.6):
    """ Calculate dynamic font size for PIL to fit text within plate dimensions. """
    base_size = int(min(max_height, max_width) * 0.5)
    min_size = 5  # Minimum font size for legibility
    font_size = base_size
    try:
        font = ImageFont.truetype(font_path, font_size)
        while font_size > min_size:
            text_bbox = font.getbbox(text)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            if text_width <= max_width * max_text_width_ratio and text_height <= max_height * max_text_height_ratio:
                break
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font_size = base_size
    return font_size

def get_dynamic_font_size_opencv(text, max_width, max_height, max_text_width_ratio=0.8, max_text_height_ratio=0.6):
    """ Calculate dynamic font scale for OpenCV to fit text within plate dimensions. """
    font_scale = 1.0
    thickness = 2
    while font_scale > 0.1:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size
        if text_width <= max_width * max_text_width_ratio and text_height <= max_height * max_text_height_ratio:
            break
        font_scale -= 0.1
    return font_scale, thickness

def process_frame_with_replacement(frame, model, new_plate_number, font_path, logo_image, logo_image_rgb, use_custom_font=True):
    """ Process a single frame and replace number plates if detected. """
    results = model(frame, conf=0.5)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()
                
                if confidence > 0.5:  # Only process high confidence detections
                    plate_region = frame[y1:y2, x1:x2]
                    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    
                    # Find contours for quadrilateral detection
                    contours, _ = cv2.findContours(gray_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        if len(approx) == 4:
                            # Quadrilateral case - use 4-point transform
                            pts = approx.reshape(4, 2).astype(np.float32)
                            warped, M, (maxWidth, maxHeight) = four_point_transform(plate_region, pts)
                            white_plate = np.ones((maxHeight, maxWidth, 3), dtype=np.uint8) * 255
                            
                            # Draw border on the actual edge
                            border_thickness = 2
                            cv2.line(white_plate, (0, 0), (maxWidth-1, 0), (0, 0, 0), border_thickness)
                            cv2.line(white_plate, (0, maxHeight-1), (maxWidth-1, maxHeight-1), (0, 0, 0), border_thickness)
                            cv2.line(white_plate, (0, 0), (0, maxHeight-1), (0, 0, 0), border_thickness)
                            cv2.line(white_plate, (maxWidth-1, 0), (maxWidth-1, maxHeight-1), (0, 0, 0), border_thickness)
                            
                            # Add logo and text
                            image_width_ratio = 0.3
                            image_max_width = int(maxWidth * image_width_ratio)
                            image_max_height = maxHeight - 10
                            
                            if logo_image is not None:
                                logo_height, logo_width = logo_image.shape[:2]
                                scale = min(image_max_width / logo_width, image_max_height / logo_height)
                                new_logo_width = int(logo_width * scale)
                                new_logo_height = int(logo_height * scale)
                                logo_resized = cv2.resize(logo_image, (new_logo_width, new_logo_height), interpolation=cv2.INTER_AREA)
                                logo_y = (maxHeight - new_logo_height) // 2
                                logo_x = 5
                            else:
                                new_logo_width = 0
                                logo_y, logo_x = 0, 0
                            
                            text_max_width = maxWidth - new_logo_width - 15
                            
                            if use_custom_font:
                                pil_image = Image.fromarray(cv2.cvtColor(white_plate, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_image)
                                font_size = get_dynamic_font_size_pil(font_path, new_plate_number, text_max_width, maxHeight)
                                try:
                                    font = ImageFont.truetype(font_path, font_size)
                                except Exception:
                                    font = None
                                
                                if font:
                                    text_bbox = font.getbbox(new_plate_number)
                                    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                                    start_x = logo_x + new_logo_width + 10
                                    start_y = (maxHeight - text_height) // 2
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
                            
                            # Transform back and replace
                            dst_pts = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype=np.float32)
                            src_pts = order_points(pts)
                            inverse_M = cv2.getPerspectiveTransform(dst_pts, src_pts)
                            
                            warped_back = cv2.warpPerspective(white_plate, inverse_M, (x2 - x1, y2 - y1))
                            mask = np.ones((maxHeight, maxWidth), dtype=np.uint8) * 255
                            warped_mask = cv2.warpPerspective(mask, inverse_M, (x2 - x1, y2 - y1))
                            
                            plate_roi = frame[y1:y2, x1:x2]
                            for c in range(3):
                                plate_roi[:, :, c] = np.where(warped_mask > 0, warped_back[:, :, c], plate_roi[:, :, c])
                            frame[y1:y2, x1:x2] = plate_roi
                            
                            # Draw border on quadrilateral edges
                            frame_pts = pts + np.array([x1, y1])
                            draw_quadrilateral_border(frame, frame_pts, thickness=3, color=(0, 0, 0))
                        
                        else:
                            # Rectangular fallback
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                            
                            # Add logo and text for rectangular case
                            plate_width = x2 - x1
                            plate_height = y2 - y1
                            
                            if logo_image is not None:
                                image_max_width = int(plate_width * 0.3)
                                image_max_height = plate_height - 10
                                logo_height, logo_width = logo_image.shape[:2]
                                scale = min(image_max_width / logo_width, image_max_height / logo_height)
                                new_logo_width = int(logo_width * scale)
                                new_logo_height = int(logo_height * scale)
                                logo_resized = cv2.resize(logo_image, (new_logo_width, new_logo_height), interpolation=cv2.INTER_AREA)
                                logo_y = y1 + (plate_height - new_logo_height) // 2
                                logo_x = x1 + 5
                                
                                if logo_image.shape[2] == 4:
                                    alpha = logo_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width, c] = \
                                            (1 - alpha) * frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width, c] + \
                                            alpha * logo_resized[:, :, c]
                                else:
                                    frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width] = logo_resized
                                
                                text_start_x = x1 + new_logo_width + 10
                            else:
                                text_start_x = x1 + 10
                            
                            # Add text
                            font_scale, thickness = get_dynamic_font_size_opencv(new_plate_number, plate_width - (text_start_x - x1) - 10, plate_height)
                            text_size, _ = cv2.getTextSize(new_plate_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                            text_y = y1 + (plate_height + text_size[1]) // 2
                            cv2.putText(frame, new_plate_number, (text_start_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return frame

def main():
    """ Main function to process entire video and replace number plates in all frames. """
    MODEL_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Models/best .pt"
    VIDEO_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Assets /videoplayback.mp4"
    OUTPUT_DIR = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/output"
    CSV_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Dynamic Values.csv"
    FONT_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/fonts/dealerplate california.otf"
    FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.otf"
    IMAGE_PATH = "/home/harsh/Downloads/Darsh/yolo_model_16epochs/yolo model testing/Assets /IND number plate.png"

    try:
        # Read the CSV file to get the new plate number
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

        print("Setting up font path...")
        font_path = FONT_PATH
        if not os.path.exists(FONT_PATH):
            print(f"Custom font {FONT_PATH} not found. Trying fallback font {FALLBACK_FONT_PATH}.")
            font_path = FALLBACK_FONT_PATH

        # Load YOLO model
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        
        # Load logo image
        try:
            logo_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
            if logo_image is None:
                raise ValueError(f"Could not load image from {IMAGE_PATH}")
            if logo_image.shape[2] == 4:
                logo_image_rgb = cv2.cvtColor(logo_image, cv2.COLOR_BGRA2RGBA)
            else:
                logo_image_rgb = cv2.cvtColor(logo_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {IMAGE_PATH}: {str(e)}. Proceeding without image.")
            logo_image = None
            logo_image_rgb = None
        
        # Create output directory
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video {VIDEO_PATH}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer
        video_name = Path(VIDEO_PATH).stem
        output_video_path = os.path.join(OUTPUT_DIR, f"{video_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print(f"Processing video and saving to: {output_video_path}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with number plate replacement
            processed_frame = process_frame_with_replacement(
                frame, model, new_plate_number, font_path, logo_image, logo_image_rgb, 
                use_custom_font=os.path.exists(font_path)
            )
            
            # Write processed frame to output video
            out.write(processed_frame)
            
            # Show progress
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nVideo processing complete!")
        print(f"Processed {frame_count} frames")
        print(f"Output saved to: {output_video_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
