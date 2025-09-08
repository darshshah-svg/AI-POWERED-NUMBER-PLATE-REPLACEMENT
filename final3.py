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
    """ Order coordinates in a quadrilateral (top-left, top-right, bottom-right, bottom-left). """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """ Apply perspective transform to warp a quadrilateral into a rectangle. """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M, (maxWidth, maxHeight)

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
                return font_size
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
        return max(min_size, font_size)
    except Exception:
        return min_size

def get_dynamic_font_size_opencv(text, max_width, max_height, max_text_width_ratio=0.8, max_text_height_ratio=0.6):
    """ Calculate dynamic font scale for OpenCV to fit text within plate dimensions. """
    base_scale = 0.5
    min_scale = 0.1  # Minimum font scale for legibility
    font_scale = base_scale
    thickness = max(1, int(font_scale * 2))
    while font_scale > min_scale:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size
        if text_width <= max_width * max_text_width_ratio and text_height <= max_height * max_text_height_ratio:
            return font_scale, thickness
        font_scale -= 0.05
        thickness = max(1, int(font_scale * 2))
    return max(min_scale, font_scale), max(1, int(font_scale * 2))

class NumberPlateFrameExtractor:
    """ A class to extract frames containing number plates from video at 3 frames per second,
        select the best quality frame based on sharpness, create a blank white number plate,
        impose a new number with matching font and seamless blending. """
   
    def __init__(self, model_path, video_path, output_dir="output_frames", new_plate_number="ABC123", font_path=None):
        """ Initialize the frame extractor. """
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = output_dir
        self.new_plate_number = new_plate_number
        self.font_path = font_path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.video_name = Path(self.video_path).stem
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.detected_frames = []
        self.frame_metrics = []
        self.fps = None
        self.frame_count = None
        self.video_width = None
        self.video_height = None
        self.use_custom_font = False
        if self.font_path and os.path.exists(self.font_path):
            try:
                ImageFont.truetype(self.font_path, size=10)  # Test font loading
                self.use_custom_font = True
                print(f"Using custom font: {self.font_path}")
            except Exception as e:
                print(f"Error loading font {self.font_path}: {str(e)}. Falling back to default font.")
                self.use_custom_font = False
        else:
            print(f"Font file {self.font_path} not found. Falling back to default font.")
            self.use_custom_font = False

    def calculate_sharpness(self, image):
        """ Calculate sharpness of an image using Laplacian variance. """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def calculate_additional_metrics(self, image, detection_box):
        """ Calculate additional quality metrics for the detected number plate region. """
        x1, y1, x2, y2 = map(int, detection_box)
        plate_region = image[y1:y2, x1:x2]
        if plate_region.size == 0:
            return {"error": "Invalid detection box"}
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        metrics = {
            "plate_area": (x2 - x1) * (y2 - y1),
            "mean_brightness": np.mean(gray_plate),
            "brightness_std": np.std(gray_plate),
            "contrast": np.std(gray_plate),
            "plate_sharpness": cv2.Laplacian(gray_plate, cv2.CV_64F).var()
        }
        return metrics

    def detect_number_plates(self, frame):
        """ Detect number plates in a frame using YOLO model. """
        results = self.model(frame, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            return True, results[0]
        return False, None

    def process_video(self, confidence_threshold=0.5, frame_interval=0.333):
        """ Process the video to extract frames with number plate detections at 3 FPS. """
        print(f"Processing video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video properties: {self.video_width}x{self.video_height}, {self.fps:.2f} FPS, {self.frame_count} frames")
        frame_skip = int(self.fps * frame_interval)
        frame_idx = 0
        detection_count = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            progress = (frame_idx / self.frame_count) * 100
            print(f"Progress: {progress:.1f}% (Frame {frame_idx}/{self.frame_count})")
            has_detection, detection_results = self.detect_number_plates(frame)
            if has_detection:
                boxes = detection_results.boxes
                confidences = boxes.conf.cpu().numpy()
                valid_detections = confidences >= confidence_threshold
                if np.any(valid_detections):
                    detection_count += 1
                    best_detection_idx = np.argmax(confidences)
                    best_box = boxes.xyxy[best_detection_idx].cpu().numpy()
                    best_confidence = confidences[best_detection_idx]
                    frame_sharpness = self.calculate_sharpness(frame)
                    additional_metrics = self.calculate_additional_metrics(frame, best_box)
                    frame_data = {
                        "frame_idx": frame_idx,
                        "timestamp": frame_idx / self.fps,
                        "sharpness": frame_sharpness,
                        "confidence": float(best_confidence),
                        "detection_box": best_box.tolist(),
                        "additional_metrics": additional_metrics,
                        "frame": frame.copy()
                    }
                    self.detected_frames.append(frame_data)
                    self.frame_metrics.append(frame_sharpness)
                    print(f"Detection {detection_count}: Frame {frame_idx}, "
                          f"Sharpness: {frame_sharpness:.2f}, Confidence: {best_confidence:.3f}")
            frame_idx += frame_skip
            if frame_idx >= self.frame_count:
                break
        cap.release()
        total_detections = len(self.detected_frames)
        if total_detections == 0:
            print("No number plates detected in the video!")
            return {
                "total_frames_processed": frame_idx,
                "total_detections": 0,
                "best_frame": None
            }
        print(f"\nProcessing complete! Total detections found: {total_detections}")
        return {
            "total_frames_processed": frame_idx,
            "total_detections": total_detections,
            "detection_rate": total_detections / (frame_idx / frame_skip) if frame_idx > 0 else 0,
            "video_properties": {
                "fps": self.fps,
                "width": self.video_width,
                "height": self.video_height,
                "duration": self.frame_count / self.fps
            }
        }

    def select_best_frame(self, method="sharpness", weight_factors=None):
        """ Select the best quality frame from detected frames. """
        if not self.detected_frames:
            return None
        if method == "sharpness":
            best_idx = np.argmax([frame["sharpness"] for frame in self.detected_frames])
        elif method == "confidence":
            best_idx = np.argmax([frame["confidence"] for frame in self.detected_frames])
        elif method == "combined":
            if weight_factors is None:
                weight_factors = {
                    "sharpness": 0.4,
                    "confidence": 0.3,
                    "plate_area": 0.2,
                    "contrast": 0.1
                }
            scores = []
            for frame in self.detected_frames:
                sharpness_norm = frame["sharpness"] / max([f["sharpness"] for f in self.detected_frames])
                confidence_norm = frame["confidence"]
                plate_area = frame["additional_metrics"].get("plate_area", 0)
                plate_area_norm = plate_area / max([f["additional_metrics"].get("plate_area", 1) for f in self.detected_frames])
                contrast = frame["additional_metrics"].get("contrast", 0)
                contrast_norm = contrast / max([f["additional_metrics"].get("contrast", 1) for f in self.detected_frames])
                score = (weight_factors["sharpness"] * sharpness_norm +
                         weight_factors["confidence"] * confidence_norm +
                         weight_factors["plate_area"] * plate_area_norm +
                         weight_factors["contrast"] * contrast_norm)
                scores.append(score)
            best_idx = np.argmax(scores)
        best_frame = self.detected_frames[best_idx]
        sharpness_rank = sorted(enumerate([f["sharpness"] for f in self.detected_frames]),
                              key=lambda x: x[1], reverse=True).index((best_idx, best_frame["sharpness"])) + 1
        confidence_rank = sorted(enumerate([f["confidence"] for f in self.detected_frames]),
                               key=lambda x: x[1], reverse=True).index((best_idx, best_frame["confidence"])) + 1
        return {
            "selected_frame": best_frame,
            "selection_method": method,
            "frame_index": best_frame["frame_idx"],
            "timestamp": best_frame["timestamp"],
            "sharpness_score": best_frame["sharpness"],
            "confidence_score": best_frame["confidence"],
            "sharpness_rank": f"{sharpness_rank}/{len(self.detected_frames)}",
            "confidence_rank": f"{confidence_rank}/{len(self.detected_frames)}",
            "detection_box": best_frame["detection_box"],
            "additional_metrics": best_frame["additional_metrics"],
            "frame": best_frame["frame"]
        }

    def generate_report(self, best_frame_info, save_report=True):
        """ Generate a comprehensive ROI-focused report of the processing results. """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
        NUMBER PLATE DETECTION AND FRAME SELECTION REPORT
        ================================================
        Generated: {timestamp}
       
        INPUT SPECIFICATIONS:
        - Video Path: {self.video_path}
        - Model Path: {self.model_path}
        - Output Directory: {self.output_dir}
        - Font Path: {self.font_path if self.font_path else "Default (Hershey Simplex)"}
       
        VIDEO PROPERTIES:
        - Resolution: {self.video_width}x{self.video_height}
        - Frame Rate: {self.fps:.2f} FPS
        - Total Frames: {self.frame_count}
        - Duration: {self.frame_count/self.fps:.2f} seconds
       
        DETECTION RESULTS:
        - Total Detections: {len(self.detected_frames)}
        - Detection Rate: {len(self.detected_frames)/(self.frame_count/(self.fps*0.333))*100:.2f}%
        - Frames with Detections: {len(self.detected_frames)}/{self.frame_count}
       
        QUALITY METRICS SUMMARY:
        - Sharpness Range: {min(self.frame_metrics):.2f} - {max(self.frame_metrics):.2f}
        - Average Sharpness: {np.mean(self.frame_metrics):.2f}
        - Sharpness Std Dev: {np.std(self.frame_metrics):.2f}
       
        SELECTED BEST FRAME:
        - Frame Index: {best_frame_info['frame_index']}
        - Timestamp: {best_frame_info['timestamp']:.2f} seconds
        - Selection Method: {best_frame_info['selection_method']}
        - Sharpness Score: {best_frame_info['sharpness_score']:.2f}
        - Detection Confidence: {best_frame_info['confidence_score']:.3f}
        - Sharpness Rank: {best_frame_info['sharpness_rank']}
        - Confidence Rank: {best_frame_info['confidence_rank']}
       
        DETECTION BOX COORDINATES:
        - X1: {best_frame_info['detection_box'][0]:.1f}
        - Y1: {best_frame_info['detection_box'][1]:.1f}
        - X2: {best_frame_info['detection_box'][2]:.1f}
        - Y2: {best_frame_info['detection_box'][3]:.1f}
        - Width: {best_frame_info['detection_box'][2] - best_frame_info['detection_box'][0]:.1f}
        - Height: {best_frame_info['detection_box'][3] - best_frame_info['detection_box'][1]:.1f}
       
        ADDITIONAL METRICS:
        """
        if best_frame_info['additional_metrics']:
            for key, value in best_frame_info['additional_metrics'].items():
                if key != "error":
                    report += f"- {key.replace('_', ' ').title()}: {value:.2f}\n"
       
        report += f"\nRECOMMENDATIONS:\n"
        report += f"- Selected frame shows good quality for text recognition.\n"
        report += f"- Number plate region has been replaced with a blank white area and new text '{self.new_plate_number}' added.\n"
        if best_frame_info['sharpness_score'] > np.mean(self.frame_metrics) + np.std(self.frame_metrics):
            report += f"- Sharpness is above average + 1 standard deviation (excellent).\n"
        elif best_frame_info['sharpness_score'] > np.mean(self.frame_metrics):
            report += f"- Sharpness is above average (good).\n"
        else:
            report += f"- Consider checking if better quality frames are available.\n"
       
        if save_report:
            report_path = os.path.join(self.output_dir, "processing_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {report_path}")
        return report

def main():
    """ Main function to run the number plate detection, frame selection, and text imposition process with an image on the left. """
    MODEL_PATH = "/home/divyesh/Documents/Malhar/AI powered number plate replacement/best.pt"
    VIDEO_PATH = "/home/divyesh/Documents/Malhar/AI powered number plate replacement/IndianNumberPlate.mp4"
    OUTPUT_DIR = "/home/divyesh/Documents/Malhar/AI powered number plate replacement/output/whitebg"
    CSV_PATH = "/home/divyesh/Documents/Malhar/AI powered number plate replacement/Dynamic Values.csv"
    FONT_PATH = "/home/divyesh/Documents/Malhar/AI powered number plate replacement/dealerplate california.otf"
    FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.otf"
    IMAGE_PATH = "/home/divyesh/Documents/Malhar/AI powered number plate replacement/IND number plate.png"  # Path to the image to place on the left

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

        print("Initializing Number Plate Frame Extractor...")
        font_path = FONT_PATH
        if not os.path.exists(FONT_PATH):
            print(f"Custom font {FONT_PATH} not found. Trying fallback font {FALLBACK_FONT_PATH}.")
            font_path = FALLBACK_FONT_PATH
        extractor = NumberPlateFrameExtractor(MODEL_PATH, VIDEO_PATH, OUTPUT_DIR, new_plate_number, font_path)
        print("Starting video processing...")
        processing_results = extractor.process_video(confidence_threshold=0.5, frame_interval=0.333)
        if processing_results["total_detections"] == 0:
            print("No number plates detected. Please check your video and model.")
            return

        print("\nSelecting best frame using sharpness method...")
        best_frame_sharpness = extractor.select_best_frame(method="sharpness")

        print("\nCreating blank white number plate with image on left and new text...")
        frame = best_frame_sharpness["frame"].copy()
        x1, y1, x2, y2 = map(int, best_frame_sharpness["detection_box"])
        plate_region = frame[y1:y2, x1:x2].copy()
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Load the image to place on the left
        try:
            logo_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
            if logo_image is None:
                raise ValueError(f"Could not load image from {IMAGE_PATH}")
            # Convert to RGB if it has an alpha channel for PIL compatibility
            if logo_image.shape[2] == 4:
                logo_image_rgb = cv2.cvtColor(logo_image, cv2.COLOR_BGRA2RGBA)
            else:
                logo_image_rgb = cv2.cvtColor(logo_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {IMAGE_PATH}: {str(e)}. Proceeding without image.")
            logo_image = None
            logo_image_rgb = None

        # Ensure the plate region is filled with white and bordered with black
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        plate_width = x2 - x1
        plate_height = y2 - y1

        # Calculate image placement dimensions
        image_width_ratio = 0.3  # Image takes up 30% of plate width
        image_max_width = int(plate_width * image_width_ratio)
        image_max_height = plate_height - 10  # Leave some padding
        if logo_image is not None:
            logo_height, logo_width = logo_image.shape[:2]
            scale = min(image_max_width / logo_width, image_max_height / logo_height)
            new_logo_width = int(logo_width * scale)
            new_logo_height = int(logo_height * scale)
            logo_resized = cv2.resize(logo_image, (new_logo_width, new_logo_height), interpolation=cv2.INTER_AREA)
            logo_y = y1 + (plate_height - new_logo_height) // 2
            logo_x = x1 + 5
        else:
            new_logo_width = 0
            logo_y, logo_x = 0, 0

        # Adjust text placement to be right of the image
        text_max_width = plate_width - new_logo_width - 15  # Padding between image and text
        if extractor.use_custom_font:
            pil_image = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font_size = get_dynamic_font_size_pil(extractor.font_path, extractor.new_plate_number, text_max_width, plate_height)
            try:
                font = ImageFont.truetype(extractor.font_path, font_size)
            except Exception as e:
                print(f"Error loading font: {str(e)}. Falling back to default font.")
                font = None
            if font:
                text_bbox = font.getbbox(extractor.new_plate_number)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                start_x = new_logo_width + 10
                start_y = (plate_height - text_height) // 2
                print(f"Text metrics: width={text_width}, height={text_height}, "
                      f"plate width={plate_width}, height={plate_height}, font_size={font_size}")
                draw.text((start_x, start_y), extractor.new_plate_number, fill=(0, 0, 0), font=font)
                if logo_image is not None:
                    pil_logo = Image.fromarray(logo_image_rgb)
                    pil_logo = pil_logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)
                    mask = pil_logo.split()[3] if logo_image.shape[2] == 4 else None
                    try:
                        pil_image.paste(pil_logo, (5, (plate_height - new_logo_height) // 2), mask)
                    except Exception as e:
                        print(f"Error pasting logo: {str(e)}. Pasting without mask.")
                        pil_image.paste(pil_logo, (5, (plate_height - new_logo_height) // 2))
                frame[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                font_scale, thickness = get_dynamic_font_size_opencv(extractor.new_plate_number, text_max_width, plate_height)
                text_size, _ = cv2.getTextSize(extractor.new_plate_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                start_x = x1 + new_logo_width + 10
                start_y = y1 + (plate_height + text_size[1]) // 2
                print(f"Text metrics: width={text_size[0]}, height={text_size[1]}, "
                      f"plate width={plate_width}, height={plate_height}, font_scale={font_scale}")
                cv2.putText(frame, extractor.new_plate_number, (start_x, start_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                if logo_image is not None:
                    if logo_image.shape[2] == 4:
                        alpha = logo_resized[:, :, 3] / 255.0
                        for c in range(3):
                            frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width, c] = \
                                (1 - alpha) * frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width, c] + \
                                alpha * logo_resized[:, :, c]
                    else:
                        frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width] = logo_resized
        else:
            font_scale, thickness = get_dynamic_font_size_opencv(extractor.new_plate_number, text_max_width, plate_height)
            text_size, _ = cv2.getTextSize(extractor.new_plate_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            start_x = x1 + new_logo_width + 10
            start_y = y1 + (plate_height + text_size[1]) // 2
            print(f"Text metrics: width={text_size[0]}, height={text_size[1]}, "
                  f"plate width={plate_width}, height={plate_height}, font_scale={font_scale}")
            cv2.putText(frame, extractor.new_plate_number, (start_x, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            if logo_image is not None:
                if logo_image.shape[2] == 4:
                    alpha = logo_resized[:, :, 3] / 255.0
                    for c in range(3):
                        frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width, c] = \
                            (1 - alpha) * frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width, c] + \
                            alpha * logo_resized[:, :, c]
                else:
                    frame[logo_y:logo_y+new_logo_height, logo_x:logo_x+new_logo_width] = logo_resized

        best_frame_path = os.path.join(OUTPUT_DIR, f"frame_{extractor.video_name}.jpg")
        cv2.imwrite(best_frame_path, frame)
        print(f"Modified best frame with new plate '{extractor.new_plate_number}' and image saved to: {best_frame_path}")

        best_frame_data = {
            "video_properties": {
                "resolution": f"{extractor.video_width}x{extractor.video_height}",
                "frame_rate": f"{extractor.fps:.2f} FPS",
                "total_frames": extractor.frame_count,
                "duration": f"{extractor.frame_count/extractor.fps:.2f} seconds"
            },
            "detection_results": {
                "total_detections": len(extractor.detected_frames),
                "detection_rate": f"{len(extractor.detected_frames)/(extractor.frame_count/(extractor.fps*0.333))*100:.2f}%",
                "frames_with_detections": f"{len(extractor.detected_frames)}/{extractor.frame_count}"
            },
            "quality_metrics_summary": {
                "sharpness_range": f"{min(extractor.frame_metrics):.2f} - {max(extractor.frame_metrics):.2f}",
                "average_sharpness": f"{np.mean(extractor.frame_metrics):.2f}",
                "sharpness_std_dev": f"{np.std(extractor.frame_metrics):.2f}"
            },
            "selected_best_frame": {
                "frame_index": best_frame_sharpness["frame_index"],
                "timestamp": f"{best_frame_sharpness['timestamp']:.2f} seconds",
                "selection_method": best_frame_sharpness["selection_method"],
                "sharpness_score": f"{best_frame_sharpness['sharpness_score']:.2f}",
                "detection_confidence": f"{best_frame_sharpness['confidence_score']:.3f}",
                "sharpness_rank": best_frame_sharpness["sharpness_rank"],
                "confidence_rank": best_frame_sharpness["confidence_rank"]
            },
            "detection_box_coordinates": {
                "x1": f"{best_frame_sharpness['detection_box'][0]:.1f}",
                "y1": f"{best_frame_sharpness['detection_box'][1]:.1f}",
                "x2": f"{best_frame_sharpness['detection_box'][2]:.1f}",
                "y2": f"{best_frame_sharpness['detection_box'][3]:.1f}",
                "width": f"{best_frame_sharpness['detection_box'][2] - best_frame_sharpness['detection_box'][0]:.1f}",
                "height": f"{best_frame_sharpness['detection_box'][3] - best_frame_sharpness['detection_box'][1]:.1f}"
            },
            "additional_metrics": {
                key: f"{value:.2f}" for key, value in best_frame_sharpness["additional_metrics"].items() if key != "error"
            },
            "new_plate_number": extractor.new_plate_number,
            "image_added": IMAGE_PATH if logo_image is not None else "None"
        }
        best_frame_json_path = os.path.join(OUTPUT_DIR, f"best_frame_{extractor.video_name}.json")
        with open(best_frame_json_path, 'w') as f:
            json.dump(best_frame_data, f, indent=2)
        print(f"Best frame JSON saved to: {best_frame_json_path}")

        x1, y1, x2, y2 = best_frame_sharpness["detection_box"]
        bbox_data = [
            {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": best_frame_sharpness["confidence_score"],
                "class_name": "License_Plate"
            }
        ]
        bbox_json_path = os.path.join(OUTPUT_DIR, f"bbox_{extractor.video_name}.json")
        with open(bbox_json_path, 'w') as f:
            json.dump(bbox_data, f, indent=2)
        print(f"Bounding box JSON saved to: {bbox_json_path}")

        print(f"\nGenerating comprehensive report...")
        report = extractor.generate_report(best_frame_sharpness, save_report=True)
        print(report)

        print(f"\nProcessing complete! All results saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()