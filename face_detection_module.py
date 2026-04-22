import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='yolov5s.pt', conf=0.3):
        # 1. Initialize YOLOv5s Face Detector using ultralytics (faster/modern)
        print(f"Initializing YOLO Face Detector from {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.conf = conf
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

        # 2. Initialize MediaPipe Face Landmarker (Iris, Mesh, Eyelids)
        face_landmarker_path = 'face_landmarker.task'
        print("Initializing MediaPipe Face Landmarker for Landmarks...")
        if os.path.exists(face_landmarker_path):
            base_options = python.BaseOptions(model_asset_path=face_landmarker_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.landmarks_ready = True
        else:
            print(f"WARNING: Face Landmarker model not found at {face_landmarker_path}.")
            self.landmarks_ready = False

    def get_face_box(self, frame):
        """
        Detect faces in a frame and return the bounding box of the largest face.
        Returns: [xmin, ymin, xmax, ymax] or None if no face is detected.
        """
        if self.model is None:
            return None
            
        # Run inference (classes=0 filters for 'person')
        results = self.model.predict(frame, classes=0, conf=self.conf, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return None
        
        # Find the most prominent box (largest and closest to center)
        best_box = None
        max_score = -1.0
        h_img, w_img = frame.shape[:2]
        boxes = results[0].boxes
        
        for box in boxes:
            coords = box.xyxy[0].tolist()
            bw = coords[2] - coords[0]
            bh = coords[3] - coords[1]
            area = bw * bh
            
            # Distance from center (penalty)
            cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            dist_center = np.sqrt((cx - w_img/2)**2 + (cy - h_img/2)**2)
            
            # Preference score: Area weighted by center proximity
            score = area / (1.0 + 0.001 * dist_center)
            
            if score > max_score:
                max_score = score
                best_box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
        
        return best_box

    def get_all_faces(self, frame):
        """Detect all faces in a frame."""
        if self.model is None: return []
        results = self.model.predict(frame, classes=0, conf=self.conf, verbose=False)
        if not results or len(results[0].boxes) == 0: return []
        return [box.xyxy[0].tolist() for box in results[0].boxes]

    def detect_with_landmarks(self, frame):
        """
        1. Detect Person ROI (YOLOv5s)
        2. Extract landmarks (MediaPipe)
        3. Refine Bbox to Face only
        Returns: {bbox, face_roi, landmarks}
        """
        # 1. Attempt Target ROI detection (YOLOv5s Person)
        person_bbox = self.get_face_box(frame)
        h, w = frame.shape[:2]
        
        if person_bbox:
            px1, py1, px2, py2 = person_bbox
            px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
        else:
            # FALLBACK: Use full frame if YOLO fails to find 'person'
            px1, py1, px2, py2 = 0, 0, w, h
            
        person_roi = frame[py1:py2, px1:px2]
        
        face_bbox = person_bbox # Default to person if landmarks fail
        face_roi = person_roi
        landmarks = None
        
        if self.landmarks_ready and person_roi.size > 0:
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
            detection_result = self.landmarker.detect(mp_image)
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                p_w = px2 - px1
                p_h = py2 - py1
                
                # Transform landmarks to be global-relative (0-1 of full frame)
                # Create copies to avoid modifying internal MediaPipe objects (prevents frozen pose)
                class Landmark:
                    def __init__(self, x, y, z):
                        self.x = x
                        self.y = y
                        self.z = z
                
                global_landmarks = []
                for lm in landmarks:
                    gx = (px1 + lm.x * p_w) / w
                    gy = (py1 + lm.y * p_h) / h
                    global_landmarks.append(Landmark(gx, gy, lm.z))
                
                landmarks = global_landmarks
                
                # REFINEMENT: Calculate tight face bbox from global landmarks
                min_lx = min([lm.x for lm in landmarks])
                max_lx = max([lm.x for lm in landmarks])
                min_ly = min([lm.y for lm in landmarks])
                max_ly = max([lm.y for lm in landmarks])
                
                # Tight Face Bbox (Global Pixels)
                fx1 = int(min_lx * w)
                fy1 = int(min_ly * h)
                fx2 = int(max_lx * w)
                fy2 = int(max_ly * h)
                
                # Add a small margin (20%)
                margin_w = int((fx2 - fx1) * 0.2)
                margin_h = int((fy2 - fy1) * 0.2)
                fx1, fy1 = max(0, fx1 - margin_w), max(0, fy1 - margin_h)
                fx2, fy2 = min(w, fx2 + margin_w), min(h, fy2 + margin_h)
                
                face_bbox = [fx1, fy1, fx2, fy2]
                face_roi = frame[fy1:fy2, fx1:fx2]
                
        return {
            'bbox': face_bbox,
            'face_roi': face_roi,
            'landmarks': landmarks
        }

if __name__ == "__main__":
    # Test code
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        bbox = detector.get_face_box(frame)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()