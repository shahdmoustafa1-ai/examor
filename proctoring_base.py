import cv2
import time
import numpy as np
import torch
from numpy.linalg import norm
import os
import json
import datetime
import threading
import queue

# Modules
from face_detection_module import FaceDetector
from head_pose_estimation_module import HeadPoseEstimator
from eye_gaze_tracking_module import EyeGazeTracker

# InsightFace
try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("InsightFace not found.")

class VideoRecorder:
    """High-performance background video recorder."""
    def __init__(self, filename, fps=20.0, size=(640, 480)):
        self.filename = filename
        self.fps = fps
        self.size = size
        self.frame_queue = queue.Queue()
        self.stopped = False
        self.writer = cv2.VideoWriter(
            filename, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            size
        )
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()

    def _record(self):
        while not self.stopped or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is not None:
                    # Resize if necessary to match initialization size
                    if (frame.shape[1], frame.shape[0]) != self.size:
                        frame = cv2.resize(frame, self.size)
                    self.writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def add_frame(self, frame):
        if not self.stopped:
            self.frame_queue.put(frame.copy())

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.writer.release()
        print(f"Video saved: {self.filename}")

class ProctoringSystem:
    def __init__(self, registered_image_path, similarity_threshold=0.65, max_trials=3):
        self.registered_image_path = registered_image_path
        self.similarity_threshold = similarity_threshold
        self.max_trials = max_trials
        self.trials_exhausted = 0
        self.blocked = False
        
        # Warning Management
        self.major_warnings = 0
        self.max_major_warnings = 3
        self.last_alert_popup_time = 0
        self.alert_cooldown = 4.0
        self.last_id_score = 1.0 
        
        # Recording State
        self.active_recorder = None
        self.is_recording = False
        self.event_start_time = None
        
        # Cloud Storage Simulation
        self.cloud_dir = "cloud_storage"
        if not os.path.exists(self.cloud_dir):
            os.makedirs(self.cloud_dir)
        
        self.active_notification = None # {title, msg, color, expiry}
        
        # 1. Load YOLOv5s
        print("Loading YOLOv5s model for object detection...")
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.conf = 0.25 # Lower confidence for reliability
        self.yolo_model.classes = [0, 67, 63, 73, 64, 65] # person, cell phone, laptop, book, mouse, keyboard
        
        # 2. Load InsightFace
        print("Loading ResNet-50 + ArcFace model...")
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # 3. Register Student
        self.registered_embedding, _ = self._get_face_embedding(self.registered_image_path, is_path=True)
        if self.registered_embedding is None:
            raise ValueError(f"Could not detect student in {self.registered_image_path}")
        print(f"Student registered successfully.")
        
        # 4. Pipeline Components
        self.detector = FaceDetector()
        self.pose_estimator = HeadPoseEstimator(fsa_net_path='fsanet.onnx')
        self.gaze_tracker = EyeGazeTracker()
        
    def _get_face_embedding(self, image_source, is_path=False):
        img = cv2.imread(image_source) if is_path else image_source
        if img is None: return None, 0.0
        faces = self.app.get(img)
        if not faces: return None, 0.0
        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return largest_face.normed_embedding, largest_face.det_score

    def cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

    def check_objects(self, frame, face_detected=False):
        results = self.yolo_model(frame)
        df = results.pandas().xyxy[0]
        
        # Standard COCO names for YOLOv5
        monitored_map = {
            'cell phone': 'PHONE',
            'laptop': 'LAPTOP',
            'book': 'BOOK',
            'mouse': 'OFF-SCREEN DEVICE',
            'keyboard': 'OFF-SCREEN DEVICE'
        }
        
        detected_names = df['name'].tolist()
        detected_conf = df['confidence'].tolist()
        
        alerts = []
        violations = set()
        object_scores = {} # label -> max_conf
        
        # Count persons
        person_count = detected_names.count('person')
        if person_count > 1:
            alerts.append("MULTIPLE PERSONS!")
            violations.add("multiple_persons")
        
        for name, conf in zip(detected_names, detected_conf):
            if name in monitored_map:
                label = monitored_map[name]
                alerts.append(f"FORBIDDEN: {label} ({conf:.2f})")
                violations.add(name)
                # Track max confidence per object type
                object_scores[name] = max(object_scores.get(name, 0), conf)
            
        return alerts, violations, np.squeeze(results.render()), object_scores

    def detect_obstruction(self, frame):
        """Detects if camera is obstructed via extreme darkness or low variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        variance = np.var(gray)
        
        # Obstruction if very dark (mean < 15) or very flat/blurry (variance < 100)
        is_obstructed = mean_brightness < 15 or variance < 100
        return is_obstructed, mean_brightness, variance

    def verify_student(self):
        """Pre-exam identity verification with HUD (Frame #, Sim, Best)."""
        if self.blocked: return False
        
        while self.trials_exhausted < self.max_trials:
            cap = cv2.VideoCapture(0)
            captured = 0
            best_similarity = -1.0
            frame_idx = 0
            
            while captured < 5:
                ret, frame = cap.read()
                if not ret: continue
                
                frame_idx += 1
                current_sim = 0.0
                emb, conf = self._get_face_embedding(frame)
                if emb is not None:
                    current_sim = self.cosine_similarity(self.registered_embedding, emb)
                    if current_sim > best_similarity: 
                        best_similarity = current_sim
                    captured += 1
                    
                # Verification HUD
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                cv2.putText(frame, f"FRAME: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"SIMILARITY: {current_sim:.4f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"MAX SCORE: {best_similarity:.4f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Progress: {captured}/5", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Progress Bars
                cv2.line(frame, (20, 85), (20 + int(max(0, current_sim) * 250), 85), (0, 255, 255), 5)
                cv2.line(frame, (20, 120), (20 + int(max(0, best_similarity) * 250), 120), (0, 255, 0), 5)
                
                cv2.imshow('Identity Verification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
            cap.release()
            cv2.destroyAllWindows()
            
            self.trials_exhausted += 1
            if best_similarity >= self.similarity_threshold:
                self.last_id_score = best_similarity
                return True
                
            # If we reach here, this trial failed.
            if self.trials_exhausted < self.max_trials:
                self.show_popup_message(
                    "TRIAL FAILED", 
                    f"Identity mismatch.\nRetrying in 3s...\nTrial {self.trials_exhausted}/{self.max_trials}", 
                    color=(0, 165, 255), 
                    duration_ms=3000
                )

        return False

    def run_pose_calibration(self, pose_estimator):
        """Interactive 5-stage calibration: Forward, Up, Down, Left, Right"""
        stages = [
            ("FORWARD", (0, 255, 0)),
            ("UP", (255, 255, 0)),
            ("DOWN", (0, 255, 255)),
            ("LEFT", (255, 0, 255)),
            ("RIGHT", (0, 0, 255))
        ]
        
        cap = cv2.VideoCapture(0)
        calibration_data = {}
        
        print("Starting 5-Stage Calibration Sequence...")
        for stage_name, color in stages:
            start_time = time.time()
            collected_poses = []
            
            while time.time() - start_time < 3.0:
                ret, frame = cap.read()
                if not ret: continue
                
                # Instruction HUD
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (620, 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                rem = max(0, 3.0 - (time.time() - start_time))
                cv2.putText(frame, f"CALIBRATION STAGE: {stage_name}", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                cv2.putText(frame, f"Hold position... {rem:.1f}s", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                det = self.detector.detect_with_landmarks(frame)
                if det and det['face_roi'].size > 0:
                    h, w = frame.shape[:2]
                    pose = pose_estimator.estimate_pose(det['face_roi'], det['landmarks'], global_wh=(w, h))
                    if isinstance(pose, (tuple, list)) and len(pose) >= 3:
                        collected_poses.append(pose)
                    
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    cv2.putText(frame, "NO FACE DETECTED", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Calibration', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            if collected_poses:
                calibration_data[stage_name] = np.mean(collected_poses, axis=0)
                print(f"Captured {stage_name}: {calibration_data[stage_name]}")
            else:
                calibration_data[stage_name] = [0.0, 0.0, 0.0]
                
        cap.release()
        cv2.destroyWindow('Calibration')
        
        # We explicitly lock in these boundaries on the estimator module
        pose_estimator.set_calibration_data(calibration_data)
        return True

    def on_suspicious_behavior(self, frame, behavior_res):
        """Handle per-frame parallel proctoring alerts and dynamic recording."""
        if self.blocked: return
        
        is_suspicious = behavior_res.get('suspicious', False)
        
        # 1. DYNAMIC RECORDING LOGIC
        if is_suspicious and not self.is_recording:
            # START RECORDING
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.cloud_dir, f"event_MAJOR_{timestamp}.mp4")
            h, w = frame.shape[:2]
            self.active_recorder = VideoRecorder(filename, fps=20.0, size=(w, h))
            self.is_recording = True
            self.event_start_time = time.time()
            print(f"Started Recording: {filename}")
            
            # Initial Snapshot and Metadata
            cv2.imwrite(os.path.join(self.cloud_dir, f"event_MAJOR_{timestamp}.jpg"), frame)
            self._log_metadata(timestamp, behavior_res)

        if self.is_recording:
            self.active_recorder.add_frame(frame)
            
        if not is_suspicious and self.is_recording:
            # STOP RECORDING
            self.active_recorder.stop()
            self.is_recording = False
            self.active_recorder = None

        # 2. POPUP WARNINGS (Rate Limited)
        if is_suspicious:
            curr_time = time.time()
            if curr_time - self.last_alert_popup_time > self.alert_cooldown:
                self.major_warnings += 1
                msg = f"{behavior_res.get('reason', 'Suspicion')}\nWarning: {self.major_warnings}/{self.max_major_warnings}"
                if self.major_warnings >= self.max_major_warnings:
                    self.blocked = True
                    self.show_popup_message("TERMINATED", "3 WARNINGS REACHED\nEXAM KILLED", color=(0, 0, 255), duration_ms=5000)
                else:
                    self.show_popup_message("MAJOR WARNING", msg, color=(0, 0, 255))
                self.last_alert_popup_time = curr_time

    def _log_metadata(self, timestamp, behavior_res):
        event_id = f"event_MAJOR_{timestamp}"
        meta = {
            "event_id": event_id, "timestamp": timestamp, "type": "MAJOR",
            "id_score": self.last_id_score, "reason": behavior_res.get('reason', ''),
            "warning_count": self.major_warnings + 1
        }
        with open(os.path.join(self.cloud_dir, f"{event_id}.json"), 'w') as f:
            json.dump(meta, f, indent=4)

    def _draw_modern_overlay(self, frame, title, message, color=(180, 105, 255)):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Center coordinates
        cx, cy = w // 2, h // 2
        rw, rh = 240, 90 
        
        # 1. Main Draw (Rounded Rectangle Simulation)
        # Background
        cv2.rectangle(overlay, (cx - rw, cy - rh), (cx + rw, cy + rh), (35, 35, 35), -1)
        
        # Rounded Corners (Corners ellipses)
        rad = 20
        cv2.ellipse(overlay, (cx - rw + rad, cy - rh + rad), (rad, rad), 180, 0, 90, (35, 35, 35), -1)
        cv2.ellipse(overlay, (cx + rw - rad, cy - rh + rad), (rad, rad), 270, 0, 90, (35, 35, 35), -1)
        cv2.ellipse(overlay, (cx - rw + rad, cy + rh - rad), (rad, rad), 90, 0, 90, (35, 35, 35), -1)
        cv2.ellipse(overlay, (cx + rw - rad, cy + rh - rad), (rad, rad), 0, 0, 90, (35, 35, 35), -1)

        # 2. Merge with Alpha
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        
        # 3. Text Rendering
        # Title (Status)
        (tw, th), _ = cv2.getTextSize(title.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, title.upper(), (cx - tw // 2, cy - rh + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Divider Line
        cv2.line(frame, (cx - 180, cy - rh + 50), (cx + 180, cy - rh + 50), (80, 80, 80), 1)

        # Message Body
        lines = message.split('\n')
        y_cursor = cy - rh + 85
        for line in lines:
            (mw, mh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, line, (cx - mw // 2, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
            y_cursor += 40

    def show_popup_message(self, title, message, duration_ms=2500, color=(180, 105, 255)):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release() # Release immediately as we just need one frame for background
        if not ret: frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._draw_modern_overlay(frame, title, message, color=color)
        cv2.imshow("Notification", frame)
        cv2.waitKey(duration_ms)
        try:
            cv2.destroyWindow("Notification")
        except cv2.error:
            pass # Window already closed or not found