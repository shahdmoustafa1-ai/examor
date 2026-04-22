import cv2
import time
import numpy as np
import os
import json
import datetime
import threading
import queue
from proctoring_base import ProctoringSystem, VideoRecorder
from fusion_model import predict_fusion_vector
from collections import defaultdict, deque

class InvigilationSystem(ProctoringSystem):
    # --- SYSTEM CONFIGURATION ---
    CONFIG = {
        "COOLDOWN_MINOR": 10,
        "COOLDOWN_MAJOR": 5,
        "MINOR_TO_MAJOR_THRESHOLD": 3,
        "MAJOR_TO_FLAG_THRESHOLD": 2,
        "MAJOR_TO_TERMINATE": 3,
        "GAZE_MINOR_DURATION": 3,
        "GAZE_MAJOR_DURATION": 7,
        "FACE_MISSING_MINOR": 3,
        "FACE_MISSING_MAJOR": 7,
        "OBJECT_CONFIDENCE_MINOR": 0.4,
        "OBJECT_CONFIDENCE_MAJOR": 0.7,
        "FRAME_VALIDATION_COUNT": 5,
        "SHORT_TIME_WINDOW": 5.0 # Seconds
    }

    def __init__(self, registered_image_path, similarity_threshold=0.65, max_trials=3):
        super().__init__(registered_image_path, similarity_threshold, max_trials)
        print("Invigilation System Starting (XGBoost Fusion)...")
        
        # 1. Heuristic Rules State
        self.minor_warning_count = 0
        self.major_warning_count = 0
        self.is_flagged = False
        
        # Rule Validation & Cooldowns
        self.consecutive_frames = defaultdict(int) # rule_id -> count
        self.rule_cooldowns = defaultdict(float)    # rule_id -> timestamp
        
        # Behavior History
        self.gaze_shift_history = deque() # list of (timestamp, angles)
        self.last_gaze_angles = (0, 0)
        
        # 2. Performance & Temporal State
        self.face_missing_start = None
        self.face_missing_duration = 0
        self.gaze_away_start = None
        self.gaze_away_duration = 0
        self.gaze_away_frames = 0
        self.total_exam_frames = 0
        
        # Performance
        self.fps_queue = queue.Queue(maxsize=30)
        self.last_fps_time = time.time()

    def get_monitoring_features(self, frame, results):
        """Standardizes the 15-feature vector for the XGBoost model."""
        # 1. Identity & Detection
        identity_score = self.last_id_score
        similarity_score = self.last_id_score
        face_detected = 1 if results.get('bbox') is not None else 0
        
        # Objects (from YOLO)
        alerts = results.get('alerts', [])
        detected_names = []
        for a in alerts:
             if "FORBIDDEN:" in a: detected_names.append(a.split(": ")[1].lower())
             if "MULTIPLE" in a: detected_names.append("person")
             
        person_count = 1 + (1 if "person" in detected_names else 0)
        phone_detected = 1 if "phone" in detected_names else 0
        book_detected = 1 if "book" in detected_names else 0
        laptop_detected = 1 if "laptop" in detected_names else 0
        
        # 3. Pose & Gaze
        yaw, pitch, roll = results.get('pose', (0,0,0))
        gaze_h, gaze_v = (0, 0)
        gaze_data = results.get('gaze_data')
        if gaze_data and 'angles' in gaze_data:
            gaze_h, gaze_v = gaze_data['angles']
            
        # 4. Temporal Features
        self.total_exam_frames += 1
        is_gazing_away = abs(gaze_h) > 0.6 or abs(gaze_v) > 0.6
        if is_gazing_away: self.gaze_away_frames += 1
        gaze_away_ratio = self.gaze_away_frames / max(1, self.total_exam_frames)
        
        face_missing_duration = 0
        if not face_detected:
            if self.face_missing_start is None:
                self.face_missing_start = time.time()
            face_missing_duration = time.time() - self.face_missing_start
        else:
            self.face_missing_start = None
            
        return {
            "identity_score": identity_score, "similarity_score": similarity_score,
            "face_detected": face_detected, "person_count": person_count,
            "phone_detected": phone_detected, "book_detected": book_detected,
            "laptop_detected": laptop_detected, "yaw": yaw, "pitch": pitch, "roll": roll,
            "gaze_horizontal": gaze_h, "gaze_vertical": gaze_v,
            "gaze_away_ratio": 0.0, "face_missing_duration": face_missing_duration,
            "repeated_violation_flag": 1 if self.major_warning_count >= self.CONFIG['MAJOR_TO_FLAG_THRESHOLD'] else 0,
            "object_scores": results.get('object_scores', {})
        }

    def _check_face_visibility(self, landmarks):
        """Returns True if face is partially obscured (landmarks missing or very close to edge)."""
        if landmarks is None: return False
        
        # 1. Count percentage of clearly detected landmarks
        # (MediaPipe traditionally returns all, but we can check if they are very close to edges)
        edge_threshold = 0.02
        obscured_count = 0
        for lm in landmarks:
            if lm.x < edge_threshold or lm.x > (1 - edge_threshold) or \
               lm.y < edge_threshold or lm.y > (1 - edge_threshold):
                obscured_count += 1
                
        percentage_obscured = obscured_count / len(landmarks)
        # Rule 2: more than 20% landmarks obscured is considered "partially visible"
        return percentage_obscured > 0.2

    def _track_gaze_shifts(self, current_gaze):
        """Detects if gaze shift frequency is high."""
        now = time.time()
        gh, gv = current_gaze
        prev_gh, prev_gv = self.last_gaze_angles
        
        # Calculate angular displacement
        dist = np.sqrt((gh - prev_gh)**2 + (gv - prev_gv)**2)
        
        # Definition: A "shift" is a > 0.1 normalized unit change (approx 10 degrees)
        if dist > 0.1:
            self.gaze_shift_history.append(now)
            
        self.last_gaze_angles = current_gaze
        
        # Clean old shifts
        while self.gaze_shift_history and now - self.gaze_shift_history[0] > self.CONFIG["SHORT_TIME_WINDOW"]:
            self.gaze_shift_history.popleft()
            
        return len(self.gaze_shift_history)

    def evaluate_heuristic_rules(self, frame, results, features):
        """Master rules engine for proctoring violations."""
        now = time.time()
        
        # 1. Gather Conditions
        presence = results.get('bbox') is not None
        landmarks = results.get('landmarks')
        gaze = (features['gaze_horizontal'], features['gaze_vertical'])
        pose = results.get('pose', (0,0,0))
        
        # a. Gaze Durations
        is_gazing_away = abs(gaze[0]) > 0.6 or abs(gaze[1]) > 0.6
        if is_gazing_away:
            if self.gaze_away_start is None: self.gaze_away_start = now
            self.gaze_away_duration = now - self.gaze_away_start
        else:
            self.gaze_away_start = None
            self.gaze_away_duration = 0
            
        # b. Face Absence
        if not presence:
            if self.face_missing_start is None: self.face_missing_start = now
            self.face_missing_duration = now - self.face_missing_start
        else:
            self.face_missing_start = None
            self.face_missing_duration = 0
            
        # c. Object Confidence
        obj_scores = features.get('object_scores', {})
        max_forbidden_conf = 0.0
        for obj in ['cell phone', 'laptop', 'book']:
            max_forbidden_conf = max(max_forbidden_conf, obj_scores.get(obj, 0.0))
            
        # d. Obstruction
        is_obstructed, brightness, variance = self.detect_obstruction(frame)
        
        # 2. EVALUATE RULES (Minor)
        rules_triggered = []
        
        # Gaze Minor
        if self.CONFIG['GAZE_MINOR_DURATION'] <= self.gaze_away_duration < self.CONFIG['GAZE_MAJOR_DURATION']:
            rules_triggered.append(("gaze_minor", "MINOR", "Looking away from screen"))
            
        # Gaze Shifts
        if self._track_gaze_shifts(gaze) > 5:
            rules_triggered.append(("gaze_shift", "MINOR", "Frequent gaze shifts"))
            
        # Part Visibility
        if presence and self._check_face_visibility(landmarks):
            rules_triggered.append(("face_visibility", "MINOR", "Face not fully visible"))
            
        # Face Missing Minor
        if self.CONFIG['FACE_MISSING_MINOR'] <= self.face_missing_duration < self.CONFIG['FACE_MISSING_MAJOR']:
            rules_triggered.append(("face_missing_minor", "MINOR", "Temporary absence"))
            
        # Head Pose Minor
        if presence and (abs(pose[0]) > 15 or abs(pose[1]) > 15) and self.gaze_away_duration < self.CONFIG['GAZE_MAJOR_DURATION']:
             rules_triggered.append(("head_pose_minor", "MINOR", "Minor head movement"))

        # 3. EVALUATE RULES (Major)
        
        # Gaze Major
        if self.gaze_away_duration >= self.CONFIG['GAZE_MAJOR_DURATION']:
            rules_triggered.append(("gaze_major", "MAJOR", "Prolonged gaze away"))
            
        # Face Missing Major
        if self.face_missing_duration >= self.CONFIG['FACE_MISSING_MAJOR']:
            rules_triggered.append(("face_missing_major", "MAJOR", "Face not detected"))
            
        # Multiple People
        if features['person_count'] > 1:
            rules_triggered.append(("multiple_faces", "MAJOR", "Multiple persons detected"))
            
        # Identity
        if presence and features['identity_score'] < self.similarity_threshold:
            rules_triggered.append(("identity_mismatch", "MAJOR", "Identity mismatch"))
            
        # Forbidden Objects
        if max_forbidden_conf >= self.CONFIG['OBJECT_CONFIDENCE_MAJOR']:
            rules_triggered.append(("object_major", "MAJOR", "Unauthorized object detected"))
            
        # Head Pose Major
        if presence and (abs(pose[0]) > 30 or abs(pose[1]) > 30) and self.gaze_away_duration >= self.CONFIG['GAZE_MAJOR_DURATION']:
            rules_triggered.append(("head_pose_major", "MAJOR", "Head turned away"))
            
        # Obstruction & Out of Frame
        if is_obstructed and self.face_missing_duration >= self.CONFIG['FACE_MISSING_MAJOR']:
            rules_triggered.append(("camera_obstruction", "MAJOR", "Camera obstruction detected"))
        elif not presence and self.face_missing_duration >= self.CONFIG['FACE_MISSING_MAJOR']:
            rules_triggered.append(("out_of_frame", "MAJOR", "Student left frame"))

        # 4. PROCESS WARNINGS (Validation & Cooldowns)
        final_alerts = []
        for rule_id, level, reason in rules_triggered:
            # FRAME VALIDATION
            self.consecutive_frames[rule_id] += 1
            if self.consecutive_frames[rule_id] >= self.CONFIG['FRAME_VALIDATION_COUNT']:
                # COOLDOWN Check
                cooldown = self.CONFIG[f'COOLDOWN_{level}']
                if now - self.rule_cooldowns[rule_id] > cooldown:
                    self.issue_violation(level, reason)
                    self.rule_cooldowns[rule_id] = now
                    final_alerts.append(f"{level}: {reason}")
        
        # Reset counters for rules not triggered this frame
        triggered_ids = [r[0] for r in rules_triggered]
        for rid in list(self.consecutive_frames.keys()):
            if rid not in triggered_ids:
                self.consecutive_frames[rid] = 0
                
        return final_alerts

    def issue_violation(self, level, reason):
        """Handles counters and escalations."""
        if level == "MINOR":
            self.minor_warning_count += 1
            print(f"--- MINOR WARNING ({self.minor_warning_count}): {reason} ---")
            
            # Escalation
            if self.minor_warning_count >= self.CONFIG['MINOR_TO_MAJOR_THRESHOLD']:
                self.minor_warning_count = 0
                self.issue_violation("MAJOR", "Escalation from repeated minor warnings")
        else:
            self.major_warning_count += 1
            print(f"!!! MAJOR WARNING ({self.major_warning_count}): {reason} !!!")
            
            # Terminator & Flagging
            if self.major_warning_count >= self.CONFIG['MAJOR_TO_TERMINATE']:
                self.blocked = True
                print("CRITICAL: EXAM TERMINATED")
            elif self.major_warning_count >= self.CONFIG['MAJOR_TO_FLAG_THRESHOLD']:
                self.is_flagged = True
                print("FLAG: EXAM SESSION MARKED FOR REVIEW")

    def fusion_behavior_callback(self, frame, results):
        """Real-time decision making using HEURISTICS + XGBoost Fusion."""
        if self.blocked: return
        
        features = self.get_monitoring_features(frame, results)
        
        # 1. PRIMARY: Heuristic Rules Engine
        heuristic_alerts = self.evaluate_heuristic_rules(frame, results, features)
        
        # 2. SECONDARY: XGBoost for subtle patterns
        prediction = predict_fusion_vector(**features)
        
        # --- Recording Logic ---
        # Trigger recording on ANY heuristic alert or Major ML prediction
        has_forbidden = features['phone_detected'] or features['book_detected'] or features['laptop_detected']
        has_multiple_people = features['person_count'] > 1
        critical_condition = has_forbidden or has_multiple_people or (len(heuristic_alerts) > 0)

        should_start_recording = (prediction >= 2 or critical_condition)
        
        if should_start_recording and not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.cloud_dir, f"event_LEVEL{prediction}_{timestamp}.mp4")
            h, w = frame.shape[:2]
            self.active_recorder = VideoRecorder(filename, fps=20.0, size=(w, h))
            self.is_recording = True
            
            # Save Snapshot
            cv2.imwrite(os.path.join(self.cloud_dir, f"event_LEVEL{prediction}_{timestamp}.jpg"), frame)
            self._log_metadata(timestamp, {
                "level": int(prediction), 
                "heuristic_alerts": heuristic_alerts,
                "features": features
            })

        if self.is_recording:
            self.active_recorder.add_frame(frame)
            
            # Stop condition
            if prediction < 1 and not critical_condition:
                self.active_recorder.stop()
                self.is_recording = False
                self.active_recorder = None

        # --- Popup Notification handling ---
        # The rules engine already handles self.issue_violation which prints to console.
        # We can also trigger a popup if any MAJOR heuristic alert happened.
        for alert in heuristic_alerts:
            if "MAJOR" in alert:
                self.show_popup_message("MAJOR VIOLATION", alert.split(": ")[1], color=(0, 0, 255))

    def start_exam(self):
        """Starts the high-performance proctoring engine."""
        print("Starting Parallel Proctoring Engine (Target: 20-30 FPS)...")
        # We start the webcam process from the pose estimator, which 
        # now supports our fusion_behavior_callback.
        self.pose_estimator.process_webcam(
            detector=self.detector,
            object_detector_callback=self.check_objects,
            eye_gaze_callback=self.gaze_tracker.get_gaze_direction,
            behavior_callback=self.fusion_behavior_callback
        )

if __name__ == "__main__":
    orchestrator = InvigilationSystem(registered_image_path="student2_id.jpeg")
    
    # 1. Identity Phase
    print("Beginning Identity Verification...")
    if orchestrator.verify_student():
        orchestrator.show_popup_message("VERIFIED", "CALIBRATING...\nPlease follow prompts.", color=(0, 255, 0))
        
        # 1.5 Calibration Phase
        print("Beginning 5-Stage Head Pose Calibration...")
        orchestrator.run_pose_calibration(orchestrator.pose_estimator)
        
        orchestrator.show_popup_message("CALIBRATION COMPLETE", "EXAM STARTING\nSystem Online", color=(0, 255, 0))
        
        # 2. Main Monitoring Phase
        orchestrator.start_exam()
    else:
        orchestrator.show_popup_message("DENIED", "IDENTITY MISMATCH\nAccess Blocked", color=(0, 0, 255), duration_ms=5000)