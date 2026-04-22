import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import time
from math import cos, sin
import os
from face_detection_module import FaceDetector
import threading
from queue import Queue, Empty
from collections import deque
from multi_model_fusion_layer import BehaviorClassifier

class HeadPoseEstimator:
    def __init__(self, fsa_net_path='fsanet.onnx', face_landmarker_path='face_landmarker.task'):
        # 1. Models and State
        self.smooth_pose = [0.0, 0.0, 0.0] 
        self.pose_bias = [0.0, 0.0, 0.0] 
        self.reference_poses = {} 
        self.calibrated = False
        self.ema_alpha = 0.3

        # 3. Load FSA-Net ONNX model
        self.fsa_net_path = fsa_net_path
        try:
            if os.path.exists(fsa_net_path):
                print(f"Loading FSA-Net from {fsa_net_path}...")
                self.ort_session = ort.InferenceSession(fsa_net_path, providers=['CPUExecutionProvider'])
                self.model_loaded = True
                print("FSA-Net loaded successfully.")
            else:
                print(f"WARNING: FSA-Net model not found at {fsa_net_path}.")
                self.model_loaded = False
        except Exception as e:
            print(f"WARNING: FSA-Net model could not be loaded: {e}")
            self.model_loaded = False
        
        print("Initialization complete.")

    def get_face_box(self, frame):
        """No-op as box is now passed from central detector."""
        return None

    def set_calibration_data(self, data):
        self.reference_poses = data
        if "FORWARD" in data:
            self.pose_bias = data["FORWARD"]
        self.calibrated = True
        print(f"Calibration locked! Baseline Pose Bias: {self.pose_bias}")

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        """Draw pose axes on the image."""
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis (red)
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis (green)
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (blue)
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)

        return img

    def estimate_pose(self, face_roi, landmarks=None, calibrated=True, global_wh=None):
        """
        Estimate head pose using FSA-Net, with optional calibration adjustment.
        Returns: (yaw, pitch, roll) in degrees.
        """
        if not self.model_loaded:
            if landmarks:
                # If landmarks are global-normalized, use global WH. Otherwise use ROI WH.
                w, h = global_wh if global_wh else (face_roi.shape[1], face_roi.shape[0])
                return self._estimate_pose_from_modern_landmarks(landmarks, w, h)
            return 0.0, 0.0, 0.0

        try:
            # 1. PREPROCESSING
            face_img = cv2.resize(face_roi, (64, 64))
            face_img = face_img.astype(np.float32)
            face_img = (face_img / 255.0 - 0.5) / 0.5
            
            # 2. MATCH MODEL INPUT SHAPE
            input_shape = self.ort_session.get_inputs()[0].shape
            if input_shape[1] == 3:
                # NCHW
                blob = face_img.transpose(2, 0, 1)
            else:
                # NHWC
                blob = face_img
            blob = np.expand_dims(blob, axis=0)
            
            # 3. RUN INFERENCE
            input_name = self.ort_session.get_inputs()[0].name
            output = self.ort_session.run(None, {input_name: blob})
            
            # 3. PARSE OUTPUT
            res = np.array(output).flatten()
            if len(res) >= 3:
                y, p, r = res[0], res[1], res[2]
            else:
                y, p, r = 0.0, 0.0, 0.0
                
            # 4. SMOOTHING (EMA)
            if self.smooth_pose is None:
                self.smooth_pose = [y, p, r]
            else:
                self.smooth_pose[0] = self.ema_alpha * y + (1 - self.ema_alpha) * self.smooth_pose[0]
                self.smooth_pose[1] = self.ema_alpha * p + (1 - self.ema_alpha) * self.smooth_pose[1]
                self.smooth_pose[2] = self.ema_alpha * r + (1 - self.ema_alpha) * self.smooth_pose[2]
                
            return tuple(self.smooth_pose)
            
        except Exception as e:
            print(f"Error during Head Pose Estimation: {e}")
            return 0.0, 0.0, 0.0

    def _estimate_pose_from_modern_landmarks(self, face_landmarks, img_w, img_h):
        """Estimate head pose using MediaPipe Task landmarks and solvePnP."""
        model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),      
            (-150.0, -150.0, -125.0),    
            (150.0, -150.0, -125.0)      
        ])

        indices = [1, 152, 33, 263, 61, 291]
        image_points = []
        for idx in indices:
            lm = face_landmarks[idx]
            image_points.append((lm.x * img_w, lm.y * img_h))
        image_points = np.array(image_points, dtype="double")

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        return angles[1], angles[0], angles[2]

    def process_webcam(self, detector=None, object_detector_callback=None, eye_gaze_callback=None, behavior_callback=None):
        """Sequential Proctoring Loop: Detector -> Head Pose -> Eye Gaze"""
        if detector is None:
            detector = FaceDetector()

        class VideoStream:
            def __init__(self, src=0):
                self.stream = cv2.VideoCapture(src)
                self.stream.set(cv2.CAP_PROP_FPS, 30)
                (self.grabbed, self.frame) = self.stream.read()
                self.stopped = False
            def start(self): 
                threading.Thread(target=self.update, args=(), daemon=True).start()
                return self
            def update(self):
                while not self.stopped:
                    if not self.grabbed: self.stop(); return
                    (self.grabbed, self.frame) = self.stream.read()
            def read(self): return self.frame
            def stop(self): self.stopped = True

        vs = VideoStream(src=0).start()
        print("\n--- Zero-Lag Head Pose Estimation Started ---")
        
        results_lock = threading.Lock()
        classifier = BehaviorClassifier()
        glance_start_time = None
        
        latest_results = {
            'bbox': None,
            'pose': (0, 0, 0),
            'mode': "Loading...",
            'gaze_data': None,
            'alerts': [],
            'suspicious': False,
            'reason': ""
        }

        def inference_worker():
            nonlocal latest_results, glance_start_time
            while not vs.stopped:
                try:
                    frame = vs.read()
                    if frame is None: continue
                    
                    # 1. Pipeline execution
                    h, w = frame.shape[:2]
                    detection = detector.detect_with_landmarks(frame)
                    
                    id_score = 1.0 
                    if hasattr(object_detector_callback, "__self__") and hasattr(object_detector_callback.__self__, "last_id_score"):
                        id_score = object_detector_callback.__self__.last_id_score

                    gaze_data = None
                    gaze_angles = (0, 0)
                    if eye_gaze_callback and detection:
                        _, res = eye_gaze_callback(detection['landmarks'])
                        gaze_data = res
                        if res: gaze_angles = res.get('angles', (0,0))

                    head_pose = (0, 0, 0)
                    if detection and detection['face_roi'].size > 0:
                        # Pass global (w, h) because landmarks are global-normalized
                        head_pose = self.estimate_pose(detection['face_roi'], detection['landmarks'], global_wh=(w, h))
                    
                    # 2. DYNAMIC CALIBRATION GLANCE BLOCK
                    yaw_adj = head_pose[0] - self.pose_bias[0]
                    pitch_adj = head_pose[1] - self.pose_bias[1]
                    
                    # Default bounds if not perfectly configured
                    yaw_bound = 35
                    pitch_bound = 25
                    
                    if self.calibrated and "LEFT" in self.reference_poses and "RIGHT" in self.reference_poses:
                        # Dynamically calculate boundaries from calibration extremes
                        # Assuming symmetric tolerance based on extremes
                        left_yaw_extreme = abs(self.reference_poses["LEFT"][0] - self.pose_bias[0])
                        right_yaw_extreme = abs(self.reference_poses["RIGHT"][0] - self.pose_bias[0])
                        yaw_bound = max(20, min(left_yaw_extreme, right_yaw_extreme) - 10)
                        
                        up_pitch_extreme = abs(self.reference_poses["UP"][1] - self.pose_bias[1])
                        down_pitch_extreme = abs(self.reference_poses["DOWN"][1] - self.pose_bias[1])
                        pitch_bound = max(15, min(up_pitch_extreme, down_pitch_extreme) - 10)
                    
                    glancing = abs(yaw_adj) > yaw_bound or abs(pitch_adj) > pitch_bound or \
                               (gaze_angles and (abs(gaze_angles[0]) > 0.8 or abs(gaze_angles[1]) > 0.8))
                    
                    if glancing:
                        if not glance_start_time:
                            glance_start_time = time.time()
                    else:
                        glance_start_time = None
                    
                    # 3. REASON GENERATION (DESCRIPTIVE)
                    sustained_glance = False
                    reason_parts = []
                    if glancing and glance_start_time and (time.time() - glance_start_time > 3.0):
                        reason_parts.append("GLANCED AWAY (3S)")
                        sustained_glance = True
                        
                    # 4. EXTERNAL OBJECT DETECTION
                    alerts = []
                    obj_scores = {}
                    if object_detector_callback:
                        alerts, _, _, obj_scores = object_detector_callback(frame, face_detected=(detection is not None))
                        if alerts:
                            clean_alerts = [a.replace("FORBIDDEN: ", "") for a in alerts]
                            reason_parts.extend(clean_alerts)
                            
                    # 5. FEATURE FUSION
                    # Replace the generic XGBoost classification with actual temporal heuristic
                    is_suspicious = (sustained_glance == True) or (len(alerts) > 0)
                    if id_score < 0.60:
                        is_suspicious = True
                        reason_parts.append("IDENTITY UNVERIFIED")

                    reason = " | ".join(set(reason_parts)) if reason_parts else ""

                    new_res = {
                        'bbox': detection['bbox'] if detection else None,
                        'landmarks': detection['landmarks'] if detection else None,
                        'pose': head_pose,
                        'mode': "FSA-Net + XGBoost",
                        'gaze_data': gaze_data,
                        'alerts': alerts,
                        'suspicious': is_suspicious,
                        'reason': reason,
                        'id_score': id_score,
                        'object_scores': obj_scores
                    }
                    
                    with results_lock:
                        latest_results = new_res
                    
                    # Trigger behavior logic (Fusion Layer)
                    if behavior_callback:
                        behavior_callback(frame, new_res)
                    elif is_suspicious and hasattr(object_detector_callback, "__self__"):
                        object_detector_callback.__self__.on_suspicious_behavior(frame, new_res)

                    time.sleep(0.01)
                except Exception as e:
                    import traceback
                    print(f"Error in inference worker: {e}")
                    traceback.print_exc()
                    time.sleep(0.5)

        threading.Thread(target=inference_worker, daemon=True).start()
        
        prev_time = 0
        while not vs.stopped:
            frame = vs.read()
            if frame is None: continue
            frame = frame.copy()
            h_img, w_img = frame.shape[:2]
            
            with results_lock:
                res = latest_results.copy()
            
            if res['bbox']:
                x1, y1, x2, y2 = res['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if res.get('gaze_data'):
                    gd = res['gaze_data']
                    for eye in ['left', 'right']:
                        bbox_key = f'{eye}_bbox'
                        pupil_key = f'{eye}_pupil'
                        if gd.get(bbox_key) and gd.get(pupil_key):
                            bb = gd[bbox_key]
                            p = gd[pupil_key]
                            # Global coordinates
                            ex1, ey1, ex2, ey2 = int(bb[0]*w_img), int(bb[1]*h_img), int(bb[2]*w_img), int(bb[3]*h_img)
                            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)
                            # Accurate Pupil/Iris Center
                            cv2.circle(frame, (int(p[0]*w_img), int(p[1]*h_img)), 3, (0, 0, 255), -1)

                status_color = (0, 0, 255) if res['suspicious'] else (0, 255, 0)
                status_txt = f"STATUS: {'SUSPICIOUS' if res['suspicious'] else 'NORMAL'}"
                if res['reason']: status_txt += f" ({res['reason']})"
                cv2.putText(frame, status_txt, (20, h_img - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                y, p, r = res['pose']
                info = f"Head Pose(Yaw:{y:.1f}, Pitch:{p:.1f}, Roll:{r:.1f})"
                # Changed from bright yellow to Dark Blue (150, 0, 0) in BGR
                cv2.putText(frame, info, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 0, 0), 1)
                self.draw_axis(frame, y, p, r, (x1+x2)//2, (y1+y2)//2)
                
                if res['alerts']:
                    cv2.putText(frame, "ALERTS: " + " | ".join(res['alerts']), (20, h_img-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                 cv2.putText(frame, "No face detected", (w_img//2 - 130, h_img//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

            new_time = time.time()
            fps = 1/(new_time-prev_time) if (new_time-prev_time) > 0 else 0
            prev_time = new_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Robust Head Pose Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): vs.stop(); break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    estimator = HeadPoseEstimator(fsa_net_path='fsanet.onnx')
    estimator.process_webcam()