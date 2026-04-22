import cv2
import numpy as np
import os
try:
    import xgboost as xgb
except ImportError:
    xgb = None

class EyeGazeTracker:
    def __init__(self, model_path="gaze_regressor.xgb"):
        # iris indices:
        self.left_eye_indices = [33, 133, 159, 145]
        self.right_eye_indices = [362, 263, 386, 374]
        
        self.model = None
        if xgb and os.path.exists(model_path):
            try:
                self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
                print(f"XGBoost Gaze Model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load XGBoost model: {e}")
        
        print("Eye Gaze Tracker Initialized (Hybrid Mode)...")

    def _extract_features(self, landmarks):
        """Extract the relative iris displacement as features for the regressor."""
        def get_displacement(iris_idx, corners):
            iris = landmarks[iris_idx]
            # [outer, inner, top, bottom]
            c_outer = landmarks[corners[0]]
            c_inner = landmarks[corners[1]]
            c_top = landmarks[corners[2]]
            c_bottom = landmarks[corners[3]]
            
            w = max(1e-6, abs(c_inner.x - c_outer.x))
            h = max(1e-6, abs(c_bottom.y - c_top.y))
            dx = (iris.x - (c_outer.x + c_inner.x)/2) / (w / 2)
            dy = (iris.y - (c_top.y + c_bottom.y)/2) / (h / 2)
            return [dx, dy]

        l_disp = get_displacement(468, self.left_eye_indices)
        r_disp = get_displacement(473, self.right_eye_indices)
        return np.array(l_disp + r_disp).reshape(1, -1)

    def get_gaze_direction(self, landmarks):
        """
        Estimate precise gaze angles and pupil locations.
        Returns: {status, left_pupil, right_pupil, left_eye_bbox, right_eye_bbox, angles}
        """
        if not landmarks or len(landmarks) < 468:
            return "No Face Landmarks", None

        has_iris = len(landmarks) >= 474

        def process_eye(iris_idx, corners):
            c_outer = landmarks[corners[0]]
            c_inner = landmarks[corners[1]]
            c_top = landmarks[corners[2]]
            c_bottom = landmarks[corners[3]]
            
            # Tighter eye corner bounds
            min_x = min([landmarks[idx].x for idx in corners])
            max_x = max([landmarks[idx].x for idx in corners])
            min_y = min([landmarks[idx].y for idx in corners])
            max_y = max([landmarks[idx].y for idx in corners])
            bbox = [min_x, min_y, max_x, max_y]

            if has_iris:
                iris = landmarks[iris_idx]
                pupil_x, pupil_y = iris.x, iris.y
            else:
                # Fallback to center of eye bounding box
                pupil_x, pupil_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            
            # Scale invariant targets
            eye_width = max(1e-6, abs(max_x - min_x))
            target_yaw = (pupil_x - (min_x + max_x)/2) / (eye_width / 2)
            eye_height = max(1e-6, abs(max_y - min_y))
            target_pitch = (pupil_y - (min_y + max_y)/2) / (eye_height / 2)
            
            return (pupil_x, pupil_y), bbox, (target_yaw, target_pitch)

        # iris_idx 468 = left iris, 473 = right iris
        lp, lbb, lang = process_eye(468, self.left_eye_indices)
        rp, rbb, rang = process_eye(473, self.right_eye_indices)
        
        # Priority: XGBoost Gaze Estimation
        if self.model and has_iris:
            features = self._extract_features(landmarks)
            preds = self.model.predict(features)[0] 
            avg_yaw, avg_pitch = float(preds[0]), float(preds[1])
            status = f"Gaze (XGB): [{avg_yaw:.2f}, {avg_pitch:.2f}]"
        else:
            # Fallback heuristic
            avg_yaw = (lang[0] + rang[0]) / 2
            avg_pitch = (lang[1] + rang[1]) / 2
            status = "Gaze (H): "
            if abs(avg_yaw) < 0.25 and abs(avg_pitch) < 0.25:
                status += "Forward"
            else:
                if avg_yaw > 0.3: status += "Right "
                elif avg_yaw < -0.3: status += "Left "
                if avg_pitch > 0.3: status += "Down "
                elif avg_pitch < -0.3: status += "Up "
            
        res = {
            'status': status.strip(),
            'left_pupil': lp,
            'right_pupil': rp,
            'left_bbox': lbb,
            'right_bbox': rbb,
            'angles': (avg_yaw, avg_pitch)
        }
        return status.strip(), res

if __name__ == "__main__":
    from face_detection_module import FaceDetector
    detector = FaceDetector()
    tracker = EyeGazeTracker()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        det = detector.detect_with_landmarks(frame)
        if det and det['landmarks']:
            status, res = tracker.get_gaze_direction(det['landmarks'])
            cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Gaze tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()