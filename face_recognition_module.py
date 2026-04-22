import cv2
import time
import numpy as np
import os
# Base Proctoring Logic
from proctoring_base import ProctoringSystem, VideoRecorder
if __name__ == "__main__":
    image_name = "student2_id.jpeg"
    if not os.path.exists(image_name): image_name = "student2_id.jpeg"
    try:
        proctor = ProctoringSystem(registered_image_path=image_name, max_trials=3)
        # Onboarding Delay
        proctor.show_popup_message("PREPARING", "Identity verification starting in 5s\nPlease center your face", duration_ms=5000)
        # Identity Check Phase
        success = False
        while proctor.trials_exhausted < proctor.max_trials:
            if proctor.verify_student():
                success = True; break
            else:
                remaining = proctor.max_trials - proctor.trials_exhausted
                if remaining > 0:
                    msg = f"VERIFICATION FAILED\n{remaining} Trials Remaining\nNext trial starts in 5s"
                    proctor.show_popup_message("Warning", msg, color=(0, 0, 255), duration_ms=5000)
                else: break
        if success:
            proctor.show_popup_message("Verified", "EXAM STARTING\nIdentity Confirmed", color=(0, 255, 0))
            print("Starting Parallel Proctoring Engine (Target: 20-30 FPS)...")
            proctor.pose_estimator.process_webcam(
                detector=proctor.detector,
                object_detector_callback=proctor.check_objects,
                eye_gaze_callback=proctor.gaze_tracker.get_gaze_direction
            )
        else:
            proctor.show_popup_message("Denied", "IDENTITY MISMATCH\nAccount is Blocked", color=(0, 0, 255))
    except Exception as e:
        print(f"Error initializing system: {e}")
        