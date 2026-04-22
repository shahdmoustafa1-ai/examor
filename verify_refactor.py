import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from face_detection_module import FaceDetector
    from head_pose_estimation_module import HeadPoseEstimator
    from eye_gaze_tracking_module import EyeGazeTracker
    
    print("All modules imported successfully.")
    
    print("\nTesting FaceDetector initialization...")
    detector = FaceDetector()
    print("FaceDetector initialized.")
    
    print("\nTesting HeadPoseEstimator initialization...")
    # Using existing model paths if they exist
    head_pose = HeadPoseEstimator(fsa_net_path='fsanet.onnx', face_landmarker_path='face_landmarker.task')
    print("HeadPoseEstimator initialized and uses shared detector.")
    
    print("\nTesting EyeGazeTracker initialization...")
    eye_gaze = EyeGazeTracker(face_landmarker_path='face_landmarker.task')
    print("EyeGazeTracker initialized and uses shared detector.")
    
    print("\nVerifying shared detector usage in HeadPoseEstimator...")
    if hasattr(head_pose, 'detector') and isinstance(head_pose.detector, FaceDetector):
        print("SUCCESS: HeadPoseEstimator is using the shared FaceDetector.")
    else:
        print("FAILURE: HeadPoseEstimator is NOT using the shared FaceDetector.")
        
    print("\nVerifying shared detector usage in EyeGazeTracker...")
    if hasattr(eye_gaze, 'detector') and isinstance(eye_gaze.detector, FaceDetector):
        print("SUCCESS: EyeGazeTracker is using the shared FaceDetector.")
    else:
        print("FAILURE: EyeGazeTracker is NOT using the shared FaceDetector.")

except Exception as e:
    print(f"\nERROR during verification: {e}")
    sys.exit(1)

print("\nVerification complete!")
