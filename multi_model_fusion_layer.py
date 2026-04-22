import os
import numpy as np

try:
    import xgboost as xgb
    import joblib
except ImportError:
    xgb = None
    joblib = None

class BehaviorClassifier:
    """
    Centralized Fusion Layer for multi-modal proctoring signals.
    Combines: [Identity Confidence, Head Pose (3), Eye Gaze (2)]
    """
    def __init__(self):
        # We bypass faulty XGB bias by using explicit heuristic temporal evaluation
        self.model = None
        print("Behavior Classifier initialized (Temporal Heuristic Mode).")

    def create_fusion_vector(self, kwargs):
        """Standardizes the feature vector for classification."""
        return [
            kwargs.get('identity_score', 1.0),
            kwargs.get('similarity_score', 1.0),
            kwargs.get('face_detected', 1),
            kwargs.get('person_count', 1),
            kwargs.get('phone_detected', 0),
            kwargs.get('book_detected', 0),
            kwargs.get('laptop_detected', 0),
            kwargs.get('yaw', 0.0),
            kwargs.get('pitch', 0.0),
            kwargs.get('roll', 0.0),
            kwargs.get('gaze_horizontal', 0.0),
            kwargs.get('gaze_vertical', 0.0),
            kwargs.get('gaze_away_ratio', 0.0),
            kwargs.get('face_missing_duration', 0.0),
            kwargs.get('repeated_violation_flag', 0)
        ]

    def classify(self, **kwargs):
        """
        Input: Raw signals as kwargs.
        Returns: 1 for Suspicious, 0 for Normal
        """
        features = self.create_fusion_vector(kwargs)
        
        if self.model:
            # Predict returns the class level (0-3 usually). We check if >= 1
            pred = self.model.predict(np.array(features).reshape(1, -1))[0]
            return 1 if pred >= 1 else 0
        
        # Heuristic Fusion Logic (Fallback)
        suspicion_score = 0
        id_score = kwargs.get('identity_score', 1.0)
        
        # 1. Identity Penalty
        if id_score < 0.60: 
            suspicion_score += 2 # Major penalty for person mismatch
            
        # 2. Temporal Behavior Flags (passed directly from the tracking engine)
        if kwargs.get('sustained_glance', False):
            suspicion_score += 1
            
        # 3. Object Penalty
        if kwargs.get('phone_detected', 0) > 0: suspicion_score += 2
        if kwargs.get('person_count', 1) > 1: suspicion_score += 2
        
        return 1 if suspicion_score >= 1 else 0

if __name__ == "__main__":
    # Test block
    classifier = BehaviorClassifier()
    result = classifier.classify(identity_score=1.0)
    print(f"Test Status (Normal): {result}")
    
    result_fail = classifier.classify(identity_score=0.3)
    print(f"Test Status (Mismatch): {result_fail}")
