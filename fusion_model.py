import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

# 1. Prediction function
def predict_fusion_vector(
    identity_score=1.0,
    similarity_score=1.0,
    face_detected=1,
    person_count=1,
    phone_detected=0,
    book_detected=0,
    laptop_detected=0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    gaze_horizontal=0.0,
    gaze_vertical=0.0,
    gaze_away_ratio=0.0,
    face_missing_duration=0.0,
    repeated_violation_flag=0,
    **kwargs
):
    """
    Predict the invigilation status based on multi-modal features.
    0 -> Normal behavior
    1 -> Minor warning
    2 -> Major warning
    3 -> Block exam
    """
    level = 0
    
    # 1. Major Violations (Level 2)
    if face_detected == 0 and face_missing_duration > 5.0:
        level = max(level, 2)
    if person_count > 1:
        level = max(level, 2)
    if phone_detected == 1:
        level = max(level, 2)
    if book_detected == 1:
        level = max(level, 2)
    if laptop_detected == 1:
        level = max(level, 2)
        
    # 2. Minor Violations (Level 1)
    if level == 0:
        # Note: Head pose and gaze are handled strictly by head_pose_estimation_module via 'suspicious' flags
        # to ensure perfect temporal durations (no instant false alarms here).
        if face_detected == 0 and face_missing_duration > 2.0:
            level = max(level, 1)
        elif identity_score < 0.60:
            level = max(level, 1)
            
    # 3. Critical Block (Level 3)
    if repeated_violation_flag == 1 and level >= 2:
        level = 3
        
    return level

def train_fusion_model():
    dataset_path = "xgboost_fusion_dataset_with_pose_gaze.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)

    # Split dataset
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train XGBoost model
    print("Training Fusion XGBoost Model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "fusion_xgboost_model.pkl")
    print("Model saved to fusion_xgboost_model.pkl")

    # Print accuracy
    accuracy = model.score(X_test, y_test)
    print("Fusion model accuracy:", accuracy)

if __name__ == "__main__":
    print("Running Fusion Model Training...")
    train_fusion_model()