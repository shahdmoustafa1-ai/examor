import numpy as np

yaw_thresh = 8.0
pitch_thresh = 8.0

def get_direction(yaw, pitch, bias):
    c_yaw = yaw - bias[0]
    c_pitch = pitch - bias[1]
    
    user_yaw = -c_yaw
    user_pitch = -c_pitch
    
    if abs(user_yaw) < yaw_thresh and abs(user_pitch) < pitch_thresh:
        return "Looking Forward"
    
    direction = "Looking "
    if user_yaw > yaw_thresh:
        direction += "Right "
    elif user_yaw < -yaw_thresh:
        direction += "Left "
        
    if user_pitch > pitch_thresh:
        direction += "Up "
    elif user_pitch < -pitch_thresh:
        direction += "Down "
        
    return direction.strip()

# Test Cases (assuming bias is zeros for simplicity)
bias = [0, 0, 0]

print(f"Forward (0,0): {get_direction(0, 0, bias)}")
print(f"Right (10,0): {get_direction(10, 0, bias)}")
print(f"Left (-10,0): {get_direction(-10, 0, bias)}")
print(f"Up (0,-10): {get_direction(0, -10, bias)}")
print(f"Down (0,10): {get_direction(0, 10, bias)}")
