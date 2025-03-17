from ultralytics import YOLO
import cv2
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Path to the YOLO-pose model or just the name")
parser.add_argument("--cam_idx", type=int, default=0, help="Camera index")

args = parser.parse_args()

model = YOLO(args.model)  # Load model

def preprocess_frame(frame):
    """Preprocess the frame before inference."""
    return frame  # YOLO model handles preprocessing internally

# Open webcam
cap = cv2.VideoCapture(args.cam_idx)  # Use 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define YOLO pose keypoint connections
skeleton_connections = [(0, 1), (1, 3), (5, 7), (7, 9), (11, 13), (13, 15), (1, 2), (3, 5), (4, 6),
                        (0, 2), (2, 4), (6, 8), (8, 10), (12, 14), (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]

# Rule-based filtering for keypoints

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Preprocess (if needed, YOLO handles this internally)
    processed_frame = preprocess_frame(frame)
    
    # Inference
    results = model(processed_frame)
    
    # Draw pose on frame
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        
        for keypoint in keypoints:
            # Filter out invalid keypoints (ignore extreme or negative values)
            valid_keypoints = [(int(x), int(y)) for x, y in keypoint if x > 0 and y > 0 and x < frame.shape[1] and y < frame.shape[0]]
            
            for x, y in valid_keypoints:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw skeleton
            for connection in skeleton_connections:
                pt1, pt2 = connection
                if pt1 < len(valid_keypoints) and pt2 < len(valid_keypoints):
                    x1, y1 = valid_keypoints[pt1]
                    x2, y2 = valid_keypoints[pt2]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow("YOLO Pose Estimation", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()