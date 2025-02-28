from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time



 #used for calibration for original position
calibrated_variable = None
calibration_threshold = 0.1 

def grab_iposition():
    return x, y, w, h

def is_within_threshold(initial, current, threshold):
    return abs(initial - current) <= threshold

def calibrate():
    global calibrated_variable

    # Step 1: Capture initial positions
    initial_positions = grab_iposition()

    # (Optional) Wait for some time or another button press
    time.sleep(3)  # Adjust as needed for your application

    # Step 2: Check current positions
    current_positions = (x, y, w, h)

# Step 3: Verify that each position is within the threshold
    if all(is_within_threshold(initial, current, calibration_threshold) 
        for initial, current in zip(initial_positions, current_positions)):
# Step 4: Set calibrated_variable if within threshold
        calibrated_variable = current_positions
        print("Calibration successful:", calibrated_variable)
    else:
        print("Calibration failed: values changed too much.")
        return calibrated_variable(x), calibrated_variable(y), calibrated_variable(w), calibrated_variable(h)




# Load the YOLO model
model = YOLO("yolo11n-seg.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize results on the frame
        annotated_frame = results[0].plot()

        # Plot tracks and display bounding box values
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box  # Center x, center y, width, height
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point

            if len(track) > 30:  # Retain up to 30 points for smooth tracking lines
                track.pop(0)

            # Draw tracking lines
            points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Draw bounding box around object
            top_left = (int((x + w - 50) / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            #cv2.rectangle(annotated_frame, top_left, bottom_right, (0, 255, 0), 2)

            # Prepare text for the box (track ID, x, y, width, height)
            box_info = f"x: {int(x)}, y: {int(y)}, w: {int(w)}, h: {int(h)}"
            text_origin = (top_left[0], top_left[1] - 10)  # Position above the bounding box
            
            # Draw text on the frame
            cv2.putText(annotated_frame, box_info, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking with Box Info", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("c"):
            calibrate()
            calibrated_box_info =  f"cx: {int(calibrated_variable(x))}, cy: {int(calibrated_variable(y))}, cw: {int(calibrated_variable(w))}, ch: {int(calibrated_variable(h))}"
            cv2.putText(calibrated_box_info, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()