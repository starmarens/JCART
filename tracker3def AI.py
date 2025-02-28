from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

def initialize_yolo_model():
    """Initialize and return the YOLO model."""
    return YOLO("yolo11n-seg.pt")

def initialize_webcam():
    """Initialize and return the webcam capture object."""
    return cv2.VideoCapture(0)

def process_frame(model, frame, track_history, reference_coords):
    """
    Process a frame with YOLO model and update tracking history and movement detection.
    
    Args:
        model: YOLO model for object detection and tracking.
        frame: The current frame from the video feed.
        track_history: Dictionary to store track history of each object.
        reference_coords: Dictionary to store reference coordinates on key press.

    Returns:
        annotated_frame: Frame with annotations and tracking lines.
    """
    results = model.track(frame, persist=True)

    # Ensure results and boxes are not None
    if not results or not results[0].boxes:
        print("No detections in this frame.")
        return frame  # Return the unprocessed frame

    # Extract detections
    boxes = results[0].boxes.xywh.cpu()  # Bounding box coordinates
    track_ids = results[0].boxes.id  # Tracking IDs

    # Handle cases where IDs might be None
    if track_ids is not None:
        track_ids = track_ids.int().cpu().tolist()
    else:
        track_ids = []

    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box  # Center x, center y, width, height
        track = track_history[track_id]
        track.append((float(x), float(y)))  # Store x, y center point

        # Retain up to 30 points for smooth tracking lines
        if len(track) > 30:
            track.pop(0)

        # Store reference coordinates on key press
        if cv2.waitKey(1) & 0xFF == ord("i"):
            reference_coords["x"], reference_coords["y"] = float(x), float(y)
            reference_coords["w"], reference_coords["h"] = float(w), float(h)

        # Check if reference coordinates are valid
        if all(val is not None for val in reference_coords.values()):
            ref_x, ref_y = reference_coords["x"], reference_coords["y"]
            ref_w, ref_h = reference_coords["w"], reference_coords["h"]

            movement = ""
            if float(x) < ref_x:  # Ensure tensors are converted to floats
                movement = "moving left"
            elif float(x) > ref_x:
                movement = "moving right"

            if float(w) < ref_w or float(h) < ref_h:
                movement = "moving away"

            # Display movement message
            text_origin = (int(x - w / 2), int(y - h / 2) - 20)
            cv2.putText(annotated_frame, movement, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw bounding box and display track info
        top_left = (int((x + w - 50) / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(annotated_frame, top_left, bottom_right, (0, 255, 0), 2)

        # Prepare and draw text for box information
        box_info = f"x: {int(x)}, y: {int(y)}, w: {int(w)}, h: {int(h)}"
        text_origin = (top_left[0], top_left[1] - 10)
        cv2.putText(annotated_frame, box_info, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated_frame

def display_frame(annotated_frame):
    """
    Display the frame and handle exit on 'q' press.
    
    Args:
        annotated_frame: Frame with annotations to be displayed.
        
    Returns:
        bool: True if 'q' is pressed, else False.
    """
    cv2.imshow("YOLO11 Tracking with Box Info", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    return False

def main():
    """Main function to execute the YOLO tracking pipeline."""
    model = initialize_yolo_model()
    cap = initialize_webcam()
    track_history = defaultdict(lambda: [])
    reference_coords = {"x": None, "y": None, "w": None, "h": None}  # For storing reference point

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Exit loop if video feed ends or frame not read

        annotated_frame = process_frame(model, frame, track_history, reference_coords)
        if display_frame(annotated_frame):
            break  # Exit on 'q' key press

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
