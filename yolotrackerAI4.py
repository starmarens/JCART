from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


def initialize_yolo_model():
    """Initialize and return the YOLO model."""
    model = YOLO("yolo11n-seg.pt")
    model.export(format = "ncnn")
    return YOLO("yolo11n-seg_ncnn_model")


def initialize_webcam():
    """Initialize and return the webcam capture object."""
    cv2.VideoCapture(0)
    
    return cv2.VideoCapture(0)


def process_frame(model, frame, track_history, home_position):
    """
    Process a frame with YOLO model and update tracking history.
    
    Args:
        model: YOLO model for object detection and tracking.
        frame: The current frame from the video feed.
        track_history: Dictionary to store track history of each object.
        home_position: Dictionary storing "home" positions of tracked objects.
        
    Returns:
        annotated_frame: Frame with annotations and tracking lines.
    """
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else []
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box  # Center x, center y, width, height
        track = track_history[track_id]
        track.append((float(x), float(y)))  # Store x, y center point

        # Retain up to 30 points for smooth tracking lines
        if len(track) > 30:
            track.pop(0)

        # Check if the "i" key is pressed to initialize home position
        if cv2.waitKey(1) & 0xFF == ord("i"):
            print("Home position initialized for object", track_id)
            home_position[track_id] = (x, y, w, h)

        # Movement detection
        if track_id in home_position:
            x_home, y_home, _, _ = home_position[track_id]
            if x < x_home - 5:
                print(f"Object {track_id} moving left")
            elif x > x_home + 5:
                print(f"Object {track_id} moving right")
            else:
                print(f"Object {track_id} stationary")

        # Draw tracking lines
        points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Prepare and draw text for box information
        box_info = f"x: {int(x)}, y: {int(y)}, w: {int(w)}, h: {int(h)}"
        text_origin = (int(x), int(y) - 10)
        cv2.putText(annotated_frame, box_info, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
    track_history = defaultdict(list)
    home_position = {}  # Store "home" positions for tracked objects

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam. Exiting...")
            break

        annotated_frame = process_frame(model, frame, track_history, home_position)
        if display_frame(annotated_frame):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()