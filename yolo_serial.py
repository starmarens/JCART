from collections import defaultdict
import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO

def initialize_yolo_model():
    model = YOLO("yolo11n-seg.pt")
    #model.export(format="ncnn")
    return YOLO("/Users/georgelopez/Desktop/yolo11n-seg_ncnn_model")

def send_command(arduino, command):
    arduino.write(command.encode())
    time.sleep(0.1)

def process_frame(model, frame, track_history, home_position, arduino):
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else []
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box  # Center x, center y, width, height
        track = track_history[track_id]
        track.append((float(x), float(y)))


        if cv2.waitKey(1) & 0xFF == ord("i"):
            print("Home position initialized for object", track_id)
            home_position[track_id] = (x, y, w, h)
            

        # Movement detection and sending serial commands
        if track_id in home_position:
            x_home, y_home, w_home, h_home = home_position[track_id]
            x_offset = x - x_home
            y_offset = y - y_home
            w_offset = w - w_home
            h_offset = h - h_home
            if x_offset > 0 or x_offset < 0:
                send_command(arduino, f"x {x_offset}\n")
            elif y_offset > 0 or y_offset < 0:
                send_command(arduino, f"y {y_offset}\n")
            elif h_offset > 0 or h_offset < 0:
                send_command(arduino, f"h {h_offset}\n")
            elif w_offset > 0 or w_offset < 0:
                send_command(arduino, f"w {w_offset}\n")
            else:
                print(f"Object {track_id} stationary")
                send_command(arduino, "STOP\n")

        points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

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
    model = initialize_yolo_model()
    cap = cv2.VideoCapture(0)
    track_history = defaultdict(list)
    home_position = {}
    arduino = serial.Serial("/dev/cu.usbmodem1201", 9600, timeout=1)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam. Exiting...")
            break

        annotated_frame = process_frame(model, frame, track_history, home_position, arduino)
        if display_frame(annotated_frame):
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
