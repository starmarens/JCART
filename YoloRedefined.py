import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
from collections import defaultdict

def initialize_yolo_model():
    model = YOLO("yolo11n-seg.pt")
    model.export(format="ncnn")
    return YOLO("yolo11n-seg_ncnn_model")

def send_command(serial_conn, command):
    serial_conn.write(command.encode())
    time.sleep(0.1)

def process_frame(model, frame, track_history, home_position, arduino):
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes.xywh is not None else []
    annotated_frame = results[0].plot()

    for box in boxes:
        x, y, w, h = box  # Center x, center y, width, height
        box_key = tuple(box)  # Convert box to a tuple to use as a dictionary key
        track = track_history[box_key]
        track.append((float(x), float(y)))

        if cv2.waitKey(1) & 0xFF == ord("i"):
            print("home position")
            print(x, y, w, h)
            home_position[box_key] = (x, y, w, h)

    for box in boxes:
        box_key = tuple(box)
        if box_key in home_position:
            x, y, w, h = box
            x_home, y_home, w_home, h_home = home_position[box_key]
            x_offset = x - x_home
            y_offset = y - y_home
            w_offset = w - w_home
            h_offset = h - h_home
            if x_offset != 0:
                send_command(arduino, f"x {x_offset}\n")
            if y_offset != 0:
                send_command(arduino, f"y {y_offset}\n")
            if h_offset != 0:
                send_command(arduino, f"h {h_offset}\n")
            if w_offset != 0:
                send_command(arduino, f"w {w_offset}\n")
            if x_offset == 0 and y_offset == 0 and h_offset == 0 and w_offset == 0:
                send_command(arduino, f"x 0\n")
                send_command(arduino, f"y 0\n")
                send_command(arduino, f"h 0\n")
                send_command(arduino, f"w 0\n")

            box_info = f"x: {int(x)}, y: {int(y)}, w: {int(w)}, h: {int(h)}"
            text_origin = (int(x), int(y) - 10)
            cv2.putText(annotated_frame, box_info, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame

def display_frame(annotated_frame):
    cv2.imshow("YOLO11 Tracking with Box Info", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    return False

def main():
    #sets yolo model
    model = initialize_yolo_model()
    #opens serial port
    arduino = serial.Serial("/dev/cu.usbmodem1201", 9600, timeout=1)
    #opens camera
    cap = cv2.VideoCapture(0)
    serial_conn = arduino
    track_history = defaultdict(list)
    home_position = {}

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam. Exiting...")
                break

            annotated_frame = process_frame(model, frame, track_history, home_position, arduino)
            if display_frame(annotated_frame):
                break
    finally:
        cap.release()
        serial_conn.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()