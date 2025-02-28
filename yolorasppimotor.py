from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time


def initialize_yolo_model():
    """Initialize and return the YOLO model."""
    model = YOLO("yolo11n-seg.pt")
    model.export(format="ncnn")
    return YOLO("yolo11n-seg_ncnn_model")


def initialize_webcam():
    """Initialize and return the webcam capture object."""
    return cv2.VideoCapture(0)


def initialize_gpio(servo_pin=18, esc_pin=19):
    """
    Initialize GPIO for servo and ESC control.
    
    Args:
        servo_pin: GPIO pin connected to the servo.
        esc_pin: GPIO pin connected to the ESC.
        
    Returns:
        dict: PWM objects for the servo and ESC.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin, GPIO.OUT)
    GPIO.setup(esc_pin, GPIO.OUT)

    servo_pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz for servo
    esc_pwm = GPIO.PWM(esc_pin, 50)  # 50 Hz for ESC

    # Start PWM with 0 duty cycle (stopped position)
    servo_pwm.start(0)
    esc_pwm.start(0)

    return {"servo": servo_pwm, "esc": esc_pwm}


def process_frame(model, frame, track_history, home_position, motor_control):
    """
    Process a frame with YOLO model and update tracking history and motor control.
    
    Args:
        model: YOLO model for object detection and tracking.
        frame: The current frame from the video feed.
        track_history: Dictionary to store track history of each object.
        home_position: Dictionary storing "home" positions of tracked objects.
        motor_control: Dictionary containing servo and ESC PWM objects.
        
    Returns:
        tuple: Annotated frame, a list of tracked object coordinates, and their movement status.
    """
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else []
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    annotated_frame = results[0].plot()

    movement_status = {}  # Dictionary to store movement status for each object

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box  # Center x, center y, width, height
        track = track_history[track_id]
        track.append((float(x), float(y)))  # Store x, y center point

        # Retain up to 30 points for smooth tracking lines
        if len(track) > 30:
            track.pop(0)

        # Initialize home position
        if cv2.waitKey(1) & 0xFF == ord("i"):
            home_position[track_id] = (x, y, w, h)

        # Movement detection
        if track_id in home_position:
            x_home, y_home, _, _ = home_position[track_id]
            movement = "stationary"

            if x < x_home - 5:
                movement = "moving left"
                motor_control["servo"].ChangeDutyCycle(7.5)  # Adjust servo to left
            elif x > x_home + 5:
                movement = "moving right"
                motor_control["servo"].ChangeDutyCycle(12.5)  # Adjust servo to right
            else:
                motor_control["servo"].ChangeDutyCycle(10.0)  # Center servo

            if y < y_home - 5:
                movement = "moving forward"
                motor_control["esc"].ChangeDutyCycle(7.5)  # Increase ESC throttle
            elif y > y_home + 5:
                movement = "moving backward"
                motor_control["esc"].ChangeDutyCycle(5.5)  # Decrease ESC throttle
            else:
                motor_control["esc"].ChangeDutyCycle(6.5)  # Neutral ESC

            movement_status[track_id] = movement

    return annotated_frame, movement_status


def main():
    """Main function to execute the YOLO tracking pipeline."""
    model = initialize_yolo_model()
    cap = initialize_webcam()
    motor_control = initialize_gpio()
    track_history = defaultdict(list)
    home_position = {}

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam. Exiting...")
                break

            annotated_frame, movement_status = process_frame(
                model, frame, track_history, home_position, motor_control
            )
            cv2.imshow("YOLO11 Tracking with Box Info", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Cleanup resources
        cap.release()
        cv2.destroyAllWindows()
        motor_control["servo"].stop()
        motor_control["esc"].stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
