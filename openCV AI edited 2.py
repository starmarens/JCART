import cv2  # Import OpenCV
import numpy as np
import os
import sys
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Open the MacBook's default camera (usually at index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Capture the first frame and define a bounding box
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the camera.")
    cap.release()
    exit()

# Define a static bounding box (you can change this or use cv2.selectROI())
bbox = (100, 100, 200, 200)  # Adjust the coordinates as needed

# Initialize the GOTURN tracker with the first frame and bounding box
tracker = cv2.TrackerGOTURN_create()  # Assuming this is correctly set up
ok = tracker.init(frame, bbox)

# Main loop for tracking
while True:
    # Capture a new frame from the camera
    ok, frame = cap.read()
    if not ok:
        print("Failed to grab frame.")
        break

    # Start timer for FPS calculation
    timer = cv2.getTickCount()

    # Update the tracker with the new frame
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box if tracking is successful
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # Display FPS on the frame
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))

    # Show the frame with tracking info
    cv2.imshow('Camera Feed with Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
