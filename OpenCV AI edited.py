import cv2  # Import OpenCV
import numpy as np


# Set up tracker
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# Change the index to change the tracker type

    tracker = cv2.TrackerGOTURN.create()

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# 1. Open the MacBook's default camera (usually at index 0)
cap = cv2.VideoCapture(0)
# 2. Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera resolution:", width, "x", height)

# 3. Loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Failed to grab frame.")
        break

    # Define a bounding box
    bbox = (100, 100, 200, 200)

    # Draw the rectangle on the frame
    drawRectangle(frame, bbox)

    # Optional: Draw some text on the frame
    #drawText(frame, "Camera Feed", (10, 30))

    # Display the captured frame with the rectangle
    cv2.imshow('Camera Feed with Rectangle', frame)

    ret = ok = tracker.init(frame, bbox)

    # 5. Press 'q' to exit the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    ok, frame = cap.read()

    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # Display Info
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))


# 6. Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
