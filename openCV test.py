import cv2  # Import OpenCV
import numpy as np
import matplotlib.pyplot as plt

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")


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
    print("height", cv2.CAP_PROP_FRAME_HEIGHT)
    print("width", cv2.CAP_PROP_FRAME_WIDTH )


# 3. Loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Failed to grab frame.")
        break

    # 4. Display the captured frame
    cv2.imshow('Camera Feed', frame)

    # Define a bounding box
    bbox = (100, 100, 200, 200)
    # bbox = cv2.selectROI(frame, False)
    # print(bbox)
    displayRectangle(frame, bbox)



    # 5. Press 'q' to exit the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()