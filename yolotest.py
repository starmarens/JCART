from ultralytics import solutions
from ultralytics import YOLO
import cv2
import math 

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load YOLO model
using_model = "yolo11.pt"
model = YOLO(using_model)

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush",]

# Initialize the distance calculation object
distance_calculator = solutions.DistanceCalculation(model=using_model, show=False)  # Disable box display in `show`


while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam. Exiting...")
        break

    # Perform detection
    #results = model(img, stream=True)
    results = model.track(img, stream=True, tracker="bytetrack.yaml")  # with ByteTrack
    
    # Process each detection
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100
            if confidence < 0.5:
                continue  # Skip low-confidence detections

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Calculate distance
            #distance_info = distance_calculator.calculate(img)  # Assuming this returns a distance measurement

            # Display class name and distance
            org = (x1, y1 - 10)  # Position slightly above the bounding box
            text = f"{class_name}m"  # Customize this text format as needed
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the processed frame
    cv2.imshow('Webcam', img)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
