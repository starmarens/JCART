from ultralytics import YOLO
import cv2
import numpy as np
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load YOLO model with segmentation support
using_model = "yolo11n-pos.pt"
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
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam. Exiting...")
        break

    # Perform detection with segmentation
    results = model(img, stream=True)

    # Process each detection
    for r in results:
        masks = r.masks  # Segmentation masks
        boxes = r.boxes

        for i, box in enumerate(boxes):
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100
            if confidence < 0.5:
                continue  # Skip low-confidence detections

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Extract and apply segmentation mask
            if masks is not None and i < len(masks):
                mask_data = masks[i].numpy()  # Convert mask to NumPy array

                # Check if mask_data is 2D; some models output multi-dimensional masks
                if mask_data.ndim == 2:
                    mask_resized = cv2.resize(mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold mask for binary representation

                    # Create a color overlay for the mask
                    color = (0, 255, 0)  # Set color for segmentation mask
                    colored_mask = np.zeros_like(img)
                    colored_mask[mask_binary == 255] = color

                    # Blend the mask with the original image
                    img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

            # Display class name
            org = (x1, y1 - 10)  # Position slightly above the bounding box
            text = f"{class_name} {confidence:.2f}"
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the processed frame
    cv2.imshow('Webcam', img)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()