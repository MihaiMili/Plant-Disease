import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r'C:\Users\mihai\runs\detect\train14\weights\best.pt')  # path to your trained model

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # YOLO expects BGR images directly
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.85)

    # Results contains a list (one element per image/frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box corners

            # Class id and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Get class name
            class_name = model.names[cls_id]

            # Draw bounding box and label
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Plant Disease Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
