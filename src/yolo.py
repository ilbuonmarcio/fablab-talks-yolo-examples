import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("yolo11n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction on the frame (results object)
    results = model.predict(frame, verbose=False)

    # Get the annotated frame with detections drawn
    annotated_frame = results[0].plot()

    # Show the annotated frame in a window
    cv2.imshow("YOLOv11 Live Detection", annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()