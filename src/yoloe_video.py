import cv2
from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-v8l-seg-pf.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["muffin"]
model.set_classes(names, model.get_text_pe(names))

# Open webcam (0 = default camera)
cap = cv2.VideoCapture("video.mp4")

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
    cv2.imshow("YOLOE LLM Live Detection", annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()