import cv2
import tempfile
import streamlit as st
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model
model = YOLO("Weights/best.pt")
classNames = ['With Helmet', 'Without Helmet']

st.title("YOLO Helmet Detection Web App")

# File uploader for video input
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Initialize video capture
    cap = cv2.VideoCapture(temp_path)

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Perform YOLO detection
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5, colorR=(0, 255, 0))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(
                    img,
                    f'{classNames[cls]} {conf}',
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                    colorR=(0, 255, 0),
                    offset=10,
                )

        # Convert BGR image (OpenCV) to RGB for Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the current frame
        frame_placeholder.image(img_rgb, channels="RGB")

    cap.release()
    st.success("Processing Complete!")