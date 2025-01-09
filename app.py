import cv2
import math
import cvzone
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Load YOLO model
model = YOLO("Weights/best.pt")
class_labels = ['With Helmet', 'Without Helmet']

# Streamlit App Title
st.title("Counter Pelanggaran Lalu Lintas (Helm Detection)")

# Subtitle
st.subheader("Nama Kelompok: CPL")
st.write("""
1. Muhammad Ismail Ardhafillah (201111013)  
2. Atha An Naufal (211111012)  
3. Marcellino Andriano Dressel (211111018)  
4. Robeth Ahmad Kirom Sholeh (211111022)
5. Alwarits Akbar Nuranda (221111024)
""")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read the uploaded image as a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(img)

    # Initialize counters
    with_helmet_count = 0
    without_helmet_count = 0

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5, colorR=(0, 255, 0))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.1:
                # Update counters based on class
                if class_labels[cls] == 'With Helmet':
                    with_helmet_count += 1
                elif class_labels[cls] == 'Without Helmet':
                    without_helmet_count += 1

                # Add labels to the image
                cvzone.putTextRect(
                    img,
                    f'{class_labels[cls]} {conf}',
                    (max(0, x1), max(35, y1)),
                    scale=0.8,
                    thickness=1,
                    colorR=(255, 0, 0),
                    offset=10,
                )

    # Convert BGR image (OpenCV) to RGB for display in Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the processed image
    st.image(img_rgb, channels="RGB", caption="Processed Image")

    # Display detection counts
    st.write(f"**With Helmet:** {with_helmet_count}")
    st.write(f"**Without Helmet:** {without_helmet_count}")