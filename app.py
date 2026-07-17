import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from detector import detect_frame
from video_processor import process_video

st.set_page_config(
    page_title="Real-Time PPE Detector",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 Real-Time PPE Detector")
st.write("Detect Helmet and Mask compliance using YOLOv8")

# Initialize session state for frames
if 'latest_frame' not in st.session_state:
    st.session_state['latest_frame'] = None
frame_count = 0

mode = st.radio(
    "Select Input Type",
    ["Image", "Video", "Webcam"],
    horizontal=True
)

# ---------------- WEBCAM ---------------- #
if mode == "Webcam":
    st.subheader("Live PPE Detection")
    
    def video_frame_callback(frame):
        global frame_count
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 3rd frame to reduce lag
        if frame_count % 3 == 0:
            annotated, _ = detect_frame(img)
            st.session_state['latest_frame'] = annotated
        
        frame_count += 1
        return av.VideoFrame.from_ndarray(st.session_state['latest_frame'] if st.session_state['latest_frame'] is not None else img, format="bgr24")

    webrtc_streamer(
        key="ppe-live",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "height": {"ideal": 360}}, 
            "audio": False
        },
        mode=WebRtcMode.SENDRECV
    )

# ---------------- IMAGE ---------------- #
elif mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, stats = detect_frame(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detection Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
        with col2:
            st.metric("People", stats.get("people", 0))
            st.metric("Helmet", stats.get("helmet", 0))
            st.metric("No Helmet", stats.get("no_helmet", 0))
            st.metric("Mask", stats.get("mask", 0))
            st.metric("No Mask", stats.get("no_mask", 0))
            
            st.subheader("Logs")
            logs = stats.get("logs", [])
            for log in logs:
                p = log.get('Person', 'N/A')
                h = log.get('Helmet', False)
                m = log.get('Mask', False)
                st.write(f"Person {p} | Helmet {'✅' if h else '❌'} | Mask {'✅' if m else '❌'}")

# ---------------- VIDEO ---------------- #
else:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.subheader("Original Video")
        st.video(uploaded_video)
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_input.write(uploaded_video.read())
                temp_input.close()
                os.makedirs("outputs", exist_ok=True)
                output_path = os.path.join("outputs", "processed_video.mp4")
                process_video(temp_input.name, output_path)
                if os.path.exists(temp_input.name):
                    os.remove(temp_input.name)
            st.success("Processing Complete!")
            st.subheader("Processed Video")
            st.video(output_path)
            with open(output_path, "rb") as video_file:
                st.download_button("Download Processed Video", video_file, "processed_video.mp4", "video/mp4")