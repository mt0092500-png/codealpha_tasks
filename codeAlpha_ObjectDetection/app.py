
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Object Detection & Tracking")
st.write("Upload a video to detect and track objects using AI.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    if st.button("Start Detection"):
        model = load_model()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_file:
            temp_file.write(uploaded_file.read())
            input_path = temp_file.name

        output_path = input_path.replace(".mp4", "_tracked.mp4")

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        progress = st.progress(0)
        status = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                frame_count += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

            status.success("Detection and tracking completed!")

        finally:
            cap.release()
            writer.release()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.subheader("Processed Video")
            st.video(output_path)

            with open(output_path, "rb") as video:
                st.download_button(
                    "Download Processed Video",
                    data=video.read(),
                    file_name="tracked_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Could not process this video. Please try another.")

        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
else:
    st.info("Please upload a video to get started.")
