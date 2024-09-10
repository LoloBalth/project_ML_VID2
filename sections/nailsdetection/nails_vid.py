import streamlit as st
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# WebRTC configuration for STUN server
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN server
    ]
})

# Custom UDP Transport Class for Socket Communication
class MyTransport:
    def __init__(self):
        self._sock = None
        self._protocol = None

    async def initialize_socket(self):
        loop = asyncio.get_event_loop()
        try:
            self._protocol = asyncio.DatagramProtocol()
            self._sock = await loop.create_datagram_endpoint(lambda: self._protocol, local_addr=('localhost', 12345))
            logging.info("Socket initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing socket: {e}")

    async def send_data(self, data, addr):
        if self._sock is None:
            await self.initialize_socket()
        try:
            if self._sock is not None:
                self._sock.sendto(data, addr)
            else:
                logging.error("Socket is not initialized.")
        except Exception as e:
            logging.error(f"Error sending data: {e}")

# Transformer class for video processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model_id: str, confidence_threshold: float):
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="yVnoBqLgjl2tRxWIWMvx"
        )
    
    def transform(self, frame: np.ndarray) -> np.ndarray:
        try:
            # Convert the frame to PIL image for inference
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result: Dict[str, Any] = self.client.infer(frame_pil, model_id=self.model_id)
            output_dict: Dict[str, Any] = result
            
            def filter_predictions(predictions: List[Dict[str, Any]], confidence_threshold: float) -> List[Dict[str, Any]]:
                return [
                    pred for pred in predictions
                    if pred.get('confidence', 0) >= confidence_threshold
                ]
            
            # Filter predictions by confidence threshold
            filtered_predictions = filter_predictions(output_dict.get('predictions', []), self.confidence_threshold)
            
            if filtered_predictions:
                frame = self.draw_polygons_on_frame(frame, filtered_predictions)
        
        except Exception as e:
            st.error(f"Error processing frame: {e}")
        return frame

    def draw_polygons_on_frame(self, frame: np.ndarray, predictions: List[Dict[str, Any]]) -> np.ndarray:
        for prediction in predictions:
            if 'points' in prediction:
                points = prediction['points']
                polygon_points = [(int(p['x']), int(p['y'])) for p in points]
                
                if len(polygon_points) >= 3:
                    # Draw polygons on the frame
                    cv2.polylines(frame, [np.array(polygon_points)], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Add class name and confidence level as text
                    cv2.putText(frame, f"{prediction['class']} ({prediction['confidence']:.2f})",
                                (polygon_points[0][0], polygon_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
        return frame

# Sidebar for setting detection confidence
def side_bar_nails():
    st.write('#### Set detection confidence threshold.')
    
    confidence_threshold: float = st.slider(
        'Confidence threshold slider',
        0.0, 1.0, 0.5, 0.01, key="nailsvid",
        label_visibility="hidden"  # Hide label but keep accessibility
    )
    st.write(f"Confidence threshold set to: {confidence_threshold}")
    return confidence_threshold

# Main page function for handling video streaming
def nails_page():
    confidence_threshold = side_bar_nails()
    model_id = "laurent/1"
    
    # Initialize session state keys
    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False

    col1, col2 = st.columns(2)

    # Start webcam button
    with col1:
        if st.button("Run"):
            st.session_state["run_webcam"] = True

    # Stop webcam button
    with col2:
        if st.button("Stop"):
            st.session_state["run_webcam"] = False

    # Run the webcam stream if active
    if st.session_state["run_webcam"]:
        try:
            webrtc_streamer(
                key="nails-detection",
                video_transformer_factory=lambda: VideoTransformer(model_id, confidence_threshold),
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": True},
                async_processing=True,
            )
        except Exception as e:
            st.error(f"WebRTC session failed: {e}")
    else:
        logging.info("WebRTC streaming stopped.")

# Handling asyncio exceptions to prevent 'NoneType' errors
def handle_asyncio_exceptions(loop):
    def handle_exception(loop, context):
        # Log the error
        msg = context.get("exception", context["message"])
        st.error(f"AsyncIO Exception: {msg}")

    loop.set_exception_handler(handle_exception)

# Main function for Streamlit app
def main():
    st.title("Nail Detection with WebRTC")

    # Handle asyncio exceptions
    loop = asyncio.get_event_loop()
    handle_asyncio_exceptions(loop)

#     # Run the nails detection page
#     nails_page()

# # Run the app
# if __name__ == "__main__":
#     main()
