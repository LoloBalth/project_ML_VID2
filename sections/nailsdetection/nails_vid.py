import streamlit as st
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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
            # Convert the frame to PIL image for the model inference
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result: Dict[str, Any] = self.client.infer(frame_pil, model_id=self.model_id)
            output_dict: Dict[str, Any] = result
            
            def filter_predictions(predictions: List[Dict[str, Any]], confidence_threshold: float) -> List[Dict[str, Any]]:
                return [
                    pred for pred in predictions
                    if pred.get('confidence', 0) >= confidence_threshold
                ]
            
            # Filter predictions based on the confidence threshold
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

def side_bar_nails():
    st.write('#### Set detection confidence threshold.')
    
    # Utiliser un label non vide et le masquer pour l'accessibilité
    confidence_threshold: float = st.slider(
        'Confidence threshold slider',  # Label non vide pour l'accessibilité
        0.0, 1.0, 0.5, 0.01, key="nailsvid",
        label_visibility="hidden"  # Masquer le label mais garder l'accessibilité
    )
    st.write(f"Confidence threshold set to: {confidence_threshold}")
    return confidence_threshold

def nails_page():
    # Récupérer le seuil de confiance depuis la barre latérale
    confidence_threshold = side_bar_nails()
    model_id = "laurent/1"
    
    # Initialiser les clés dans le session_state si elles n'existent pas
    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False
    
    if "nails-detection:frontend" not in st.session_state:
        st.session_state["nails-detection:frontend"] = None

    col1, col2 = st.columns(2)

    # Bouton pour démarrer la webcam
    with col1:
        if st.button("Run"):
            st.session_state["run_webcam"] = True

    # Bouton pour arrêter la webcam
    with col2:
        if st.button("Stop"):
            st.session_state["run_webcam"] = False

    # Lancer la webcam si l'état de la session est actif
    if st.session_state["run_webcam"]:
        try:
            webrtc_streamer(
                key="nails-detection",
                video_transformer_factory=lambda: VideoTransformer(model_id, confidence_threshold)
            )
        except Exception as e:
            st.error(f"WebRTC session failed: {e}")
    else:
        logging.info("WebRTC streaming stopped.")
