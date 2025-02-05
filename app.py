import streamlit as st
import cv2
import subprocess
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import backend as K
from metrics import mcc_loss, mcc_metric, dice_coef, dice_loss, f1, tversky, tversky_loss, focal_tversky_loss, bce_dice_loss_new, jaccard, bce_dice_loss
from SandBoilNet import PCALayer, spatial_pooling_block, attention_block, initial_conv2d_bn, conv2d_bn, iterLBlock, decoder_block
# SandboilNet_Dropout, old_attention_block
import gc
import os
import time
from io import BytesIO
from PIL import Image
import zipfile

import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import os
import time

# #--- GPU Management ---
def kill_gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], stdout=subprocess.PIPE)
    pids = result.stdout.decode('utf-8').strip().split('\n')
    for pid in pids:
        if pid.isdigit():
            try:
                os.kill(int(pid), 9)
                print(f"Killed process with PID: {pid}")
            except Exception as e:
                print(f"Could not kill process {pid}: {e}")
#kill_gpu_processes()

def enable_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs.")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")

enable_memory_growth()

def clear_tf_memory():
    K.clear_session()
    gc.collect()
    print("Cleared TensorFlow session and garbage collected.")

# Sidebar slider for distance threshold
distance_threshold = st.sidebar.slider("Distance Threshold Between Overlays (pixels)", min_value=5, max_value=50, value=20)

def constrained_flood_fill(sandboil_mask, seepage_mask, distance_threshold):
    # Find contours for both masks
    sandboil_contours, _ = cv2.findContours(sandboil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seepage_contours, _ = cv2.findContours(seepage_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create copies of masks to modify
    updated_sandboil_mask = sandboil_mask.copy()
    updated_seepage_mask = seepage_mask.copy()

    # Iterate through each sandboil contour
    for sandboil_cnt in sandboil_contours:
        for seepage_cnt in seepage_contours:
            # Calculate minimum distance between the two contours
            dist = cv2.pointPolygonTest(sandboil_cnt, tuple(map(int, seepage_cnt[0][0])), measureDist=True)
            if abs(dist) < distance_threshold:  # If within threshold
                # Remove the overlapping or nearby seepage region
                cv2.drawContours(updated_seepage_mask, [seepage_cnt], -1, 0, -1)  # Erase from seepage mask

    return updated_sandboil_mask, updated_seepage_mask

def prioritize_sandboil_over_seepage(sandboil_mask, seepage_mask):
    """
    Ensures that sandboil regions take precedence over seepage regions in overlapping areas.
    """
    combined_mask = np.where(sandboil_mask > 0, 1, 0)  # Sandboil takes priority
    
    # Resize seepage_mask to match sandboil_mask
    seepage_mask_resized = cv2.resize(seepage_mask, (sandboil_mask.shape[1], sandboil_mask.shape[0]))

    # Step 2: Remove overlapping seepage pixels
    updated_seepage_mask = np.where(combined_mask == 1, 0, seepage_mask_resized)


    return sandboil_mask, updated_seepage_mask

def remove_smaller_overlaps(mask1, mask2, distance_threshold):
    # Find contours for both masks
    contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create copies of masks to modify
    updated_mask1 = mask1.copy()
    updated_mask2 = mask2.copy()

    # Iterate through each contour in mask1
    for cnt1 in contours1:
        area1 = cv2.contourArea(cnt1)
        for cnt2 in contours2:
            area2 = cv2.contourArea(cnt2)

            # Calculate minimum distance between contours
            # dist = cv2.pointPolygonTest(cnt1, tuple(cnt2[0][0]), measureDist=True)
            dist = cv2.pointPolygonTest(cnt1, (int(cnt2[0][0][0]), int(cnt2[0][0][1])), measureDist=True)

            if abs(dist) < distance_threshold:
                # If regions overlap, remove the smaller one
                if area1 < area2:
                    cv2.drawContours(updated_mask1, [cnt1], -1, 0, -1)  # Erase from mask1
                else:
                    cv2.drawContours(updated_mask2, [cnt2], -1, 0, -1)  # Erase from mask2

    return updated_mask1, updated_mask2


# Define custom objects for loading models
custom_objects = {
    'Addons>GroupNormalization': tfa.layers.GroupNormalization,
    'mcc_loss': mcc_loss,
    'mcc_metric': mcc_metric,
    'dice_coef': dice_coef,
    'dice_loss': dice_loss,
    'f1': f1,
    'tversky': tversky,
    'tversky_loss': tversky_loss,
    'focal_tversky_loss': focal_tversky_loss,
    'bce_dice_loss_new': bce_dice_loss_new,
    'jaccard': jaccard,
    'PCALayer': PCALayer,
    'bce_dice_loss': bce_dice_loss
}

@st.cache_resource
def load_sandboil_model():
    return load_model('sandboil_best_model.h5', custom_objects=custom_objects)

@st.cache_resource
def load_seepage_model():
    return load_model('seepage_best_model.h5', custom_objects=custom_objects)

# Allow user to choose between image or video processing
processing_choice = st.radio("Choose the type of input you want to process:", ("Image", "Video"))
# Allow user to choose between overlay or bounding box detection
detection_choice = st.radio("Choose the type of detection:", ("Overlay", "Bounding Box"))

# UI Setup for Streamlit
st.title("Levee Fault Detection WebApp")

# Checkboxes for model selection
sandboil_selected = st.checkbox("Detect Sandboils")
seepage_selected = st.checkbox("Detect Seepage")
crack_selected = st.checkbox("Detect Crack")
potholes_selected = st.checkbox("Detect Potholes")
encroachment_selected = st.checkbox("Detect Encroachment")
rutting_selected = st.checkbox("Detect Rutting")
animal_burrow_selected = st.checkbox("Detect Animal Burrow")
vegetation_selected = st.checkbox("Detect Vegetation")

# Sidebar legend for detection types and their corresponding colors
@st.cache_data
def render_legend():
    legend_html = """
    <span style='color:green'>■ Sandboils</span><br>
    <span style='color:pink'>■ Seepage</span><br>
    <span style='color:blue'>■ Crack</span><br>
    <span style='color:orange'>■ Potholes</span><br>
    <span style='color:red'>■ Encroachment</span><br>
    <span style='color:purple'>■ Rutting</span><br>
    <span style='color:brown'>■ Animal Burrow</span><br>
    <span style='color:darkgreen'>■ Vegetation</span>
    """
    return legend_html

st.sidebar.write("### Legend")
st.sidebar.markdown(render_legend(), unsafe_allow_html=True)

# Optional: Add logic to highlight the selected items in the main content area
if sandboil_selected:
    st.write("Sandboils detection is selected.")
if seepage_selected:
    st.write("Seepage detection is selected.")
if crack_selected:
    st.write("Crack detection is selected.")
if potholes_selected:
    st.write("Potholes detection is selected.")
if encroachment_selected:
    st.write("Encroachment detection is selected.")
if rutting_selected:
    st.write("Rutting detection is selected.")
if animal_burrow_selected:
    st.write("Animal Burrow detection is selected.")
if vegetation_selected:
    st.write("Vegetation detection is selected.")
        
@st.cache_data
def preprocess_image(image, model_type, resolution_factor=1.0, brightness_factor=0, contrast_factor=0,
                     blur_amount=1, edge_detection=False, flip_horizontal=False, flip_vertical=False,
                     rotate_angle=0):
    """
    Preprocess the image based on the selected model type and additional transformations.
    """
    # Set input dimensions dynamically based on the model type
    if model_type == "sandboil":
        input_width, input_height = 512, 512  # Sandboil model dimensions
    elif model_type == "seepage":
        input_width, input_height = 256, 256  # Seepage model dimensions
    else:
        raise ValueError("Invalid model type. Choose 'sandboil' or 'seepage'.")

    # Resize image to match the model's expected input size
    image_resized = cv2.resize(image, (input_width, input_height))

    # Apply resolution scaling
    new_width = int(image.shape[1] * resolution_factor)
    new_height = int(image.shape[0] * resolution_factor)
    image = cv2.resize(image, (new_width, new_height))

    # Brightness and contrast adjustments
    if brightness_factor != 0 or contrast_factor != 0:
        image = cv2.convertScaleAbs(image, alpha=1 + contrast_factor / 100.0, beta=brightness_factor)

    # Gaussian blur
    if blur_amount > 1:
        image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # Edge detection
    if edge_detection:
        image = cv2.Canny(image, threshold1=100, threshold2=200)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

    # Flip horizontally or vertically
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)

    # Rotate the image
    if rotate_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    # Normalize and add batch dimension
    return np.expand_dims(image_resized / 255.0, axis=0)

def process_frame(frame, input_width, input_height):
    """Process a single frame."""
    frame_resized = cv2.resize(frame, (input_width, input_height))
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    K.clear_session()
    return np.expand_dims(frame_normalized, axis=0)

def apply_model_to_frame(model, frame, input_width, input_height):
    """Apply the model to a single frame."""
    processed_frame = process_frame(frame, input_width, input_height)
    predictions = model.predict(processed_frame)
    predicted_mask = np.squeeze(predictions) 
    K.clear_session()
    return predicted_mask

def overlay_mask_on_frame(frame, mask, alpha=0.5, color=(0, 255, 0)):
    """Overlay the segmentation mask on the original frame."""
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_colored = np.stack([mask_resized] * 3, axis=-1) if len(mask_resized.shape) == 2 else mask_resized
    mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])
    overlaid_frame = cv2.addWeighted(frame.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)
    K.clear_session()
    return overlaid_frame

def draw_bounding_boxes(frame, mask):
    """Draw bounding boxes around detected regions."""
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    contours, _ = cv2.findContours((mask_resized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)  # Green bounding box
    K.clear_session()
    return frame 

# Sidebar form for sliders
with st.sidebar.form("slider_form"):
    st.write("### Customize Image Processing")

    # Sliders for image processing customization
    resolution_factor = st.slider("Adjust Image Resolution Scaling Factor", 0.1, 2.0, 1.0)
    brightness_factor = st.slider("Adjust Brightness", -100, 100, 0)
    contrast_factor = st.slider("Adjust Contrast", -100, 100, 0)
    blur_amount = st.slider("Apply Gaussian Blur (Kernel Size)", 1, 15, step=2)
    edge_detection = st.checkbox("Apply Edge Detection (Canny)")
    flip_horizontal = st.checkbox("Flip Horizontally")
    flip_vertical = st.checkbox("Flip Vertically")
    rotate_angle = st.slider("Rotate Image (Degrees)", -180, 180, step=1)

    # Submit button
    submitted = st.form_submit_button("Submit")

def apply_model(selected_model, image, model_type):
    """
    Apply the selected model to a preprocessed image.
    """
    # Preprocess the image based on the model type
    processed_image = preprocess_image(
        image=image.copy(),
        model_type=model_type,
        resolution_factor=resolution_factor,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        blur_amount=blur_amount,
        edge_detection=edge_detection,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        rotate_angle=rotate_angle
    )
    
    # Predict using the selected model
    predictions = selected_model.predict(processed_image)
    # Remove batch and channel dimensions if necessary
    predicted_mask = np.squeeze(predictions)
    # Clear session and invoke garbage collection to free up GPU memory
    K.clear_session()
    gc.collect()
    return predicted_mask

def resolve_overlaps(sandboil_mask, seepage_mask, distance_threshold):
    """
    Enhanced overlap resolution combining constrained flood fill, smaller overlap removal,
    and prioritization of sandboil regions over seepage.
    """
    # Step 1: Apply constrained flood fill
    sandboil_mask, seepage_mask = constrained_flood_fill(sandboil_mask, seepage_mask, distance_threshold)

    # Step 2: Remove smaller overlaps
    sandboil_mask, seepage_mask = remove_smaller_overlaps(sandboil_mask, seepage_mask, distance_threshold)

    # Step 3: Prioritize sandboil regions over seepage
    sandboil_mask, seepage_mask = prioritize_sandboil_over_seepage(sandboil_mask, seepage_mask)

    return sandboil_mask, seepage_mask


# Function to overlay mask on image with color and intensity options
def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    # Ensure mask is in the same size as the image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Convert grayscale mask to color if needed (e.g., binary or single-channel)
    if len(mask_resized.shape) == 2:  # If it's a single-channel grayscale or binary mask
        mask_colored = np.stack([mask_resized] * 3, axis=-1)  # Convert to 3-channel RGB
        
        # Apply custom color based on detection type
        mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])  # Use selected color
    
    else:
        mask_colored = mask_resized
    
    # Blend images using addWeighted
    overlaid_image = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)
    K.clear_session()
    return overlaid_image

# Slider to control overlay intensity (transparency)
overlay_intensity = st.sidebar.slider("Overlay Intensity", 0.0, 1.0, 0.5)

# === Modified Helper Functions (remove per-frame session clearing) ===

def process_frame(frame, input_width, input_height):
    frame_resized = cv2.resize(frame, (input_width, input_height))
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    # Removed: K.clear_session()
    return np.expand_dims(frame_normalized, axis=0)

def apply_model_to_frame(model, frame, input_width, input_height):
    processed_frame = process_frame(frame, input_width, input_height)
    predictions = model.predict(processed_frame)
    predicted_mask = np.squeeze(predictions)
    # Removed: K.clear_session()
    return predicted_mask

def overlay_mask_on_frame(frame, mask, alpha=0.5, color=(0, 255, 0)):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_colored = np.stack([mask_resized] * 3, axis=-1) if len(mask_resized.shape) == 2 else mask_resized
    mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])
    overlaid_frame = cv2.addWeighted(frame.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)
    # Removed: K.clear_session()
    return overlaid_frame

def draw_bounding_boxes(frame, mask):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    contours, _ = cv2.findContours((mask_resized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)  # Green bounding box
    # Removed: K.clear_session()
    return frame 

def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    if len(mask_resized.shape) == 2:  # Single-channel mask
        mask_colored = np.stack([mask_resized] * 3, axis=-1)
        mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])
    else:
        mask_colored = mask_resized
    
    overlaid_image = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)
    # Removed: K.clear_session()
    return overlaid_image

def apply_model(selected_model, image, model_type):
    processed_image = preprocess_image(
        image=image.copy(),
        model_type=model_type,
        resolution_factor=resolution_factor,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        blur_amount=blur_amount,
        edge_detection=edge_detection,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        rotate_angle=rotate_angle
    )
    
    predictions = selected_model.predict(processed_image)
    predicted_mask = np.squeeze(predictions)
    # Removed: K.clear_session() and gc.collect()
    return predicted_mask





if processing_choice == "Image":
    # Image Upload Section
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png'])
    if uploaded_image is not None:
        # Read and decode uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image[..., ::-1], caption="Uploaded Image", use_container_width=True)

        # Initialize combined overlay
        combined_overlay = np.zeros_like(image, dtype=np.uint8)

        # Load both models if selected
        sandboil_model = None
        seepage_model = None

        if sandboil_selected:
            st.write("Loading Sandboil model...")
            sandboil_model = load_sandboil_model()

        if seepage_selected:
            st.write("Loading Seepage model...")
            seepage_model = load_seepage_model()

        # Sidebar sliders for thresholds
        sandboil_threshold = st.sidebar.slider(
            "Sandboil Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

        seepage_threshold = st.sidebar.slider(
            "Seepage Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.98,
            step=0.01
        )

        distance_threshold = st.sidebar.slider(
            "Minimum Distance Between Sandboil and Seepage Overlays (pixels)",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

        adjusted_slider_value = seepage_threshold + 1

        # If both sandboil and seepage are selected
        if sandboil_selected and seepage_selected:
            st.write("Running Sandboil and Seepage Detection...")

            # Generate predictions for both models
            sandboil_predictions = apply_model(sandboil_model, image, model_type="sandboil")
            seepage_predictions = apply_model(seepage_model, image, model_type="seepage")

            # Convert predictions to binary masks based on thresholds
            sandboil_mask = (sandboil_predictions > sandboil_threshold).astype(np.uint8)
            seepage_mask = (seepage_predictions > seepage_threshold).astype(np.uint8)

            # Apply erosion to clean up the seepage mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            seepage_mask = cv2.erode(seepage_mask, kernel, iterations=1)

            # Resolve overlaps
            sandboil_mask, seepage_mask = resolve_overlaps(sandboil_mask, seepage_mask, distance_threshold)

            # # Apply morphological operations to clean up the seepage mask
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # seepage_mask = cv2.erode(seepage_mask, kernel, iterations=1)

            # Apply constrained flood fill to resolve overlaps
            sandboil_mask, seepage_mask = constrained_flood_fill(sandboil_mask, seepage_mask, distance_threshold)
            sandboil_mask, seepage_mask = remove_smaller_overlaps(sandboil_mask, seepage_mask, distance_threshold)

            # Overlay both masks on the original image
            sandboil_overlay = overlay_mask_on_image(image, sandboil_mask, alpha=overlay_intensity, color=(0, 255, 0))
            seepage_overlay = overlay_mask_on_image(image, seepage_mask, alpha=overlay_intensity, color=(255, 105, 180))

            combined_overlay = cv2.addWeighted(sandboil_overlay, 0.5, seepage_overlay, 0.5, 0)

        # If only sandboil is selected
        elif sandboil_selected and sandboil_model is not None:
            st.write("Running Sandboil Detection...")
            sandboil_predictions = apply_model(sandboil_model, image, model_type="sandboil")

            # Apply confidence threshold to filter predictions
            sandboil_mask = (sandboil_predictions > sandboil_threshold).astype(np.uint8)

            if detection_choice == "Bounding Box":
                image_with_boxes = draw_bounding_boxes(image.copy(), sandboil_mask)
                combined_overlay = cv2.addWeighted(combined_overlay, 1.0, image_with_boxes, 1.0, 0)
            elif detection_choice == "Overlay":
                sandboil_overlay = overlay_mask_on_image(
                    image,
                    sandboil_mask,
                    alpha=overlay_intensity,
                    color=(0, 255, 0)
                )
                combined_overlay = cv2.addWeighted(combined_overlay, 1.0, sandboil_overlay, 1.0, 0)

        # If only seepage is selected
        elif seepage_selected and seepage_model is not None:
            st.write("Running Seepage Detection...")
            seepage_predictions = apply_model(seepage_model, image, model_type="seepage")

            # Apply confidence threshold and clean up mask using erosion
            seepage_mask = (seepage_predictions > seepage_threshold).astype(np.uint8)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            seepage_mask = cv2.erode(seepage_mask, kernel, iterations=1)

            if detection_choice == "Bounding Box":
                image_with_boxes = draw_bounding_boxes(image.copy(), seepage_mask)
                combined_overlay = cv2.addWeighted(combined_overlay, 1.0, image_with_boxes, 1.0, 0)
            elif detection_choice == "Overlay":
                seepage_overlay = overlay_mask_on_image(
                    image,
                    seepage_mask,
                    alpha=overlay_intensity,
                    color=(255, 105, 180)
                )
                combined_overlay = cv2.addWeighted(combined_overlay, 1.0, seepage_overlay, 1.0, 0)

        # Display the final result
        st.image(combined_overlay[..., ::-1], caption='Detection Results', use_container_width=True)
        # --- Download Button for Processed Image ---
        ret, buffer = cv2.imencode('.png', combined_overlay)
        if ret:
            download_bytes = buffer.tobytes()
            st.download_button("Download Processed Image", data=download_bytes, file_name="processed_image.png", mime="image/png")
    else:
        st.warning("Please upload an image to proceed.")

# === Video Processing Section ===
if processing_choice == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps if fps > 0 else 0

        # Sidebar controls
        timeline_offset = st.sidebar.slider("Select starting video time (seconds)", min_value=0.0, max_value=video_duration, value=0.0, step=0.1)
        cap.set(cv2.CAP_PROP_POS_MSEC, timeline_offset * 1000)
        speed_multiplier = st.sidebar.slider("Playback Speed Multiplier", min_value=0.5, max_value=4.0, value=1.0, step=0.1)
        real_time_playback = st.sidebar.checkbox("Real-time playback (no artificial delay)", value=True)

        # Load models if selected
        sandboil_model = load_sandboil_model() if sandboil_selected else None
        seepage_model = load_seepage_model() if seepage_selected else None

        # Threshold sliders for detections
        sandboil_threshold = st.sidebar.slider("Sandboil Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        seepage_threshold = st.sidebar.slider("Seepage Confidence Threshold", 0.0, 1.0, 0.6, 0.01)
        distance_threshold = st.sidebar.slider("Min Distance Between Overlays (pixels)", 5, 100, 20, 5)

        frame_placeholder = st.empty()
        debug_placeholder = st.empty()

        if real_time_playback:
            frame_delay = 0
        else:
            frame_delay = 1.0 / (fps * speed_multiplier)

        # List to store processed frames for later saving
        processed_frames = []

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if sandboil_selected and seepage_selected:
                sandboil_predictions = apply_model(sandboil_model, frame, model_type="sandboil")
                seepage_predictions = apply_model(seepage_model, frame, model_type="seepage")
                sandboil_mask = (sandboil_predictions > sandboil_threshold).astype(np.uint8)
                seepage_mask = (seepage_predictions > seepage_threshold).astype(np.uint8)
                sandboil_mask, seepage_mask = resolve_overlaps(sandboil_mask, seepage_mask, distance_threshold)
                sandboil_overlay = overlay_mask_on_image(frame, sandboil_mask, alpha=overlay_intensity, color=(0, 255, 0))
                seepage_overlay = overlay_mask_on_image(frame, seepage_mask, alpha=overlay_intensity, color=(255, 105, 180))
                combined_overlay = cv2.addWeighted(sandboil_overlay, 0.5, seepage_overlay, 0.5, 0)
            elif sandboil_selected and sandboil_model is not None:
                sandboil_predictions = apply_model(sandboil_model, frame, model_type="sandboil")
                sandboil_mask = (sandboil_predictions > sandboil_threshold).astype(np.uint8)
                if detection_choice == "Bounding Box":
                    frame_with_boxes = draw_bounding_boxes(frame.copy(), sandboil_mask)
                    combined_overlay = cv2.addWeighted(frame, 1.0, frame_with_boxes, 1.0, 0)
                else:
                    sandboil_overlay = overlay_mask_on_image(frame, sandboil_mask, alpha=overlay_intensity, color=(0, 255, 0))
                    combined_overlay = cv2.addWeighted(frame, 1.0, sandboil_overlay, 1.0, 0)
            elif seepage_selected and seepage_model is not None:
                seepage_predictions = apply_model(seepage_model, frame, model_type="seepage")
                seepage_mask = (seepage_predictions > seepage_threshold).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                seepage_mask = cv2.erode(seepage_mask, kernel, iterations=1)
                if detection_choice == "Bounding Box":
                    frame_with_boxes = draw_bounding_boxes(frame.copy(), seepage_mask)
                    combined_overlay = cv2.addWeighted(frame, 1.0, frame_with_boxes, 1.0, 0)
                else:
                    seepage_overlay = overlay_mask_on_image(frame, seepage_mask, alpha=overlay_intensity, color=(255, 105, 180))
                    combined_overlay = cv2.addWeighted(frame, 1.0, seepage_overlay, 1.0, 0)
            else:
                combined_overlay = frame

            # Optional debug output
            debug_text = ""
            if sandboil_selected and sandboil_model is not None:
                max_sandboil = np.max(sandboil_predictions)
                debug_text += f"Max Sandboil probability: {max_sandboil:.3f}\n"
            if seepage_selected and seepage_model is not None:
                max_seepage = np.max(seepage_predictions)
                debug_text += f"Max Seepage probability: {max_seepage:.3f}\n"
            debug_placeholder.text(debug_text)

            # Append the processed frame for later saving
            processed_frames.append(combined_overlay.copy())

            if frame_delay > 0:
                elapsed_time = time.time() - start_time
                sleep_time = frame_delay - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            frame_placeholder.image(combined_overlay[..., ::-1], channels="RGB", use_container_width=True)

        cap.release()
        os.remove(tfile.name)

        # --- Save Processed Video to File and Provide Download Button ---
        if processed_frames:
            height, width, _ = processed_frames[0].shape
            out_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            for frame in processed_frames:
                writer.write(frame)
            writer.release()
            with open(out_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button("Download Processed Video", video_bytes, file_name="processed_video.mp4", mime="video/mp4")
            os.remove(out_path)
    else:
        st.warning("Please upload a video file to proceed.")
