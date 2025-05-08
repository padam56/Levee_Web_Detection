import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_addons as tfa
import gc
import os
import time
from PIL import Image
import tempfile
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import json


# Import custom modules (assuming they exist)
from metrics import mcc_loss, mcc_metric, dice_coef, dice_loss, f1, tversky, tversky_loss, focal_tversky_loss, bce_dice_loss_new, jaccard, bce_dice_loss
from SandBoilNet import PCALayer, spatial_pooling_block, attention_block, initial_conv2d_bn, conv2d_bn, iterLBlock, decoder_block

# --- GPU Management ---
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
    tf.keras.backend.clear_session()
    gc.collect()

# --- Custom Objects ---
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

# --- Model Loading ---
@st.cache_resource
def load_model_by_type(model_type):
    model_paths = {
        'sandboil': 'sandboil_best_model.h5',
        'seepage': 'seepage_best_model.h5',
        'rutting': 'rutting_best_model.h5'
        # Add paths for other models here
    }
    if model_type in model_paths:
        return load_model(model_paths[model_type], custom_objects=custom_objects)
    raise ValueError(f"No model defined for {model_type}")

def load_selected_models(selected_types):
    models = {}
    for model_type in selected_types:
        models[model_type] = load_model_by_type(model_type)
    return models

# --- Automatic Thresholding ---
def otsu_threshold(predictions):
    """Apply Otsu's method to prediction probabilities."""
    flat_preds = (predictions * 255).astype(np.uint8).flatten()
    threshold, _ = cv2.threshold(flat_preds, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold / 255.0

def adaptive_threshold(predictions, block_size=11, C=2):
    """Apply adaptive thresholding to prediction probabilities."""
    preds_uint8 = (predictions * 255).astype(np.uint8)
    thresh = cv2.adaptiveThreshold(preds_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    return thresh / 255.0

def get_auto_threshold(predictions, method='otsu'):
    if method == 'otsu':
        return otsu_threshold(predictions)
    elif method == 'adaptive':
        return adaptive_threshold(predictions)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

# --- Preprocessing ---
@st.cache_data
def preprocess_image(image, model_type, resolution_factor=1.0, brightness_factor=0, contrast_factor=0,
                     blur_amount=1, edge_detection=False, flip_horizontal=False, flip_vertical=False,
                     rotate_angle=0):
    dims = {'sandboil': (512, 512), 'seepage': (256, 256)}  # Add other model dimensions
    input_width, input_height = dims.get(model_type, (512, 512))
    
    image_resized = cv2.resize(image, (input_width, input_height))
    if resolution_factor != 1.0:
        new_width, new_height = int(image.shape[1] * resolution_factor), int(image.shape[0] * resolution_factor)
        image = cv2.resize(image, (new_width, new_height))
    if brightness_factor != 0 or contrast_factor != 0:
        image = cv2.convertScaleAbs(image, alpha=1 + contrast_factor / 100.0, beta=brightness_factor)
    if blur_amount > 1:
        image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
    if edge_detection:
        image = cv2.Canny(image, 100, 200)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)
    if rotate_angle != 0:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    return np.expand_dims(image_resized / 255.0, axis=0)

# --- Overlap Resolution ---
def constrained_flood_fill(sandboil_mask, seepage_mask, distance_threshold):
    sandboil_contours, _ = cv2.findContours(sandboil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seepage_contours, _ = cv2.findContours(seepage_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    updated_sandboil_mask = sandboil_mask.copy()
    updated_seepage_mask = seepage_mask.copy()
    for sandboil_cnt in sandboil_contours:
        for seepage_cnt in seepage_contours:
            dist = cv2.pointPolygonTest(sandboil_cnt, tuple(map(int, seepage_cnt[0][0])), measureDist=True)
            if abs(dist) < distance_threshold:
                cv2.drawContours(updated_seepage_mask, [seepage_cnt], -1, 0, -1)
    return updated_sandboil_mask, updated_seepage_mask

def remove_nearby_seepage(sandboil_mask, seepage_mask, distance_threshold):
    kernel_size = distance_threshold if distance_threshold % 2 == 1 else distance_threshold + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sandboil_dilated = cv2.dilate(sandboil_mask, kernel, iterations=1)
    updated_seepage_mask = np.where(sandboil_dilated > 0, 0, seepage_mask)
    return sandboil_mask, updated_seepage_mask

def prioritize_sandboil_over_seepage(sandboil_mask, seepage_mask):
    combined_mask = np.where(sandboil_mask > 0, 1, 0)
    seepage_mask_resized = cv2.resize(seepage_mask, (sandboil_mask.shape[1], sandboil_mask.shape[0]))
    updated_seepage_mask = np.where(combined_mask == 1, 0, seepage_mask_resized)
    return sandboil_mask, updated_seepage_mask

def remove_smaller_overlaps(mask1, mask2, distance_threshold):
    contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    updated_mask1, updated_mask2 = mask1.copy(), mask2.copy()
    for cnt1 in contours1:
        area1 = cv2.contourArea(cnt1)
        for cnt2 in contours2:
            area2 = cv2.contourArea(cnt2)
            dist = cv2.pointPolygonTest(cnt1, (int(cnt2[0][0][0]), int(cnt2[0][0][1])), measureDist=True)
            if abs(dist) < distance_threshold:
                if area1 < area2:
                    cv2.drawContours(updated_mask1, [cnt1], -1, 0, -1)
                else:
                    cv2.drawContours(updated_mask2, [cnt2], -1, 0, -1)
    return updated_mask1, updated_mask2

def resolve_overlaps(masks, distance_threshold, priority_order=None):
    """Generalized overlap resolution for multiple masks."""
    if len(masks) <= 1:
        return masks
    updated_masks = masks.copy()
    types = list(masks.keys())
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            t1, t2 = types[i], types[j]
            if t1 == 'sandboil' and t2 == 'seepage':
                updated_masks[t1], updated_masks[t2] = constrained_flood_fill(updated_masks[t1], updated_masks[t2], distance_threshold)
                updated_masks[t1], updated_masks[t2] = remove_smaller_overlaps(updated_masks[t1], updated_masks[t2], distance_threshold)
                updated_masks[t1], updated_masks[t2] = prioritize_sandboil_over_seepage(updated_masks[t1], updated_masks[t2])
                updated_masks[t1], updated_masks[t2] = remove_nearby_seepage(updated_masks[t1], updated_masks[t2], distance_threshold)
            else:
                updated_masks[t1], updated_masks[t2] = remove_smaller_overlaps(updated_masks[t1], updated_masks[t2], distance_threshold)
    return updated_masks

# --- Visualization ---
def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    if len(mask_resized.shape) == 2:
        mask_colored = np.stack([mask_resized] * 3, axis=-1)
        mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])
    else:
        mask_colored = mask_resized
    return cv2.addWeighted(image.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)

def draw_bounding_boxes(frame, mask, color=(0, 255, 0)):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    contours, _ = cv2.findContours((mask_resized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
    return frame

def get_color_for_detection(detection_type):
    colors = {
        'sandboil': (0, 255, 0),      # Green
        'seepage': (255, 105, 180),     # Pink
        'crack': (0, 0, 255),         # Blue
        'potholes': (255, 165, 0),    # Orange
        'encroachment': (255, 0, 0),  # Red
        'rutting': (128, 0, 128),     # Purple
        'animal_burrow': (165, 42, 42),  # Brown
        'vegetation': (0, 100, 0)     # Dark Green
    }
    return colors.get(detection_type, (255, 255, 255))  # Default white

# --- Core Processing ---
def process_frame(frame, models, thresholds, threshold_types, detection_choice, overlay_intensity, distance_threshold):
    masks = {}
    # 1) Run inference & threshold each model
    for detection_type, model in models.items():
        preds = apply_model(model, frame, detection_type)
        thresh = (thresholds[detection_type]
                  if threshold_types[detection_type] == 'Manual'
                  else get_auto_threshold(preds, 'otsu'))
        mask = (preds > thresh).astype(np.uint8)
        if detection_type == 'seepage':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)
        masks[detection_type] = mask

    # 2) Resolve overlaps if needed
    if len(masks) > 1:
        masks = resolve_overlaps(masks, distance_threshold)

    # 3A) Bounding Box mode: draw all boxes on a single copy
    if detection_choice == "Bounding Box":
        boxed_frame = frame.copy()
        for dt, mask in masks.items():
            color = get_color_for_detection(dt)
            boxed_frame = draw_bounding_boxes(boxed_frame, mask, color)
        return boxed_frame, masks

    # 3B) Overlay mode: stack your translucent masks as before
    combined_overlay = np.zeros_like(frame, dtype=np.uint8)
    for dt, mask in masks.items():
        color = get_color_for_detection(dt)
        overlay = overlay_mask_on_image(frame, mask, overlay_intensity, color)
        combined_overlay = cv2.addWeighted(combined_overlay, 1.0, overlay, 1.0, 0)
    return combined_overlay, masks


def apply_model(model, image, model_type):
    processed_image = preprocess_image(
        image.copy(), model_type,
        resolution_factor=st.session_state.get('resolution_factor', 1.0),
        brightness_factor=st.session_state.get('brightness_factor', 0),
        contrast_factor=st.session_state.get('contrast_factor', 0),
        blur_amount=st.session_state.get('blur_amount', 1),
        edge_detection=st.session_state.get('edge_detection', False),
        flip_horizontal=st.session_state.get('flip_horizontal', False),
        flip_vertical=st.session_state.get('flip_vertical', False),
        rotate_angle=st.session_state.get('rotate_angle', 0)
    )
    predictions = model.predict(processed_image)
    return np.squeeze(predictions)

# --- UI Setup ---
st.title("Levee Fault Detection WebApp")

# Detection Type Selection
st.sidebar.write("### Select Detection Types")
detection_types = {
    'sandboil': st.sidebar.checkbox("Detect Sandboils"),
    'seepage': st.sidebar.checkbox("Detect Seepage"),
    'crack': st.sidebar.checkbox("Detect Crack"),
    'potholes': st.sidebar.checkbox("Detect Potholes"),
    'encroachment': st.sidebar.checkbox("Detect Encroachment"),
    'rutting': st.sidebar.checkbox("Detect Rutting"),
    'animal_burrow': st.sidebar.checkbox("Detect Animal Burrow"),
    'vegetation': st.sidebar.checkbox("Detect Vegetation")
}
selected_types = [t for t, selected in detection_types.items() if selected]

# Detection Choice
detection_choice = st.sidebar.radio("Choose Detection Type", ("Overlay", "Bounding Box"))

# Threshold Settings
st.sidebar.write("### Threshold Settings")
thresholds = {}
threshold_types = {}
for detection_type in selected_types:
    threshold_types[detection_type] = st.sidebar.radio(f"{detection_type.capitalize()} Threshold Type", ["Manual", "Automatic"], key=f"{detection_type}_thresh_type")
    if threshold_types[detection_type] == "Manual":
        default = 0.5 if detection_type == 'sandboil' else 0.98 if detection_type == 'seepage' else 0.5
        thresholds[detection_type] = st.sidebar.slider(f"{detection_type.capitalize()} Confidence Threshold", 0.0, 1.0, default, 0.01, key=f"{detection_type}_thresh")
    else:
        thresholds[detection_type] = None

# Other Settings
distance_threshold = st.sidebar.slider("Distance Threshold Between Overlays (pixels)", 5, 100, 20)
overlay_intensity = st.sidebar.slider("Overlay Intensity", 0.0, 1.0, 0.5)
with st.sidebar.form("image_processing"):
    st.write("### Customize Image Processing")
    st.session_state['resolution_factor'] = st.slider("Resolution Scaling Factor", 0.1, 2.0, 1.0)
    st.session_state['brightness_factor'] = st.slider("Brightness", -100, 100, 0)
    st.session_state['contrast_factor'] = st.slider("Contrast", -100, 100, 0)
    st.session_state['blur_amount'] = st.slider("Gaussian Blur (Kernel Size)", 1, 15, 1, step=2)
    st.session_state['edge_detection'] = st.checkbox("Edge Detection (Canny)")
    st.session_state['flip_horizontal'] = st.checkbox("Flip Horizontally")
    st.session_state['flip_vertical'] = st.checkbox("Flip Vertically")
    st.session_state['rotate_angle'] = st.slider("Rotate Image (Degrees)", -180, 180, 0)
    st.form_submit_button("Apply")

# Legend
st.sidebar.write("### Legend")
legend_html = "".join([f"<span style='color:rgb{get_color_for_detection(t)}'>■ {t.capitalize()}</span><br>" for t in detection_types])
st.sidebar.markdown(legend_html, unsafe_allow_html=True)

# Processing Choice
processing_choice = st.radio("Choose Input Type", ("Image", "Video"))

# Load Models
if selected_types:
    models = load_selected_models(selected_types)
if "save_success" not in st.session_state:
    st.session_state.save_success = False



# —— Imports at top of file ——
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import time

# —— Session-state init ——
if "canvas_counter" not in st.session_state:
    st.session_state.canvas_counter = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []

@st.cache_data(show_spinner=False)
def get_cached_detection(image_array, selected_types, thresholds, threshold_types,
                          detection_choice, overlay_intensity, distance_threshold):
    return process_frame(
        image_array, models, thresholds, threshold_types,
        detection_choice, overlay_intensity, distance_threshold
    )



# —— IMAGE PROCESSING BRANCH ——
if processing_choice == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg','png'])
    if not uploaded_image:
        st.warning("Please upload an image to get started.")
    else:
        # Decode & inference
        img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        if not selected_types:
            st.warning("Please select at least one detection type.")
        else:
            proc_img, pred_masks = get_cached_detection(
                img, selected_types, thresholds, threshold_types,
                detection_choice, overlay_intensity, distance_threshold
            )

            # Show overlay + download
            st.image(proc_img[..., ::-1], caption="Detection Results", use_container_width=True)
            ret, buf = cv2.imencode('.png', proc_img)
            if ret:
                st.download_button("Download Processed Image", buf.tobytes(), "processed_image.png", "image/png")

            # Show masks
            with st.expander("Show Predicted Masks"):
                for dt, mask in pred_masks.items():
                    disp = (mask * 255).astype(np.uint8)
                    st.image(disp, caption=f"{dt.capitalize()} Mask", use_container_width=True)

            # Re-annotation toggle
            if st.checkbox("🖊️ Do you want to re-annotate this overlay?"):
                st.markdown("### ✍️ Human-in-the-Loop Re-annotation")
                target = st.selectbox("Which detection are you correcting?", [t.capitalize() for t in selected_types])

                # Delete Last / Clear All
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🗑️ Delete Last Annotation"):
                        if st.session_state.annotations:
                            st.session_state.annotations.pop()
                            st.session_state.canvas_counter += 1
                with c2:
                    if st.button("🗑️ Clear All Annotations"):
                        st.session_state.annotations = []
                        st.session_state.canvas_counter += 1

                st.markdown("**Instructions:** Click to place points. **Double-click** to close the polygon. Use buttons above to delete.")

                # Prepare canvas
                overlay = Image.fromarray(proc_img[..., ::-1])
                max_w = 600
                scale = min(1.0, max_w/overlay.width)
                w, h = int(overlay.width*scale), int(overlay.height*scale)

                # Always render the canvas
                canvas_result = st_canvas(
                    background_image=overlay,
                    drawing_mode="polygon",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    fill_color="rgba(255,0,0,0.2)",
                    width=w, height=h,
                    key=f"canvas_{st.session_state.canvas_counter}",
                    initial_drawing={"objects": st.session_state.annotations}
                )

                # Grab shapes into session state
                if canvas_result.json_data and "objects" in canvas_result.json_data:
                    st.session_state.annotations = canvas_result.json_data["objects"]


                # Save Annotations button
                if st.button("💾 Save Annotations"):
                    shapes = st.session_state.annotations
                    if not shapes:
                        st.warning("No polygons to save—draw one first!")
                    else:
                        path = "annotation_feedback.json"
                        try:
                            with open(path, "r") as f:
                                data = json.load(f)
                        except FileNotFoundError:
                            data = []

                        for obj in shapes:
                            if obj.get("type") == "polygon":
                                data.append({
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "target": target,
                                    "coords": obj["path"]
                                })

                        with open(path, "w") as f:
                            json.dump(data, f, indent=2)

                        # set our flag and keep annotations intact for review
                        st.session_state.save_success = True

                # **After** that Save button, add this once per run to show the saved output:
                if st.session_state.save_success:
                    st.success(f"✅ Saved {len(st.session_state.annotations)} annotation(s)!")
                    # show and offer download
                    try:
                        with open("annotation_feedback.json", "r") as f:
                            saved_data = json.load(f)
                        st.json(saved_data)
                        st.download_button(
                            "Download All Annotations",
                            json.dumps(saved_data, indent=2),
                            "annotation_feedback.json",
                            "application/json"
                        )
                    except Exception as e:
                        st.error(f"Could not read back JSON: {e}")

                    # Optionally let the user clear the success message when they're done:
                    if st.button("Done Reviewing Saved Annotations"):
                        st.session_state.save_success = False




                        
                        
# Video Processing
if processing_choice == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps
        
        timeline_offset = st.sidebar.slider("Start Time (seconds)", 0.0, video_duration, 0.0, 0.1)
        cap.set(cv2.CAP_PROP_POS_MSEC, timeline_offset * 1000)
        speed_multiplier = st.sidebar.slider("Playback Speed Multiplier", 0.5, 4.0, 1.0, 0.1)
        real_time_playback = st.sidebar.checkbox("Real-time Playback", value=True)
        
        frame_placeholder = st.empty()
        processed_frames = []
        frame_delay = 0 if real_time_playback else 1.0 / (fps * speed_multiplier)
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            if selected_types:
                processed_frame, _ = process_frame(frame, models, thresholds, threshold_types, detection_choice, overlay_intensity, distance_threshold)
            else:
                processed_frame = frame
            processed_frames.append(processed_frame.copy())
            frame_placeholder.image(processed_frame[..., ::-1], channels="RGB", use_container_width=True)
            if frame_delay > 0:
                elapsed = time.time() - start_time
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        cap.release()
        os.remove(tfile.name)
        
        if processed_frames and selected_types:
            height, width = processed_frames[0].shape[:2]
            out_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            for frame in processed_frames:
                writer.write(frame)
            writer.release()
            with open(out_path, 'rb') as f:
                st.download_button("Download Processed Video", f.read(), "processed_video.mp4", "video/mp4")
            os.remove(out_path)
    else:
        st.warning("Please upload a video.")
