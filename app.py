import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import io

# ---------------------
# Utility functions
# ---------------------

def get_face_embedding(img, model_name="VGG-Face", detector_backend="opencv"):
    """
    Given an image (PIL or CV2 format), compute its embedding using DeepFace.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img.convert('RGB'))
    # Represent returns a list of embeddings (one per face)
    # but if enforce_detection=False, it might return an empty or single list. 
    embedding = DeepFace.represent(
        img_path=img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )
    return embedding  # This is a list of embeddings if multiple faces are found.


def detect_and_draw_faces(input_image, known_faces, model_name, detector_backend, distance_metric, threshold):
    """
    Detect faces from the input_image using DeepFace, 
    then compare each face with known_faces to find the best match if within threshold.
    Returns an annotated image (with bounding boxes and name labels).
    """
    if isinstance(input_image, Image.Image):
        # Convert PIL Image to numpy array
        input_image = np.array(input_image.convert('RGB'))

    # Convert to BGR for OpenCV drawing
    bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Use DeepFace's detectFace or analyze function to get face detections
    # The simplest approach: use DeepFace's 'analyze' or 'detectFace' function per detection.
    # Alternatively, we can rely on DeepFace's "verify" or "find" function repeatedly.

    # We'll detect faces and bounding boxes using detectFace:
    #   "DeepFace.extract_faces" returns a list of { "face": face_array, "facial_area": bounding_box, ... }
    detections = DeepFace.extract_faces(
    img_array=input_image,
    target_size=(224, 224),
    detector_backend=detector_backend,
    enforce_detection=False
)


    # For each detected face, we get its bounding box and a cropped face
    for detection in detections:
        facial_area = detection["facial_area"]  # x, y, w, h
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        crop = detection['face']  # This is the cropped face in RGB

        # Compute embedding of this face
        embedding = DeepFace.represent(
            img_path=crop,
            model_name=model_name,
            detector_backend='skip',  # we already have a cropped face, skip detection
            enforce_detection=False
        )

        # Compare with known faces to find the best match
        best_match_name = "Unknown"
        min_distance = float("inf")

        for item in known_faces:
            known_name = item["name"]
            known_embedding = item["embedding"]
            # DeepFace has a built-in function to compute distance:
            distance = DeepFace.distance(
                embedding[0]['embedding'],
                known_embedding[0]['embedding'],
                distance_metric
            )
            if distance < min_distance:
                min_distance = distance
                best_match_name = known_name

        # Check if min_distance is under the threshold
        if min_distance > threshold:
            best_match_name = "Unknown"

        # Draw the bounding box and name on the BGR image
        cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(bgr_image, best_match_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert back to RGB
    annotated_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return annotated_image


def add_new_face(name, face_image, model_name, detector_backend):
    """
    Given a name and an uploaded face image, compute its embedding and store it.
    """
    embedding = DeepFace.represent(
        img_path=np.array(face_image.convert('RGB')),
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )
    return {"name": name, "embedding": embedding}


# ---------------------
# Streamlit App
# ---------------------

def main():
    st.title("Face Recognition with DeepFace")

    # Initialize session_state for known faces
    if "known_faces" not in st.session_state:
        st.session_state["known_faces"] = []

    # Sidebar: Add new face
    st.sidebar.header("Add a New Face")
    uploaded_face = st.sidebar.file_uploader("Upload Face", type=["jpg", "png", "jpeg"])
    new_face_name = st.sidebar.text_input("Name for the Face", "")

    # Sidebar: DeepFace configuration
    st.sidebar.header("DeepFace Configuration")
    model_name = st.sidebar.selectbox("Model", 
                                      ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace"], 
                                      index=0)
    detector_backend = st.sidebar.selectbox("Detector Backend", 
                                            ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe"], 
                                            index=0)
    distance_metric = st.sidebar.selectbox("Distance Metric", 
                                           ["cosine", "euclidean", "euclidean_l2"], 
                                           index=0)
    threshold = st.sidebar.slider("Recognition Threshold", 0.0, 2.0, 0.6, 0.01)

    # Button to add the new face
    if st.sidebar.button("Add Face"):
        if uploaded_face and new_face_name.strip():
            face_image = Image.open(uploaded_face)
            record = add_new_face(new_face_name.strip(), face_image, model_name, detector_backend)
            st.session_state["known_faces"].append(record)
            st.sidebar.success(f"Added face for '{new_face_name}'")
        else:
            st.sidebar.warning("Please provide both image and name.")

    # Sidebar: List known faces
    st.sidebar.header("Registered Faces")
    if len(st.session_state["known_faces"]) == 0:
        st.sidebar.write("No faces registered yet.")
    else:
        for i, face_data in enumerate(st.session_state["known_faces"]):
            st.sidebar.write(f"{i+1}. {face_data['name']}")

    # Main area: Upload an image for face recognition
    st.subheader("Face Recognition on Uploaded Photo")
    recognition_img = st.file_uploader("Upload an image (single or group)", type=["jpg", "jpeg", "png"])

    if recognition_img:
        input_image = Image.open(recognition_img)
        st.image(input_image, caption="Original Image", use_column_width=True)

        if st.button("Detect and Recognize Faces"):
            if len(st.session_state["known_faces"]) == 0:
                st.warning("No registered faces. Please add faces in the sidebar first.")
            else:
                annotated_image = detect_and_draw_faces(
                    input_image=input_image,
                    known_faces=st.session_state["known_faces"],
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    threshold=threshold
                )
                st.image(annotated_image, caption="Detected Faces", use_column_width=True)


if __name__ == "__main__":
    main()
