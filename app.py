import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace  # <-- Make sure your version actually supports DeepFace.represent
from PIL import Image

# ---------------------
# Utility Functions
# ---------------------

def compute_distance(embedding1, embedding2, distance_metric="cosine"):
    """
    Compute the distance between two embedding vectors according to the specified metric.
    Supported metrics: "cosine", "euclidean", "euclidean_l2".
    """
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)

    if distance_metric == "cosine":
        # Cosine similarity -> distance = 1 - cos_sim
        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        cos_sim = dot / (norm1 * norm2 + 1e-8)
        return 1 - cos_sim

    elif distance_metric == "euclidean":
        # Standard L2 distance
        return np.linalg.norm(e1 - e2)

    elif distance_metric == "euclidean_l2":
        # L2 squared (squared Euclidean)
        diff = e1 - e2
        return np.dot(diff, diff)

    else:
        raise ValueError(f"Unsupported distance metric '{distance_metric}'")


def get_face_embedding(img, model_name="VGG-Face", detector_backend="opencv"):
    """
    Given a PIL or CV2 image, compute its embedding using DeepFace.represent.
    Returns a list of dictionaries, each dict has 'embedding' and 'facial_area'.
    """
    if isinstance(img, Image.Image):
        # Convert PIL Image to an RGB NumPy array
        img = np.array(img.convert("RGB"))

    # DeepFace expects BGR arrays when passing an array to 'img_path'.
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    embedding_objs = DeepFace.represent(
        img_path=bgr,  # pass the BGR array
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )
    return embedding_objs  # list of dicts, each with "embedding" key


def add_new_face(name, face_image, model_name, detector_backend):
    """
    Given a name and an uploaded face image (PIL), compute its embedding and return a record dict.
    """
    # Convert the PIL image to BGR
    rgb_array = np.array(face_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    embedding_objs = DeepFace.represent(
        img_path=bgr_array,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )
    # Store them in a record
    record = {"name": name, "embedding": embedding_objs}
    return record


def detect_and_draw_faces(input_image, known_faces, model_name, detector_backend, distance_metric, threshold):
    """
    1) Detect faces in `input_image` via DeepFace.extract_faces.
    2) For each detected face, compute embedding.
    3) Compare embedding to known_faces using compute_distance(...).
    4) Draw bounding box and label with recognized name or "Unknown".
    5) Return the annotated image.
    """

    # Convert PIL to NumPy (RGB)
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image.convert("RGB"))

    # Convert to BGR for both detection and OpenCV drawing
    bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Extract faces
    detections = DeepFace.extract_faces(
        img_path=bgr_image,  # pass the BGR array
        detector_backend=detector_backend,
        enforce_detection=False
    )

    for detection in detections:
        facial_area = detection["facial_area"]  # {x, y, w, h}
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        face_crop = detection["face"]  # The cropped face, by default in 'rgb' if color_face='rgb'

        # Resize if your model expects 224x224
        face_crop = cv2.resize(face_crop, (224, 224))

        # Represent / embed the face (skip detection because it's already cropped)
        face_embed = DeepFace.represent(
            img_path=face_crop,
            model_name=model_name,
            detector_backend="skip",  # we already have the cropped face
            enforce_detection=False
        )

        if len(face_embed) == 0:
            # no embedding
            recognized_name = "Unknown"
        else:
            current_embedding = face_embed[0]["embedding"]

            # Compare with known faces
            recognized_name = "Unknown"
            min_dist = float("inf")

            for known in known_faces:
                known_name = known["name"]
                known_embedding = known["embedding"]
                if len(known_embedding) == 0:
                    continue

                known_vector = known_embedding[0]["embedding"]

                dist = compute_distance(current_embedding, known_vector, distance_metric)
                if dist < min_dist:
                    min_dist = dist
                    recognized_name = known_name

            # Check threshold
            if min_dist > threshold:
                recognized_name = "Unknown"

        # Draw bounding box & name
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            bgr_image,
            recognized_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Convert back to RGB for Streamlit display
    annotated_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return annotated_image


# ---------------------
# Streamlit App
# ---------------------

def main():
    st.title("Face Recognition with DeepFace (Custom Distance)")

    # Initialize session_state for known faces
    if "known_faces" not in st.session_state:
        st.session_state["known_faces"] = []

    # Sidebar - Add new face
    st.sidebar.header("Add a New Face")
    uploaded_face = st.sidebar.file_uploader("Upload Face", type=["jpg", "png", "jpeg"])
    new_face_name = st.sidebar.text_input("Name for the Face", "")

    # Sidebar - DeepFace configuration
    st.sidebar.header("DeepFace Configuration")
    model_name = st.sidebar.selectbox(
        "Model",
        ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "Dlib", "SFace", "GhostFaceNet"],
        index=0
    )
    detector_backend = st.sidebar.selectbox(
        "Detector Backend",
        ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe", "dlib", "yolov8", "centerface"],
        index=0
    )
    distance_metric = st.sidebar.selectbox(
        "Distance Metric",
        ["cosine", "euclidean", "euclidean_l2"],
        index=0
    )
    threshold = st.sidebar.slider("Recognition Threshold", 0.0, 2.0, 0.6, 0.01)

    # Button: Add new face
    if st.sidebar.button("Add Face"):
        if uploaded_face and new_face_name.strip():
            face_image = Image.open(uploaded_face)
            record = add_new_face(new_face_name.strip(), face_image, model_name, detector_backend)
            st.session_state["known_faces"].append(record)
            st.sidebar.success(f"Added face for '{new_face_name}'")
        else:
            st.sidebar.warning("Please provide both an image and a name.")

    # Sidebar - List known faces
    st.sidebar.header("Registered Faces")
    if len(st.session_state["known_faces"]) == 0:
        st.sidebar.write("No faces registered yet.")
    else:
        for i, face_data in enumerate(st.session_state["known_faces"]):
            st.sidebar.write(f"{i + 1}. {face_data['name']}")

    # Main area - Upload an image for recognition
    st.subheader("Face Recognition on Uploaded Photo")
    recognition_img = st.file_uploader("Upload an image (single or group)", type=["jpg", "jpeg", "png"])

    if recognition_img:
        input_image = Image.open(recognition_img)
        st.image(input_image, caption="Original Image", use_container_width=True)

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
