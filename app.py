import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os
import json
import sqlite3
from datetime import datetime

DB_PATH = "faces_data.db"
FACES_FOLDER = "faces_db"

def add_new_face(name, face_image, model_name, detector_backend):
    """
    Given a name and an uploaded face (PIL Image), compute its embedding, 
    save the image to disk, and return a record dict.
    """
    # Convert the PIL image to BGR
    rgb_array = np.array(face_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Get embedding
    embedding_objs = DeepFace.represent(
        img_path=bgr_array,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )

    # Construct a filename
    os.makedirs(FACES_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{name}_{timestamp}.jpg"
    photo_path = os.path.join(FACES_FOLDER, file_name)

    # Save the uploaded image to disk
    face_image.save(photo_path)

    # Return the dict
    record = {
        "name": name, 
        "embedding": embedding_objs,
        "photo_path": photo_path
    }
    return record


def init_db():
    """Create table if not exists."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            photo_path TEXT
        )
        """)
        conn.commit()

def load_known_faces_from_db():
    results = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding_json, photo_path FROM known_faces")
        rows = cursor.fetchall()

    for (name, embedding_json, photo_path) in rows:
        embedding_list = json.loads(embedding_json)
        record = {
            "name": name,
            "embedding": embedding_list,
            "photo_path": photo_path
        }
        results.append(record)
    return results

def save_face_to_db(name, embedding_list, photo_path):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        embedding_json = json.dumps(embedding_list)
        cursor.execute("""
        INSERT INTO known_faces (name, embedding_json, photo_path)
        VALUES (?, ?, ?)
        """, (name, embedding_json, photo_path))
        conn.commit()

def compute_distance(embedding1, embedding2, distance_metric="cosine"):
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)

    if distance_metric == "cosine":
        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        cos_sim = dot / (norm1 * norm2 + 1e-8)
        return 1 - cos_sim

    elif distance_metric == "euclidean":
        return np.linalg.norm(e1 - e2)

    elif distance_metric == "euclidean_l2":
        diff = e1 - e2
        return np.dot(diff, diff)

    else:
        raise ValueError(f"Unsupported distance metric '{distance_metric}'")

def detect_and_draw_faces(
    input_image, known_faces, model_name, detector_backend,
    distance_metric, threshold
):
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image.convert("RGB"))

    bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    detections = DeepFace.extract_faces(
        img_path=bgr_image,
        detector_backend=detector_backend,
        enforce_detection=False
    )

    for detection in detections:
        facial_area = detection["facial_area"]
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        face_crop = detection["face"]
        face_crop = cv2.resize(face_crop, (224, 224))

        face_embed = DeepFace.represent(
            img_path=face_crop,
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False
        )

        if len(face_embed) == 0:
            recognized_name = "Unknown"
        else:
            current_embedding = face_embed[0]["embedding"]
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

            if min_dist > threshold:
                recognized_name = "Unknown"

        cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            bgr_image, recognized_name, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
        )

    annotated_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return annotated_image

def main():
    st.title("Face Recognition with Persistence (SQLite + Local Images)")

    # 1) Init DB / table if not exists
    init_db()

    # 2) If we haven't loaded known_faces yet, do so
    if "known_faces" not in st.session_state:
        # Load from DB
        loaded_faces = load_known_faces_from_db()
        st.session_state["known_faces"] = loaded_faces

    # Sidebar - Add new face
    st.sidebar.header("Add a New Face")
    uploaded_face = st.sidebar.file_uploader("Upload Face", type=["jpg", "png", "jpeg"])
    new_face_name = st.sidebar.text_input("Name for the Face", "")

    # Sidebar - DeepFace Config
    st.sidebar.header("DeepFace Configuration")
    model_name = st.sidebar.selectbox(
        "Model",
        ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "Dlib", "SFace", "GhostFaceNet"],
        index=0
    )
    detector_backend = st.sidebar.selectbox(
        "Detector Backend",
        ["retinaface","opencv", "ssd", "mtcnn", "mediapipe", "dlib", "yolov8", "centerface"],
        index=0
    )
    distance_metric = st.sidebar.selectbox(
        "Distance Metric",
        ["cosine", "euclidean", "euclidean_l2"],
        index=0
    )
    threshold = st.sidebar.slider("Recognition Threshold", 0.0, 2.0, 0.25, 0.01)

    # Button: Add new face
    if st.sidebar.button("Add Face"):
        if uploaded_face and new_face_name.strip():
            pil_image = Image.open(uploaded_face)
            # Compute embedding + store on disk
            record = add_new_face(new_face_name.strip(), pil_image, model_name, detector_backend)
            # Save to DB
            save_face_to_db(record["name"], record["embedding"], record["photo_path"])
            # Also add to session_state
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
            txt = f"{i+1}. {face_data['name']}"
            if "photo_path" in face_data:
                txt += f" (photo: {os.path.basename(face_data['photo_path'])})"
            st.sidebar.write(txt)

    # Main - Upload image for recognition
    st.subheader("Face Recognition on Uploaded Photo")
    recognition_img = st.file_uploader("Upload an image (single or group)", type=["jpg", "jpeg", "png"])

    if recognition_img:
        input_image = Image.open(recognition_img)
        st.image(input_image, caption="Original Image", use_container_width=True)

        if st.button("Detect and Recognize Faces"):
            if len(st.session_state["known_faces"]) == 0:
                st.warning("No registered faces. Please add faces in the sidebar first.")
            else:
                annotated = detect_and_draw_faces(
                    input_image, 
                    st.session_state["known_faces"],
                    model_name,
                    detector_backend,
                    distance_metric,
                    threshold
                )
                st.image(annotated, caption="Detected Faces", use_container_width=True)

if __name__ == "__main__":
    main()
