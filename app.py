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

    detection_results = []
    
    for i, detection in enumerate(detections):
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

        detection_result = {
            "face_id": i + 1,
            "facial_area": facial_area,
            "confidence": detection.get("confidence", "N/A"),
            "recognized_name": "Unknown",
            "min_distance": float("inf"),
            "all_matches": []
        }

        if len(face_embed) == 0:
            detection_result["recognized_name"] = "Unknown"
            detection_result["min_distance"] = float("inf")
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
                
                # Add to all matches
                detection_result["all_matches"].append({
                    "name": known_name,
                    "distance": float(dist),
                    "photo_path": known.get("photo_path", "N/A")
                })
                
                if dist < min_dist:
                    min_dist = dist
                    recognized_name = known_name

            detection_result["recognized_name"] = recognized_name
            detection_result["min_distance"] = float(min_dist)

            if min_dist > threshold:
                detection_result["recognized_name"] = "Unknown"

        # Sort all matches by distance
        detection_result["all_matches"].sort(key=lambda x: x["distance"])

        cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            bgr_image, f"{detection_result['recognized_name']} ({detection_result['min_distance']:.3f})", 
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
        )
        
        detection_results.append(detection_result)

    annotated_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return annotated_image, detection_results

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
    
    # Input method selection for adding faces
    add_input_method = st.sidebar.radio("Choose input method:", ["Upload Image", "Take Photo"])
    
    if add_input_method == "Upload Image":
        uploaded_face = st.sidebar.file_uploader("Upload Face", type=["jpg", "png", "jpeg"])
        face_image = None
        if uploaded_face:
            face_image = Image.open(uploaded_face)
            st.sidebar.image(face_image, caption="Uploaded Face", width=200)
    else:  # Take Photo
        camera_photo = st.sidebar.camera_input("Take a photo of the face")
        face_image = None
        if camera_photo:
            face_image = Image.open(camera_photo)
            st.sidebar.image(face_image, caption="Captured Face", width=200)
    
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
    threshold = st.sidebar.slider("Recognition Threshold", 0.0, 2.0, 0.40, 0.01)

    # Button: Add new face
    if st.sidebar.button("Add Face"):
        if face_image and new_face_name.strip():
            # Compute embedding + store on disk
            record = add_new_face(new_face_name.strip(), face_image, model_name, detector_backend)
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
            col1, col2 = st.sidebar.columns([1, 3])
            
            # Display the face image if available
            if "photo_path" in face_data and os.path.exists(face_data["photo_path"]):
                try:
                    face_img = Image.open(face_data["photo_path"])
                    col1.image(face_img, width=60)
                except:
                    col1.write("📷")
            else:
                col1.write("📷")
            
            # Display name and filename
            txt = f"**{face_data['name']}**"
            if "photo_path" in face_data:
                txt += f"\n`{os.path.basename(face_data['photo_path'])}`"
            col2.write(txt)
            
            st.sidebar.markdown("---")

    # Main - Upload image for recognition
    st.subheader("Face Recognition on Uploaded Photo")
    
    # Input method selection for recognition
    recognition_input_method = st.radio("Choose input method for recognition:", ["Upload Image", "Take Photo"])
    
    if recognition_input_method == "Upload Image":
        recognition_img = st.file_uploader("Upload an image (single or group)", type=["jpg", "jpeg", "png"])
        input_image = None
        if recognition_img:
            input_image = Image.open(recognition_img)
            st.image(input_image, caption="Original Image", use_container_width=True)
    else:  # Take Photo
        recognition_camera = st.camera_input("Take a photo for recognition")
        input_image = None
        if recognition_camera:
            input_image = Image.open(recognition_camera)
            st.image(input_image, caption="Captured Image", use_container_width=True)

    if input_image and st.button("Detect and Recognize Faces"):
        if len(st.session_state["known_faces"]) == 0:
            st.warning("No registered faces. Please add faces in the sidebar first.")
        else:
            annotated, detection_results = detect_and_draw_faces(
                input_image, 
                st.session_state["known_faces"],
                model_name,
                detector_backend,
                distance_metric,
                threshold
            )
            st.image(annotated, caption="Detected Faces", use_container_width=True)
            
            # Display detailed JSON report
            st.subheader("Detailed Detection Report")
            
            # Create a comprehensive report
            full_report = {
                "detection_summary": {
                    "total_faces_detected": len(detection_results),
                    "model_used": model_name,
                    "detector_backend": detector_backend,
                    "distance_metric": distance_metric,
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat()
                },
                "face_detections": detection_results
            }
            
            # Display as expandable JSON
            with st.expander("📊 View Complete Detection Report (JSON)", expanded=True):
                st.json(full_report)
            
            # Display summary table
            st.subheader("Detection Summary")
            if detection_results:
                summary_data = []
                for result in detection_results:
                    summary_data.append({
                        "Face ID": result["face_id"],
                        "Recognized As": result["recognized_name"],
                        "Min Distance": f"{result['min_distance']:.4f}",
                        "Confidence": result["confidence"],
                        "Total Matches": len(result["all_matches"])
                    })
                
                st.table(summary_data)
                
                # Show top matches for each face
                for result in detection_results:
                    if result["all_matches"]:
                        st.write(f"**Face {result['face_id']} - Top 3 Matches:**")
                        top_matches = result["all_matches"][:3]
                        for match in top_matches:
                            st.write(f"  • {match['name']}: {match['distance']:.4f}")
                        st.write("---")

if __name__ == "__main__":
    main()
