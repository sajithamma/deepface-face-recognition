# Face Detection using DeepFace

## Introduction
This project is a simple implementation of face detection using the DeepFace library. The DeepFace library is a deep learning based face recognition library that is built on top of TensorFlow and Keras. It is a lightweight and user-friendly face recognition library that is easy to use and provides high accuracy. The library provides a number of pre-trained models for face recognition and face detection, and it also allows you to train your own models on custom datasets.

## Installation
To install the DeepFace library, you can use the following command:
```bash
pip install -r requirements.txt
```

## Run
To run the face detection script, you can use the following command:
```bash
streamlit run app.py
```

## Code
The code for the face detection script is as follows:
```python
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
```


