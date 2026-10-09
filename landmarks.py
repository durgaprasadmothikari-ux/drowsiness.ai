import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Load Face Landmarker model
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open camera")
    exit()


timestamp_ms = 0


# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read frame")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp
    timestamp_ms += 33


    # Detect landmarks
    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # -----------------------------
    # Check detection
    # -----------------------------

    face_count = len(results.face_landmarks)

    print(
        "Faces detected:",
        face_count,
        end="\r"
    )


    # -----------------------------
    # Draw landmarks
    # -----------------------------

    if face_count > 0:

        for face_landmarks in results.face_landmarks:

            for landmark in face_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )


    # Display
    cv2.imshow(
        "Face Landmark Detection",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

camera.release()
landmarker.close()
cv2.destroyAllWindows()