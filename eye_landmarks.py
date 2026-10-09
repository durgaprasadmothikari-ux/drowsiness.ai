import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load Face Landmarker model
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


# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open camera")
    exit()


timestamp_ms = 0


# Eye landmark indexes
LEFT_EYE = [
    33,   # left corner
    160,  # upper-left
    158,  # upper-right
    133,  # right corner
    153,  # lower-right
    144   # lower-left
]

RIGHT_EYE = [
    362,  # left corner
    385,  # upper-left
    387,  # upper-right
    263,  # right corner
    373,  # lower-right
    380   # lower-left
]


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


    # Detect face landmarks
    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # Draw selected eye landmarks
    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]


        # LEFT EYE
        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 0, 0),
                -1
            )


        # RIGHT EYE
        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 0, 0),
                -1
            )


    # Display
    cv2.imshow(
        "Eye Landmarks",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
landmarker.close()
cv2.destroyAllWindows()