import cv2
import math
import time
import winsound
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Load Face Landmarker
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
# Eye landmark indexes
# -----------------------------

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# -----------------------------
# Distance
# -----------------------------

def distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


# -----------------------------
# EAR
# -----------------------------

def calculate_ear(eye_points):

    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    return (vertical1 + vertical2) / (2 * horizontal)


# -----------------------------
# Start camera
# -----------------------------

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("Could not open camera")
    exit()


# -----------------------------
# Settings
# -----------------------------

timestamp_ms = 0

EAR_THRESHOLD = 0.20

DROWSY_TIME = 1.5

eyes_closed_start = None

alarm_active = False


# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = camera.read()

    if not ret:

        print("Could not read frame")
        break


    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp_ms += 33


    # Detect face
    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    status = "NO FACE"

    average_ear = 0.0


    # -----------------------------
    # Face detected
    # -----------------------------

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]

        left_points = []
        right_points = []


        # Left eye
        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            left_points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )


        # Right eye
        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            right_points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )


        # Calculate EAR
        left_ear = calculate_ear(left_points)

        right_ear = calculate_ear(right_points)

        average_ear = (
            left_ear + right_ear
        ) / 2


        # -----------------------------
        # Eye closure detection
        # -----------------------------

        if average_ear < EAR_THRESHOLD:

            if eyes_closed_start is None:

                eyes_closed_start = time.time()


            closed_duration = (
                time.time() - eyes_closed_start
            )


            if closed_duration >= DROWSY_TIME:

                status = "DROWSINESS DETECTED"

                # Start alarm
                if not alarm_active:

                    winsound.PlaySound(
                        "SystemExclamation",
                        winsound.SND_ALIAS |
                        winsound.SND_ASYNC |
                        winsound.SND_LOOP
                    )

                    alarm_active = True


            else:

                status = "EYES CLOSED"


        else:

            eyes_closed_start = None

            status = "EYES OPEN"


            # Stop alarm
            if alarm_active:

                winsound.PlaySound(
                    None,
                    winsound.SND_ASYNC
                )

                alarm_active = False


        # -----------------------------
        # Display EAR
        # -----------------------------

        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.2f}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Average EAR: {average_ear:.2f}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # -----------------------------
    # Display status
    # -----------------------------

    cv2.putText(
        frame,
        status,
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )


    # Display alarm status
    if alarm_active:

        cv2.putText(
            frame,
            "!!! ALARM !!!",
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


    # Show camera
    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# -----------------------------
# Stop alarm
# -----------------------------

if alarm_active:

    winsound.PlaySound(
        None,
        winsound.SND_ASYNC
    )


# Cleanup
camera.release()

landmarker.close()

cv2.destroyAllWindows()