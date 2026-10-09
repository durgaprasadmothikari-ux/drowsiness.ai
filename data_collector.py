import cv2
import math
import csv
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# FACE LANDMARKER
# ==========================================

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


# ==========================================
# EYE LANDMARKS
# ==========================================

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


# ==========================================
# DISTANCE
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ==========================================
# EAR CALCULATION
# ==========================================

def calculate_ear(points):

    vertical1 = distance(
        points[1],
        points[5]
    )

    vertical2 = distance(
        points[2],
        points[4]
    )

    horizontal = distance(
        points[0],
        points[3]
    )

    if horizontal == 0:
        return 0

    return (
        (vertical1 + vertical2)
        / (2 * horizontal)
    )


# ==========================================
# CREATE DATASET
# ==========================================

file_name = "drowsiness_dataset.csv"

file = open(
    file_name,
    "w",
    newline=""
)

writer = csv.writer(file)

writer.writerow([
    "left_ear",
    "right_ear",
    "average_ear",
    "label"
])


# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not camera.isOpened():

    print("Could not open camera")

    file.close()
    landmarker.close()

    exit()


print()
print("Camera opened successfully.")
print()


# ==========================================
# VARIABLES
# ==========================================

timestamp_ms = 0

current_label = None

samples_saved = 0

last_save_time = 0


print("=======================================")
print("      DROWSINESS DATA COLLECTOR")
print("=======================================")
print()

print("Press A = Awake")
print("Press D = Drowsy")
print("Press Q = Quit")

print()

print("Current state: NOT COLLECTING")
print()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = camera.read()


    if not ret:

        print("Could not read camera frame")

        break


    # Mirror the camera
    frame = cv2.flip(
        frame,
        1
    )


    # ======================================
    # CONVERT BGR TO RGB
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # CREATE MEDIAPIPE IMAGE
    # ======================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ======================================
    # TIMESTAMP
    # ======================================

    timestamp_ms += 33


    # ======================================
    # DETECT FACE LANDMARKS
    # ======================================

    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # Default values

    left_ear = 0.0

    right_ear = 0.0

    average_ear = 0.0


    # ======================================
    # FACE DETECTED
    # ======================================

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]


        # ==================================
        # LEFT EYE
        # ==================================

        left_points = []


        for index in LEFT_EYE:

            landmark = face_landmarks[index]


            x = int(
                landmark.x *
                frame.shape[1]
            )

            y = int(
                landmark.y *
                frame.shape[0]
            )


            left_points.append(
                (x, y)
            )


            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )


        # ==================================
        # RIGHT EYE
        # ==================================

        right_points = []


        for index in RIGHT_EYE:

            landmark = face_landmarks[index]


            x = int(
                landmark.x *
                frame.shape[1]
            )

            y = int(
                landmark.y *
                frame.shape[0]
            )


            right_points.append(
                (x, y)
            )


            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )


        # ==================================
        # CALCULATE EAR
        # ==================================

        left_ear = calculate_ear(
            left_points
        )

        right_ear = calculate_ear(
            right_points
        )

        average_ear = (
            left_ear +
            right_ear
        ) / 2


        # ==================================
        # SAVE DATA
        # ==================================

        current_time = time.time()


        if (
            current_label is not None
            and
            current_time - last_save_time >= 0.1
        ):

            writer.writerow([
                round(left_ear, 4),
                round(right_ear, 4),
                round(average_ear, 4),
                current_label
            ])


            file.flush()


            samples_saved += 1


            last_save_time = current_time


    # ======================================
    # LABEL DISPLAY
    # ======================================

    if current_label is None:

        label_text = "NOT COLLECTING"

    elif current_label == 0:

        label_text = "AWAKE"

    else:

        label_text = "DROWSY"


    # ======================================
    # DISPLAY EAR
    # ======================================

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


    # ======================================
    # DISPLAY LABEL
    # ======================================

    cv2.putText(
        frame,
        f"Label: {label_text}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ======================================
    # DISPLAY SAMPLE COUNT
    # ======================================

    cv2.putText(
        frame,
        f"Samples: {samples_saved}",
        (30, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # INSTRUCTIONS
    # ======================================

    cv2.putText(
        frame,
        "A = Awake | D = Drowsy | Q = Quit",
        (30, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Drowsiness Data Collector",
        frame
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("a"):

        current_label = 0

        print("Collecting AWAKE data...")


    elif key == ord("d"):

        current_label = 1

        print("Collecting DROWSY data...")


    elif key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

landmarker.close()

file.close()

cv2.destroyAllWindows()


print()
print("=======================================")
print("DATA COLLECTION FINISHED")
print("=======================================")

print()

print(
    "Total samples:",
    samples_saved
)

print(
    "Dataset:",
    file_name
)