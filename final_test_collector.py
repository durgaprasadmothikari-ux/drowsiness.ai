import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "face_landmarker.task"
DATASET_PATH = "final_test_dataset.csv"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Save one sample approximately every 150 ms
SAMPLE_INTERVAL = 0.15

# Labels
AWAKE = 0
DROWSY = 1


# ============================================================
# MEDIAPIPE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# ============================================================
# CSV HEADER
# ============================================================

FEATURE_NAMES = [
    "left_ear",
    "right_ear",
    "average_ear",

    "mar",

    "blink_left",
    "blink_right",

    "jaw_open",

    "brow_left",
    "brow_right",

    "mouth_smile_left",
    "mouth_smile_right",

    "pitch",
    "yaw",
    "roll",

    "brightness",
    "contrast",
    "sharpness",
    "motion",

    "eye_closure_duration",
    "yawn_duration",
    "head_down_duration",

    "label"
]


# ============================================================
# CREATE CSV IF NEEDED
# ============================================================

if not os.path.exists(DATASET_PATH):

    with open(DATASET_PATH, "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(FEATURE_NAMES)

    print("Created:", DATASET_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def eye_aspect_ratio(landmarks, indices):

    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks):

    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = distance(top, bottom)
    horizontal = distance(left, right)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def get_blendshape(blendshapes, name):

    if not blendshapes:
        return 0.0

    for category in blendshapes:

        if category.category_name == name:
            return float(category.score)

    return 0.0


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(landmarks, width, height):

    image_points = np.array([
        [landmarks[1].x * width, landmarks[1].y * height],       # Nose
        [landmarks[152].x * width, landmarks[152].y * height],   # Chin
        [landmarks[33].x * width, landmarks[33].y * height],     # Left eye
        [landmarks[263].x * width, landmarks[263].y * height],   # Right eye
        [landmarks[61].x * width, landmarks[61].y * height],     # Left mouth
        [landmarks[291].x * width, landmarks[291].y * height]    # Right mouth
    ], dtype=np.float64)

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.0, -12.0],
        [-43.0, 32.0, -26.0],
        [43.0, 32.0, -26.0],
        [-28.0, -28.0, -24.0],
        [28.0, -28.0, -24.0]
    ], dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = math.sqrt(
        rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        pitch = math.atan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )

        yaw = math.atan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = math.atan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0]
        )

    else:

        pitch = math.atan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1]
        )

        yaw = math.atan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = 0

    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll)
    )


# ============================================================
# ENVIRONMENT ANALYSIS
# ============================================================

def analyze_environment(frame, previous_gray):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    motion = 0.0

    if previous_gray is not None:

        difference = cv2.absdiff(gray, previous_gray)

        motion = float(np.mean(difference))

    return (
        brightness,
        contrast,
        sharpness,
        motion,
        gray
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    raise SystemExit


print()
print("==============================================")
print(" ADVANCED DROWSINESS DATA COLLECTOR")
print("==============================================")
print()
print("A = Awake")
print("D = Drowsy")
print("Q = Quit")
print()
print("Press A or D before collecting samples.")
print()


# ============================================================
# STATE VARIABLES
# ============================================================

current_label = None

previous_gray = None

timestamp_ms = 0

last_sample_time = 0

eye_closed_start = None
yawn_start = None
head_down_start = None

eye_closure_duration = 0.0
yawn_duration = 0.0
head_down_duration = 0.0

total_samples = 0

awake_samples = 0
drowsy_samples = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame could not be read.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    timestamp_ms += 33

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    brightness, contrast, sharpness, motion, gray = \
        analyze_environment(frame, previous_gray)

    previous_gray = gray

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    left_ear = 0.0
    right_ear = 0.0
    average_ear = 0.0

    mar = 0.0

    blink_left = 0.0
    blink_right = 0.0

    jaw_open = 0.0

    brow_left = 0.0
    brow_right = 0.0

    mouth_smile_left = 0.0
    mouth_smile_right = 0.0

    pitch = 0.0
    yaw = 0.0
    roll = 0.0

    face_detected = False

    # --------------------------------------------------------
    # FACE DETECTED
    # --------------------------------------------------------

    if result.face_landmarks:

        face_detected = True

        landmarks = result.face_landmarks[0]

        # ----------------------------------------------
        # EYE FEATURES
        # ----------------------------------------------

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear +
            right_ear
        ) / 2.0

        # ----------------------------------------------
        # MOUTH
        # ----------------------------------------------

        mar = mouth_aspect_ratio(
            landmarks
        )

        # ----------------------------------------------
        # BLENDSHAPES
        # ----------------------------------------------

        blendshapes = None

        if result.face_blendshapes:

            blendshapes = result.face_blendshapes[0]

        blink_left = get_blendshape(
            blendshapes,
            "eyeBlinkLeft"
        )

        blink_right = get_blendshape(
            blendshapes,
            "eyeBlinkRight"
        )

        jaw_open = get_blendshape(
            blendshapes,
            "jawOpen"
        )

        brow_left = get_blendshape(
            blendshapes,
            "browDownLeft"
        )

        brow_right = get_blendshape(
            blendshapes,
            "browDownRight"
        )

        mouth_smile_left = get_blendshape(
            blendshapes,
            "mouthSmileLeft"
        )

        mouth_smile_right = get_blendshape(
            blendshapes,
            "mouthSmileRight"
        )

        # ----------------------------------------------
        # HEAD POSE
        # ----------------------------------------------

        pitch, yaw, roll = calculate_head_pose(
            landmarks,
            width,
            height
        )

        # ----------------------------------------------
        # TEMPORAL FEATURES
        # ----------------------------------------------

        current_time = time.time()

        eyes_closed = (
            average_ear < 0.21 or
            (blink_left > 0.6 and blink_right > 0.6)
        )

        mouth_open = (
            mar > 0.55 or
            jaw_open > 0.5
        )

        head_down = (
            pitch > 15
        )

        # Eye closure duration

        if eyes_closed:

            if eye_closed_start is None:
                eye_closed_start = current_time

            eye_closure_duration = (
                current_time -
                eye_closed_start
            )

        else:

            eye_closed_start = None
            eye_closure_duration = 0.0

        # Yawn duration

        if mouth_open:

            if yawn_start is None:
                yawn_start = current_time

            yawn_duration = (
                current_time -
                yawn_start
            )

        else:

            yawn_start = None
            yawn_duration = 0.0

        # Head-down duration

        if head_down:

            if head_down_start is None:
                head_down_start = current_time

            head_down_duration = (
                current_time -
                head_down_start
            )

        else:

            head_down_start = None
            head_down_duration = 0.0

    else:

        eye_closed_start = None
        yawn_start = None
        head_down_start = None

        eye_closure_duration = 0.0
        yawn_duration = 0.0
        head_down_duration = 0.0

    # ========================================================
    # DATA COLLECTION
    # ========================================================

    current_time = time.time()

    if (
        current_label is not None
        and face_detected
        and current_time - last_sample_time >= SAMPLE_INTERVAL
    ):

        row = [
            round(left_ear, 5),
            round(right_ear, 5),
            round(average_ear, 5),

            round(mar, 5),

            round(blink_left, 5),
            round(blink_right, 5),

            round(jaw_open, 5),

            round(brow_left, 5),
            round(brow_right, 5),

            round(mouth_smile_left, 5),
            round(mouth_smile_right, 5),

            round(pitch, 5),
            round(yaw, 5),
            round(roll, 5),

            round(brightness, 5),
            round(contrast, 5),
            round(sharpness, 5),
            round(motion, 5),

            round(eye_closure_duration, 5),
            round(yawn_duration, 5),
            round(head_down_duration, 5),

            current_label
        ]

        with open(
            DATASET_PATH,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(row)

        total_samples += 1

        if current_label == AWAKE:
            awake_samples += 1

        elif current_label == DROWSY:
            drowsy_samples += 1

        last_sample_time = current_time

    # ========================================================
    # UI
    # ========================================================

    if current_label == AWAKE:

        label_text = "CURRENT LABEL: AWAKE"

    elif current_label == DROWSY:

        label_text = "CURRENT LABEL: DROWSY"

    else:

        label_text = "PRESS A = AWAKE | D = DROWSY"

    cv2.putText(
        frame,
        label_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"EAR: {average_ear:.3f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"MAR: {mar:.3f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {total_samples}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Awake: {awake_samples}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Drowsy: {drowsy_samples}",
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        2
    )

    cv2.putText(
        frame,
        "A: Awake   D: Drowsy   Q: Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Advanced Drowsiness Data Collector",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):

        current_label = AWAKE

        print("Label changed to: AWAKE")

    elif key == ord("d"):

        current_label = DROWSY

        print("Label changed to: DROWSY")

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print()
print("==============================================")
print(" DATA COLLECTION FINISHED")
print("==============================================")
print()
print("Total samples :", total_samples)
print("Awake samples :", awake_samples)
print("Drowsy samples:", drowsy_samples)
print()
print("Dataset saved to:")
print(DATASET_PATH)
print()