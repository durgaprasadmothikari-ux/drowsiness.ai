import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
import time
import threading
import winsound
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "advanced_drowsiness_model.pkl"
FACE_MODEL_PATH = "face_landmarker.task"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Drowsiness decision settings
DROWSY_THRESHOLD = 0.65
AWAKE_THRESHOLD = 0.35

DROWSY_FRAMES_REQUIRED = 12
AWAKE_FRAMES_REQUIRED = 8

# Alarm settings
ALARM_START_DELAY = 1.5
ALARM_REPEAT_DELAY = 0.8

# EAR/MAR thresholds are used only for temporal analysis
EYE_CLOSED_EAR = 0.21
MOUTH_OPEN_MAR = 0.55
HEAD_DOWN_PITCH = 15.0


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit


if not os.path.exists(FACE_MODEL_PATH):

    print("ERROR: Face Landmarker model not found:")
    print(FACE_MODEL_PATH)
    raise SystemExit


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)


FEATURE_NAMES = model["feature_names"]
MEANS = model["means"]
STDS = model["stds"]

X_TRAIN = model["X_train"]
Y_TRAIN = model["y_train"]

K = model["k"]


print()
print("=" * 65)
print("        ADVANCED DROWSINESS AI - LIVE PREDICTION")
print("=" * 65)

print()
print("Model type :", model["model_type"])
print("Features   :", len(FEATURE_NAMES))
print("K          :", K)


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL_PATH
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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# LANDMARKS
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# ============================================================
# GLOBAL ALARM STATE
# ============================================================

alarm_active = False
alarm_thread = None


def alarm_loop():

    global alarm_active

    while alarm_active:

        try:

            winsound.Beep(
                1200,
                350
            )

            time.sleep(
                ALARM_REPEAT_DELAY
            )

        except:

            break


def start_alarm():

    global alarm_active
    global alarm_thread

    if alarm_active:
        return

    alarm_active = True

    alarm_thread = threading.Thread(
        target=alarm_loop,
        daemon=True
    )

    alarm_thread.start()


def stop_alarm():

    global alarm_active

    alarm_active = False


# ============================================================
# MATH FUNCTIONS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def eye_aspect_ratio(
    landmarks,
    indices
):

    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    vertical_1 = distance(
        p2,
        p6
    )

    vertical_2 = distance(
        p3,
        p5
    )

    horizontal = distance(
        p1,
        p4
    )

    if horizontal == 0:

        return 0.0

    return (
        vertical_1 +
        vertical_2
    ) / (
        2.0 *
        horizontal
    )


def mouth_aspect_ratio(
    landmarks
):

    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]

    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = distance(
        top,
        bottom
    )

    horizontal = distance(
        left,
        right
    )

    if horizontal == 0:

        return 0.0

    return vertical / horizontal


def get_blendshape(
    blendshapes,
    name
):

    if not blendshapes:

        return 0.0

    for category in blendshapes:

        if category.category_name == name:

            return float(
                category.score
            )

    return 0.0


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(
    landmarks,
    width,
    height
):

    image_points = np.array([

        [
            landmarks[1].x * width,
            landmarks[1].y * height
        ],

        [
            landmarks[152].x * width,
            landmarks[152].y * height
        ],

        [
            landmarks[33].x * width,
            landmarks[33].y * height
        ],

        [
            landmarks[263].x * width,
            landmarks[263].y * height
        ],

        [
            landmarks[61].x * width,
            landmarks[61].y * height
        ],

        [
            landmarks[291].x * width,
            landmarks[291].y * height
        ]

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

        [
            focal_length,
            0,
            width / 2
        ],

        [
            0,
            focal_length,
            height / 2
        ],

        [
            0,
            0,
            1
        ]

    ], dtype=np.float64)


    dist_coeffs = np.zeros(
        (4, 1)
    )


    success, rotation_vector, translation_vector = cv2.solvePnP(

        model_points,

        image_points,

        camera_matrix,

        dist_coeffs,

        flags=cv2.SOLVEPNP_ITERATIVE

    )


    if not success:

        return 0.0, 0.0, 0.0


    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )


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

        roll = 0.0


    return (

        math.degrees(pitch),

        math.degrees(yaw),

        math.degrees(roll)

    )


# ============================================================
# ENVIRONMENT ANALYSIS
# ============================================================

def analyze_environment(
    frame,
    previous_gray
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )

    motion = 0.0

    if previous_gray is not None:

        difference = cv2.absdiff(
            gray,
            previous_gray
        )

        motion = float(
            np.mean(difference)
        )

    return (
        brightness,
        contrast,
        sharpness,
        motion,
        gray
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    features
):

    normalized = []

    for i in range(
        len(features)
    ):

        value = (
            features[i] -
            MEANS[i]
        ) / STDS[i]

        normalized.append(
            value
        )

    return normalized


# ============================================================
# KNN PREDICTION
# ============================================================

def euclidean_distance(
    a,
    b
):

    total = 0.0

    for i in range(
        len(a)
    ):

        difference = (
            a[i] -
            b[i]
        )

        total += (
            difference *
            difference
        )

    return math.sqrt(
        total
    )


def predict_knn(
    sample
):

    distances = []

    for i in range(
        len(X_TRAIN)
    ):

        distance = euclidean_distance(

            sample,

            X_TRAIN[i]

        )

        distances.append(
            (
                distance,
                Y_TRAIN[i]
            )
        )


    distances.sort(
        key=lambda x: x[0]
    )


    nearest = distances[:K]


    awake_votes = 0
    drowsy_votes = 0


    for distance, label in nearest:

        if label == 0:

            awake_votes += 1

        else:

            drowsy_votes += 1


    if drowsy_votes > awake_votes:

        prediction = 1

        confidence = (
            drowsy_votes / K
        )

    else:

        prediction = 0

        confidence = (
            awake_votes / K
        )


    return prediction, confidence


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


if not cap.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    print()
    raise SystemExit


# ============================================================
# STATE VARIABLES
# ============================================================

timestamp_ms = 0

previous_gray = None

previous_prediction = 0

drowsy_frames = 0
awake_frames = 0

eye_closed_start = None
yawn_start = None
head_down_start = None

eye_closure_duration = 0.0
yawn_duration = 0.0
head_down_duration = 0.0

drowsiness_score = 0.0

last_warning_time = 0

fps_counter = 0
fps_start = time.time()
fps = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Camera frame could not be read.")

        break


    frame = cv2.flip(
        frame,
        1
    )


    height, width = frame.shape[:2]


    timestamp_ms += 33


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1

    current_time = time.time()

    elapsed = (
        current_time -
        fps_start
    )


    if elapsed >= 1.0:

        fps = (
            fps_counter /
            elapsed
        )

        fps_counter = 0

        fps_start = current_time


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    (
        brightness,
        contrast,
        sharpness,
        motion,
        gray
    ) = analyze_environment(

        frame,

        previous_gray

    )


    previous_gray = gray


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )


    result = landmarker.detect_for_video(

        mp_image,

        timestamp_ms

    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

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


    # ========================================================
    # FACE ANALYSIS
    # ========================================================

    if result.face_landmarks:

        face_detected = True

        landmarks = (
            result.face_landmarks[0]
        )


        # ----------------------------------------------------
        # EYES
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MOUTH
        # ----------------------------------------------------

        mar = mouth_aspect_ratio(
            landmarks
        )


        # ----------------------------------------------------
        # BLENDSHAPES
        # ----------------------------------------------------

        blendshapes = None


        if result.face_blendshapes:

            blendshapes = (
                result.face_blendshapes[0]
            )


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


        # ----------------------------------------------------
        # HEAD POSE
        # ----------------------------------------------------

        pitch, yaw, roll = calculate_head_pose(

            landmarks,

            width,

            height

        )


        # ====================================================
        # TEMPORAL FEATURES
        # ====================================================

        current_time = time.time()


        eyes_closed = (

            average_ear <
            EYE_CLOSED_EAR

            or

            (
                blink_left > 0.6
                and
                blink_right > 0.6
            )

        )


        mouth_open = (

            mar >
            MOUTH_OPEN_MAR

            or

            jaw_open >
            0.5

        )


        head_down = (

            pitch >
            HEAD_DOWN_PITCH

        )


        # ----------------------------------------------------
        # EYE CLOSURE
        # ----------------------------------------------------

        if eyes_closed:

            if eye_closed_start is None:

                eye_closed_start = (
                    current_time
                )


            eye_closure_duration = (

                current_time -
                eye_closed_start

            )

        else:

            eye_closed_start = None

            eye_closure_duration = 0.0


        # ----------------------------------------------------
        # YAWN
        # ----------------------------------------------------

        if mouth_open:

            if yawn_start is None:

                yawn_start = (
                    current_time
                )


            yawn_duration = (

                current_time -
                yawn_start

            )

        else:

            yawn_start = None

            yawn_duration = 0.0


        # ----------------------------------------------------
        # HEAD DOWN
        # ----------------------------------------------------

        if head_down:

            if head_down_start is None:

                head_down_start = (
                    current_time
                )


            head_down_duration = (

                current_time -
                head_down_start

            )

        else:

            head_down_start = None

            head_down_duration = 0.0


        # ====================================================
        # BUILD 21 FEATURES
        # ====================================================

        features = [

            left_ear,
            right_ear,
            average_ear,

            mar,

            blink_left,
            blink_right,

            jaw_open,

            brow_left,
            brow_right,

            mouth_smile_left,
            mouth_smile_right,

            pitch,
            yaw,
            roll,

            brightness,
            contrast,
            sharpness,
            motion,

            eye_closure_duration,
            yawn_duration,
            head_down_duration

        ]


        # ====================================================
        # NORMALIZE
        # ====================================================

        normalized_features = normalize_features(
            features
        )


        # ====================================================
        # ML PREDICTION
        # ====================================================

        prediction, confidence = predict_knn(

            normalized_features

        )


        # ====================================================
        # DROWSINESS SCORE
        # ====================================================

        if prediction == 1:

            ml_score = confidence

        else:

            ml_score = (
                1.0 -
                confidence
            )


        # Additional behavioral evidence

        behavioral_score = 0.0


        if average_ear < 0.21:

            behavioral_score += 0.30


        if eye_closure_duration > 0.5:

            behavioral_score += 0.25


        if eye_closure_duration > 1.0:

            behavioral_score += 0.15


        if yawn_duration > 0.8:

            behavioral_score += 0.10


        if head_down_duration > 0.7:

            behavioral_score += 0.20


        if pitch > 20:

            behavioral_score += 0.10


        behavioral_score = min(
            behavioral_score,
            1.0
        )


        # Combine ML and temporal evidence

        drowsiness_score = (

            ml_score * 0.70 +

            behavioral_score * 0.30

        )


        drowsiness_score = max(

            0.0,

            min(
                drowsiness_score,
                1.0
            )

        )


        # ====================================================
        # TEMPORAL FILTER
        # ====================================================

        if drowsiness_score >= DROWSY_THRESHOLD:

            drowsy_frames += 1

            awake_frames = 0

        elif drowsiness_score <= AWAKE_THRESHOLD:

            awake_frames += 1

            drowsy_frames = 0

        else:

            drowsy_frames = max(
                0,
                drowsy_frames - 1
            )

            awake_frames = max(
                0,
                awake_frames - 1
            )


        # ====================================================
        # FINAL STATE
        # ====================================================

        if drowsy_frames >= DROWSY_FRAMES_REQUIRED:

            current_state = "DROWSY"

        elif awake_frames >= AWAKE_FRAMES_REQUIRED:

            current_state = "AWAKE"

        else:

            if previous_prediction == 1:

                current_state = "DROWSY"

            else:

                current_state = "AWAKE"


        previous_prediction = (

            1
            if current_state == "DROWSY"
            else 0

        )


        # ====================================================
        # ALARM
        # ====================================================

        if (

            current_state == "DROWSY"

            and

            drowsy_frames >=
            DROWSY_FRAMES_REQUIRED

        ):

            if (
                current_time -
                last_warning_time
                >=
                ALARM_START_DELAY
            ):

                start_alarm()

                last_warning_time = (
                    current_time
                )

        else:

            stop_alarm()


    else:

        # ====================================================
        # NO FACE
        # ====================================================

        stop_alarm()

        drowsy_frames = 0
        awake_frames = 0

        eye_closed_start = None
        yawn_start = None
        head_down_start = None

        eye_closure_duration = 0.0
        yawn_duration = 0.0
        head_down_duration = 0.0

        drowsiness_score = 0.0

        current_state = "NO FACE"

        confidence = 0.0


    # ========================================================
    # USER INTERFACE
    # ========================================================

    # Dark translucent-style top area

    overlay = frame.copy()

    cv2.rectangle(

        overlay,

        (0, 0),

        (width, 145),

        (10, 10, 10),

        -1

    )

    frame = cv2.addWeighted(

        overlay,

        0.72,

        frame,

        0.28,

        0

    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if current_state == "DROWSY":

        status_color = (
            0,
            0,
            255
        )

        status_text = "DROWSY"

    elif current_state == "AWAKE":

        status_color = (
            0,
            255,
            0
        )

        status_text = "AWAKE"

    else:

        status_color = (
            0,
            255,
            255
        )

        status_text = "NO FACE"


    cv2.putText(

        frame,

        f"DRIVER STATUS: {status_text}",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        status_color,

        2

    )


    # --------------------------------------------------------
    # DROWSINESS SCORE
    # --------------------------------------------------------

    score_percent = (
        drowsiness_score *
        100
    )


    cv2.putText(

        frame,

        f"Drowsiness Score: {score_percent:.1f}%",

        (20, 68),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # MODEL CONFIDENCE
    # --------------------------------------------------------

    confidence_percent = (
        confidence *
        100
    )


    cv2.putText(

        frame,

        f"ML Confidence: {confidence_percent:.1f}%",

        (20, 98),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.52,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (20, 128),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (180, 180, 180),

        1

    )


    # ========================================================
    # LEFT INFORMATION PANEL
    # ========================================================

    panel_x = 20
    panel_y = 170


    cv2.putText(

        frame,

        "FACE ANALYSIS",

        (panel_x, panel_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Left EAR      : {left_ear:.3f}",

        (panel_x, panel_y + 32),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Right EAR     : {right_ear:.3f}",

        (panel_x, panel_y + 57),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Average EAR   : {average_ear:.3f}",

        (panel_x, panel_y + 82),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Mouth MAR     : {mar:.3f}",

        (panel_x, panel_y + 107),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Jaw Open      : {jaw_open:.3f}",

        (panel_x, panel_y + 132),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    # ========================================================
    # HEAD PANEL
    # ========================================================

    head_x = 300
    head_y = 170


    cv2.putText(

        frame,

        "HEAD POSE",

        (head_x, head_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Pitch         : {pitch:6.1f}",

        (head_x, head_y + 32),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Yaw           : {yaw:6.1f}",

        (head_x, head_y + 57),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Roll          : {roll:6.1f}",

        (head_x, head_y + 82),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Eye Closure   : {eye_closure_duration:.2f}s",

        (head_x, head_y + 107),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Yawn Duration : {yawn_duration:.2f}s",

        (head_x, head_y + 132),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    # ========================================================
    # ENVIRONMENT PANEL
    # ========================================================

    env_x = 20
    env_y = 350


    cv2.putText(

        frame,

        "ENVIRONMENT",

        (env_x, env_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Brightness : {brightness:.1f}",

        (env_x, env_y + 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (200, 200, 200),

        1

    )


    cv2.putText(

        frame,

        f"Contrast   : {contrast:.1f}",

        (env_x, env_y + 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (200, 200, 200),

        1

    )


    cv2.putText(

        frame,

        f"Sharpness  : {sharpness:.1f}",

        (env_x, env_y + 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (200, 200, 200),

        1

    )


    cv2.putText(

        frame,

        f"Motion     : {motion:.1f}",

        (env_x, env_y + 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (200, 200, 200),

        1

    )


    # ========================================================
    # DROWSINESS MONITOR
    # ========================================================

    monitor_x = 300
    monitor_y = 350


    cv2.putText(

        frame,

        "DROWSINESS MONITOR",

        (monitor_x, monitor_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Drowsy frames : {drowsy_frames}",

        (monitor_x, monitor_y + 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    cv2.putText(

        frame,

        f"Head down     : {head_down_duration:.2f}s",

        (monitor_x, monitor_y + 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1

    )


    if alarm_active:

        cv2.putText(

            frame,

            "!!! DROWSINESS ALERT !!!",

            (monitor_x, monitor_y + 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 0, 255),

            2

        )


    else:

        cv2.putText(

            frame,

            "Alarm: OFF",

            (monitor_x, monitor_y + 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            1

        )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(

        frame,

        "Q = Quit",

        (width - 100, height - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (200, 200, 200),

        1

    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(

        "Advanced Driver Drowsiness AI",

        frame

    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

stop_alarm()

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print()
print("=" * 65)
print("DROWSINESS AI STOPPED")
print("=" * 65)
print()