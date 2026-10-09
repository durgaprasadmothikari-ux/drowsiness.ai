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

from session_logger import SessionLogger


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "drowsiness_model_v3.pkl"
FACE_MODEL_PATH = "face_landmarker.task"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

EYE_CLOSED_EAR = 0.21
MOUTH_OPEN_MAR = 0.55
HEAD_DOWN_PITCH = 15.0

DROWSY_THRESHOLD = 0.62
AWAKE_THRESHOLD = 0.35

DROWSY_FRAMES_REQUIRED = 12
AWAKE_FRAMES_REQUIRED = 8


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: V3 model not found.")
    print(MODEL_PATH)
    raise SystemExit

if not os.path.exists(FACE_MODEL_PATH):
    print("ERROR: face_landmarker.task not found.")
    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


FEATURE_NAMES = model["feature_names"]
MEANS = model["means"]
STDS = model["stds"]

X_TRAIN = model["X_train"]
Y_TRAIN = model["y_train"]

K = model["k"]

WEIGHT_DICT = model["feature_weights"]

WEIGHTS = [
    WEIGHT_DICT[name]
    for name in FEATURE_NAMES
]

MODEL_THRESHOLD = model.get(
    "decision_threshold",
    0.55
)


# ============================================================
# START SESSION LOGGER
# ============================================================

logger = SessionLogger(
    "session_logs.csv"
)


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

LEFT_EYE = [
    33, 160, 158,
    133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387,
    263, 373, 380
]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# ============================================================
# ALARM
# ============================================================

alarm_active = False


def alarm_loop():

    global alarm_active

    while alarm_active:

        try:

            winsound.Beep(
                1200,
                350
            )

            time.sleep(0.8)

        except:

            break


def start_alarm():

    global alarm_active

    if alarm_active:
        return

    alarm_active = True

    threading.Thread(
        target=alarm_loop,
        daemon=True
    ).start()


def stop_alarm():

    global alarm_active

    alarm_active = False


# ============================================================
# MATH
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

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


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


def get_blendshape(
    blendshapes,
    name
):

    if not blendshapes:
        return 0.0

    for category in blendshapes:

        if category.category_name == name:
            return float(category.score)

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


    success, rotation_vector, _ = cv2.solvePnP(

        model_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
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


    if sy >= 1e-6:

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
# ENVIRONMENT
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

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    motion = 0.0

    if previous_gray is not None:

        motion = float(
            np.mean(
                cv2.absdiff(
                    gray,
                    previous_gray
                )
            )
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

def normalize_features(features):

    return [

        (
            features[i] -
            MEANS[i]
        ) / STDS[i]

        for i in range(
            len(features)
        )

    ]


# ============================================================
# WEIGHTED KNN
# ============================================================

def weighted_distance(a, b):

    total = 0.0

    for i in range(len(a)):

        difference = a[i] - b[i]

        total += (
            difference *
            difference *
            WEIGHTS[i]
        )

    return math.sqrt(total)


def predict_knn(sample):

    distances = []

    for i in range(len(X_TRAIN)):

        d = weighted_distance(
            sample,
            X_TRAIN[i]
        )

        distances.append(
            (d, Y_TRAIN[i])
        )


    distances.sort(
        key=lambda item: item[0]
    )


    nearest = distances[:K]

    awake_score = 0.0
    drowsy_score = 0.0


    for d, label in nearest:

        vote = 1.0 / (
            d + 0.000001
        )

        if label == 0:
            awake_score += vote
        else:
            drowsy_score += vote


    total = (
        awake_score +
        drowsy_score
    )


    if total == 0:
        return 0, 0.5


    probability = (
        drowsy_score /
        total
    )


    if probability >= MODEL_THRESHOLD:

        return 1, probability

    return 0, 1.0 - probability


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

    print("ERROR: Camera could not be opened.")
    raise SystemExit


# ============================================================
# STATE
# ============================================================

timestamp_ms = 0

previous_gray = None

eye_closed_start = None
yawn_start = None
head_down_start = None

eye_closure_duration = 0.0
yawn_duration = 0.0
head_down_duration = 0.0

drowsy_frames = 0
awake_frames = 0

current_state = "NO FACE"

drowsiness_score = 0.0
confidence = 0.0

fps = 0.0
fps_count = 0
fps_start = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame error.")
            break


        frame = cv2.flip(
            frame,
            1
        )

        height, width = frame.shape[:2]

        timestamp_ms += 33


        # ====================================================
        # FPS
        # ====================================================

        fps_count += 1

        now = time.time()

        if now - fps_start >= 1.0:

            fps = (
                fps_count /
                (now - fps_start)
            )

            fps_count = 0
            fps_start = now


        # ====================================================
        # ENVIRONMENT
        # ====================================================

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


        # ====================================================
        # MEDIAPIPE
        # ====================================================

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


        # ====================================================
        # DEFAULTS
        # ====================================================

        left_ear = 0.0
        right_ear = 0.0
        average_ear = 0.0

        mar = 0.0

        blink_left = 0.0
        blink_right = 0.0

        jaw_open = 0.0

        brow_left = 0.0
        brow_right = 0.0

        smile_left = 0.0
        smile_right = 0.0

        pitch = 0.0
        yaw = 0.0
        roll = 0.0


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # ------------------------------------------------
            # EYES
            # ------------------------------------------------

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


            # ------------------------------------------------
            # MOUTH
            # ------------------------------------------------

            mar = mouth_aspect_ratio(
                landmarks
            )


            # ------------------------------------------------
            # BLENDSHAPES
            # ------------------------------------------------

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

            smile_left = get_blendshape(
                blendshapes,
                "mouthSmileLeft"
            )

            smile_right = get_blendshape(
                blendshapes,
                "mouthSmileRight"
            )


            # ------------------------------------------------
            # HEAD
            # ------------------------------------------------

            pitch, yaw, roll = calculate_head_pose(
                landmarks,
                width,
                height
            )


            # =================================================
            # TEMPORAL FEATURES
            # =================================================

            now = time.time()


            eyes_closed = (

                average_ear < EYE_CLOSED_EAR

                or

                (
                    blink_left > 0.6
                    and
                    blink_right > 0.6
                )

            )


            mouth_open = (

                mar > MOUTH_OPEN_MAR

                or

                jaw_open > 0.5

            )


            head_down = (
                pitch > HEAD_DOWN_PITCH
            )


            # ------------------------------------------------
            # EYE CLOSURE
            # ------------------------------------------------

            if eyes_closed:

                if eye_closed_start is None:
                    eye_closed_start = now

                eye_closure_duration = (
                    now -
                    eye_closed_start
                )

            else:

                eye_closed_start = None
                eye_closure_duration = 0.0


            # ------------------------------------------------
            # YAWN
            # ------------------------------------------------

            if mouth_open:

                if yawn_start is None:
                    yawn_start = now

                yawn_duration = (
                    now -
                    yawn_start
                )

            else:

                yawn_start = None
                yawn_duration = 0.0


            # ------------------------------------------------
            # HEAD DOWN
            # ------------------------------------------------

            if head_down:

                if head_down_start is None:
                    head_down_start = now

                head_down_duration = (
                    now -
                    head_down_start
                )

            else:

                head_down_start = None
                head_down_duration = 0.0


            # =================================================
            # 21 FEATURES
            # =================================================

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

                smile_left,
                smile_right,

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


            # =================================================
            # MACHINE LEARNING
            # =================================================

            normalized = normalize_features(
                features
            )

            prediction, confidence = predict_knn(
                normalized
            )


            if prediction == 1:

                ml_score = confidence

            else:

                ml_score = (
                    1.0 -
                    confidence
                )


            # =================================================
            # BEHAVIOR SCORE
            # =================================================

            behavior_score = 0.0


            if average_ear < 0.21:
                behavior_score += 0.25


            if eye_closure_duration > 0.5:
                behavior_score += 0.20


            if eye_closure_duration > 1.0:
                behavior_score += 0.15


            if yawn_duration > 0.8:
                behavior_score += 0.10


            if head_down_duration > 0.7:
                behavior_score += 0.20


            if pitch > 20:
                behavior_score += 0.10


            behavior_score = min(
                behavior_score,
                1.0
            )


            # =================================================
            # FINAL SCORE
            # =================================================

            drowsiness_score = (

                ml_score * 0.70

                +

                behavior_score * 0.30

            )


            drowsiness_score = max(
                0.0,
                min(
                    drowsiness_score,
                    1.0
                )
            )


            # =================================================
            # TEMPORAL FILTER
            # =================================================

            if (
                drowsiness_score >=
                DROWSY_THRESHOLD
            ):

                drowsy_frames += 1
                awake_frames = 0

            elif (
                drowsiness_score <=
                AWAKE_THRESHOLD
            ):

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


            # =================================================
            # STATE
            # =================================================

            if (
                drowsy_frames >=
                DROWSY_FRAMES_REQUIRED
            ):

                current_state = "DROWSY"

            elif (
                awake_frames >=
                AWAKE_FRAMES_REQUIRED
            ):

                current_state = "AWAKE"


            # =================================================
            # ALARM
            # =================================================

            if (
                current_state == "DROWSY"
                and
                drowsy_frames >=
                DROWSY_FRAMES_REQUIRED
            ):

                start_alarm()

            else:

                stop_alarm()


        else:

            # =================================================
            # NO FACE
            # =================================================

            current_state = "NO FACE"

            drowsy_frames = 0
            awake_frames = 0

            drowsiness_score = 0.0
            confidence = 0.0

            eye_closed_start = None
            yawn_start = None
            head_down_start = None

            eye_closure_duration = 0.0
            yawn_duration = 0.0
            head_down_duration = 0.0

            stop_alarm()


        # ====================================================
        # SESSION LOGGER
        # ====================================================

        logger.update(

            state=current_state,

            drowsiness_score=drowsiness_score,

            eye_closure_duration=
                eye_closure_duration,

            yawn_duration=
                yawn_duration,

            head_down_duration=
                head_down_duration

        )


        # ====================================================
        # UI
        # ====================================================

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


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if current_state == "DROWSY":

            status_color = (
                0,
                0,
                255
            )

        elif current_state == "AWAKE":

            status_color = (
                0,
                255,
                0
            )

        else:

            status_color = (
                0,
                255,
                255
            )


        cv2.putText(

            frame,

            f"DRIVER STATUS: {current_state}",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            status_color,

            2

        )


        cv2.putText(

            frame,

            f"Drowsiness Score: "
            f"{drowsiness_score * 100:.1f}%",

            (20, 68),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.58,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"ML Confidence: "
            f"{confidence * 100:.1f}%",

            (20, 98),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            1

        )


        cv2.putText(

            frame,

            f"FPS: {fps:.1f}",

            (20, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (180, 180, 180),

            1

        )


        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "FACE",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (20, 202),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (20, 227),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Eye closure: "
            f"{eye_closure_duration:.2f}s",
            (20, 252),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "HEAD",
            (300, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (300, 202),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (300, 227),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Roll: {roll:.1f}",
            (300, 252),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Head down: "
            f"{head_down_duration:.2f}s",
            (300, 277),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # ----------------------------------------------------
        # SESSION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "SESSION",
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Events: {logger.drowsiness_events}",
            (20, 357),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Max score: "
            f"{logger.maximum_drowsiness_score * 100:.1f}%",
            (20, 382),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # ----------------------------------------------------
        # ALARM
        # ----------------------------------------------------

        if alarm_active:

            cv2.putText(

                frame,

                "!!! DROWSINESS ALERT !!!",

                (20, 425),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 0, 255),

                2

            )

        else:

            cv2.putText(

                frame,

                "Alarm: OFF",

                (20, 425),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (0, 255, 0),

                1

            )


        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Light:{brightness:.0f} "
            f"Sharp:{sharpness:.0f}",

            (300, 330),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (180, 180, 180),

            1

        )


        cv2.putText(

            frame,

            "Q = Quit",

            (width - 95, height - 15),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (180, 180, 180),

            1

        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Driver Drowsiness AI - V3",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


finally:

    # ========================================================
    # STOP ALARM
    # ========================================================

    stop_alarm()


    # ========================================================
    # FINISH LOGGER
    # ========================================================

    summary = logger.finish()


    # ========================================================
    # CAMERA CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()


    # ========================================================
    # SESSION SUMMARY
    # ========================================================

    print()
    print("=" * 65)
    print("                 SESSION COMPLETE")
    print("=" * 65)

    print()

    print(
        "Session ID          :",
        summary["session_id"]
    )

    print(
        "Duration            :",
        f"{summary['duration']:.2f} seconds"
    )

    print(
        "Frames analyzed     :",
        summary["total_frames"]
    )

    print(
        "Drowsiness events   :",
        summary["drowsiness_events"]
    )

    print(
        "Total alert time    :",
        f"{summary['total_alert_time']:.2f} seconds"
    )

    print(
        "Longest drowsiness  :",
        f"{summary['longest_drowsy_event']:.2f} seconds"
    )

    print(
        "Longest eye closure:",
        f"{summary['longest_eye_closure']:.2f} seconds"
    )

    print(
        "Longest yawn        :",
        f"{summary['longest_yawn']:.2f} seconds"
    )

    print(
        "Longest head down  :",
        f"{summary['longest_head_down']:.2f} seconds"
    )

    print(
        "Maximum score       :",
        f"{summary['maximum_score'] * 100:.1f}%"
    )

    print(
        "Average score       :",
        f"{summary['average_score'] * 100:.1f}%"
    )

    print()

    print(
        "Saved to:"
    )

    print(
        "session_logs.csv"
    )

    print()

    print("=" * 65)