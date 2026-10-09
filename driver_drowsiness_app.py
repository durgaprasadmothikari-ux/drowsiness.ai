import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
import time
import threading
import winsound
import os
import tkinter as tk

from PIL import Image, ImageTk

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

CAMERA_FPS = 30


# ============================================================
# COLORS
# ============================================================

BG = "#070a0f"
PANEL = "#0d1219"
PANEL_2 = "#111923"

TEXT = "#e9eef3"
MUTED = "#7d8996"

GREEN = "#35e58a"
RED = "#ff4d5a"
BLUE = "#4da3ff"
YELLOW = "#ffc857"
CYAN = "#48d7ff"


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(MODEL_PATH):

    print(
        "ERROR: Model not found:",
        MODEL_PATH
    )

    raise SystemExit


if not os.path.exists(FACE_MODEL_PATH):

    print(
        "ERROR: MediaPipe model not found:",
        FACE_MODEL_PATH
    )

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

with open(
    MODEL_PATH,
    "rb"
) as file:

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
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(

    model_asset_path=
    FACE_MODEL_PATH

)


options = vision.FaceLandmarkerOptions(

    base_options=base_options,

    running_mode=
    vision.RunningMode.VIDEO,

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
# LANDMARK INDICES
# ============================================================

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


MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


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

    print(
        "ERROR: Camera could not be opened."
    )

    landmarker.close()

    raise SystemExit


# ============================================================
# SESSION LOGGER
# ============================================================

logger = SessionLogger(
    "session_logs.csv"
)


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

            time.sleep(
                0.8
            )

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
# MATH FUNCTIONS
# ============================================================

def distance(
    p1,
    p2
):

    return math.sqrt(

        (p1.x - p2.x) ** 2

        +

        (p1.y - p2.y) ** 2

    )


def eye_aspect_ratio(

    landmarks,

    indices

):

    p1 = landmarks[
        indices[0]
    ]

    p2 = landmarks[
        indices[1]
    ]

    p3 = landmarks[
        indices[2]
    ]

    p4 = landmarks[
        indices[3]
    ]

    p5 = landmarks[
        indices[4]
    ]

    p6 = landmarks[
        indices[5]
    ]


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

        vertical_1
        +
        vertical_2

    ) / (

        2.0 *
        horizontal

    )


def mouth_aspect_ratio(
    landmarks
):

    top = landmarks[
        MOUTH_TOP
    ]

    bottom = landmarks[
        MOUTH_BOTTOM
    ]

    left = landmarks[
        MOUTH_LEFT
    ]

    right = landmarks[
        MOUTH_RIGHT
    ]


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


    return (
        vertical /
        horizontal
    )


# ============================================================
# BLENDSHAPE
# ============================================================

def get_blendshape(

    blendshapes,

    name

):

    if not blendshapes:

        return 0.0


    for category in blendshapes:

        if (
            category.category_name
            ==
            name
        ):

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
            landmarks[1].x *
            width,

            landmarks[1].y *
            height
        ],

        [
            landmarks[152].x *
            width,

            landmarks[152].y *
            height
        ],

        [
            landmarks[33].x *
            width,

            landmarks[33].y *
            height
        ],

        [
            landmarks[263].x *
            width,

            landmarks[263].y *
            height
        ],

        [
            landmarks[61].x *
            width,

            landmarks[61].y *
            height
        ],

        [
            landmarks[291].x *
            width,

            landmarks[291].y *
            height
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

        return (
            0.0,
            0.0,
            0.0
        )


    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )


    sy = math.sqrt(

        rotation_matrix[0, 0] ** 2

        +

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
# MODEL FUNCTIONS
# ============================================================

def normalize_features(

    features

):

    return [

        (

            features[i]

            -

            MEANS[i]

        )

        /

        STDS[i]

        for i in range(
            len(features)
        )

    ]


def weighted_distance(

    a,

    b

):

    total = 0.0


    for i in range(
        len(a)
    ):

        difference = (

            a[i]

            -

            b[i]

        )


        total += (

            difference *

            difference *

            WEIGHTS[i]

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

        d = weighted_distance(

            sample,

            X_TRAIN[i]

        )


        distances.append(

            (
                d,

                Y_TRAIN[i]

            )

        )


    distances.sort(

        key=lambda item:
        item[0]

    )


    nearest = distances[
        :K
    ]


    awake_score = 0.0

    drowsy_score = 0.0


    for d, label in nearest:

        vote = 1.0 / (

            d +

            0.000001

        )


        if label == 0:

            awake_score += vote

        else:

            drowsy_score += vote


    total = (

        awake_score

        +

        drowsy_score

    )


    if total == 0:

        return (
            0,
            0.5
        )


    probability = (

        drowsy_score

        /

        total

    )


    if (
        probability
        >=
        MODEL_THRESHOLD
    ):

        return (

            1,

            probability

        )


    return (

        0,

        1.0 - probability

    )


# ============================================================
# APPLICATION
# ============================================================

class DriverDrowsinessApp:

    def __init__(

        self,

        root

    ):

        self.root = root


        self.root.title(

            "Driver Drowsiness AI"

        )


        self.root.geometry(

            "1280x820"

        )


        self.root.minsize(

            1100,

            700

        )


        self.root.configure(

            bg=BG

        )


        # ====================================================
        # STATE
        # ====================================================

        self.running = True

        self.timestamp_ms = 0

        self.previous_gray = None

        self.eye_closed_start = None
        self.yawn_start = None
        self.head_down_start = None

        self.eye_closure_duration = 0.0
        self.yawn_duration = 0.0
        self.head_down_duration = 0.0

        self.drowsy_frames = 0
        self.awake_frames = 0

        self.current_state = "NO FACE"

        self.drowsiness_score = 0.0
        self.confidence = 0.0

        self.left_ear = 0.0
        self.right_ear = 0.0
        self.average_ear = 0.0

        self.mar = 0.0

        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

        self.fps = 0.0

        self.fps_count = 0

        self.fps_start = time.time()


        # ====================================================
        # HEADER
        # ====================================================

        header = tk.Frame(

            root,

            bg=BG

        )

        header.pack(

            fill="x",

            padx=25,

            pady=(20, 10)

        )


        tk.Label(

            header,

            text="DRIVER DROWSINESS AI",

            bg=BG,

            fg=TEXT,

            font=(
                "Segoe UI",
                22,
                "bold"
            )

        ).pack(

            side="left"

        )


        tk.Label(

            header,

            text="  /  LIVE MONITOR",

            bg=BG,

            fg=CYAN,

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        ).pack(

            side="left",

            pady=(7, 0)

        )


        self.connection = tk.Label(

            header,

            text="● CAMERA ONLINE",

            bg=BG,

            fg=GREEN,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        )

        self.connection.pack(

            side="right",

            pady=5

        )


        # ====================================================
        # MAIN AREA
        # ====================================================

        main = tk.Frame(

            root,

            bg=BG

        )

        main.pack(

            fill="both",

            expand=True,

            padx=25,

            pady=10

        )


        # ====================================================
        # CAMERA PANEL
        # ====================================================

        camera_panel = tk.Frame(

            main,

            bg=PANEL

        )

        camera_panel.pack(

            side="left",

            fill="both",

            expand=True,

            padx=(0, 8)

        )


        self.camera_label = tk.Label(

            camera_panel,

            bg="#000000"

        )

        self.camera_label.pack(

            fill="both",

            expand=True,

            padx=10,

            pady=10

        )


        # ====================================================
        # RIGHT PANEL
        # ====================================================

        right = tk.Frame(

            main,

            bg=BG,

            width=350

        )

        right.pack(

            side="right",

            fill="y"

        )


        right.pack_propagate(
            False
        )


        # ====================================================
        # STATUS PANEL
        # ====================================================

        status_panel = tk.Frame(

            right,

            bg=PANEL

        )

        status_panel.pack(

            fill="x",

            pady=(0, 8)

        )


        tk.Label(

            status_panel,

            text="DRIVER STATUS",

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(15, 3)

        )


        self.status_value = tk.Label(

            status_panel,

            text="NO FACE",

            bg=PANEL,

            fg=YELLOW,

            font=(
                "Segoe UI",
                22,
                "bold"
            )

        )

        self.status_value.pack(

            anchor="w",

            padx=18,

            pady=(0, 12)

        )


        # ====================================================
        # SCORE
        # ====================================================

        self.score_label = tk.Label(

            status_panel,

            text="DROWSINESS SCORE  0.0%",

            bg=PANEL,

            fg=TEXT,

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        )

        self.score_label.pack(

            anchor="w",

            padx=18,

            pady=(0, 5)

        )


        self.score_canvas = tk.Canvas(

            status_panel,

            height=22,

            bg=PANEL,

            highlightthickness=0

        )

        self.score_canvas.pack(

            fill="x",

            padx=18,

            pady=(0, 15)

        )


        # ====================================================
        # FACE FEATURES
        # ====================================================

        face_panel = tk.Frame(

            right,

            bg=PANEL

        )

        face_panel.pack(

            fill="x",

            pady=8

        )


        tk.Label(

            face_panel,

            text="FACE ANALYSIS",

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(15, 8)

        )


        self.ear_value = self.metric(
            face_panel,
            "Average EAR"
        )

        self.mar_value = self.metric(
            face_panel,
            "Mouth MAR"
        )

        self.eye_value = self.metric(
            face_panel,
            "Eye closure"
        )

        self.yawn_value = self.metric(
            face_panel,
            "Yawn duration"
        )


        # ====================================================
        # HEAD
        # ====================================================

        head_panel = tk.Frame(

            right,

            bg=PANEL

        )

        head_panel.pack(

            fill="x",

            pady=8

        )


        tk.Label(

            head_panel,

            text="HEAD POSITION",

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(15, 8)

        )


        self.pitch_value = self.metric(
            head_panel,
            "Pitch"
        )

        self.yaw_value = self.metric(
            head_panel,
            "Yaw"
        )

        self.roll_value = self.metric(
            head_panel,
            "Roll"
        )

        self.head_value = self.metric(
            head_panel,
            "Head-down"
        )


        # ====================================================
        # SESSION
        # ====================================================

        session_panel = tk.Frame(

            right,

            bg=PANEL

        )

        session_panel.pack(

            fill="x",

            pady=8

        )


        tk.Label(

            session_panel,

            text="SESSION",

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(15, 8)

        )


        self.events_value = self.metric(
            session_panel,
            "Drowsiness events"
        )

        self.alert_value = self.metric(
            session_panel,
            "Alert time"
        )

        self.max_value = self.metric(
            session_panel,
            "Maximum score"
        )

        self.fps_value = self.metric(
            session_panel,
            "FPS"
        )


        # ====================================================
        # BOTTOM BAR
        # ====================================================

        bottom = tk.Frame(

            root,

            bg=BG

        )

        bottom.pack(

            fill="x",

            padx=25,

            pady=(5, 18)

        )


        tk.Label(

            bottom,

            text=(
                "Q  •  Stop session     "
                "DROWSINESS AI V3"
            ),

            bg=BG,

            fg=MUTED,

            font=(
                "Segoe UI",
                9
            )

        ).pack(

            side="left"

        )


        self.alarm_label = tk.Label(

            bottom,

            text="● ALARM OFF",

            bg=BG,

            fg=GREEN,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        )

        self.alarm_label.pack(

            side="right"

        )


        # ====================================================
        # START
        # ====================================================

        self.root.protocol(

            "WM_DELETE_WINDOW",

            self.close

        )


        self.update_frame()


    # ========================================================
    # METRIC
    # ========================================================

    def metric(

        self,

        parent,

        name

    ):

        row = tk.Frame(

            parent,

            bg=PANEL

        )

        row.pack(

            fill="x",

            padx=18,

            pady=3

        )


        tk.Label(

            row,

            text=name,

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                9
            )

        ).pack(

            side="left"

        )


        value = tk.Label(

            row,

            text="0",

            bg=PANEL,

            fg=TEXT,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        )

        value.pack(

            side="right"

        )


        return value


    # ========================================================
    # DRAW SCORE
    # ========================================================

    def draw_score(self):

        self.score_canvas.delete(
            "all"
        )


        width = (
            self.score_canvas.winfo_width()
        )


        if width < 20:

            width = 300


        self.score_canvas.create_rectangle(

            0,

            3,

            width,

            19,

            fill=PANEL_2,

            outline=""

        )


        fill = (

            width *

            self.drowsiness_score

        )


        if self.drowsiness_score < 0.35:

            color = GREEN

        elif self.drowsiness_score < 0.65:

            color = YELLOW

        else:

            color = RED


        self.score_canvas.create_rectangle(

            0,

            3,

            fill,

            19,

            fill=color,

            outline=""

        )


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(

        self,

        frame

    ):

        height, width = frame.shape[:2]


        self.timestamp_ms += 33


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.fps_count += 1

        now = time.time()

        elapsed = (
            now -
            self.fps_start
        )


        if elapsed >= 1.0:

            self.fps = (

                self.fps_count
                /
                elapsed

            )

            self.fps_count = 0

            self.fps_start = now


        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        (
            brightness,
            contrast,
            sharpness,
            motion,
            gray

        ) = analyze_environment(

            frame,

            self.previous_gray

        )


        self.previous_gray = gray


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=rgb

        )


        result = landmarker.detect_for_video(

            mp_image,

            self.timestamp_ms

        )


        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # -----------------------------------------------
            # EAR
            # -----------------------------------------------

            self.left_ear = eye_aspect_ratio(

                landmarks,

                LEFT_EYE

            )


            self.right_ear = eye_aspect_ratio(

                landmarks,

                RIGHT_EYE

            )


            self.average_ear = (

                self.left_ear
                +
                self.right_ear

            ) / 2.0


            # -----------------------------------------------
            # MAR
            # -----------------------------------------------

            self.mar = mouth_aspect_ratio(

                landmarks

            )


            # -----------------------------------------------
            # BLENDSHAPES
            # -----------------------------------------------

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


            # -----------------------------------------------
            # HEAD
            # -----------------------------------------------

            (

                self.pitch,

                self.yaw,

                self.roll

            ) = calculate_head_pose(

                landmarks,

                width,

                height

            )


            # -----------------------------------------------
            # TEMPORAL CONDITIONS
            # -----------------------------------------------

            now = time.time()


            eyes_closed = (

                self.average_ear
                <
                EYE_CLOSED_EAR

                or

                (

                    blink_left > 0.6

                    and

                    blink_right > 0.6

                )

            )


            mouth_open = (

                self.mar
                >
                MOUTH_OPEN_MAR

                or

                jaw_open > 0.5

            )


            head_down = (

                self.pitch
                >
                HEAD_DOWN_PITCH

            )


            # -----------------------------------------------
            # EYE CLOSURE
            # -----------------------------------------------

            if eyes_closed:

                if (
                    self.eye_closed_start
                    is None
                ):

                    self.eye_closed_start = now


                self.eye_closure_duration = (

                    now
                    -
                    self.eye_closed_start

                )

            else:

                self.eye_closed_start = None

                self.eye_closure_duration = 0.0


            # -----------------------------------------------
            # YAWN
            # -----------------------------------------------

            if mouth_open:

                if (
                    self.yawn_start
                    is None
                ):

                    self.yawn_start = now


                self.yawn_duration = (

                    now
                    -
                    self.yawn_start

                )

            else:

                self.yawn_start = None

                self.yawn_duration = 0.0


            # -----------------------------------------------
            # HEAD DOWN
            # -----------------------------------------------

            if head_down:

                if (
                    self.head_down_start
                    is None
                ):

                    self.head_down_start = now


                self.head_down_duration = (

                    now
                    -
                    self.head_down_start

                )

            else:

                self.head_down_start = None

                self.head_down_duration = 0.0


            # -----------------------------------------------
            # FEATURES
            # -----------------------------------------------

            features = [

                self.left_ear,

                self.right_ear,

                self.average_ear,

                self.mar,

                blink_left,

                blink_right,

                jaw_open,

                brow_left,

                brow_right,

                smile_left,

                smile_right,

                self.pitch,

                self.yaw,

                self.roll,

                brightness,

                contrast,

                sharpness,

                motion,

                self.eye_closure_duration,

                self.yawn_duration,

                self.head_down_duration

            ]


            # -----------------------------------------------
            # ML
            # -----------------------------------------------

            normalized = normalize_features(

                features

            )


            prediction, self.confidence = predict_knn(

                normalized

            )


            if prediction == 1:

                ml_score = self.confidence

            else:

                ml_score = (

                    1.0
                    -
                    self.confidence

                )


            # -----------------------------------------------
            # BEHAVIOR SCORE
            # -----------------------------------------------

            behavior_score = 0.0


            if self.average_ear < 0.21:

                behavior_score += 0.25


            if self.eye_closure_duration > 0.5:

                behavior_score += 0.20


            if self.eye_closure_duration > 1.0:

                behavior_score += 0.15


            if self.yawn_duration > 0.8:

                behavior_score += 0.10


            if self.head_down_duration > 0.7:

                behavior_score += 0.20


            if self.pitch > 20:

                behavior_score += 0.10


            behavior_score = min(

                behavior_score,

                1.0

            )


            # -----------------------------------------------
            # FINAL SCORE
            # -----------------------------------------------

            self.drowsiness_score = (

                ml_score * 0.70

                +

                behavior_score * 0.30

            )


            self.drowsiness_score = max(

                0.0,

                min(

                    self.drowsiness_score,

                    1.0

                )

            )


            # -----------------------------------------------
            # TEMPORAL FILTER
            # -----------------------------------------------

            if (

                self.drowsiness_score
                >=
                DROWSY_THRESHOLD

            ):

                self.drowsy_frames += 1

                self.awake_frames = 0

            elif (

                self.drowsiness_score
                <=
                AWAKE_THRESHOLD

            ):

                self.awake_frames += 1

                self.drowsy_frames = 0

            else:

                self.drowsy_frames = max(

                    0,

                    self.drowsy_frames - 1

                )

                self.awake_frames = max(

                    0,

                    self.awake_frames - 1

                )


            # -----------------------------------------------
            # STATE
            # -----------------------------------------------

            if (

                self.drowsy_frames
                >=
                DROWSY_FRAMES_REQUIRED

            ):

                self.current_state = "DROWSY"


            elif (

                self.awake_frames
                >=
                AWAKE_FRAMES_REQUIRED

            ):

                self.current_state = "AWAKE"


            # -----------------------------------------------
            # ALARM
            # -----------------------------------------------

            if (

                self.current_state
                ==
                "DROWSY"

                and

                self.drowsy_frames
                >=
                DROWSY_FRAMES_REQUIRED

            ):

                start_alarm()

            else:

                stop_alarm()


        else:

            self.current_state = "NO FACE"

            self.drowsy_frames = 0

            self.awake_frames = 0

            self.drowsiness_score = 0.0

            self.confidence = 0.0

            self.eye_closed_start = None

            self.yawn_start = None

            self.head_down_start = None

            self.eye_closure_duration = 0.0

            self.yawn_duration = 0.0

            self.head_down_duration = 0.0

            stop_alarm()


        # ----------------------------------------------------
        # LOGGER
        # ----------------------------------------------------

        logger.update(

            state=self.current_state,

            drowsiness_score=
            self.drowsiness_score,

            eye_closure_duration=
            self.eye_closure_duration,

            yawn_duration=
            self.yawn_duration,

            head_down_duration=
            self.head_down_duration

        )


        return frame


    # ========================================================
    # UPDATE UI
    # ========================================================

    def update_ui(self):

        state = self.current_state


        if state == "DROWSY":

            color = RED

        elif state == "AWAKE":

            color = GREEN

        else:

            color = YELLOW


        self.status_value.config(

            text=state,

            fg=color

        )


        self.score_label.config(

            text=(

                "DROWSINESS SCORE  "

                f"{self.drowsiness_score * 100:.1f}%"

            )

        )


        self.ear_value.config(

            text=f"{self.average_ear:.3f}"

        )


        self.mar_value.config(

            text=f"{self.mar:.3f}"

        )


        self.eye_value.config(

            text=(

                f"{self.eye_closure_duration:.2f}s"

            )

        )


        self.yawn_value.config(

            text=(

                f"{self.yawn_duration:.2f}s"

            )

        )


        self.pitch_value.config(

            text=f"{self.pitch:.1f}°"

        )


        self.yaw_value.config(

            text=f"{self.yaw:.1f}°"

        )


        self.roll_value.config(

            text=f"{self.roll:.1f}°"

        )


        self.head_value.config(

            text=(

                f"{self.head_down_duration:.2f}s"

            )

        )


        self.events_value.config(

            text=str(

                logger.drowsiness_events

            )

        )


        self.alert_value.config(

            text=(

                f"{logger.total_alert_time:.1f}s"

            )

        )


        self.max_value.config(

            text=(

                f"{logger.maximum_drowsiness_score * 100:.1f}%"

            )

        )


        self.fps_value.config(

            text=f"{self.fps:.1f}"

        )


        if alarm_active:

            self.alarm_label.config(

                text="● ALARM ACTIVE",

                fg=RED

            )

        else:

            self.alarm_label.config(

                text="● ALARM OFF",

                fg=GREEN

            )


        self.draw_score()


    # ========================================================
    # CAMERA UPDATE
    # ========================================================

    def update_frame(self):

        if not self.running:

            return


        ret, frame = cap.read()


        if not ret:

            self.connection.config(

                text="● CAMERA ERROR",

                fg=RED

            )

            self.root.after(

                100,

                self.update_frame

            )

            return


        frame = cv2.flip(

            frame,

            1

        )


        frame = self.process_frame(

            frame

        )


        # ----------------------------------------------------
        # CAMERA IMAGE
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        image = Image.fromarray(

            frame_rgb

        )


        # Fit image into available camera area

        image = image.resize(

            (

                max(
                    640,
                    self.camera_label.winfo_width()
                ),

                max(
                    480,
                    self.camera_label.winfo_height()
                )

            )

        )


        photo = ImageTk.PhotoImage(

            image=image

        )


        self.camera_label.configure(

            image=photo

        )


        self.camera_label.image = photo


        self.update_ui()


        self.root.after(

            15,

            self.update_frame

        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if not self.running:

            return


        self.running = False


        stop_alarm()


        # ----------------------------------------------------
        # SAVE SESSION
        # ----------------------------------------------------

        summary = logger.finish()


        # ----------------------------------------------------
        # RELEASE RESOURCES
        # ----------------------------------------------------

        cap.release()

        landmarker.close()


        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print()
        print("=" * 65)
        print("                 SESSION COMPLETE")
        print("=" * 65)

        print()

        print(
            "Session ID:",
            summary["session_id"]
        )

        print(
            "Duration:",
            f"{summary['duration']:.2f}s"
        )

        print(
            "Drowsiness events:",
            summary["drowsiness_events"]
        )

        print(
            "Alert time:",
            f"{summary['total_alert_time']:.2f}s"
        )

        print(
            "Maximum score:",
            f"{summary['maximum_score'] * 100:.1f}%"
        )

        print(
            "Average score:",
            f"{summary['average_score'] * 100:.1f}%"
        )

        print()

        print(
            "Saved to session_logs.csv"
        )

        print()


        self.root.destroy()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DriverDrowsinessApp(
        root
    )

    root.mainloop()