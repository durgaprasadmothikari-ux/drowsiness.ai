import cv2
import mediapipe as mp
import pickle
import math
import time
import threading
import queue
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import winsound

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from session_logger import SessionLogger


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "drowsiness_model_v3.pkl"
LANDMARK_MODEL = "face_landmarker.task"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Analyze every Nth frame.
# Camera frames are still displayed continuously.
PROCESS_EVERY_N_FRAMES = 2

# UI refresh rate
UI_UPDATE_MS = 30

# Drowsiness thresholds
DROWSY_THRESHOLD = 0.65
AWAKE_THRESHOLD = 0.40

# Temporal filtering
DROWSY_FRAMES_REQUIRED = 12
AWAKE_FRAMES_REQUIRED = 6

# Eye closure
EYE_CLOSURE_THRESHOLD = 0.21

# Yawn
YAWN_THRESHOLD = 0.55

# Head-down threshold
HEAD_DOWN_THRESHOLD = 20.0

# Alarm
ALARM_SCORE_THRESHOLD = 0.75


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Head pose points
POSE_LANDMARKS = [
    1,      # Nose
    152,    # Chin
    33,     # Left eye
    263,    # Right eye
    61,     # Left mouth
    291     # Right mouth
]


# ============================================================
# 3D FACE MODEL FOR HEAD POSE
# ============================================================

MODEL_POINTS = __import__("numpy").array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.0, -12.0),
        (-43.0, 32.0, -26.0),
        (43.0, 32.0, -26.0),
        (-28.0, -28.0, -24.0),
        (28.0, -28.0, -24.0)
    ],
    dtype="double"
)


# ============================================================
# FEATURE NAMES
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
    "smile_left",
    "smile_right",
    "pitch",
    "yaw",
    "roll",
    "brightness",
    "contrast",
    "sharpness",
    "motion",
    "eye_closure_duration",
    "yawn_duration",
    "head_down_duration"
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_ear(points, indices):
    try:
        p1 = points[indices[0]]
        p2 = points[indices[1]]
        p3 = points[indices[2]]
        p4 = points[indices[3]]
        p5 = points[indices[4]]
        p6 = points[indices[5]]

        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    except Exception:
        return 0.0


def calculate_mar(points):
    try:
        vertical = euclidean_distance(
            points[MOUTH_TOP],
            points[MOUTH_BOTTOM]
        )

        horizontal = euclidean_distance(
            points[MOUTH_LEFT],
            points[MOUTH_RIGHT]
        )

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    except Exception:
        return 0.0


def get_blendshape(blendshapes, name):
    try:
        for category in blendshapes:
            if category.category_name == name:
                return float(category.score)

        return 0.0

    except Exception:
        return 0.0


# ============================================================
# MAIN APPLICATION
# ============================================================

class DriverDrowsinessApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Driver Drowsiness AI — V2")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)

        self.root.configure(bg="#080b10")

        # ----------------------------------------------------
        # Application state
        # ----------------------------------------------------

        self.running = True
        self.camera_running = False

        self.frame_counter = 0

        self.latest_frame = None
        self.latest_result = None

        self.frame_lock = threading.Lock()

        self.result_queue = queue.Queue(maxsize=2)

        self.worker_thread = None

        self.camera = None
        self.landmarker = None

        self.previous_gray = None

        self.timestamp_ms = 0

        # Drowsiness state
        self.drowsy_frames = 0
        self.awake_frames = 0

        self.is_drowsy = False

        # Duration tracking
        self.eye_closure_start = None
        self.yawn_start = None
        self.head_down_start = None

        self.eye_closure_duration = 0.0
        self.yawn_duration = 0.0
        self.head_down_duration = 0.0

        # Alarm
        self.alarm_active = False
        self.alarm_thread = None
        self.alarm_stop_event = threading.Event()

        # Session
        self.session_logger = SessionLogger()

        self.session_start_time = time.time()

        self.total_frames = 0
        self.face_frames = 0
        self.drowsiness_events = 0

        self.current_score = 0.0
        self.max_score = 0.0

        self.last_event_state = False

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.load_model()

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # Start camera worker
        # ----------------------------------------------------

        self.start_camera()

        # ----------------------------------------------------
        # Window close
        # ----------------------------------------------------

        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        # Start UI update loop
        self.root.after(UI_UPDATE_MS, self.update_ui)


    # ========================================================
    # MODEL
    # ========================================================

    def load_model(self):

        try:

            with open(MODEL_PATH, "rb") as file:
                self.model = pickle.load(file)

            self.feature_names = self.model.get(
                "feature_names",
                FEATURE_NAMES
            )

            self.means = self.model.get(
                "means",
                [0.0] * len(self.feature_names)
            )

            self.stds = self.model.get(
                "stds",
                [1.0] * len(self.feature_names)
            )

            self.X_train = self.model["X_train"]
            self.y_train = self.model["y_train"]

            self.k = self.model.get("k", 5)

            self.feature_weights = self.model.get(
                "feature_weights",
                [1.0] * len(self.feature_names)
            )

            self.decision_threshold = self.model.get(
                "decision_threshold",
                0.5
            )

            print("Model loaded successfully.")
            print("Features:", len(self.feature_names))

        except Exception as error:

            messagebox.showerror(
                "Model Error",
                f"Could not load {MODEL_PATH}\n\n{error}"
            )

            self.root.destroy()


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        # Header
        header = tk.Frame(
            self.root,
            bg="#0d1118",
            height=70
        )

        header.pack(
            fill="x",
            side="top"
        )

        title = tk.Label(
            header,
            text="DRIVER DROWSINESS AI",
            font=("Segoe UI", 24, "bold"),
            fg="#ffffff",
            bg="#0d1118"
        )

        title.pack(
            side="left",
            padx=25,
            pady=15
        )

        self.status_label = tk.Label(
            header,
            text="● INITIALIZING",
            font=("Segoe UI", 12, "bold"),
            fg="#00e5ff",
            bg="#0d1118"
        )

        self.status_label.pack(
            side="right",
            padx=25
        )

        # Main container
        main = tk.Frame(
            self.root,
            bg="#080b10"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        # ====================================================
        # LEFT CAMERA
        # ====================================================

        camera_frame = tk.Frame(
            main,
            bg="#11161e",
            highlightbackground="#202a36",
            highlightthickness=1
        )

        camera_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        camera_title = tk.Label(
            camera_frame,
            text="LIVE CAMERA",
            font=("Segoe UI", 13, "bold"),
            fg="#8be9fd",
            bg="#11161e"
        )

        camera_title.pack(
            anchor="w",
            padx=15,
            pady=10
        )

        self.camera_label = tk.Label(
            camera_frame,
            bg="#05070a",
            text="Starting camera...",
            fg="#6b7280",
            font=("Segoe UI", 14)
        )

        self.camera_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # ====================================================
        # RIGHT ANALYTICS
        # ====================================================

        analytics = tk.Frame(
            main,
            bg="#0d1118",
            width=390
        )

        analytics.pack(
            side="right",
            fill="y"
        )

        analytics.pack_propagate(False)

        # Risk
        risk_card = tk.Frame(
            analytics,
            bg="#11161e",
            highlightbackground="#202a36",
            highlightthickness=1
        )

        risk_card.pack(
            fill="x",
            pady=(0, 10)
        )

        tk.Label(
            risk_card,
            text="DROWSINESS RISK",
            font=("Segoe UI", 11, "bold"),
            fg="#9ca3af",
            bg="#11161e"
        ).pack(
            anchor="w",
            padx=15,
            pady=(12, 2)
        )

        self.score_label = tk.Label(
            risk_card,
            text="0%",
            font=("Segoe UI", 34, "bold"),
            fg="#00e5ff",
            bg="#11161e"
        )

        self.score_label.pack(
            pady=3
        )

        self.risk_label = tk.Label(
            risk_card,
            text="ATTENTIVE",
            font=("Segoe UI", 13, "bold"),
            fg="#00ff88",
            bg="#11161e"
        )

        self.risk_label.pack(
            pady=(0, 12)
        )

        # Face analysis
        self.create_section(
            analytics,
            "FACE ANALYSIS"
        )

        self.face_label = tk.Label(
            analytics,
            text="Face: --\nLeft EAR: --\nRight EAR: --\nAverage EAR: --\nMAR: --",
            justify="left",
            font=("Consolas", 11),
            fg="#d1d5db",
            bg="#0d1118"
        )

        self.face_label.pack(
            anchor="w",
            padx=15,
            pady=(0, 12)
        )

        # Head
        self.create_section(
            analytics,
            "HEAD POSITION"
        )

        self.head_label = tk.Label(
            analytics,
            text="Pitch: --\nYaw: --\nRoll: --\nHead Down: --",
            justify="left",
            font=("Consolas", 11),
            fg="#d1d5db",
            bg="#0d1118"
        )

        self.head_label.pack(
            anchor="w",
            padx=15,
            pady=(0, 12)
        )

        # Environment
        self.create_section(
            analytics,
            "ENVIRONMENT"
        )

        self.environment_label = tk.Label(
            analytics,
            text="Brightness: --\nContrast: --\nSharpness: --\nMotion: --",
            justify="left",
            font=("Consolas", 11),
            fg="#d1d5db",
            bg="#0d1118"
        )

        self.environment_label.pack(
            anchor="w",
            padx=15,
            pady=(0, 12)
        )

        # Session
        self.create_section(
            analytics,
            "SESSION"

        )

        self.session_label = tk.Label(
            analytics,
            text="Frames: 0\nFace Frames: 0\nDrowsiness Events: 0\nMaximum Risk: 0%",
            justify="left",
            font=("Consolas", 11),
            fg="#d1d5db",
            bg="#0d1118"
        )

        self.session_label.pack(
            anchor="w",
            padx=15,
            pady=(0, 12)
        )

        # Alarm
        self.alarm_label = tk.Label(
            analytics,
            text="● ALARM OFF",
            font=("Segoe UI", 12, "bold"),
            fg="#6b7280",
            bg="#0d1118"
        )

        self.alarm_label.pack(
            pady=10
        )

        # Footer
        footer = tk.Label(
            self.root,
            text="V2 PERFORMANCE MODE  •  MediaPipe + Machine Learning + Behavioral Analysis",
            font=("Segoe UI", 9),
            fg="#4b5563",
            bg="#080b10"
        )

        footer.pack(
            side="bottom",
            pady=5
        )


    def create_section(self, parent, text):

        frame = tk.Frame(
            parent,
            bg="#0d1118"
        )

        frame.pack(
            fill="x"
        )

        tk.Label(
            frame,
            text=text,
            font=("Segoe UI", 10, "bold"),
            fg="#8be9fd",
            bg="#0d1118"
        ).pack(
            anchor="w",
            padx=15,
            pady=(5, 5)
        )


    # ========================================================
    # CAMERA
    # ========================================================

    def start_camera(self):

        self.camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.camera.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Could not open the camera."
            )

            self.running = False
            return

        # MediaPipe
        base_options = python.BaseOptions(
            model_asset_path=LANDMARK_MODEL
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

        self.landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

        self.camera_running = True

        self.worker_thread = threading.Thread(
            target=self.camera_worker,
            daemon=True
        )

        self.worker_thread.start()

        self.status_label.config(
            text="● SYSTEM ACTIVE",
            fg="#00ff88"
        )


    # ========================================================
    # CAMERA WORKER THREAD
    # ========================================================

    def camera_worker(self):

        while self.running and self.camera_running:

            success, frame = self.camera.read()

            if not success:
                time.sleep(0.01)
                continue

            self.total_frames += 1

            # Always keep newest frame
            with self.frame_lock:
                self.latest_frame = frame.copy()

            self.frame_counter += 1

            # Process only selected frames
            if self.frame_counter % PROCESS_EVERY_N_FRAMES != 0:
                continue

            try:

                result = self.process_frame(frame)

                if result is not None:

                    # Keep only latest result
                    try:
                        while not self.result_queue.empty():
                            self.result_queue.get_nowait()
                    except queue.Empty:
                        pass

                    self.result_queue.put_nowait(result)

            except Exception as error:

                print(
                    "Processing error:",
                    error
                )

        print("Camera worker stopped.")


    # ========================================================
    # FRAME PROCESSING
    # ========================================================

    def process_frame(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp_ms += 33

        detection = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        # Environment
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(gray.mean())
        contrast = float(gray.std())

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        motion = 0.0

        if self.previous_gray is not None:

            motion = float(
                cv2.absdiff(
                    gray,
                    self.previous_gray
                ).mean()
            )

        self.previous_gray = gray

        # No face
        if not detection.face_landmarks:

            self.face_frames = self.face_frames

            return {
                "frame": frame.copy(),
                "face": False,
                "score": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "average_ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "motion": motion,
                "eye_closure_duration": 0.0,
                "yawn_duration": 0.0,
                "head_down_duration": 0.0,
                "drowsy": False
            }

        self.face_frames += 1

        landmarks = detection.face_landmarks[0]

        # ----------------------------------------------------
        # Landmark coordinates
        # ----------------------------------------------------

        points = []

        for landmark in landmarks:

            points.append(
                (
                    landmark.x * FRAME_WIDTH,
                    landmark.y * FRAME_HEIGHT
                )
            )

        # ----------------------------------------------------
        # EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            points,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            points,
            RIGHT_EYE
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0

        # ----------------------------------------------------
        # MAR
        # ----------------------------------------------------

        mar = calculate_mar(points)

        # ----------------------------------------------------
        # Blendshapes
        # ----------------------------------------------------

        blink_left = 0.0
        blink_right = 0.0
        jaw_open = 0.0
        brow_left = 0.0
        brow_right = 0.0
        smile_left = 0.0
        smile_right = 0.0

        if detection.face_blendshapes:

            blendshapes = detection.face_blendshapes[0]

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

        # ----------------------------------------------------
        # Eye closure duration
        # ----------------------------------------------------

        current_time = time.time()

        eyes_closed = (
            average_ear < EYE_CLOSURE_THRESHOLD
            or
            (
                blink_left > 0.65
                and blink_right > 0.65
            )
        )

        if eyes_closed:

            if self.eye_closure_start is None:
                self.eye_closure_start = current_time

            self.eye_closure_duration = (
                current_time -
                self.eye_closure_start
            )

        else:

            self.eye_closure_start = None
            self.eye_closure_duration = 0.0

        # ----------------------------------------------------
        # Yawn duration
        # ----------------------------------------------------

        yawning = (
            mar > YAWN_THRESHOLD
            or jaw_open > 0.65
        )

        if yawning:

            if self.yawn_start is None:
                self.yawn_start = current_time

            self.yawn_duration = (
                current_time -
                self.yawn_start
            )

        else:

            self.yawn_start = None
            self.yawn_duration = 0.0

        # ----------------------------------------------------
        # Head pose
        # ----------------------------------------------------

        pitch, yaw, roll = self.calculate_head_pose(
            points
        )

        head_down = pitch > HEAD_DOWN_THRESHOLD

        if head_down:

            if self.head_down_start is None:
                self.head_down_start = current_time

            self.head_down_duration = (
                current_time -
                self.head_down_start
            )

        else:

            self.head_down_start = None
            self.head_down_duration = 0.0

        # ----------------------------------------------------
        # Feature vector
        # ----------------------------------------------------

        feature_dict = {

            "left_ear": left_ear,
            "right_ear": right_ear,
            "average_ear": average_ear,

            "mar": mar,

            "blink_left": blink_left,
            "blink_right": blink_right,

            "jaw_open": jaw_open,

            "brow_left": brow_left,
            "brow_right": brow_right,

            "smile_left": smile_left,
            "smile_right": smile_right,

            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,

            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "motion": motion,

            "eye_closure_duration":
                self.eye_closure_duration,

            "yawn_duration":
                self.yawn_duration,

            "head_down_duration":
                self.head_down_duration
        }

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        ml_score = self.predict_knn(
            feature_dict
        )

        # ----------------------------------------------------
        # Behavioral score
        # ----------------------------------------------------

        behavior_score = self.calculate_behavior_score(
            average_ear,
            mar,
            blink_left,
            blink_right,
            jaw_open,
            pitch,
            self.eye_closure_duration,
            self.yawn_duration,
            self.head_down_duration
        )

        # Combine ML + behavior
        final_score = (
            0.70 * ml_score +
            0.30 * behavior_score
        )

        final_score = clamp(
            final_score
        )

        self.current_score = final_score

        if final_score > self.max_score:
            self.max_score = final_score

        # ----------------------------------------------------
        # Temporal filtering
        # ----------------------------------------------------

        if final_score >= DROWSY_THRESHOLD:

            self.drowsy_frames += 1
            self.awake_frames = 0

        elif final_score <= AWAKE_THRESHOLD:

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

        previous_state = self.is_drowsy

        if self.drowsy_frames >= DROWSY_FRAMES_REQUIRED:

            self.is_drowsy = True

        elif self.awake_frames >= AWAKE_FRAMES_REQUIRED:

            self.is_drowsy = False

        # New drowsiness event
        if (
            self.is_drowsy
            and not previous_state
        ):

            self.drowsiness_events += 1

        # Alarm
        if self.is_drowsy or final_score >= ALARM_SCORE_THRESHOLD:

            self.start_alarm()

        else:

            self.stop_alarm()

        # ----------------------------------------------------
        # Draw camera HUD
        # ----------------------------------------------------

        display_frame = frame.copy()

        self.draw_hud(
            display_frame,
            final_score,
            left_ear,
            right_ear,
            average_ear,
            mar,
            pitch,
            yaw,
            roll
        )

        return {

            "frame": display_frame,

            "face": True,

            "score": final_score,

            "left_ear": left_ear,
            "right_ear": right_ear,
            "average_ear": average_ear,

            "mar": mar,

            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,

            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "motion": motion,

            "eye_closure_duration":
                self.eye_closure_duration,

            "yawn_duration":
                self.yawn_duration,

            "head_down_duration":
                self.head_down_duration,

            "drowsy":
                self.is_drowsy
        }


    # ========================================================
    # HEAD POSE
    # ========================================================

    def calculate_head_pose(self, points):

        try:

            import numpy as np

            image_points = np.array(
                [
                    points[1],
                    points[152],
                    points[33],
                    points[263],
                    points[61],
                    points[291]
                ],
                dtype="double"
            )

            focal_length = FRAME_WIDTH

            center = (
                FRAME_WIDTH / 2,
                FRAME_HEIGHT / 2
            )

            camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ],
                dtype="double"
            )

            dist_coeffs = np.zeros(
                (4, 1)
            )

            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS,
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

                x = math.atan2(
                    rotation_matrix[2, 1],
                    rotation_matrix[2, 2]
                )

                y = math.atan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                z = math.atan2(
                    rotation_matrix[1, 0],
                    rotation_matrix[0, 0]
                )

            else:

                x = math.atan2(
                    -rotation_matrix[1, 2],
                    rotation_matrix[1, 1]
                )

                y = math.atan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                z = 0

            pitch = math.degrees(x)
            yaw = math.degrees(y)
            roll = math.degrees(z)

            return pitch, yaw, roll

        except Exception:

            return 0.0, 0.0, 0.0


    # ========================================================
    # KNN
    # ========================================================

    def predict_knn(self, feature_dict):

        try:

            values = []

            for feature in self.feature_names:

                values.append(
                    float(
                        feature_dict.get(
                            feature,
                            0.0
                        )
                    )
                )

            normalized = []

            for i, value in enumerate(values):

                mean = float(
                    self.means[i]
                )

                std = float(
                    self.stds[i]
                )

                if std == 0:
                    std = 1.0

                normalized_value = (
                    value - mean
                ) / std

                weight = float(
                    self.feature_weights[i]
                )

                normalized_value *= math.sqrt(
                    max(weight, 0.0001)
                )

                normalized.append(
                    normalized_value
                )

            distances = []

            for i, training_sample in enumerate(
                self.X_train
            ):

                distance = 0.0

                for j in range(
                    len(normalized)
                ):

                    difference = (
                        normalized[j]
                        -
                        float(training_sample[j])
                    )

                    distance += (
                        difference ** 2
                    )

                distance = math.sqrt(
                    distance
                )

                distances.append(
                    (
                        distance,
                        self.y_train[i]
                    )
                )

            distances.sort(
                key=lambda x: x[0]
            )

            neighbors = distances[
                :self.k
            ]

            drowsy_votes = 0.0
            total_weight = 0.0

            for distance, label in neighbors:

                weight = 1.0 / (
                    distance + 0.000001
                )

                total_weight += weight

                if int(label) == 1:
                    drowsy_votes += weight

            if total_weight == 0:
                return 0.0

            return clamp(
                drowsy_votes /
                total_weight
            )

        except Exception as error:

            print(
                "KNN error:",
                error
            )

            return 0.0


    # ========================================================
    # BEHAVIOR SCORE
    # ========================================================

    def calculate_behavior_score(
        self,
        ear,
        mar,
        blink_left,
        blink_right,
        jaw_open,
        pitch,
        eye_duration,
        yawn_duration,
        head_duration
    ):

        score = 0.0

        # Eyes
        if ear < 0.18:
            score += 0.35

        elif ear < 0.21:
            score += 0.20

        elif ear < 0.24:
            score += 0.08

        # Blink intensity
        if (
            blink_left > 0.7
            and blink_right > 0.7
        ):
            score += 0.10

        # Long eye closure
        if eye_duration > 1.0:
            score += 0.25

        elif eye_duration > 0.5:
            score += 0.12

        # Yawning
        if (
            mar > 0.65
            or jaw_open > 0.7
        ):
            score += 0.15

        # Long yawn
        if yawn_duration > 1.5:
            score += 0.10

        # Head down
        if abs(pitch) > HEAD_DOWN_THRESHOLD:
            score += 0.10

        # Long head-down duration
        if head_duration > 1.5:
            score += 0.15

        return clamp(
            score
        )


    # ========================================================
    # HUD
    # ========================================================

    def draw_hud(
        self,
        frame,
        score,
        left_ear,
        right_ear,
        average_ear,
        mar,
        pitch,
        yaw,
        roll
    ):

        height, width = frame.shape[:2]

        # Face status
        status = (
            "DROWSINESS DETECTED"
            if self.is_drowsy
            else "ATTENTIVE"
        )

        # Top panel
        cv2.rectangle(
            frame,
            (10, 10),
            (340, 115),
            (10, 15, 22),
            -1
        )

        cv2.putText(
            frame,
            status,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (
                (0, 0, 255)
                if self.is_drowsy
                else (0, 255, 120)
            ),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {score * 100:.1f}%",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"EAR: {average_ear:.2f}   MAR: {mar:.2f}",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 220, 255),
            1
        )

        # Bottom panel
        cv2.rectangle(
            frame,
            (10, height - 105),
            (350, height - 10),
            (10, 15, 22),
            -1
        )

        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (25, height - 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Yaw:   {yaw:.1f}",
            (25, height - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Roll:  {roll:.1f}",
            (25, height - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1
        )


    # ========================================================
    # ALARM
    # ========================================================

    def start_alarm(self):

        if self.alarm_active:
            return

        self.alarm_active = True

        self.alarm_stop_event.clear()

        self.alarm_thread = threading.Thread(
            target=self.alarm_worker,
            daemon=True
        )

        self.alarm_thread.start()


    def alarm_worker(self):

        while (
            self.alarm_active
            and self.running
            and not self.alarm_stop_event.is_set()
        ):

            try:

                winsound.Beep(
                    1200,
                    400
                )

            except Exception:
                pass

            if self.alarm_stop_event.wait(
                0.2
            ):
                break


    def stop_alarm(self):

        if not self.alarm_active:
            return

        self.alarm_active = False

        self.alarm_stop_event.set()


    # ========================================================
    # UI UPDATE
    # ========================================================

    def update_ui(self):

        if not self.running:
            return

        # ----------------------------------------------------
        # Camera frame
        # ----------------------------------------------------

        with self.frame_lock:

            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )

        if frame is not None:

            # Display latest processed frame if available
            result = None

            try:

                while True:
                    result = self.result_queue.get_nowait()

            except queue.Empty:
                pass

            if result is not None:

                frame = result["frame"]

                self.latest_result = result

            # Convert BGR -> RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                rgb
            )

            # Get available camera panel size
            panel_width = max(
                640,
                self.camera_label.winfo_width()
            )

            panel_height = max(
                480,
                self.camera_label.winfo_height()
            )

            # Preserve aspect ratio
            image.thumbnail(
                (
                    panel_width - 20,
                    panel_height - 20
                ),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                image=image
            )

            self.camera_label.configure(
                image=photo,
                text=""
            )

            self.camera_label.image = photo

        # ----------------------------------------------------
        # Analytics
        # ----------------------------------------------------

        result = self.latest_result

        if result is not None:

            score = result["score"]

            # Risk
            self.score_label.config(
                text=f"{score * 100:.0f}%"
            )

            if result["drowsy"]:

                self.risk_label.config(
                    text="DROWSY",
                    fg="#ff3b30"
                )

                self.status_label.config(
                    text="● DROWSINESS ALERT",
                    fg="#ff3b30"
                )

            elif score >= 0.65:

                self.risk_label.config(
                    text="HIGH RISK",
                    fg="#ff9f0a"
                )

                self.status_label.config(
                    text="● HIGH RISK",
                    fg="#ff9f0a"
                )

            elif score >= 0.40:

                self.risk_label.config(
                    text="MODERATE",
                    fg="#ffd60a"
                )

                self.status_label.config(
                    text="● MONITORING",
                    fg="#ffd60a"
                )

            else:

                self.risk_label.config(
                    text="ATTENTIVE",
                    fg="#00ff88"
                )

                self.status_label.config(
                    text="● SYSTEM ACTIVE",
                    fg="#00ff88"
                )

            # Face
            if result["face"]:

                self.face_label.config(
                    text=(
                        "Face: DETECTED\n"
                        f"Left EAR: {result['left_ear']:.3f}\n"
                        f"Right EAR: {result['right_ear']:.3f}\n"
                        f"Average EAR: {result['average_ear']:.3f}\n"
                        f"MAR: {result['mar']:.3f}\n"
                        f"Eye closure: "
                        f"{result['eye_closure_duration']:.2f}s\n"
                        f"Yawn: "
                        f"{result['yawn_duration']:.2f}s"
                    )
                )

            else:

                self.face_label.config(
                    text=(
                        "Face: NOT DETECTED\n"
                        "Waiting for driver..."
                    )
                )

            # Head
            self.head_label.config(
                text=(
                    f"Pitch: {result['pitch']:.1f}°\n"
                    f"Yaw: {result['yaw']:.1f}°\n"
                    f"Roll: {result['roll']:.1f}°\n"
                    f"Head Down: "
                    f"{result['head_down_duration']:.2f}s"
                )
            )

            # Environment
            self.environment_label.config(
                text=(
                    f"Brightness: "
                    f"{result['brightness']:.1f}\n"
                    f"Contrast: "
                    f"{result['contrast']:.1f}\n"
                    f"Sharpness: "
                    f"{result['sharpness']:.1f}\n"
                    f"Motion: "
                    f"{result['motion']:.2f}"
                )
            )

            # Session
            self.session_label.config(
                text=(
                    f"Frames: {self.total_frames}\n"
                    f"Face Frames: {self.face_frames}\n"
                    f"Drowsiness Events: "
                    f"{self.drowsiness_events}\n"
                    f"Maximum Risk: "
                    f"{self.max_score * 100:.1f}%"
                )
            )

            # Alarm
            if self.alarm_active:

                self.alarm_label.config(
                    text="● ALARM ACTIVE",
                    fg="#ff3b30"
                )

            else:

                self.alarm_label.config(
                    text="● ALARM OFF",
                    fg="#6b7280"
                )

        self.root.after(
            UI_UPDATE_MS,
            self.update_ui
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close_application(self):

        if not self.running:
            return

        self.running = False
        self.camera_running = False

        # Stop alarm
        self.stop_alarm()

        # Release camera
        if self.camera is not None:

            try:
                self.camera.release()

            except Exception:
                pass

        # Close MediaPipe
        if self.landmarker is not None:

            try:
                self.landmarker.close()

            except Exception:
                pass

        # Save session
        try:

            duration = (
                time.time() -
                self.session_start_time
            )

            self.session_logger.end_session(
                total_frames=self.total_frames,
                face_detected_frames=self.face_frames,
                drowsiness_events=self.drowsiness_events,
                total_alert_time=0.0,
                longest_drowsy_event=0.0,
                longest_eye_closure=0.0,
                longest_yawn=0.0,
                longest_head_down=0.0,
                maximum_drowsiness_score=self.max_score,
                average_drowsiness_score=self.current_score
            )

            print()
            print("=" * 50)
            print("SESSION ENDED")
            print("=" * 50)
            print(
                f"Duration: {duration:.1f} seconds"
            )
            print(
                f"Frames: {self.total_frames}"
            )
            print(
                f"Face frames: {self.face_frames}"
            )
            print(
                f"Drowsiness events: "
                f"{self.drowsiness_events}"
            )
            print(
                f"Maximum risk: "
                f"{self.max_score * 100:.1f}%"
            )
            print("=" * 50)

        except Exception as error:

            print(
                "Session logging error:",
                error
            )

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