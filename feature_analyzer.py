import cv2
import math
import time
import threading
import winsound
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# DRIVER MONITORING AI
# FACE + EXPRESSION + HEAD + ENVIRONMENT + AUDIO ALERT
# ============================================================

MODEL_PATH = "face_landmarker.task"


# ============================================================
# DARK FUTURISTIC COLOR THEME
# ============================================================

BACKGROUND = (8, 10, 14)
PANEL = (15, 18, 24)
PANEL_LIGHT = (21, 25, 32)
BORDER = (45, 53, 65)

WHITE = (235, 240, 245)
GRAY = (135, 145, 155)
DARK_GRAY = (65, 75, 85)

BLUE = (220, 120, 35)
LIGHT_BLUE = (255, 170, 70)

GREEN = (100, 220, 120)
ORANGE = (80, 165, 255)
RED = (70, 70, 245)

FACE_POINT = (130, 95, 65)
EYE_POINT = (255, 150, 45)


# ============================================================
# FACE LANDMARKS
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

NOSE_TIP = 1
CHIN = 152

LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# ============================================================
# AUDIO CONTROL
# ============================================================

alarm_active = False
warning_played = False


def drowsiness_alarm():

    global alarm_active

    while alarm_active:

        try:

            winsound.Beep(
                1200,
                500
            )

            time.sleep(0.25)

        except:

            break


def start_alarm():

    global alarm_active

    if alarm_active:
        return

    alarm_active = True

    thread = threading.Thread(
        target=drowsiness_alarm,
        daemon=True
    )

    thread.start()


def stop_alarm():

    global alarm_active

    alarm_active = False


def warning_beep():

    try:

        winsound.Beep(
            800,
            250
        )

    except:

        pass


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(

        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2

    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(
    landmarks,
    points
):

    p1, p2, p3, p4, p5, p6 = [

        landmarks[i]

        for i in points

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

        return 0


    return (

        vertical_1 +
        vertical_2

    ) / (

        2 *
        horizontal

    )


# ============================================================
# MOUTH ASPECT RATIO
# ============================================================

def calculate_mar(
    landmarks
):

    vertical = distance(

        landmarks[MOUTH_TOP],

        landmarks[MOUTH_BOTTOM]

    )


    horizontal = distance(

        landmarks[MOUTH_LEFT],

        landmarks[MOUTH_RIGHT]

    )


    if horizontal == 0:

        return 0


    return vertical / horizontal


# ============================================================
# LANDMARK TO PIXEL
# ============================================================

def to_pixel(
    point,
    width,
    height
):

    return (

        int(point.x * width),

        int(point.y * height)

    )


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(

    landmarks,

    width,

    height

):

    image_points = np.array([

        to_pixel(
            landmarks[NOSE_TIP],
            width,
            height
        ),

        to_pixel(
            landmarks[CHIN],
            width,
            height
        ),

        to_pixel(
            landmarks[LEFT_EYE_CORNER],
            width,
            height
        ),

        to_pixel(
            landmarks[RIGHT_EYE_CORNER],
            width,
            height
        ),

        to_pixel(
            landmarks[LEFT_MOUTH],
            width,
            height
        ),

        to_pixel(
            landmarks[RIGHT_MOUTH],
            width,
            height
        )

    ], dtype=np.float64)


    # Approximate 3D face model

    model_points = np.array([

        (0.0, 0.0, 0.0),

        (0.0, -63.0, -12.0),

        (-43.0, 32.0, -26.0),

        (43.0, 32.0, -26.0),

        (-28.0, -28.0, -24.0),

        (28.0, -28.0, -24.0)

    ], dtype=np.float64)


    focal_length = width


    center = (

        width / 2,

        height / 2

    )


    camera_matrix = np.array([

        [
            focal_length,
            0,
            center[0]
        ],

        [
            0,
            focal_length,
            center[1]
        ],

        [
            0,
            0,
            1
        ]

    ], dtype=np.float64)


    distortion = np.zeros(
        (4, 1)
    )


    success, rotation_vector, _ = cv2.solvePnP(

        model_points,

        image_points,

        camera_matrix,

        distortion,

        flags=cv2.SOLVEPNP_ITERATIVE

    )


    if not success:

        return 0, 0, 0


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


        roll = 0


    return (

        math.degrees(pitch),

        math.degrees(yaw),

        math.degrees(roll)

    )


# ============================================================
# BLENDSHAPE SCORE
# ============================================================

def get_blendshape_score(

    blendshapes,

    name

):

    for category in blendshapes:

        if category.category_name == name:

            return category.score


    return 0.0


# ============================================================
# ENVIRONMENT ANALYZER
# ============================================================

def analyze_environment(

    frame,

    previous_gray

):

    gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY

    )


    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    brightness = float(

        np.mean(gray)

    )


    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    contrast = float(

        np.std(gray)

    )


    # --------------------------------------------------------
    # SHARPNESS
    # --------------------------------------------------------

    sharpness = float(

        cv2.Laplacian(

            gray,

            cv2.CV_64F

        ).var()

    )


    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    motion = 0.0


    if previous_gray is not None:

        difference = cv2.absdiff(

            gray,

            previous_gray

        )


        motion = float(

            np.mean(difference)

        )


    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    edges = cv2.Canny(

        gray,

        80,

        160

    )


    edge_density = (

        np.count_nonzero(edges)

        /

        edges.size

    )


    # --------------------------------------------------------
    # LIGHT STATUS
    # --------------------------------------------------------

    if brightness < 45:

        light_status = "VERY DARK"

    elif brightness < 80:

        light_status = "LOW LIGHT"

    elif brightness > 210:

        light_status = "VERY BRIGHT"

    else:

        light_status = "NORMAL"


    # --------------------------------------------------------
    # CAMERA STATUS
    # --------------------------------------------------------

    if sharpness < 50:

        camera_status = "VERY BLURRY"

    elif sharpness < 120:

        camera_status = "BLURRY"

    else:

        camera_status = "CLEAR"


    # --------------------------------------------------------
    # VISIBILITY
    # --------------------------------------------------------

    if contrast < 25:

        visibility_status = "LOW"

    elif contrast < 45:

        visibility_status = "MEDIUM"

    else:

        visibility_status = "GOOD"


    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    if motion > 30:

        motion_status = "HIGH"

    elif motion > 10:

        motion_status = "MEDIUM"

    else:

        motion_status = "LOW"


    return {

        "brightness": brightness,

        "contrast": contrast,

        "sharpness": sharpness,

        "motion": motion,

        "edge_density": edge_density,

        "light_status": light_status,

        "camera_status": camera_status,

        "visibility_status": visibility_status,

        "motion_status": motion_status,

        "gray": gray

    }


# ============================================================
# TEXT
# ============================================================

def draw_text(

    frame,

    value,

    position,

    size=0.42,

    color=WHITE,

    thickness=1

):

    cv2.putText(

        frame,

        value,

        position,

        cv2.FONT_HERSHEY_SIMPLEX,

        size,

        color,

        thickness,

        cv2.LINE_AA

    )


# ============================================================
# PANEL
# ============================================================

def draw_panel(

    frame,

    x1,

    y1,

    x2,

    y2,

    title

):

    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        PANEL,

        -1

    )


    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        BORDER,

        1

    )


    # Header

    cv2.line(

        frame,

        (x1, y1 + 32),

        (x2, y1 + 32),

        BORDER,

        1

    )


    draw_text(

        frame,

        title,

        (x1 + 14, y1 + 22),

        0.42,

        LIGHT_BLUE,

        1

    )


# ============================================================
# CORNER HUD
# ============================================================

def draw_corner_hud(frame):

    height, width = frame.shape[:2]

    length = 18

    color = (60, 80, 100)

    thickness = 1


    corners = [

        (20, 20, 1, 1),

        (width - 20, 20, -1, 1),

        (20, height - 20, 1, -1),

        (width - 20, height - 20, -1, -1)

    ]


    for x, y, sx, sy in corners:

        cv2.line(

            frame,

            (x, y),

            (
                x + sx * length,
                y
            ),

            color,

            thickness

        )


        cv2.line(

            frame,

            (x, y),

            (
                x,
                y + sy * length
            ),

            color,

            thickness

        )


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(

    model_asset_path=MODEL_PATH

)


options = vision.FaceLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_faces=2,

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
# CAMERA
# ============================================================

cap = cv2.VideoCapture(

    0,

    cv2.CAP_DSHOW

)


cap.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    640

)


cap.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    480

)


if not cap.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    print()

    exit()


# ============================================================
# VARIABLES
# ============================================================

timestamp_ms = 0

previous_gray = None

previous_time = time.time()

fps = 0

drowsiness_counter = 0

warning_cooldown = 0


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("              DRIVER MONITORING AI")
print("       DARK FACE + ENVIRONMENT ANALYZER")
print("=" * 70)
print()
print("Features:")
print("  Face landmarks")
print("  Eye / blink analysis")
print("  Mouth / yawning analysis")
print("  Jaw analysis")
print("  Head pose")
print("  Environment analysis")
print("  Drowsiness detection")
print("  Audio warning system")
print()
print("Press Q to exit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Camera frame error.")

        break


    # Mirror camera

    frame = cv2.flip(

        frame,

        1

    )


    height, width = frame.shape[:2]


    # ========================================================
    # DARK VISUAL FILTER
    # ========================================================

    dark_layer = np.full_like(

        frame,

        BACKGROUND

    )


    frame = cv2.addWeighted(

        frame,

        0.68,

        dark_layer,

        0.32,

        0

    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()


    delta = (

        current_time -
        previous_time

    )


    if delta > 0:

        fps = 1 / delta


    previous_time = current_time


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    environment = analyze_environment(

        frame,

        previous_gray

    )


    previous_gray = environment["gray"]


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )


    timestamp_ms += 33


    results = landmarker.detect_for_video(

        mp_image,

        timestamp_ms

    )


    # ========================================================
    # HUD
    # ========================================================

    draw_corner_hud(frame)


    draw_text(

        frame,

        "DRIVER MONITORING AI",

        (35, 48),

        0.60,

        WHITE,

        2

    )


    draw_text(

        frame,

        "MULTI-MODAL VISION SYSTEM",

        (36, 70),

        0.34,

        GRAY

    )


    draw_text(

        frame,

        f"{fps:.1f} FPS",

        (width - 82, 48),

        0.38,

        GRAY

    )


    # ========================================================
    # PANELS
    # ========================================================

    draw_panel(

        frame,

        20,
        90,
        215,
        315,
        "FACE ANALYTICS"

    )


    draw_panel(

        frame,

        width - 215,
        90,
        width - 20,
        315,
        "HEAD / POSE"

    )


    draw_panel(

        frame,

        20,
        330,
        300,
        455,
        "ENVIRONMENT"

    )


    draw_panel(

        frame,

        320,
        330,
        width - 20,
        455,
        "SYSTEM STATUS"

    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    if results.face_landmarks:

        face_count = len(

            results.face_landmarks

        )


        # Driver face

        landmarks = (

            results.face_landmarks[0]

        )


        # ====================================================
        # EYES
        # ====================================================

        left_ear = calculate_ear(

            landmarks,

            LEFT_EYE

        )


        right_ear = calculate_ear(

            landmarks,

            RIGHT_EYE

        )


        average_ear = (

            left_ear +
            right_ear

        ) / 2


        # ====================================================
        # MOUTH
        # ====================================================

        mar = calculate_mar(

            landmarks

        )


        # ====================================================
        # HEAD
        # ====================================================

        pitch, yaw, roll = calculate_head_pose(

            landmarks,

            width,

            height

        )


        # ====================================================
        # BLENDSHAPES
        # ====================================================

        blink_left = 0.0

        blink_right = 0.0

        jaw_open = 0.0

        brow_left = 0.0

        brow_right = 0.0


        if results.face_blendshapes:

            blendshapes = (

                results.face_blendshapes[0]

            )


            blink_left = get_blendshape_score(

                blendshapes,

                "eyeBlinkLeft"

            )


            blink_right = get_blendshape_score(

                blendshapes,

                "eyeBlinkRight"

            )


            jaw_open = get_blendshape_score(

                blendshapes,

                "jawOpen"

            )


            brow_left = get_blendshape_score(

                blendshapes,

                "browDownLeft"

            )


            brow_right = get_blendshape_score(

                blendshapes,

                "browDownRight"

            )


        # ====================================================
        # FACE MESH
        # ====================================================

        for point in landmarks:

            px = int(

                point.x *
                width

            )


            py = int(

                point.y *
                height

            )


            if (

                0 <= px < width and

                0 <= py < height

            ):

                cv2.circle(

                    frame,

                    (px, py),

                    1,

                    FACE_POINT,

                    -1

                )


        # ====================================================
        # EYE LANDMARKS
        # ====================================================

        for index in (

            LEFT_EYE +
            RIGHT_EYE

        ):

            px = int(

                landmarks[index].x *
                width

            )


            py = int(

                landmarks[index].y *
                height

            )


            cv2.circle(

                frame,

                (px, py),

                3,

                EYE_POINT,

                -1

            )


        # ====================================================
        # FACE BOUNDING BOX
        # ====================================================

        x_values = [

            int(
                p.x * width
            )

            for p in landmarks

        ]


        y_values = [

            int(
                p.y * height
            )

            for p in landmarks

        ]


        x1 = max(

            0,

            min(x_values)

        )


        x2 = min(

            width - 1,

            max(x_values)

        )


        y1 = max(

            0,

            min(y_values)

        )


        y2 = min(

            height - 1,

            max(y_values)

        )


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (55, 80, 105),

            1

        )


        # ====================================================
        # NOSE DIRECTION
        # ====================================================

        nose_x = int(

            landmarks[NOSE_TIP].x *
            width

        )


        nose_y = int(

            landmarks[NOSE_TIP].y *
            height

        )


        direction_length = 55


        yaw_rad = math.radians(yaw)

        pitch_rad = math.radians(pitch)


        direction_x = int(

            nose_x +

            direction_length *
            math.sin(yaw_rad)

        )


        direction_y = int(

            nose_y -

            direction_length *
            math.sin(pitch_rad)

        )


        cv2.arrowedLine(

            frame,

            (nose_x, nose_y),

            (
                direction_x,
                direction_y
            ),

            LIGHT_BLUE,

            2,

            tipLength=0.25

        )


        # ====================================================
        # FACE PANEL
        # ====================================================

        draw_text(

            frame,

            f"FACE COUNT     {face_count}",

            (34, 150),

            0.36

        )


        draw_text(

            frame,

            f"LEFT EAR       {left_ear:.3f}",

            (34, 176),

            0.36

        )


        draw_text(

            frame,

            f"RIGHT EAR      {right_ear:.3f}",

            (34, 202),

            0.36

        )


        draw_text(

            frame,

            f"AVG EAR        {average_ear:.3f}",

            (34, 228),

            0.36

        )


        draw_text(

            frame,

            f"MOUTH RATIO    {mar:.3f}",

            (34, 254),

            0.36

        )


        draw_text(

            frame,

            f"JAW OPEN       {jaw_open:.2f}",

            (34, 280),

            0.36

        )


        # ====================================================
        # HEAD PANEL
        # ====================================================

        draw_text(

            frame,

            f"PITCH       {pitch:+.1f} deg",

            (width - 202, 150),

            0.36

        )


        draw_text(

            frame,

            f"YAW         {yaw:+.1f} deg",

            (width - 202, 178),

            0.36

        )


        draw_text(

            frame,

            f"ROLL        {roll:+.1f} deg",

            (width - 202, 206),

            0.36

        )


        if abs(yaw) > 25:

            head_state = "LOOKING SIDE"

        elif pitch > 18:

            head_state = "HEAD DOWN"

        elif pitch < -18:

            head_state = "HEAD UP"

        elif abs(roll) > 20:

            head_state = "HEAD TILT"

        else:

            head_state = "FORWARD"


        draw_text(

            frame,

            "STATE",

            (width - 202, 245),

            0.34,

            GRAY

        )


        draw_text(

            frame,

            head_state,

            (width - 202, 270),

            0.42,

            LIGHT_BLUE,

            1

        )


        # ====================================================
        # ENVIRONMENT PANEL
        # ====================================================

        draw_text(

            frame,

            f"LIGHT       {environment['brightness']:.1f}",

            (34, 365),

            0.36

        )


        draw_text(

            frame,

            environment["light_status"],

            (190, 365),

            0.34,

            GRAY

        )


        draw_text(

            frame,

            f"CONTRAST    {environment['contrast']:.1f}",

            (34, 391),

            0.36

        )


        draw_text(

            frame,

            environment["visibility_status"],

            (190, 391),

            0.34,

            GRAY

        )


        draw_text(

            frame,

            f"SHARPNESS   {environment['sharpness']:.0f}",

            (34, 417),

            0.36

        )


        draw_text(

            frame,

            environment["camera_status"],

            (190, 417),

            0.34,

            GRAY

        )


        # ====================================================
        # SYSTEM PANEL
        # ====================================================

        draw_text(

            frame,

            f"SCENE MOTION     {environment['motion']:.1f}",

            (335, 365),

            0.36

        )


        draw_text(

            frame,

            f"MOTION STATE     {environment['motion_status']}",

            (335, 391),

            0.36

        )


        draw_text(

            frame,

            f"BLINK LEFT       {blink_left:.2f}",

            (335, 417),

            0.36

        )


        draw_text(

            frame,

            f"BLINK RIGHT      {blink_right:.2f}",

            (500, 417),

            0.36

        )


        # ====================================================
        # DRIVER POSITION
        # ====================================================

        face_center = (

            x1 + x2

        ) / 2


        center_offset = (

            face_center -

            width / 2

        )


        if abs(center_offset) < 70:

            position = "CENTERED"

        elif center_offset < 0:

            position = "LEFT"

        else:

            position = "RIGHT"


        draw_text(

            frame,

            f"DRIVER POSITION  {position}",

            (335, 442),

            0.36,

            LIGHT_BLUE

        )


        # ====================================================
        # DROWSINESS ENGINE
        # ====================================================

        eyes_closed = (

            average_ear < 0.22

        )


        mouth_open = (

            mar > 0.65

        )


        head_down = (

            pitch > 18

        )


        # Sustained eye closure

        if eyes_closed:

            drowsiness_counter += 1

        else:

            drowsiness_counter = max(

                0,

                drowsiness_counter - 1

            )


        # Warning cooldown

        if warning_cooldown > 0:

            warning_cooldown -= 1


        # ====================================================
        # DROWSINESS STATES
        # ====================================================

        if drowsiness_counter > 30:

            status = "DROWSINESS DETECTED"

            status_color = RED


            # Start repeating alarm

            start_alarm()


            warning_played = False


        elif (

            eyes_closed or

            mouth_open or

            head_down

        ):

            status = "ATTENTION WARNING"

            status_color = ORANGE


            # Stop repeating alarm

            stop_alarm()


            # Play a short warning once

            if (

                not warning_played and

                warning_cooldown == 0

            ):

                warning_beep()

                warning_played = True

                warning_cooldown = 80


        else:

            status = "DRIVER ATTENTIVE"

            status_color = GREEN


            # Stop alarm

            stop_alarm()


            warning_played = False


    # ========================================================
    # NO FACE
    # ========================================================

    else:

        status = "NO DRIVER DETECTED"

        status_color = ORANGE


        # Stop drowsiness alarm

        stop_alarm()


        warning_played = False


        draw_text(

            frame,

            "NO DRIVER FACE DETECTED",

            (
                width // 2 - 130,
                height // 2
            ),

            0.58,

            WHITE,

            2

        )


        draw_text(

            frame,

            "CHECK CAMERA POSITION",

            (
                width // 2 - 110,
                height // 2 + 30
            ),

            0.40,

            GRAY

        )


    # ========================================================
    # STATUS BAR
    # ========================================================

    bar_x1 = 145

    bar_x2 = width - 145

    bar_y1 = height - 62

    bar_y2 = height - 25


    cv2.rectangle(

        frame,

        (
            bar_x1,
            bar_y1
        ),

        (
            bar_x2,
            bar_y2
        ),

        PANEL_LIGHT,

        -1

    )


    cv2.rectangle(

        frame,

        (
            bar_x1,
            bar_y1
        ),

        (
            bar_x2,
            bar_y2
        ),

        status_color,

        1

    )


    status_width = (

        cv2.getTextSize(

            status,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            1

        )[0][0]

    )


    cv2.putText(

        frame,

        status,

        (

            width // 2 -
            status_width // 2,

            height - 38

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        status_color,

        1,

        cv2.LINE_AA

    )


    # ========================================================
    # FOOTER
    # ========================================================

    draw_text(

        frame,

        "AI VISION",

        (25, height - 8),

        0.30,

        DARK_GRAY

    )


    draw_text(

        frame,

        "AUDIO ALERT ACTIVE",

        (width - 140, height - 8),

        0.30,

        DARK_GRAY

    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        "Driver Monitoring AI",

        frame

    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        break


# ============================================================
# CLEANUP
# ============================================================

stop_alarm()

cap.release()

landmarker.close()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("Driver Monitoring AI stopped.")
print("=" * 70)