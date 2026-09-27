import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import time
from collections import deque


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_MODEL = r"C:\TrafficSignProject\models\final\traffic_sign_detector_best.pt"
CNN_MODEL = r"C:\TrafficSignProject\models\final\traffic_classifier.h5"


# ============================================================
# SETTINGS — SAME AS WORKING VERSION
# ============================================================

CAMERA_ID = 0

YOLO_CONF = 0.30
CNN_CONF = 0.70

CONFIRM_FRAMES = 4
TRACK_DISTANCE = 80

VOICE_COOLDOWN = 5.0
MEMORY_TIME = 1.5


# ============================================================
# 43 GTSRB CLASSES
# ============================================================

CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons"
}


# ============================================================
# VOICE
# ============================================================

voice_lock = threading.Lock()
last_voice_time = 0


def speak(text):
    global last_voice_time

    now = time.time()

    if now - last_voice_time < VOICE_COOLDOWN:
        return

    last_voice_time = now

    def worker():
        with voice_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print("Voice error:", e)

    threading.Thread(target=worker, daemon=True).start()


def make_alert(label):
    label_lower = label.lower()

    if "stop" in label_lower:
        return "Stop sign detected"

    if "yield" in label_lower:
        return "Yield sign detected"

    if "no entry" in label_lower:
        return "No entry sign detected"

    if "speed limit" in label_lower:
        return label.replace("(", "").replace(")", "") + " detected"

    if "traffic signals" in label_lower:
        return "Traffic signal detected"

    if "pedestrians" in label_lower:
        return "Pedestrian warning"

    if "children" in label_lower:
        return "Children crossing warning"

    if "road work" in label_lower:
        return "Road work ahead"

    return f"{label} detected"


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading CNN...")
cnn = load_model(CNN_MODEL)

print("Models loaded successfully.")


# ============================================================
# TRACKING / TEMPORAL CONFIRMATION
# ============================================================

sign_histories = []
total_confirmed = 0


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2) ** 0.5


def confirm_detection(label, center):

    global total_confirmed

    current_time = time.time()
    matched = None

    for item in sign_histories:

        if item["label"] == label:

            if distance(item["center"], center) < TRACK_DISTANCE:
                matched = item
                break

    if matched is None:

        matched = {
            "label": label,
            "center": center,
            "history": deque(maxlen=CONFIRM_FRAMES),
            "last_seen": current_time,
            "confirmed": False
        }

        sign_histories.append(matched)

    matched["center"] = center
    matched["last_seen"] = current_time
    matched["history"].append(current_time)

    if len(matched["history"]) >= CONFIRM_FRAMES:

        if not matched["confirmed"]:

            matched["confirmed"] = True
            total_confirmed += 1

            return True

    return False


def cleanup_tracking():

    current_time = time.time()

    sign_histories[:] = [
        item
        for item in sign_histories
        if current_time - item["last_seen"] < MEMORY_TIME
    ]


# ============================================================
# MAIN GUI
# ============================================================

root = tk.Tk()

root.title("VISION ADAS - Traffic Sign Detection")
root.geometry("1450x850")
root.minsize(1100, 700)
root.configure(bg="#0b0f12")


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except Exception:
    pass

style.configure(
    "ADAS.TButton",
    font=("Segoe UI", 10, "bold"),
    padding=(15, 8)
)


# ============================================================
# VARIABLES
# ============================================================

camera_running = False
voice_enabled = True
fullscreen = False

cap = None

fps = 0.0
fps_counter = 0
fps_start = time.time()

current_sign = "No Sign Detected"
current_yolo = 0.0
current_cnn = 0.0
current_detection_count = 0


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg="#151b20",
    height=75
)

header.pack(
    fill="x",
    side="top"
)

header.pack_propagate(False)


title_frame = tk.Frame(
    header,
    bg="#151b20"
)

title_frame.pack(
    side="left",
    padx=25
)


tk.Label(
    title_frame,
    text="VISION ADAS",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#151b20"
).pack(
    side="left",
    pady=10
)


tk.Label(
    title_frame,
    text="  |  TRAFFIC SIGN INTELLIGENCE",
    font=("Segoe UI", 10),
    fg="#8b949e",
    bg="#151b20"
).pack(
    side="left",
    pady=15
)


status_label = tk.Label(
    header,
    text="● SYSTEM READY",
    font=("Segoe UI", 11, "bold"),
    fg="#4ade80",
    bg="#151b20"
)

status_label.pack(
    side="right",
    padx=25
)


# ============================================================
# MAIN CONTENT
# ============================================================

content = tk.Frame(
    root,
    bg="#0b0f12"
)

content.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


# ============================================================
# CAMERA PANEL
# ============================================================

camera_panel = tk.Frame(
    content,
    bg="#050708",
    highlightbackground="#26313a",
    highlightthickness=1
)

camera_panel.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


camera_title = tk.Frame(
    camera_panel,
    bg="#151b20",
    height=42
)

camera_title.pack(
    fill="x"
)

camera_title.pack_propagate(False)


tk.Label(
    camera_title,
    text="LIVE ROAD VIEW",
    font=("Segoe UI", 11, "bold"),
    fg="#dce3e8",
    bg="#151b20"
).pack(
    side="left",
    padx=15
)


camera_status = tk.Label(
    camera_title,
    text="● OFFLINE",
    font=("Segoe UI", 9, "bold"),
    fg="#9ca3af",
    bg="#151b20"
)

camera_status.pack(
    side="right",
    padx=15
)


video_frame = tk.Label(
    camera_panel,
    text="CAMERA OFFLINE\n\nPress START CAMERA",
    font=("Segoe UI", 22, "bold"),
    fg="#68737d",
    bg="#050708"
)

video_frame.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ============================================================
# RIGHT INFORMATION PANEL
# ============================================================

right_panel = tk.Frame(
    content,
    bg="#11171c",
    width=360
)

right_panel.pack(
    side="right",
    fill="y"
)

right_panel.pack_propagate(False)


# ------------------------------------------------------------
# CURRENT DETECTION
# ------------------------------------------------------------

section_title = tk.Label(
    right_panel,
    text="CURRENT DETECTION",
    font=("Segoe UI", 10, "bold"),
    fg="#7f8a93",
    bg="#11171c"
)

section_title.pack(
    anchor="w",
    padx=20,
    pady=(20, 5)
)


sign_label = tk.Label(
    right_panel,
    text="No Sign Detected",
    font=("Segoe UI", 19, "bold"),
    fg="white",
    bg="#11171c",
    wraplength=310,
    justify="left"
)

sign_label.pack(
    anchor="w",
    padx=20,
    pady=(5, 15)
)


# Confidence cards

confidence_frame = tk.Frame(
    right_panel,
    bg="#182027"
)

confidence_frame.pack(
    fill="x",
    padx=20
)


yolo_value = tk.Label(
    confidence_frame,
    text="YOLO\n--",
    font=("Segoe UI", 12, "bold"),
    fg="#60a5fa",
    bg="#182027"
)

yolo_value.pack(
    side="left",
    expand=True,
    pady=12
)


cnn_value = tk.Label(
    confidence_frame,
    text="CNN\n--",
    font=("Segoe UI", 12, "bold"),
    fg="#a78bfa",
    bg="#182027"
)

cnn_value.pack(
    side="right",
    expand=True,
    pady=12
)


# ------------------------------------------------------------
# SYSTEM STATUS
# ------------------------------------------------------------

tk.Label(
    right_panel,
    text="SYSTEM STATUS",
    font=("Segoe UI", 10, "bold"),
    fg="#7f8a93",
    bg="#11171c"
).pack(
    anchor="w",
    padx=20,
    pady=(25, 8)
)


def status_row(text):

    label = tk.Label(
        right_panel,
        text=text,
        font=("Segoe UI", 10, "bold"),
        fg="#4ade80",
        bg="#11171c"
    )

    label.pack(
        anchor="w",
        padx=20,
        pady=2
    )

    return label


yolo_status = status_row("●  YOLO DETECTOR       ACTIVE")
cnn_status = status_row("●  CNN RECOGNIZER      ACTIVE")
voice_status = status_row("●  VOICE ALERT         ACTIVE")


# ------------------------------------------------------------
# PERFORMANCE
# ------------------------------------------------------------

tk.Label(
    right_panel,
    text="PERFORMANCE",
    font=("Segoe UI", 10, "bold"),
    fg="#7f8a93",
    bg="#11171c"
).pack(
    anchor="w",
    padx=20,
    pady=(25, 8)
)


performance_frame = tk.Frame(
    right_panel,
    bg="#182027"
)

performance_frame.pack(
    fill="x",
    padx=20
)


fps_label = tk.Label(
    performance_frame,
    text="FPS\n--",
    font=("Segoe UI", 11, "bold"),
    fg="white",
    bg="#182027"
)

fps_label.pack(
    side="left",
    expand=True,
    pady=12
)


detection_count_label = tk.Label(
    performance_frame,
    text="DETECTIONS\n0",
    font=("Segoe UI", 11, "bold"),
    fg="white",
    bg="#182027"
)

detection_count_label.pack(
    side="right",
    expand=True,
    pady=12
)


# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

tk.Label(
    right_panel,
    text="DETECTION HISTORY",
    font=("Segoe UI", 10, "bold"),
    fg="#7f8a93",
    bg="#11171c"
).pack(
    anchor="w",
    padx=20,
    pady=(25, 8)
)


history_list = tk.Listbox(
    right_panel,
    height=7,
    bg="#0b1014",
    fg="#dbe3e8",
    selectbackground="#263640",
    selectforeground="white",
    borderwidth=0,
    highlightthickness=0,
    font=("Segoe UI", 9)
)

history_list.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=(0, 15)
)


# ============================================================
# BOTTOM CONTROL BAR
# ============================================================

control_bar = tk.Frame(
    root,
    bg="#151b20",
    height=70
)

control_bar.pack(
    fill="x",
    side="bottom"
)

control_bar.pack_propagate(False)


stats_label = tk.Label(
    control_bar,
    text="YOLOv8 + CNN  |  43 GTSRB CLASSES",
    font=("Segoe UI", 9, "bold"),
    fg="#7f8a93",
    bg="#151b20"
)

stats_label.pack(
    side="left",
    padx=20
)


# ============================================================
# CAMERA CONTROL FUNCTIONS
# ============================================================

def start_camera():

    global cap
    global camera_running

    if camera_running:
        return

    cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():

        status_label.config(
            text="● CAMERA ERROR",
            fg="#ef4444"
        )

        camera_status.config(
            text="● ERROR",
            fg="#ef4444"
        )

        return

    camera_running = True

    status_label.config(
        text="● AI SYSTEM ACTIVE",
        fg="#4ade80"
    )

    camera_status.config(
        text="● LIVE",
        fg="#4ade80"
    )

    start_button.config(
        text="STOP CAMERA"
    )

    update_frame()


def stop_camera():

    global cap
    global camera_running

    camera_running = False

    if cap is not None:

        cap.release()
        cap = None

    camera_status.config(
        text="● OFFLINE",
        fg="#9ca3af"
    )

    status_label.config(
        text="● SYSTEM READY",
        fg="#4ade80"
    )

    start_button.config(
        text="START CAMERA"
    )

    video_frame.configure(
        image="",
        text="CAMERA OFFLINE\n\nPress START CAMERA"
    )

    video_frame.image = None


def toggle_camera():

    if camera_running:
        stop_camera()
    else:
        start_camera()


# ============================================================
# VOICE CONTROL
# ============================================================

def toggle_voice():

    global voice_enabled

    voice_enabled = not voice_enabled

    if voice_enabled:

        voice_status.config(
            text="●  VOICE ALERT         ACTIVE",
            fg="#4ade80"
        )

        mute_button.config(
            text="MUTE VOICE"
        )

    else:

        voice_status.config(
            text="●  VOICE ALERT         MUTED",
            fg="#f59e0b"
        )

        mute_button.config(
            text="UNMUTE VOICE"
        )


# ============================================================
# FULLSCREEN
# ============================================================

def toggle_fullscreen():

    global fullscreen

    fullscreen = not fullscreen

    root.attributes(
        "-fullscreen",
        fullscreen
    )


# ============================================================
# FRAME UPDATE
# ============================================================

def update_frame():

    global total_confirmed
    global current_sign
    global current_yolo
    global current_cnn
    global current_detection_count
    global fps
    global fps_counter
    global fps_start

    if not camera_running:
        return

    if cap is None:
        return

    success, frame = cap.read()

    if not success:

        root.after(
            30,
            update_frame
        )

        return

    # --------------------------------------------------------
    # SAME WORKING PIPELINE
    # --------------------------------------------------------

    results = yolo(
        frame,
        conf=YOLO_CONF,
        verbose=False
    )

    current_detections = 0

    detected_this_frame = None

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            yolo_conf = float(
                box.conf[0]
            )

            h, w = frame.shape[:2]

            x1 = max(
                0,
                min(x1, w - 1)
            )

            x2 = max(
                0,
                min(x2, w)
            )

            y1 = max(
                0,
                min(y1, h - 1)
            )

            y2 = max(
                0,
                min(y2, h)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # SAME PADDING
            pad_x = int(
                (x2 - x1) * 0.15
            )

            pad_y = int(
                (y2 - y1) * 0.15
            )

            crop_x1 = max(
                0,
                x1 - pad_x
            )

            crop_y1 = max(
                0,
                y1 - pad_y
            )

            crop_x2 = min(
                w,
                x2 + pad_x
            )

            crop_y2 = min(
                h,
                y2 + pad_y
            )

            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if crop.size == 0:
                continue

            current_detections += 1

            # ------------------------------------------------
            # CNN
            # ------------------------------------------------

            cnn_img = cv2.resize(
                crop,
                (32, 32),
                interpolation=cv2.INTER_AREA
            )

            cnn_img = (
                cnn_img.astype(np.float32)
                / 255.0
            )

            cnn_img = np.expand_dims(
                cnn_img,
                axis=0
            )

            prediction = cnn.predict(
                cnn_img,
                verbose=0
            )

            class_id = int(
                np.argmax(prediction)
            )

            cnn_conf = float(
                np.max(prediction)
            )

            label = CLASSES.get(
                class_id,
                "Unknown"
            )

            # ------------------------------------------------
            # CONFIDENCE FILTER
            # ------------------------------------------------

            if cnn_conf < CNN_CONF:

                display_label = "Unknown sign"

            else:

                display_label = label

                center = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                confirmed = confirm_detection(
                    label,
                    center
                )

                if confirmed:

                    alert = make_alert(label)

                    print(
                        f"[ALERT] {alert} | "
                        f"YOLO={yolo_conf:.2f} | "
                        f"CNN={cnn_conf:.2f}"
                    )

                    if voice_enabled:
                        speak(alert)

                    # History
                    timestamp = time.strftime(
                        "%H:%M:%S"
                    )

                    history_list.insert(
                        0,
                        f"{timestamp}  {label}"
                    )

                    if history_list.size() > 8:

                        history_list.delete(
                            8,
                            tk.END
                        )

                    detected_this_frame = (
                        label,
                        yolo_conf,
                        cnn_conf
                    )

            # ------------------------------------------------
            # GUI CURRENT DETECTION DATA
            # ------------------------------------------------

            if cnn_conf >= CNN_CONF:

                current_sign = display_label
                current_yolo = yolo_conf
                current_cnn = cnn_conf

                detected_this_frame = (
                    display_label,
                    yolo_conf,
                    cnn_conf
                )

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text_y = max(
                25,
                y1 - 10
            )

            cv2.putText(
                frame,
                display_label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"YOLO {yolo_conf * 100:.1f}% | "
                f"CNN {cnn_conf * 100:.1f}%",
                (
                    x1,
                    min(
                        h - 10,
                        y2 + 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

    # --------------------------------------------------------
    # CLEAN TRACKING
    # --------------------------------------------------------

    cleanup_tracking()

    current_detection_count = current_detections

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps_counter += 1

    elapsed = time.time() - fps_start

    if elapsed >= 1.0:

        fps = fps_counter / elapsed

        fps_counter = 0
        fps_start = time.time()

    # --------------------------------------------------------
    # UPDATE GUI
    # --------------------------------------------------------

    if current_detections == 0:

        sign_label.config(
            text="No Sign Detected",
            fg="#9ca3af"
        )

        yolo_value.config(
            text="YOLO\n--"
        )

        cnn_value.config(
            text="CNN\n--"
        )

    else:

        sign_label.config(
            text=current_sign,
            fg="#ffffff"
        )

        yolo_value.config(
            text=f"YOLO\n{current_yolo * 100:.1f}%"
        )

        cnn_value.config(
            text=f"CNN\n{current_cnn * 100:.1f}%"
        )

    fps_label.config(
        text=f"FPS\n{fps:.1f}"
    )

    detection_count_label.config(
        text=f"DETECTIONS\n{current_detection_count}"
    )

    stats_label.config(
        text=(
            f"YOLOv8 + CNN  |  43 GTSRB CLASSES  |  "
            f"CONFIRMED: {total_confirmed}"
        )
    )

    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    # Fit camera image into panel
    panel_width = max(
        640,
        camera_panel.winfo_width() - 20
    )

    panel_height = max(
        400,
        camera_panel.winfo_height() - 52
    )

    image.thumbnail(
        (
            panel_width,
            panel_height
        ),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    video_frame.configure(
        image=photo,
        text=""
    )

    video_frame.image = photo

    root.after(
        15,
        update_frame
    )


# ============================================================
# BUTTONS
# ============================================================

start_button = ttk.Button(
    control_bar,
    text="START CAMERA",
    style="ADAS.TButton",
    command=toggle_camera
)

start_button.pack(
    side="left",
    padx=10,
    pady=15
)


mute_button = ttk.Button(
    control_bar,
    text="MUTE VOICE",
    style="ADAS.TButton",
    command=toggle_voice
)

mute_button.pack(
    side="left",
    padx=5,
    pady=15
)


fullscreen_button = ttk.Button(
    control_bar,
    text="FULLSCREEN",
    style="ADAS.TButton",
    command=toggle_fullscreen
)

fullscreen_button.pack(
    side="left",
    padx=5,
    pady=15
)


exit_button = ttk.Button(
    control_bar,
    text="EXIT",
    style="ADAS.TButton",
    command=lambda: close_app()
)

exit_button.pack(
    side="right",
    padx=20,
    pady=15
)


# ============================================================
# CLOSE
# ============================================================

def close_app():

    global camera_running

    camera_running = False

    if cap is not None:

        cap.release()

    root.destroy()

    cv2.destroyAllWindows()


root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# ============================================================
# START
# ============================================================

print()
print("========================================")
print("VISION ADAS PROFESSIONAL GUI")
print("========================================")
print("YOLO:", YOLO_MODEL)
print("CNN :", CNN_MODEL)
print("Camera:", CAMERA_ID)
print("========================================")
print()

root.mainloop()

if cap is not None:
    cap.release()

cv2.destroyAllWindows()