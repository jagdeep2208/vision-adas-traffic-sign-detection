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

YOLO_MODEL = r"C:\\TrafficSignProject\\models\\final\\traffic_sign_detector_best.pt"
CNN_MODEL = r"C:\\TrafficSignProject\\models\\final\\traffic_classifier.h5"


# ============================================================
# SETTINGS
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
# MODELS
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
last_seen = {}

total_confirmed = 0


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2) ** 0.5


def confirm_detection(label, center):
    """
    Require the same recognized sign to appear
    in multiple nearby frames before confirmation.
    """

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
# GUI
# ============================================================

root = tk.Tk()

root.title("VISION ADAS - Traffic Sign Detection")
root.geometry("1100x720")
root.configure(bg="#101010")

style = ttk.Style()

try:
    style.theme_use("clam")
except:
    pass


# Header

header = tk.Frame(
    root,
    bg="#151515",
    height=70
)

header.pack(
    fill="x",
    side="top"
)

title = tk.Label(
    header,
    text="VISION ADAS",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#151515"
)

title.pack(
    side="left",
    padx=25,
    pady=15
)

status_label = tk.Label(
    header,
    text="● AI VISION ACTIVE",
    font=("Segoe UI", 13, "bold"),
    fg="#00ff66",
    bg="#151515"
)

status_label.pack(
    side="right",
    padx=25
)


# Camera display

video_frame = tk.Label(
    root,
    bg="black"
)

video_frame.pack(
    padx=20,
    pady=15,
    fill="both",
    expand=True
)


# Bottom dashboard

dashboard = tk.Frame(
    root,
    bg="#151515",
    height=90
)

dashboard.pack(
    fill="x",
    side="bottom"
)


detection_label = tk.Label(
    dashboard,
    text="DETECTIONS: 0",
    font=("Segoe UI", 13, "bold"),
    fg="white",
    bg="#151515"
)

detection_label.pack(
    side="left",
    padx=30,
    pady=20
)


model_label = tk.Label(
    dashboard,
    text="YOLOv8 + CNN | 43 Classes",
    font=("Segoe UI", 12),
    fg="#bbbbbb",
    bg="#151515"
)

model_label.pack(
    side="right",
    padx=30
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not cap.isOpened():

    status_label.config(
        text="● CAMERA ERROR",
        fg="red"
    )

    raise RuntimeError("Could not open camera.")


# ============================================================
# FRAME UPDATE
# ============================================================

def update_frame():

    global total_confirmed

    success, frame = cap.read()

    if not success:

        root.after(30, update_frame)
        return

    # IMPORTANT:
    # No cv2.flip() here.
    # This prevents the mirror issue from the old version.

    results = yolo(
        frame,
        conf=YOLO_CONF,
        verbose=False
    )

    current_detections = 0

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

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            # Add padding around detected sign
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w, x2 + pad_x)
            crop_y2 = min(h, y2 + pad_y)

            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]
            if crop.size == 0:
                continue

            current_detections += 1

            # --------------------------------
            # CNN
            # --------------------------------

            cnn_img = cv2.resize(
                crop,
                (32, 32),
                interpolation=cv2.INTER_AREA
            )

            cnn_img = cnn_img.astype(
                np.float32
            ) / 255.0

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

            # --------------------------------
            # Confidence filter
            # --------------------------------

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

                    speak(alert)

            # --------------------------------
            # Draw
            # --------------------------------

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
                (x1, min(h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

    # Clean old tracks AFTER processing all detections

    cleanup_tracking()

    detection_label.config(
        text=f"DETECTIONS: {current_detections} | "
             f"CONFIRMED: {total_confirmed}"
    )

    # OpenCV BGR → RGB

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    image = image.resize(
        (900, 600),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    video_frame.configure(
        image=photo
    )

    video_frame.image = photo

    root.after(
        15,
        update_frame
    )


# ============================================================
# CLOSE
# ============================================================

def close_app():

    cap.release()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# ============================================================
# START
# ============================================================

print()
print("========================================")
print("VISION ADAS FINAL")
print("========================================")
print("YOLO:", YOLO_MODEL)
print("CNN :", CNN_MODEL)
print("Camera:", CAMERA_ID)
print("Press the window X to exit.")
print("========================================")
print()

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
