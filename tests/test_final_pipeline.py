import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =========================
# MODELS
# =========================

YOLO_MODEL = r"C:\TrafficSignProject\runs\detect\traffic_sign_detector\weights\best.pt"
CNN_MODEL = r"C:\TrafficSignProject\models\traffic_classifier.h5"

yolo = YOLO(YOLO_MODEL)
cnn = load_model(CNN_MODEL)

# =========================
# 43 GTSRB CLASSES
# =========================

classes = {
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

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("FINAL PIPELINE TEST")
print("YOLO:", YOLO_MODEL)
print("CNN :", CNN_MODEL)
print("Press Q to exit.")

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read camera frame.")
        break

    # No horizontal flip
    results = yolo(frame, conf=0.30, verbose=False)

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0])

            # Safety: keep coordinates inside frame
            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop
            cropped = frame[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            # CNN preprocessing
            img = cv2.resize(cropped, (32, 32))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # CNN prediction
            prediction = cnn.predict(img, verbose=0)

            class_id = int(np.argmax(prediction))
            cnn_conf = float(np.max(prediction))

            label = classes[class_id]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display
            text1 = f"YOLO: {yolo_conf * 100:.1f}%"
            text2 = f"{label}"
            text3 = f"CNN: {cnn_conf * 100:.1f}%"

            cv2.putText(
                frame,
                text1,
                (x1, max(20, y1 - 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                text2,
                (x1, max(20, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                text3,
                (x1, min(h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    cv2.imshow("FINAL YOLO + CNN PIPELINE TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()