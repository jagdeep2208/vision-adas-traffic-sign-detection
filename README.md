\# 🚗 VISION ADAS — Traffic Sign Detection \& Recognition



A Deep Learning based Traffic Sign Detection and Recognition system designed as a prototype of an Advanced Driver Assistance System (ADAS).



The system detects traffic signs from a live camera feed, recognizes the detected sign using a CNN classifier, displays the result through an automotive-style dashboard, and provides voice alerts to the driver.



\---



\## 📌 Project Overview



VISION ADAS combines two Deep Learning models:



1\. \*\*YOLOv8\*\* — Detects and localizes traffic signs in the camera frame.

2\. \*\*CNN Classifier\*\* — Recognizes the detected traffic sign among 43 GTSRB classes.



The system uses temporal confirmation before generating an alert, helping reduce alerts caused by short-lived detections.



\### Processing Pipeline



Camera

↓

YOLOv8 Traffic Sign Detection

↓

Bounding Box + Sign Crop

↓

CNN Traffic Sign Recognition

↓

Confidence Filtering

↓

Temporal Confirmation

↓

Dashboard + Voice Alert



\---



\## 🎯 Objectives



\- Detect traffic signs from a live camera feed.

\- Recognize traffic signs using Deep Learning.

\- Support 43 traffic sign classes.

\- Display bounding boxes around detected signs.

\- Show YOLO and CNN confidence scores.

\- Provide voice-based driver alerts.

\- Provide a real-time ADAS-style dashboard.

\- Maintain detection history.

\- Provide a foundation for future edge deployment.



\---



\## 🧠 Technologies Used



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| YOLOv8 | Traffic sign detection |

| TensorFlow / Keras | CNN classification |

| OpenCV | Camera and image processing |

| NumPy | Numerical operations |

| Pillow | GUI image rendering |

| Tkinter | ADAS dashboard |

| pyttsx3 | Voice alerts |



\---



\## 🚦 Detection Architecture



The system uses a two-stage computer vision architecture.



\### Stage 1 — Traffic Sign Detection



YOLOv8 is used as a one-class detector.



The detector identifies:



`traffic\_sign`



Instead of asking YOLO to recognize all 43 classes directly, the detection model focuses on locating traffic signs.



\### Stage 2 — Traffic Sign Recognition



The detected sign is cropped from the camera frame and passed to a CNN classifier.



The CNN predicts one of the 43 GTSRB traffic sign classes.



This architecture separates:



\- Localization

\- Recognition



and allows the CNN to specialize in traffic sign classification.



\---



\## 📊 Models



\### YOLOv8 Detector



Final model:



`models/final/traffic\_sign\_detector\_best.pt`



Validation results:



\- Precision: approximately 92.5%

\- Recall: approximately 85.3%

\- mAP@50: approximately 93.2%

\- mAP@50-95: approximately 73.6%



The detector was trained using GPU acceleration.



\### CNN Classifier



Final model:



`models/final/traffic\_classifier.h5`



The classifier supports:



\*\*43 GTSRB traffic sign classes\*\*



The original classifier achieved approximately 99.6% validation accuracy during training.



\---



\## 🚘 ADAS Dashboard



The final application provides an automotive-style interface containing:



\- Live camera feed

\- Traffic sign bounding boxes

\- Current detected sign

\- YOLO confidence

\- CNN confidence

\- FPS

\- Detection count

\- Confirmed detection count

\- Detection history

\- YOLO status

\- CNN status

\- Voice status

\- Start / Stop Camera

\- Mute / Unmute Voice

\- Fullscreen mode



\---



\## 🔊 Voice Alerts



The system generates voice warnings for recognized traffic signs.



Examples:



\- Stop sign detected

\- Yield sign detected

\- No entry sign detected

\- Speed limit 20km/h detected

\- Traffic signal detected

\- Pedestrian warning

\- Children crossing warning

\- Road work ahead



Voice alerts use `pyttsx3`.



A cooldown mechanism is used to avoid continuously repeating the same voice alert.



\---



\## ⏱️ Temporal Confirmation



The system does not immediately confirm every detection.



A recognized sign must appear consistently across multiple frames before being confirmed.



Current configuration:



```text

YOLO confidence      = 0.30

CNN confidence       = 0.70

Confirmation frames  = 4

Tracking distance    = 80 pixels

Voice cooldown       = 5 seconds

Memory time          = 1.5 seconds

