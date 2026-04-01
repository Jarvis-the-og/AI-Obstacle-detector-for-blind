import cv2
import pyttsx3
import numpy as np
import time

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- LOAD MODEL ----------------
# Using Haar Cascade (simple + explainable)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

last_speak_time = 0
cooldown = 2  # seconds between voice alerts

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects
    objects = face_cascade.detectMultiScale(gray, 1.3, 5)

    message = ""
    
    if len(objects) > 0:
        for (x, y, w, h) in objects:
            center_x = x + w // 2

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # ---------------- DIRECTION LOGIC ----------------
            if center_x < width // 3:
                message = "Move Right"
            elif center_x > 2 * width // 3:
                message = "Move Left"
            else:
                message = "Obstacle Ahead"

    # ---------------- VOICE CONTROL ----------------
    current_time = time.time()
    if message != "" and (current_time - last_speak_time > cooldown):
        speak(message)
        last_speak_time = current_time

    # Display message
    if message != "":
        cv2.putText(frame, message, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Voice Navigation System", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()