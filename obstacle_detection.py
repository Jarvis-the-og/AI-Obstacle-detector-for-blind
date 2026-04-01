"""
Real-Time Obstacle Detection with Voice Guidance
=================================================
Requirements:
    pip install ultralytics opencv-python pyttsx3

Run:
    python obstacle_detection.py
    python obstacle_detection.py --source 0          # webcam (default)
    python obstacle_detection.py --source video.mp4  # video file
    python obstacle_detection.py --conf 0.5          # confidence threshold

Controls (while running):
    Q  — quit
    S  — toggle sound on/off
    P  — pause / resume
"""

import cv2
import pyttsx3
import threading
import time
import queue
import argparse
import sys
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Classes that count as "obstacles" — subset of COCO labels.
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorbike", "bus", "truck",
    "chair", "sofa", "bench", "dining table", "toilet",
    "dog", "cat", "cow", "horse",
    "suitcase", "backpack", "umbrella",
}

# Colour per detected object type (BGR)
CLASS_COLOURS = {
    "person":       (255, 100,  50),   # blue
    "car":          (50,  200, 255),   # yellow
    "truck":        (50,  150, 255),   # orange
    "bus":          (50,  150, 255),
    "bicycle":      (180, 255,  50),   # green
    "motorbike":    (180, 255,  50),
    "dog":          (255, 180,  50),   # cyan
    "cat":          (255, 180,  50),
    "default":      (50,  255, 180),   # green fallback
}

# Seconds to wait after first alert before re-checking
COOLDOWN_SECONDS = 3.0

# Proximity zones based on bounding-box area relative to frame area
PROX_THRESHOLDS = {
    "very close": 0.25,
    "close":      0.10,
    "nearby":     0.03,
}


# ---------------------------------------------------------------------------
# Voice engine (runs in its own daemon thread)
# ---------------------------------------------------------------------------

class VoiceEngine:
    """
    Per-class cooldown state machine:
      - Each obstacle CLASS gets its own independent 3-s cooldown timer.
      - A new class → immediate alert, starts that class's timer.
      - Same class within cooldown → alert deferred to timer expiry.
      - Timer fires:
          a. Class still present → speak again, restart timer.
          b. Class gone          → speak "<Class> cleared.", remove entry.
    """

    def __init__(self):
        self._tts_q: queue.Queue[str | None] = queue.Queue()
        self._enabled = True
        self._lock = threading.Lock()

        # Per-class state:
        # { class_label -> {"timer": Timer | None, "present": bool, "alert": str} }
        self._class_state: dict[str, dict] = {}

        t = threading.Thread(target=self._tts_worker, daemon=True)
        t.start()

    # ── public API ──────────────────────────────────────────────────────────

    def update(self, detected: dict[str, str]) -> None:
        """
        Call once per frame.
        detected: { class_label -> alert_message }  — only currently visible obstacles.
        """
        with self._lock:
            # Only mark classes that have disappeared as not present.
            # Do NOT touch classes that are still detected this frame.
            for label in self._class_state:
                if label not in detected:
                    self._class_state[label]["present"] = False

            for label, alert in detected.items():
                if label not in self._class_state:
                    # Brand-new class — speak immediately and start timer
                    self._class_state[label] = {
                        "timer":   None,
                        "present": True,
                        "alert":   alert,
                    }
                    self._speak(alert)
                    self._start_timer(label)
                else:
                    # Already tracked — update alert text; timer handles re-speaking
                    self._class_state[label]["present"] = True
                    self._class_state[label]["alert"]   = alert

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled

    def shutdown(self) -> None:
        with self._lock:
            for state in self._class_state.values():
                if state["timer"]:
                    state["timer"].cancel()
        self._tts_q.put(None)

    # ── internal ────────────────────────────────────────────────────────────

    def _on_cooldown_end(self, label: str) -> None:
        """Runs on a Timer thread exactly COOLDOWN_SECONDS after the last alert."""
        with self._lock:
            if label not in self._class_state:
                return
            state = self._class_state[label]
            if state["present"]:
                # Still blocking — warn again and restart the timer
                self._speak(state["alert"])
                self._start_timer(label)
            else:
                # Gone — announce clearance and remove tracking entry
                self._speak(f"{label.capitalize()} cleared.")
                del self._class_state[label]

    def _start_timer(self, label: str) -> None:
        """(Re)start the cooldown timer for a specific class. Must be called under lock."""
        if self._class_state[label]["timer"]:
            self._class_state[label]["timer"].cancel()
        timer = threading.Timer(COOLDOWN_SECONDS, self._on_cooldown_end, args=(label,))
        timer.daemon = True
        timer.start()
        self._class_state[label]["timer"] = timer

    def _speak(self, message: str) -> None:
        if not self._enabled or not message:
            return

        # CLEAR old messages → prioritize latest obstacle
        while not self._tts_q.empty():
            try:
                self._tts_q.get_nowait()
            except:
                break

        self._tts_q.put(message)
        
    def _tts_worker(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        while True:
            msg = self._tts_q.get()
            if msg is None:
                break
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Proximity helper
# ---------------------------------------------------------------------------

def proximity_label(box_area: float, frame_area: float) -> str:
    ratio = box_area / frame_area
    for label, threshold in PROX_THRESHOLDS.items():
        if ratio >= threshold:
            return label
    return "detected"


def build_alert(label: str, proximity: str) -> str:
    if label == "person":
        return f"Person {proximity} — please stop"
    if proximity in ("very close", "close"):
        return f"Obstacle {proximity} — {label} ahead"
    return f"Obstacle ahead — {label} {proximity}"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box(frame, x1, y1, x2, y2, label, conf, colour):
    """Draw bounding box and label badge."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    badge = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 4
    by1 = max(y1 - th - pad * 2, 0)
    by2 = by1 + th + pad * 2
    bx2 = x1 + tw + pad * 2

    cv2.rectangle(frame, (x1, by1), (bx2, by2), colour, -1)
    cv2.putText(
        frame, badge,
        (x1 + pad, by2 - pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (20, 20, 20), 1, cv2.LINE_AA,
    )


def draw_hud(frame, fps: float, sound_on: bool, paused: bool,
             n_detections: int) -> None:
    """Overlay HUD info in the top-left corner."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:5.1f}",
        f"Sound: {'ON' if sound_on else 'OFF'}",
        f"Objects: {n_detections}",
        "Q=quit  S=sound  P=pause",
    ]
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 255), 3, cv2.LINE_AA)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 220, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(source, conf_threshold: float, model_name: str) -> None:
    print(f"Loading model: {model_name} ...")
    model = YOLO(model_name)

    print(f"Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {source}")

    voice = VoiceEngine()
    sound_on = True
    paused = False

    # FPS tracking
    fps = 0.0
    frame_times: list[float] = []

    print("\n[INFO] Detection running. Press Q to quit.\n")

    while True:
        t_start = time.perf_counter()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break

        # ---- Key handling --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            sound_on = voice.toggle()
            print(f"[Sound] {'ON' if sound_on else 'OFF'}")
        elif key == ord("p"):
            paused = not paused
            print(f"[{'Paused' if paused else 'Resumed'}]")

        if paused:
            draw_hud(frame, fps, sound_on, paused, 0)
            cv2.imshow("Obstacle Detection", frame)
            continue

        h, w = frame.shape[:2]
        frame_area = h * w

        # ---- Inference -----------------------------------------------------
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        n_detections = 0

        # { class_label -> (alert_text, largest_box_area) }
        # We keep the alert for the largest (closest) instance of each class.
        frame_detected: dict[str, tuple[str, float]] = {}

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            label    = model.names[cls_id]

            if label not in OBSTACLE_CLASSES:
                continue

            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            colour = CLASS_COLOURS.get(label, CLASS_COLOURS["default"])

            box_area  = (x2 - x1) * (y2 - y1)
            proximity = proximity_label(box_area, frame_area)
            alert     = build_alert(label, proximity)

            # Keep only the closest (largest-box) instance per class
            if label not in frame_detected or box_area > frame_detected[label][1]:
                frame_detected[label] = (alert, box_area)

            draw_box(frame, x1, y1, x2, y2, label, conf, colour)

            cv2.putText(frame, proximity, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        colour, 1, cv2.LINE_AA)

            n_detections += 1

        # Strip box-area bookkeeping; voice engine only needs label -> alert
        voice.update({lbl: data[0] for lbl, data in frame_detected.items()})

        # ---- FPS -----------------------------------------------------------
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        draw_hud(frame, fps, sound_on, paused, n_detections)
        cv2.imshow("Obstacle Detection", frame)

    cap.release()
    cv2.destroyAllWindows()
    voice.shutdown()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Obstacle Detection")
    parser.add_argument(
        "--source", default=0,
        help="Camera index (0, 1, …) or path to video file (default: 0)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.45,
        help="Confidence threshold 0–1 (default: 0.45)"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt — downloads automatically)"
    )
    args = parser.parse_args()

    # Convert source to int if it looks like a digit
    if str(args.source).isdigit():
        args.source = int(args.source)

    return args


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, conf_threshold=args.conf, model_name=args.model)