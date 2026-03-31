import cv2
import time
import threading
import queue
import os
import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Configuration ---
RTSP_URL = "rtsp://10.76.11.62"

# --- Global Resources ---
print("[System] Loading YOLO Model globally...")
try:
    model = YOLO("best_m.pt")
except:
    print("[Warning] 'best_m.pt' not found, falling back to 'yolov8n.pt'")
    model = YOLO("yolov8n.pt")

model_lock = threading.Lock()

# --- Shared State ---
class VideoState:
    def __init__(self):
        self.frame = None
        self.boxes = []  
        self.count = 0
        self.lock = threading.Lock()
        self.running = False 

state = VideoState()

# --- Thread Functions ---
def capture_loop():
    print(f"[Thread 1] Connecting to Stream: {RTSP_URL}...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("Stream interrupted. Reconnecting in 2s...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        with state.lock:
            state.frame = frame
    
    cap.release()
    print("[Thread 1] Stopped.")

def ai_loop():
    print("[Thread 2] AI Processing Started...")
    ai_width = 640
    ai_height = 360

    while state.running:
        working_frame = None
        with state.lock:
            if state.frame is not None:
                working_frame = state.frame.copy()

        if working_frame is None:
            time.sleep(0.05)
            continue

        orig_h, orig_w = working_frame.shape[:2]
        input_frame = cv2.resize(working_frame, (ai_width, ai_height))
        
        with model_lock:
            results = model(input_frame, conf=0.5, classes=[0], verbose=False)
        
        current_boxes = []
        if len(results) > 0:
            det_boxes = results[0].boxes.xyxy.cpu().numpy()
            x_scale = orig_w / ai_width
            y_scale = orig_h / ai_height

            for box in det_boxes:
                x1, y1, x2, y2 = box
                x1 = int(x1 * x_scale)
                y1 = int(y1 * y_scale)
                x2 = int(x2 * x_scale)
                y2 = int(y2 * y_scale)
                current_boxes.append((x1, y1, x2, y2))

        with state.lock:
            state.count = len(current_boxes)
            state.boxes = current_boxes
            
        time.sleep(0.01) 
    print("[Thread 2] Stopped.")

# --- Flask Routes ---

@app.route('/start_stream', methods=['POST'])
def start_stream():
    if not state.running:
        state.running = True
        threading.Thread(target=capture_loop, daemon=True).start()
        threading.Thread(target=ai_loop, daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    if state.running:
        state.running = False 
        time.sleep(0.5)
        with state.lock:
            state.frame = None
            state.count = 0
            state.boxes = []
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with model_lock:
            results = model(img, conf=0.5, classes=[0], verbose=False)
        count = len(results[0].boxes)
        return jsonify({"message": "Success", "detected_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate_annotated_feed():
        while True:
            if not state.running:
                break 
            
            output_frame = None
            current_boxes = []

            with state.lock:
                if state.frame is None:
                    time.sleep(0.1)
                    continue
                output_frame = state.frame.copy()
                current_boxes = list(state.boxes)

            # Draw Cyberpunk Style Boxes
            for (x1, y1, x2, y2) in current_boxes:
                # Corners only for cleaner look? Or full box. Let's do full box thin.
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # Label
                label = "TARGET"
                cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', output_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
            
    return Response(generate_annotated_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_feed')
def count_feed():
    def generate_sse_count():
        last_count = -1
        while True:
            if not state.running:
                yield f"data: 0\n\n"
                break
            with state.lock:
                c = state.count
            if c != last_count:
                yield f"data: {c}\n\n"
                last_count = c
            time.sleep(0.5)
    return Response(generate_sse_count(), mimetype='text/event-stream')

# --- Frontend Template ---
@app.route('/')
def index():
    return render_template_string("""
        <html>
            <body style="font-family: sans-serif; text-align: center; background: #222; color: #fff;">
                <h1>Zero-Lag Monitor</h1>
                <img src="/video_feed" style="width: 80%; border: 2px solid #555;"/><br/>
                <h2 style="font-size: 50px; color: #0f0;">
                    People: <span id="cnt">0</span>
                </h2>
                <script>
                    new EventSource("/count_feed").onmessage = (e) => {
                        document.getElementById("cnt").innerText = e.data;
                    };
                </script>
            </body>
        </html>
    """)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)