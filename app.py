from flask import Flask, render_template, request, Response, redirect, url_for
import os
from detect import detect_image, detect_video
import cv2
from ultralytics import YOLO

model = YOLO("yolo11_model/best.pt")  # Load your YOLO model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    output_path = detect_image(filepath)
    return render_template('result.html', input_path=filepath, output_path=output_path)

@app.route('/video', methods=['GET', 'POST'])
def video_page():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No video uploaded', 400
        file = request.files['video']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{file.filename}")
        file.save(filepath)

        detect_video(model, filepath, output_path)
        return render_template('video.html', output_video=output_path)
    return render_template('video.html', output_video=None)

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/about')
def about():
    return render_template('about.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame, imgsz=640)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


