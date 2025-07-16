from ultralytics import YOLO
import cv2
import os

model = YOLO('yolo11_model/best.pt')  # Load the YOLOv8 model

def detect_image(image_path):
    results = model(image_path)
    result_image = results[0].plot()
    output_path = os.path.join('static', 'outputs', os.path.basename(image_path))
    cv2.imwrite(output_path, result_image)
    return output_path


def detect_video(model, input_path, output_path):

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, imgsz=640)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()