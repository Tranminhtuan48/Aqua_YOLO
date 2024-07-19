import os
import cv2
from flask import Flask, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['MODEL_PATH'] = '/Users/trandungcuong/Desktop/aqua_yolo/model.pt'

model = YOLO(app.config['MODEL_PATH'])

def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    results = model.predict(source=img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            confidence = box.conf[0]
            class_id = box.cls[0]
            label = f"{model.names[int(class_id)]} {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

def process_video(image_path, output_path):
    cap = cv2.VideoCapture(image_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0]
                class_id = box.cls[0]
                label = f"{model.names[int(class_id)]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)

    cap.release()
    out.release()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0]
                class_id = box.cls[0]
                label = f"{model.names[int(class_id)]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    process_image(file_path, output_path)
                elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                    process_video(file_path, output_path)
                
                return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

@app.route('/camera', methods=['POST'])
def camera():
    return redirect(url_for('video_feed'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'output/{filename}'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
