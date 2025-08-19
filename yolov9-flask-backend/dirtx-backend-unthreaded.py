from flask import Flask, request, jsonify
import subprocess
import os
import shutil
import re
from datetime import datetime
from flask import Response
from yolov9.segment import predict

app = Flask(__name__)

dirtx_out = '/mnt/c/Users/User/Documents/DIRTX/'
yolo_model = 'yolov9/runs/train-seg/exp/weights/best.pt'

detection_args = {
    "img": 320,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "line_thickness": 2,
    "hide_labels": False,
    "hide_conf": False
}

@app.route('/arguments', methods=['POST'])
def set_arguments():
    global detection_args
    data = request.json or {}
    for key in detection_args:
        if key in data:
            if key in ("conf_thres", "iou_thres"):
                try:
                    detection_args[key] = float(data[key]) / 100.0
                except Exception:
                    pass
            else:
                detection_args[key] = data[key]
    return jsonify({"status": "success", "updated_args": detection_args})

def make_timestamped_folder():
    dt = datetime.now()
    folder_name = f"{dt.strftime('%b')}_{dt.day}_{dt.year}_{dt.strftime('%I.%M.%S_%p')}"
    return folder_name

@app.route('/infer', methods=['POST'])
def run_inference():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename or 'uploaded'
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    save_path = os.path.join(uploads_dir, filename)
    file.save(save_path)

    folder_name = make_timestamped_folder()
    target_dir = os.path.join(dirtx_out, folder_name)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    cmd = [
        'python3', 'yolov9/segment/predict.py',
        '--device', 'cpu',
        '--weights', yolo_model,
        '--img', str(detection_args['img']),
        '--conf-thres', str(detection_args['conf_thres']),
        '--iou-thres', str(detection_args['iou_thres']),
        '--line-thickness', str(detection_args['line_thickness']),
        '--source', save_path,
        '--project', dirtx_out,
        '--name', folder_name
    ]

    if detection_args.get('hide_labels'):
        cmd.append('--hide-labels')
    if detection_args.get('hide_conf'):
        cmd.append('--hide-conf')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_text = (result.stdout or '') + (result.stderr or '')
        match = re.search(r"Results saved to (.+)", output_text)
        output_dir = match.group(1) if match else os.path.join(dirtx_out, folder_name)
        return jsonify({"status": "success", "message": "Inference completed.", "output_dir": output_dir})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e), "details": e.stderr}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route('/live_infer')
def live_infer():
    print("Starting live inference...")
    def generate():
        folder_name = make_timestamped_folder()
        for frame in predict.run(
            device='cpu',
            weights=yolo_model,
            source='http://192.168.100.4:4747/video',
            imgsz=(320, 320),
            conf_thres=0.80,
            iou_thres=0.50,
            project='/mnt/c/Users/User/Documents/DIRTX/',
            name=folder_name,
            view_img=True,
        ):
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
