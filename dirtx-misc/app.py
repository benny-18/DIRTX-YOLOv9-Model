from flask import Flask, request, jsonify
import subprocess
import uuid
import os
import shutil
from pathlib import Path

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_red_tide():
    file = request.files['file']
    dest_dir = request.form.get('dest_dir')

    if not file or not dest_dir:
        return jsonify({'error': 'Missing file or destination directory'}), 400

    os.makedirs(dest_dir, exist_ok=True)

    original_name = Path(file.filename).stem
    ext = Path(file.filename).suffix
    unique_id = uuid.uuid4().hex

    # Paths for original and inferenced files
    original_file_path = os.path.join(dest_dir, f"dirtx_{original_name}_original{ext}")
    inferenced_file_path = os.path.join(dest_dir, f"dirtx_{original_name}_inferenced{ext}")

    file.save(original_file_path)

    # Run YOLOv9 inference
    yolo_dir = "/home/benny/Documents/DIRTX/yolov9"
    result = subprocess.run(
        ['python3.10',
         f'{yolo_dir}/segment/predict.py',
         '--source', original_file_path,
         '--img', '320',
         '--conf-thres', '0.80',
         '--device', 'cpu',
         '--weights', f'{yolo_dir}/runs/train-seg/exp/weights/best.pt',
         '--hide-conf'],
        capture_output=True,
        text=True
    )

    # Find the latest exp folder
    runs_dir = os.path.join(yolo_dir, 'runs', 'predict-seg')
    exp_folders = sorted(Path(runs_dir).glob('exp*'), key=os.path.getmtime)
    latest_exp = exp_folders[-1] if exp_folders else None

    if not latest_exp:
        return jsonify({'error': 'Inference output not found'}), 500

    # Locate YOLO output file
    yolo_output_file = next(latest_exp.glob(f"{Path(original_file_path).name}"), None)
    if not yolo_output_file:
        return jsonify({'error': 'No output file found from inference'}), 500

    shutil.move(str(yolo_output_file), inferenced_file_path)

    return jsonify({
        'message': 'Segmentation completed successfully.',
        'output_path': inferenced_file_path
    })


if __name__ == '__main__':
    app.run(debug=True)

