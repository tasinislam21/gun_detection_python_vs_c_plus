from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import PostProcess
import torch
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Frame skipping parameter
frame_skip = 3
frame_counter = 0

device = 'cuda'
torchmodel = torch.jit.load("../best.torchscript", map_location=device)
torchmodel.eval()

postprocessor = PostProcess.PostProcessor()

def preprocess_image(image_ori) -> torch.Tensor:
    image = cv2.resize(image_ori, (640, 640))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
    image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, image_rgb

def run_model(image):
    start_time = time.time()
    result = torchmodel(image)[0]
    end_time = time.time()
    return result, (end_time - start_time) * 1000

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    global frame_counter
    frame_counter += 1
    
    if frame_counter % frame_skip != 0:
        return  # Skip this frame
    
    # Decode the image from base64
    img_data = data_image.split(",")[1]
    img = base64.b64decode(img_data)
    npimg = np.frombuffer(img, dtype=np.uint8)
    frame = cv2.imdecode(npimg, 1)
    image, image_rgb = preprocess_image(frame)
    image = image.to(device)
    result, duration_ms = run_model(image)
    postprocessor.set_image(image_rgb)
    postprocessor.set_time(duration_ms)
    postprocessor.set_result(result)
    # Encode the frame in JPEG format
    _, buffer = cv2.imencode('.jpg', postprocessor.get_frame())
    frame_data = base64.b64encode(buffer).decode('utf-8')
    emit('response_back', 'data:image/jpeg;base64,' + frame_data)
    

if __name__ == '__main__':
    socketio.run(app, debug=True)
