FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libx11-6 libgl1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages numpy opencv-python opencv-python-headless flask flask-socketio eventlet

RUN mkdir -p web_application

ADD web_application web_application
COPY evaluation.mp4 web_application
COPY best.torchscript web_application

EXPOSE 5000
CMD ["python", "web_application/app.py"]