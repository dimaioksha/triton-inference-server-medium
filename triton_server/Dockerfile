FROM nvcr.io/nvidia/tritonserver:21.05-py3 AS builder

COPY triton_server/requirements.txt .
RUN apt-get update && apt-get install python3 && apt-get install ffmpeg libsm6 libxext6 -y
RUN python3 -m pip install --upgrade pip
RUN pip install --user --no-cache-dir -r requirements.txt