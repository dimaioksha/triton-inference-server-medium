FROM python:3.8 AS builder

WORKDIR /workspace/triton_client
COPY triton_client/requirements.txt .
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --user --no-cache-dir -r requirements.txt

FROM builder as triton-client
ENTRYPOINT ["tail", "-f", "/dev/null"]
