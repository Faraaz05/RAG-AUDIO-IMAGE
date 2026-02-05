FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir faster-whisper ctranslate2

RUN pip install --no-cache-dir pyannote.audio torch torchaudio

RUN mkdir -p /root/.cache

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY aws_audio_worker.py /app/aws_audio_worker.py

CMD ["python3", "-u", "aws_audio_worker.py"]