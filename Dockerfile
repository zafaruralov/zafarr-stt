# One image, two jobs: the Deployment's `command` decides api or worker.
FROM python:3.12-slim

# ffmpeg: decodes any audio format (mp3, webm from the mic...) into raw samples
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model (175 MB) is baked into the image, so a new worker pod starts fast (no download).
COPY model/ /model/
COPY app/ /app/

USER 1000
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
