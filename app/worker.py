# stt-worker: takes one job from the Redis queue, runs the model, saves the text.
# Many workers can run at once; Redis gives each job to exactly one of them.
import os
import signal
import socket
import subprocess
import time

import numpy as np
import onnx_asr
import onnxruntime as ort
import redis

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379)
name = socket.gethostname()     # pod name, shows which worker did the job

opts = ort.SessionOptions()
opts.intra_op_num_threads = int(os.getenv("THREADS", "2"))   # match the pod's CPU limit
model = onnx_asr.load_model(
    "nemo-conformer-ctc", os.getenv("MODEL_DIR", "/model"),
    quantization="int8", sess_options=opts, providers=["CPUExecutionProvider"],
)
print(f"{name}: model loaded, waiting for jobs", flush=True)

# Kubernetes sends SIGTERM before it stops a pod: finish the current job, then exit.
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)


def decode(audio: bytes) -> np.ndarray:
    # ffmpeg turns any format (mp3, ogg, webm from the browser mic...) into 16 kHz mono float32
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=audio, capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32)


while running:
    item = r.brpop("queue", timeout=5)   # wait up to 5 s for a job (then check `running` again)
    if item is None:
        continue
    job_id = item[1].decode()
    key = f"job:{job_id}"
    r.hset(key, mapping={"status": "processing", "worker": name})
    try:
        audio = r.get(f"audio:{job_id}")
        if audio is None:
            raise ValueError("audio expired")
        wave = decode(audio)
        start = time.time()
        text = model.recognize(wave, sample_rate=16000)
        took = time.time() - start
        r.hset(key, mapping={"status": "done", "text": text,
                             "audio_seconds": f"{len(wave) / 16000:.1f}", "took_seconds": f"{took:.2f}"})
        print(f"{name}: {job_id} done, {len(wave) / 16000:.1f}s audio in {took:.2f}s", flush=True)
    except Exception as e:
        r.hset(key, mapping={"status": "error", "error": str(e)[:300]})
        print(f"{name}: {job_id} failed: {e}", flush=True)
    finally:
        r.delete(f"audio:{job_id}")

print(f"{name}: got SIGTERM, bye", flush=True)
