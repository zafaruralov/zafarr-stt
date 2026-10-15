# stt-worker: takes one job from the Redis queue, runs the model, saves the text.
# Many workers can run at once; Redis gives each job to exactly one of them.
import os
import signal
import socket
import subprocess
import tempfile
import time
import json
import urllib.request
from datetime import date

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


# --- Text cleanup by an LLM (server side only: the browser never sees this call) ---
LLM_KEY = os.getenv("OPENAI_API_KEY", "")            # from Secret llm-secret; empty = cleanup off
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_DAILY_LIMIT = int(os.getenv("LLM_DAILY_LIMIT", "200"))   # protects the paid credit
PROMPT = (
    "You fix Uzbek speech recognition output. The text may have misheard words, "
    "joined or split words and missing punctuation. Rewrite it as the speaker most likely meant it: "
    "correct words, spelling (Latin script, use oʻ and gʻ) and punctuation. Keep the language and "
    "meaning, do not add new information, do not translate. Reply with the fixed text only."
)


def clean(text: str) -> str | None:
    if not LLM_KEY or not text.strip():
        return None
    used = r.incr(f"llm:{date.today()}")             # shared counter for all workers
    r.expire(f"llm:{date.today()}", 2 * 24 * 3600)
    if used > LLM_DAILY_LIMIT:
        return None                                    # limit reached: just give the raw text
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
            "temperature": 0.2,
        }).encode(),
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            return json.load(res)["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"{name}: cleanup failed: {e}", flush=True)
        return None                                    # never lose the job because of the LLM


def decode(audio: bytes) -> np.ndarray:
    # ffmpeg turns any format (mp3, ogg, webm/mp4 from the browser mic...) into 16 kHz mono float32.
    # A temp file, not a pipe: Safari's mp4 keeps its index at the END, so ffmpeg must jump around.
    with tempfile.NamedTemporaryFile() as f:
        f.write(audio)
        f.flush()
        out = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", f.name, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            capture_output=True,
        )
    if out.returncode != 0:
        raise ValueError("cannot read this audio: " + out.stderr.decode()[-200:].strip())
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
        fixed = clean(text)
        r.hset(key, mapping={"status": "done", "text": fixed or text, "raw": text,
                             "cleaned": "yes" if fixed else "no",
                             "audio_seconds": f"{len(wave) / 16000:.1f}", "took_seconds": f"{took:.2f}"})
        print(f"{name}: {job_id} done, {len(wave) / 16000:.1f}s audio in {took:.2f}s", flush=True)
    except Exception as e:
        r.hset(key, mapping={"status": "error", "error": str(e)[:300]})
        print(f"{name}: {job_id} failed: {e}", flush=True)
    finally:
        r.delete(f"audio:{job_id}")

print(f"{name}: got SIGTERM, bye", flush=True)
