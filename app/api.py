# stt-api: takes audio, puts a job in the Redis queue, answers fast.
# It never runs the model itself; that is the worker's job.
import os
import uuid

import redis
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379)

MAX_BYTES = 10 * 1024 * 1024    # 10 MB per file
MAX_QUEUE = 50                  # queue full → 429 instead of crashing
JOB_TTL = 60 * 60               # jobs disappear from Redis after 1 hour

app = FastAPI(title="zafarr STT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zafarr.info", "https://k8s.zafarr.info", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    r.ping()                    # not healthy if Redis is unreachable
    return {"status": "ok"}


@app.post("/transcribe", status_code=202)
async def transcribe(file: UploadFile):
    audio = await file.read(MAX_BYTES + 1)
    if len(audio) > MAX_BYTES:
        raise HTTPException(413, "File is bigger than 10 MB")
    if r.llen("queue") >= MAX_QUEUE:
        raise HTTPException(429, "Too many jobs in the queue, try again later")

    job_id = uuid.uuid4().hex
    r.set(f"audio:{job_id}", audio, ex=JOB_TTL)
    r.hset(f"job:{job_id}", mapping={"status": "queued"})
    r.expire(f"job:{job_id}", JOB_TTL)
    r.lpush("queue", job_id)    # add to the left end; workers take from the right end
    return {"id": job_id, "status": "queued", "position": r.llen("queue")}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    data = r.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(404, "Job not found (or older than 1 hour)")
    return {"id": job_id, **{k.decode(): v.decode() for k, v in data.items()}}
