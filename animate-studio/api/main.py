"""
AniMate Studio — Monetization API
FastAPI server with API‑key auth, credit tracking, async video generation, Stripe integration.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import stripe
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging & environment
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("animate_api")

load_dotenv()

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
API_KEY = os.getenv("API_KEY", "dev-api-key-change-me")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:7860")

stripe.api_key = STRIPE_SECRET_KEY

logger.info("STRIPE_PRICE_10: %s", os.getenv("STRIPE_PRICE_10"))
logger.info("STRIPE_PRICE_50: %s", os.getenv("STRIPE_PRICE_50"))
logger.info("STRIPE_PRICE_100: %s", os.getenv("STRIPE_PRICE_100"))

app = FastAPI(title="AniMate Studio API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limit_store: dict[str, list[float]] = {}

def _check_rate_limit(api_key: str) -> None:
    now = datetime.utcnow().timestamp()
    if api_key not in rate_limit_store:
        rate_limit_store[api_key] = []
    rate_limit_store[api_key] = [t for t in rate_limit_store[api_key] if now - t < 60]
    if len(rate_limit_store[api_key]) >= 5:
        raise HTTPException(429, "Rate limit exceeded. Max 5 requests per minute.")
    rate_limit_store[api_key].append(now)

class GenerateRequest(BaseModel):
    prompt: str
    character: Optional[str] = None
    duration: int = 10
    quality: str = "high"
    seed: Optional[int] = None
    narration: bool = False
    beast_mode: bool = True

class GenerateResponse(BaseModel):
    job_id: str
    status: str
    estimated_time: int

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    video_url: Optional[str] = None
    error: Optional[str] = None

class CheckoutRequest(BaseModel):
    user_id: str
    credits: int

DB_PATH = Path("api/users.json")
JOBS_DIR = Path("api/jobs")
OUTPUT_DIR = Path("output/api_generations")

DB_PATH.parent.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DB_PATH.exists():
    DB_PATH.write_text(json.dumps({}))

def load_db() -> dict:
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(data: dict) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

def verify_api_key(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = auth_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return token

def check_credits(user_id: str, amount: int = 1) -> bool:
    db = load_db()
    user = db.get(user_id, {"credits": 0})
    return user.get("credits", 0) >= amount

def deduct_credits(user_id: str, amount: int = 1) -> None:
    db = load_db()
    if user_id not in db:
        db[user_id] = {"credits": 0, "generated": 0}
    db[user_id]["credits"] -= amount
    db[user_id]["generated"] = db[user_id].get("generated", 0) + 1
    save_db(db)

def add_credits(user_id: str, amount: int) -> None:
    db = load_db()
    if user_id not in db:
        db[user_id] = {"credits": 0, "generated": 0}
    db[user_id]["credits"] += amount
    save_db(db)
    logger.info("Added %d credits to user %s", amount, user_id)

async def run_generation(job_id: str, request: GenerateRequest) -> None:
    job_file = JOBS_DIR / f"{job_id}.json"
    try:
        job_data = {"status": "processing", "progress": 0.1}
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        cmd = [
            "python", "batch_generate.py",
            "--prompt", request.prompt,
            "--character", request.character or "Billy Bunny",
            "--duration", str(request.duration),
            "--quality", request.quality,
            "--output", str(output_path),
        ]
        if request.beast_mode:
            cmd.append("--beast")
        if request.seed:
            cmd.extend(["--seed", str(request.seed)])
        if request.narration:
            cmd.append("--narrate")
        proc = await asyncio.create_subprocess_exec(
            *[arg for arg in cmd if arg],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="..",
        )
        job_data["progress"] = 0.5
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        stdout_txt = stdout.decode() if stdout else ""
        stderr_txt = stderr.decode() if stderr else ""
        if proc.returncode != 0:
            logger.error("Generation subprocess failed. stderr: %s", stderr_txt)
            raise Exception(f"Generation failed: {stderr_txt}")
        job_data = {
            "status": "completed",
            "progress": 1.0,
            "video_url": f"/download/{job_id}",
            "output_path": str(output_path),
        }
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        logger.info("Job %s completed. stdout: %s", job_id, stdout_txt[:200])
    except Exception as e:
        job_data = {"status": "failed", "error": str(e)}
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        logger.exception("Job %s failed", job_id)

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_api_key),
):
    _check_rate_limit(user_id)
    if not check_credits(user_id, amount=1):
        raise HTTPException(402, "Insufficient credits")
    job_id = str(uuid.uuid4())
    deduct_credits(user_id, amount=1)
    background_tasks.add_task(run_generation, job_id, request)
    return GenerateResponse(
        job_id=job_id,
        status="queued",
        estimated_time=request.duration * 2,
    )

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, user_id: str = Depends(verify_api_key)):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")
    with open(job_file, "r") as f:
        data = json.load(f)
    return JobStatus(
        job_id=job_id,
        status=data.get("status", "unknown"),
        progress=data.get("progress", 0.0),
        video_url=data.get("video_url"),
        error=data.get("error"),
    )

@app.get("/download/{job_id}")
async def download_video(job_id: str, user_id: str = Depends(verify_api_key)):
    video_path = OUTPUT_DIR / f"{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")

@app.get("/v1/credits")
async def get_credits(user_id: str = Depends(verify_api_key)):
    db = load_db()
    credits = db.get(user_id, {}).get("credits", 0)
    return {"credits": credits}

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid signature")
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("client_reference_id")
        credits_purchased = int(session.get("metadata", {}).get("credits", 0))
        if user_id and credits_purchased:
            add_credits(user_id, credits_purchased)
        else:
            logger.warning("Webhook missing user_id or credits metadata")
    else:
        logger.debug("Unhandled webhook event: %s", event["type"])
    return JSONResponse({"status": "success"})

@app.post("/v1/create-checkout-session")
async def create_checkout_session(request: CheckoutRequest):
    price_map = {
        10: os.getenv("STRIPE_PRICE_10"),
        50: os.getenv("STRIPE_PRICE_50"),
        100: os.getenv("STRIPE_PRICE_100"),
    }
    price_id = price_map.get(request.credits)
    if not price_id:
        raise HTTPException(400, "Invalid credit amount")
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price": price_id,
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{FRONTEND_URL}/?payment=success",
        cancel_url=f"{FRONTEND_URL}/?payment=cancelled",
        client_reference_id=request.user_id,
        metadata={"credits": str(request.credits)},
    )
    logger.info("Checkout session created for user %s (%d credits)", request.user_id, request.credits)
    return {"url": session.url}

@app.get("/health")
async def health():
    return {"status": "ok", "beast_mode": True}
