"""
AniMate Studio Monetization API (FastAPI)
- Endpoints: /generate, /credits, /webhook/stripe, /create-checkout-session
- API key authentication via Bearer token (from .env)
- Credit system using api/users.json
- Stripe for payments

Setup:
1. pip install fastapi uvicorn stripe python-dotenv pyyaml
2. Place .env with STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, API_KEY
3. Place users.json in api/ (auto-created if missing)
4. Run: uvicorn api.animate_api:app --host 0.0.0.0 --port 8001
"""

import os
import json
import uuid
import stripe
import yaml
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

# Load config and env
load_dotenv()
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
API_KEY = os.getenv("API_KEY")
CREDITS_PER_VIDEO = 1
USERS_PATH = Path(__file__).parent / "users.json"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Stripe setup
stripe.api_key = STRIPE_SECRET_KEY

# FastAPI app
app = FastAPI()

def get_users():
    if not USERS_PATH.exists():
        USERS_PATH.write_text(json.dumps({}))
    with open(USERS_PATH, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)

def require_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(403, "Invalid API key")
    return token

class GenerateRequest(BaseModel):
    theme: str
    character: str
    duration: int
    beast_mode: bool = False

@app.post("/generate")
def generate(req: GenerateRequest, token: str = Depends(require_api_key)):
    users = get_users()
    if token not in users or users[token] < CREDITS_PER_VIDEO:
        raise HTTPException(402, "Insufficient credits")
    # Import engine and generate video
    from engine.animator import generate_video  # type: ignore
    video_path = generate_video(
        theme=req.theme,
        character=req.character,
        duration=req.duration,
        beast_mode=req.beast_mode
    )
    users[token] -= CREDITS_PER_VIDEO
    save_users(users)
    return FileResponse(video_path, filename=os.path.basename(video_path))

@app.get("/credits")
def credits(token: str = Depends(require_api_key)):
    users = get_users()
    return {"credits": users.get(token, 0)}

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Webhook error: {e}")
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        api_key = session["client_reference_id"]
        amount = int(session["amount_total"] / 100)
        credits = amount  # 1 credit per $1
        users = get_users()
        users.setdefault(api_key, 0)
        users[api_key] += credits
        save_users(users)
    return {"status": "success"}

@app.post("/create-checkout-session")
def create_checkout_session(token: str = Depends(require_api_key)):
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": "AniMate Studio Credits"},
                "unit_amount": 100,  # $1 per credit
            },
            "quantity": 10,
        }],
        mode="payment",
        success_url="https://yourdomain.com/success",
        cancel_url="https://yourdomain.com/cancel",
        client_reference_id=token,
    )
    return {"checkout_url": session.url}
