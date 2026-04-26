"""
Backend for AniMate Monetization Dashboard
- Serves dashboard.html and /dashboard-data for charts.

Setup:
1. pip install fastapi uvicorn
2. Place in api/ folder.
3. Run: uvicorn api.dashboard_backend:app --host 0.0.0.0 --port 8000
"""

import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from datetime import datetime, timedelta

app = FastAPI()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text()

@app.get("/dashboard-data")
def dashboard_data():
    # Dummy data for demo; replace with real stats
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    videos = [5, 7, 6, 8, 10, 12, 9]
    revenue = [2.5 * v for v in videos]
    # Read credits from users.json
    users_path = Path(__file__).parent / "users.json"
    if users_path.exists():
        users = json.loads(users_path.read_text())
        credits = [{"api_key": k, "credits": v} for k, v in users.items()]
    else:
        credits = []
    return {
        "dates": dates,
        "videos": videos,
        "revenue": revenue,
        "credits": credits,
    }
