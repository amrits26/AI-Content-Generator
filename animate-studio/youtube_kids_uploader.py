"""
YouTube Kids Auto-Uploader for AniMate Studio
- Reads batch_themes.csv (theme, character, duration)
- Generates videos using engine.animator.generate_video
- Uploads to YouTube with kid-friendly metadata
- Tracks uploads in SQLite

Setup:
1. pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
2. Place client_secrets.json in project root.
3. python youtube_kids_uploader.py

Monetization: Automates YouTube uploads for AdSense revenue.
"""

import csv
import os
import sqlite3
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

BATCH_CSV = Path("batch_themes.csv")
DB_PATH = "youtube_uploads.db"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
    creds = flow.run_console()
    return build("youtube", "v3", credentials=creds)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS uploads (video_path TEXT PRIMARY KEY)")
    return conn

def already_uploaded(conn, video_path):
    return conn.execute("SELECT 1 FROM uploads WHERE video_path=?", (video_path,)).fetchone() is not None

def mark_uploaded(conn, video_path):
    conn.execute("INSERT INTO uploads (video_path) VALUES (?)", (video_path,))
    conn.commit()

def make_title(theme, character):
    return f"{character.title()}'s {theme.title()} Adventure 🌟"

def make_description(theme, character):
    return f"Enjoy this magical {theme} story starring {character}! Made for kids by AniMate Studio."

def main():
    from engine.animator import generate_video  # type: ignore
    conn = init_db()
    youtube = get_authenticated_service()
    with open(BATCH_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            theme, character, duration = row["theme"], row["character"], int(row["duration"])
            video_path = generate_video(theme, character, duration)
            if already_uploaded(conn, video_path):
                print(f"Already uploaded: {video_path}")
                continue
            body = {
                "snippet": {
                    "title": make_title(theme, character),
                    "description": make_description(theme, character),
                    "tags": [theme, character, "kids", "animation"],
                    "categoryId": "1",
                },
                "status": {
                    "privacyStatus": "public",
                    "madeForKids": True,
                }
            }
            media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
            response = request.execute()
            print(f"Uploaded: {video_path} (YouTube ID: {response['id']})")
            mark_uploaded(conn, video_path)

if __name__ == "__main__":
    main()
