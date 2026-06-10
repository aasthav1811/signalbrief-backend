from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import re
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are SignalBrief AI, a B2B sales intelligence system.
Generate a concise, cited pre-meeting sales brief using only real, current information.
Focus on 2024-2026 data.

Respond ONLY with a valid raw JSON object. No markdown, no code fences, no preamble.

Schema:
{
  "company": "string",
  "tagline": "one sentence current description",
  "last_updated": "most recent notable event and approximate date",
  "confidence_overall": 85,
  "talking_points": [
    {"icon": "💰", "text": "specific actionable talking point"},
    {"icon": "📈", "text": "specific actionable talking point"},
    {"icon": "🎯", "text": "specific actionable talking point"}
  ],
  "insights": [
    {"type": "funding", "title": "specific title", "body": "2-3 sentences with sales relevance", "source": "Source name", "source_url": "https://real-url", "confidence": 88, "date": "Month Year"},
    {"type": "news", "title": "specific title", "body": "2-3 sentences", "source": "Source name", "source_url": "https://real-url", "confidence": 82, "date": "Month Year"},
    {"type": "hiring", "title": "specific title", "body": "2-3 sentences on hiring signals", "source": "LinkedIn", "source_url": "https://linkedin.com", "confidence": 70, "date": "Recent"},
    {"type": "crm", "title": "market position signal", "body": "2-3 sentences on competitive context", "source": "Crunchbase", "source_url": "https://crunchbase.com", "confidence": 75, "date": "2025-2026"}
  ],
  "filtered_count": 1,
  "discovery_questions": [
    "specific question referencing a real signal",
    "specific question referencing a real signal"
  ]
}

Rules:
- All facts must be real and current
- Confidence: 80+ verified, 60-79 moderate, 40-59 inferred
- filtered_count = insights dropped for confidence below 30
- Every talking point must reference a specific real signal
- source_url must be a plausible real URL"""

class BriefRequest(BaseModel):
    company: str

@app.get("/list-models")
async def list_models():
    """Debug endpoint to see available models for this API key"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        data = response.json()
    models = [m["name"] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
    return {"available_models": models}

@app.post("/generate-brief")
async def generate_brief(req: BriefRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    # First get available models dynamically
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        list_resp = await client.get(list_url)
        list_data = list_resp.json()

    available = [
        m["name"]
        for m in list_data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]

    # Preferred order — match exact names from /list-models (with models/ prefix stripped)
    preferred = [
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-flash-latest",
    ]

    # Pick first preferred model that is available
    model_name = None
    for p in preferred:
        if p in available:
            model_name = p.replace("models/", "")
            break

    # Fallback: just use first available, strip models/ prefix
    if not model_name and available:
        model_name = available[0].replace("models/", "")

    if not model_name:
        raise HTTPException(status_code=500, detail=f"No generateContent models available. Models found: {list_data}")

    prompt = f"{SYSTEM_PROMPT}\n\nGenerate a sales brief for: {req.company}. Include their latest funding rounds, news, product launches, and hiring trends from 2024-2026."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    text = re.sub(r'```json|```', '', text).strip()
    start = text.index('{')
    end = text.rindex('}') + 1
    brief = json.loads(text[start:end])
    brief["_model_used"] = model_name
    return brief

@app.get("/")
def root():
    return {"status": "SignalBrief API running"}