from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""You are SignalBrief AI, a B2B sales intelligence system.
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
- All facts must be real and current — no hallucinations
- Confidence: 80+ verified, 60-79 moderate, 40-59 inferred
- filtered_count = insights dropped for confidence below 30
- Every talking point must reference a specific real signal
- source_url must be a plausible real URL"""
)

class BriefRequest(BaseModel):
    company: str

@app.post("/generate-brief")
async def generate_brief(req: BriefRequest):
    prompt = f"Generate a sales brief for: {req.company}. Research their latest funding rounds, news, product launches, and hiring trends from 2024-2026."
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # Strip any markdown fences if model adds them
    text = re.sub(r'```json|```', '', text).strip()
    
    # Extract JSON object
    start = text.index('{')
    end = text.rindex('}') + 1
    brief = json.loads(text[start:end])
    
    return brief

@app.get("/")
def root():
    return {"status": "SignalBrief API running"}