from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class BriefRequest(BaseModel):
    company: str

@app.post("/generate-brief")
async def generate_brief(req: BriefRequest):
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1800,
        system="""You are SignalBrief AI. Generate a real-time B2B sales brief as JSON only. 
No markdown, no preamble. Schema:
{
  "company":"string",
  "tagline":"one sentence",
  "last_updated":"most recent event and when",
  "confidence_overall":85,
  "talking_points":[{"icon":"💰","text":"specific point"},{"icon":"📈","text":"specific point"},{"icon":"🎯","text":"specific point"}],
  "insights":[
    {"type":"funding","title":"title","body":"2-3 sentences","source":"Source","source_url":"https://...","confidence":88,"date":"Month Year"},
    {"type":"news","title":"title","body":"2-3 sentences","source":"Source","source_url":"https://...","confidence":82,"date":"Month Year"},
    {"type":"hiring","title":"title","body":"2-3 sentences","source":"LinkedIn","source_url":"https://linkedin.com","confidence":70,"date":"Recent"},
    {"type":"crm","title":"title","body":"2-3 sentences","source":"Crunchbase","source_url":"https://crunchbase.com","confidence":75,"date":"2025-2026"}
  ],
  "discovery_questions":["question 1","question 2"]
}""",
        messages=[{
            "role": "user",
            "content": f"Sales brief for: {req.company}. Find recent 2024-2026 funding, news, hiring."
        }]
    )
    
    import json, re
    text = message.content[0].text
    clean = re.sub(r'```json|```', '', text).strip()
    start, end = clean.index('{'), clean.rindex('}')
    return json.loads(clean[start:end+1])

@app.get("/")
def root():
    return {"status": "SignalBrief API running"}