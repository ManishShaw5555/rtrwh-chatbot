# api_with_gemini.py
import os
import json
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional, Dict

# ================================
# CONFIG
# ================================
GEMINI_API_KEY = "AIzaSyAeV7pKkw41E9GclxTl7g8scmoTvpvFH6M"  # Directly using key for now
FAISS_INDEX_FILE = "faiss_index.bin"
DOCS_META_FILE = "docs.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# ================================
# LOAD MODELS AND INDEX
# ================================
if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(DOCS_META_FILE):
    raise RuntimeError("FAISS index or docs.json not found. Run build_index.py first.")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)

print("Loading metadata...")
with open(DOCS_META_FILE, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

documents = meta_data["documents"]
ids = meta_data["ids"]
metas = meta_data["metas"]

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ================================
# FASTAPI APP
# ================================
app = FastAPI(title="RTRWH Chatbot with Gemini + FAISS")

# ================================
# DETERMINISTIC CALCULATIONS
# ================================
def harvest_water_cubic_m(roof_m2: float, annual_rain_mm: float, runoff: float) -> float:
    """Calculate annual rainwater harvesting potential in cubic meters."""
    rainfall_m = annual_rain_mm / 1000.0
    return roof_m2 * rainfall_m * runoff

def recommend_tank_size(roof_m2: float, annual_rain_mm: float, runoff: float, months: int = 2) -> float:
    """Recommend tank size for given storage duration in months."""
    annual = harvest_water_cubic_m(roof_m2, annual_rain_mm, runoff)
    return (annual / 12) * months

# ================================
# RETRIEVAL FUNCTION
# ================================
def retrieve(query: str, k: int = 4):
    query_emb = embed_model.encode([query])
    query_emb = np.array(query_emb, dtype="float32")
    distances, indices = index.search(query_emb, k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append({
            "id": ids[idx],
            "text": documents[idx],
            "meta": metas[idx]
        })
    return results

# ================================
# GEMINI API CALL
# ================================
def call_gemini(prompt: str) -> str:
    # Use Gemini key as query parameter
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    # Debugging output
    print("Gemini raw response status:", response.status_code)
    print("Gemini raw response text:", response.text)

    # If non-200 response, raise error
    response.raise_for_status()

    try:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("Gemini response parsing error:", str(e))
        return f"Error parsing Gemini response: {response.text}"

# ================================
# API MODELS
# ================================
class ChatRequest(BaseModel):
    message: str
    roof_area: Optional[float] = None
    annual_rainfall: Optional[float] = None
    runoff: Optional[float] = 0.8

# ================================
# CHAT ENDPOINT
# ================================
@app.post("/chat")
async def chat_endpoint(req: ChatRequest = Body(...)):
    query = req.message.strip()

    # If user provides data, perform calculation
    if req.roof_area and req.annual_rainfall:
        water = harvest_water_cubic_m(req.roof_area, req.annual_rainfall, req.runoff or 0.8)
        tank_size = recommend_tank_size(req.roof_area, req.annual_rainfall, req.runoff or 0.8)

        return {
            "answer": (
                f"Based on the provided data:\n"
                f"- Roof Area: {req.roof_area} m²\n"
                f"- Annual Rainfall: {req.annual_rainfall} mm\n"
                f"- Runoff Coefficient: {req.runoff}\n\n"
                f"Estimated Harvested Water (annually): {water:.2f} m³\n"
                f"Recommended Tank Size (2 months storage): {tank_size:.2f} m³"
            ),
            "source": "calculation"
        }

    # Else, retrieve context and ask Gemini
    docs = retrieve(query)
    context = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])

    prompt = (
        "You are an expert on Rooftop Rainwater Harvesting (RTRWH) and Artificial Recharge.\n"
        "Answer the user's query using the provided context.\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer clearly and cite relevant sources by their IDs."
    )

    answer = call_gemini(prompt)

    return {
        "answer": answer,
        "sources": [d["id"] for d in docs]
    }

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
