# api.py
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
GEMINI_API_KEY = "AIzaSyAeV7pKkw41E9GclxTl7g8scmoTvpvFH6M" 
FAISS_INDEX_FILE = "faiss_index.bin"
DOCS_META_FILE = "docs.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# ================================
# LOAD MODELS AND INDEX
# ================================
if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(DOCS_META_FILE):
    raise RuntimeError(f"{FAISS_INDEX_FILE} or {DOCS_META_FILE} not found. Run build_index.py first.")

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

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# RETRIEVAL FUNCTION
# ================================
def retrieve(query: str, k: int = 10): 
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        print("Gemini raw response status:", response.status_code)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Gemini API request error: {e}")
        return f"Error: Could not connect to the generative AI service. {e}"
    except Exception as e:
        print("Gemini response parsing error:", str(e))
        return f"Error parsing Gemini response: {e}"

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

    # --- START: NEW HYBRID RETRIEVAL LOGIC ---
    
    query_lower = query.lower()
    docs = []
    
    # Check if the user is asking about the website
    if "website" in query_lower or "site" in query_lower or "jaljeevan.ai" in query_lower:
        print("Hybrid query detected. Retrieving both RWH and website context...")
        
        # 1. Get top 7 docs for the original query (this will get RWH PDFs)
        rwh_docs = retrieve(query, k=7)
        
        # 2. Get top 3 docs for a query biased towards the website
        # This forces the 'page_...' chunks to be retrieved
        website_query = query + " about the JalJeevan.ai website features and mission"
        website_docs = retrieve(website_query, k=3)
        
        # 3. Combine them and remove duplicates
        all_docs_dict = {doc['id']: doc for doc in rwh_docs + website_docs}
        docs = list(all_docs_dict.values())
        print(f"Combined docs. IDs: {[d['id'] for d in docs]}")
        
    else:
        # This is a normal RWH query
        print("Standard RWH query detected.")
        docs = retrieve(query, k=10) # Use k=10 for standard queries
    
    # --- END: NEW HYBRID RETRIEVAL LOGIC ---

    context = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])

    # The "smarter prompt" will now work because 'context' will
    # contain the 'page_' chunks it needs.
    prompt = (
        "You are an expert on Rooftop Rainwater Harvesting (RTRWH) and Artificial Recharge.\n"
        "You are also the helpful AI assistant for the 'JalJeevan.ai' website named Isha.\n\n"
        "--- RULES ---\n"
        "1. If the user asks about 'this website', 'your features', 'about the site', or 'JalJeevan.ai', you MUST prioritize context from sources that start with 'page_' (e.g., 'page_home', 'page_about').\n"
        "2. If no 'page_' sources are available, it is OK to say you don't have information about the website.\n"
        "3. For all other technical questions about RWH, you can use any relevant context.\n"
        "--- END RULES ---\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer clearly based on these rules and cite your sources."
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
    print("Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)