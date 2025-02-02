from dotenv import load_dotenv
import os
import google.generativeai as genai
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
APP_PORT = os.getenv("APP_PORT")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise ValueError("Environment variables GEMINI_API_KEY and POSTGRES_URL must be set.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Setup FastAPI
app = FastAPI()

# Setup SQLAlchemy with pgvector
engine = create_engine(POSTGRES_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Ensure the vector extension is enabled
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

# Pydantic models for request validation
class InsertRequest(BaseModel):
    text: str
    source: str

class UpdateRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class ResponseRequest(BaseModel):
    query: str

# Functions for database operations
def get_embedding(content: str, output_dimensionality: int = 768) -> List[float]:
    """Generate text embedding using Gemini API."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=content,
        output_dimensionality=output_dimensionality
    )
    return result.get("embedding", [])

def parse_embedding(embedding: List[float]) -> np.ndarray:
    """Convert embedding list to NumPy array."""
    return np.array(embedding, dtype=np.float32)

@app.post("/insert")
def insert_text(request: InsertRequest):
    """Insert text and its embedding into the database."""
    embedding = get_embedding(request.text)
    vector = parse_embedding(embedding).tolist()

    query = text("""
        INSERT INTO embeddings (vector, text, source)
        VALUES (:vector, :text, :source)
        RETURNING id
    """)
    result = session.execute(query, {"vector": vector, "text": request.text, "source": request.source})
    session.commit()
    
    return {"id": result.fetchone()[0], "message": "Inserted successfully"}

@app.put("/update/{id}")
def update_text(id: int, request: UpdateRequest):
    """Update text in the database."""
    query = text("UPDATE embeddings SET text = :text WHERE id = :id")
    result = session.execute(query, {"text": request.text, "id": id})
    session.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"id": id, "message": "Updated successfully"}

@app.delete("/delete/{id}")
def delete_text(id: int):
    """Delete a record by ID."""
    query = text("DELETE FROM embeddings WHERE id = :id")
    result = session.execute(query, {"id": id})
    session.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"id": id, "message": "Deleted successfully"}

@app.get("/search")
def search_similar(request: SearchRequest):
    """Find similar text using vector similarity."""
    query = text("""
        SELECT id, text FROM embeddings
        ORDER BY vector <-> (:vector)::vector
        LIMIT :limit
    """)
    embedding = get_embedding(request.query)
    query_vector = parse_embedding(embedding).tolist()

    result = session.execute(query, {"vector": query_vector, "limit": request.limit})
    items = [{"id": row[0], "text": row[1]} for row in result.fetchall()]

    return {"query": request.query, "results": items}

@app.post("/respond")
def respond_to_question(request: ResponseRequest):
    """Generate AI response using RAG."""
    embedding = get_embedding(request.query)
    query_vector = parse_embedding(embedding).tolist()

    search_query = text("""
        SELECT text FROM embeddings
        ORDER BY vector <-> (:vector)::vector
        LIMIT 5
    """)
    retrieved_texts = session.execute(search_query, {"vector": query_vector}).fetchall()
    context = " ".join(row[0] for row in retrieved_texts)
    
    print(f'Context: {context}')

    augmented_query = f"{request.query}\n\nContext:\n{context}"
    response = model.generate_content(augmented_query)
    
    return response.text

# Run FastAPI with Uvicorn (if executing standalone)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(APP_PORT))
