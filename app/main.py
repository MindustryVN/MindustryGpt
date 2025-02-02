import json
from dotenv import load_dotenv
import os
from fastapi.responses import StreamingResponse
import google.generativeai as genai
import numpy as np
from sqlalchemy import CursorResult, create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
APP_PORT = os.getenv("APP_PORT")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise ValueError("Environment variables GEMINI_API_KEY and POSTGRES_URL must be set.")


genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction="You are a chat companion, user is talking to you, response to them with related data, if there is no data or data is not related, answer it with your own knowledge")


app = FastAPI()


engine = create_engine(POSTGRES_URL)
Session = sessionmaker(bind=engine)
session = Session()


with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


class InsertRequest(BaseModel):
    text: str
    metadata: object

class UpdateRequest(BaseModel):
    text: str

class ResponseRequest(BaseModel):
    query: str


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

@app.post("/")
def insert_text(request: InsertRequest):
    """Insert text and its embedding into the database."""
    embedding = get_embedding(request.text)
    vector = parse_embedding(embedding).tolist()

    query = text("""
        INSERT INTO embeddings (vector, text, metadata)
        VALUES (:vector, :text, :metadata)
        RETURNING id
    """)
    result = session.execute(query, {"vector": vector, "text": request.text, "metadata": json.dumps( request.metadata)})
    session.commit()
    
    return {"id": result.fetchone()[0], "message": "Inserted successfully"}

@app.put("/{id}")
def update_text(id: int, request: UpdateRequest):
    """Update text in the database."""
    query = text("UPDATE embeddings SET text = :text WHERE id = :id")
    result = session.execute(query, {"text": request.text, "id": id})
    session.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"id": id, "message": "Updated successfully"}

@app.delete("/{id}")
def delete_text(id: int):
    """Delete a record by ID."""
    query = text("DELETE FROM embeddings WHERE id = :id")
    result = session.execute(query, {"id": id})
    session.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"id": id, "message": "Deleted successfully"}

@app.get("/")
def search_similar(page: int = 0, limit: int = 10, query: str | None = None):
    """Find similar text using vector similarity with pagination."""
    
    
    if query:
        
        embedding = get_embedding(query)
        query_vector = parse_embedding(embedding).tolist()
        
        order_clause = "ORDER BY vector <-> (:vector)::vector"
    else:
        
        query_vector = np.zeros(768).tolist()  
        order_clause = "ORDER BY id"  

    offset = (page) * limit
    
    query = text(f"""
        SELECT id, text, metadata 
        FROM embeddings
        {order_clause}
        LIMIT :limit
        OFFSET :offset
    """)
    result : CursorResult = session.execute(query, {"vector": query_vector, "limit": limit, "offset": offset})
        
    items: List[Dict[str, str]] = []

    for row in result.fetchall():
        items.append({"id": str(row[0]), "text": str(row[1]), "metadata": str(row[2])})

    return items


def stream(query: str):
    response = model.generate_content(query, stream=True)
    for chunk in response:
        yield chunk.text

@app.post("/respond")
async def respond_to_question(request: ResponseRequest):
    """Generate AI response using RAG."""
    embedding = get_embedding(request.query)
    query_vector = parse_embedding(embedding).tolist()
    
    query = text("""
        INSERT INTO embeddings (vector, text, metadata)
        VALUES (:vector, :text, :metadata)
        RETURNING id
    """)
    
    session.execute(query, {"vector": query_vector, "text": request.text, "metadata": request.metadata})
    session.commit()

    search_query = text("""
        SELECT text FROM embeddings
        ORDER BY vector <-> (:vector)::vector
        LIMIT 5
    """)
    retrieved_texts = session.execute(search_query, {"vector": query_vector}).fetchall()
    context = "\n".join(row[0] for row in retrieved_texts)
    
    print(f'Question: {request.query}\nContext: {context}')

    augmented_query = f"User question:{request.query}\n\n Related data:\n{context}"
    response = stream(augmented_query)
    
    return StreamingResponse(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(APP_PORT))
