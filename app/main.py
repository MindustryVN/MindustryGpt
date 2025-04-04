import json
import logging
from dotenv import load_dotenv
import os
from fastapi.responses import StreamingResponse
import google.generativeai as genai
import numpy as np
from sqlalchemy import CursorResult, create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, List, Union

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
APP_PORT = os.getenv("APP_PORT")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise ValueError("Environment variables GEMINI_API_KEY and POSTGRES_URL must be set.")


genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction="Bạn là một người bạn trên discord, hãy trả lời các cuộc trò truyện một cách vui vẻ")


app = FastAPI(debug=True)


class InsertRequest(BaseModel):
    text: str
    metadata: object

class UpdateRequest(BaseModel):
    text: str

class ResponseRequest(BaseModel):
    query: str
    channel: str = "default"


# Remove the global session
engine = create_engine(POSTGRES_URL)
Session = sessionmaker(bind=engine)

# Add dependency for session management
def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

# Modify route handlers to use the session dependency
@app.post("/")
def insert_text(request: InsertRequest, db: Session = Depends(get_db)):
    try:
        embedding = get_embedding(request.text)
        vector = parse_embedding(embedding).tolist()

        query = text("""
            INSERT INTO embeddings (vector, text, metadata)
            VALUES (:vector, :text, :metadata)
            RETURNING id
        """)
        result = db.execute(query, {"vector": vector, "text": request.text, "metadata": json.dumps(request.metadata)})
        db.commit()
        
        return {"id": result.fetchone()[0], "message": "Inserted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Update other route handlers similarly
@app.put("/{id}")
def update_text(id: int, request: UpdateRequest, db: Session = Depends(get_db)):
    try:
        query = text("UPDATE embeddings SET text = :text WHERE id = :id")
        result = db.execute(query, {"text": request.text, "id": id})
        db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Entry not found")

        return {"id": id, "message": "Updated successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/{id}")
def delete_text(id: int, db: Session = Depends(get_db)):
    try:
        query = text("DELETE FROM embeddings WHERE id = :id")
        result = db.execute(query, {"id": id})
        db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Entry not found")

        return {"id": id, "message": "Deleted successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def search_similar(
    page: int = 0, 
    limit: int = 10, 
    query: str | None = None,
    db: Session = Depends(get_db)
):
    try:
        if query:
            embedding = get_embedding(query)
            query_vector = parse_embedding(embedding).tolist()
            order_clause = "ORDER BY vector <-> (:vector)::vector"
        else:
            query_vector = np.zeros(768).tolist()
            order_clause = "ORDER BY id"

        offset = page * limit

        query = text(f"""
            SELECT id, text, metadata 
            FROM embeddings
            {order_clause}
            LIMIT :limit
            OFFSET :offset
        """)
        result: CursorResult = db.execute(query, {"vector": query_vector, "limit": limit, "offset": offset})
        
        items = [{"id": str(row[0]), "text": str(row[1]), "metadata": str(row[2])} for row in result.fetchall()]

        return items 

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Update stream function to accept db session
def stream(query: str, channel: str, history: List[Union[str,str]], db: Session):    
    history = [{
        "role": "user",
        "parts": i[0]
    } for i in history]

    print(f"History:  {history}, query: {query}")
    
    print(f'Question: {query}\nContext: {history}')
    
    chat_session = model.start_chat(history=history)
    
    response = chat_session.send_message(query, stream=True)
    
    full = ""
    
    for chunk in response:
        full += chunk.text
        yield chunk.text
        
    embedding = get_embedding(full)
    vector = parse_embedding(embedding).tolist()

    query = text("""
        INSERT INTO embeddings (vector, text, metadata)
        VALUES (:vector, :text, :metadata)
        RETURNING id
    """)
    
    db.execute(query, {"vector": vector, "text": full, "metadata": json.dumps({"id": "bot", "user": "you"})})
    db.commit()

@app.post("/respond")
async def respond_to_question(request: ResponseRequest, db: Session = Depends(get_db)):
    embedding = get_embedding(request.query)
    query_vector = parse_embedding(embedding).tolist()
    channel = request.channel

    if (channel == None):
        channel = "default"
    
    search_query = text("""
        SELECT text, metadata FROM embeddings
        WHERE (metadata->id)::text = :channel
        ORDER BY vector <-> (:vector)::vector
        LIMIT 3
    """)
    retrieved_texts = db.execute(search_query, {"vector": query_vector, "channel": channel}).fetchall()
    
    query = text(f"""
        SELECT id, text, metadata 
        FROM embeddings
        WHERE (metadata->id)::text = :channel
        ORDER BY id DESC
        LIMIT 3
    """)

    latest = db.execute(query, {"vector": query_vector, "channel": channel}).fetchall()

    response = stream(request.query, channel, retrieved_texts + latest, db)
    
    return StreamingResponse(response)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(APP_PORT),log_level="trace")
