from dotenv import load_dotenv

import os
import google.generativeai as genai
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List

load_dotenv()

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
POSTGRES_URL: str | None = os.getenv("POSTGRES_URL")

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name='gemini-1.5-flash')


if POSTGRES_URL is None:
    raise ValueError("POSTGRES_URL is not set.")
conn = psycopg2.connect(POSTGRES_URL)
register_vector(conn)
cursor = conn.cursor()


engine = create_engine(POSTGRES_URL)
Session = sessionmaker(bind=engine)
session = Session()


def get_embedding(content: str, output_dimensionality: int = 768) -> List[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=content,
        output_dimensionality=output_dimensionality
    )
    return result["embedding"]


def parse_embedding(embedding: List[float]) -> np.ndarray:
    return np.array(embedding, dtype=np.float32)


def save_embedding(vector: np.ndarray, content: str, source: str) -> None:
    query = text("""
        INSERT INTO embeddings (vector, text, source)
        VALUES (:vector, :text, :source)
    """)
    session.execute(query, {"vector": vector.tolist(), "text": content, "source": source})
    session.commit()


def find_similar_vectors(query_vector: np.ndarray, limit: int = 5) -> List[str]:
    query = text("""
        SELECT text FROM embeddings
        ORDER BY vector <-> (:vector)::vector
        LIMIT :limit
    """)
    result = session.execute(query, {"vector": query_vector.tolist(), "limit": limit})
    return [row[0] for row in result.fetchall()]


def generate_response(query: str) -> str:
    
    embedding = get_embedding(query)
    query_vector = parse_embedding(embedding)

    
    retrieved_texts = find_similar_vectors(query_vector)
    context = " ".join(retrieved_texts)

    
    augmented_query = f"{query}\n\nContext:\n{context}"

    
    response = model.generate_content(augmented_query)
    
    return response.text


if __name__ == "__main__":
    
    sample_text: str = "This is a sample text for retrieval."
    sample_embedding: List[float] = get_embedding(sample_text)
    save_embedding(parse_embedding(sample_embedding), sample_text, "https://ai.google.dev/api/embeddings")

    
    query: str = "What is retrieval-augmented generation?"
    response: str = generate_response(query)
    print("Generated Response:", response)
