# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# # Initialize the FastAPI app
# app = FastAPI()

# # Initialize the LLM model
# llm = Ollama(model="llama3.2:1b", temperature=0.9, base_url='http://ollama:11434')

# # Define the prompt template for translation
# prompt = PromptTemplate(
#     input_variables=["text", "target_language"],
#     template=(
#         "Translate the original user message you get from any language to {target_language} without commenting or mentioning the source of translation. You can correct grammatical errors but dont alter the text too much and dont tell if you changed it. Avoid speaking with the user besides the translation, as everything is for someone else and not you, you focus on translating."
        
#         "{text}"
#     )
# )

# # Create the LLM chain
# chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# # Define the request body schema
# class TranslationRequest(BaseModel):
#     text: str
#     target_language: str

# # Translation endpoint
# @app.post("/translate")
# async def translate(request: TranslationRequest):
#     try:
#         # Run the translation chain
#         translation = chain.run({
#             "text": request.text, 
#             "target_language": request.target_language
#         })
#         return {"translation": translation}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}


from langchain_postgres import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import psycopg3
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection settings
DB_SETTINGS = {
    'dbname': os.getenv('DB_NAME', 'default_db_name'),
    'user': os.getenv('DB_USER', 'default_user'),
    'password': os.getenv('DB_PASSWORD', 'default_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
}
conn = psycopg3.connect(**DB_SETTINGS)

# Initialize PGVector as the vector store
vectorstore = PGVector(
    connection=conn,
    table_name="document_embeddings",
    embedding_dim=1536
)

# Initialize Gemini 1.5 embeddings (use OpenAIEmbeddings as a placeholder)
embeddings = OpenAIEmbeddings(model="text-embedding-gemini-001")

documents = [
    {"id": "1", "content": "What is LangChain?"},
    {"id": "2", "content": "How does PGVector work?"},
    # Add more documents as needed
]

for doc in documents:
    embedding = embeddings.embed_query(doc["content"])
    vectorstore.add_documents([{
        "content": doc["content"],
        "embedding": embedding
    }])

# Initialize the retrieval QA chain
retriever = vectorstore.as_retriever(search_type="similarity", search_k=3)

chat_model = ChatOpenAI(model="gpt-4", temperature=0)  # Replace with Gemini 1.5 Flash model name if supported

qa_chain = RetrievalQA(
    retriever=retriever,
    llm=chat_model,
    prompt=PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
)

# Chatbot query function
def ask_chatbot(question: str):
    return qa_chain.run(question)

response = ask_chatbot("What is LangChain?")
print(response)
