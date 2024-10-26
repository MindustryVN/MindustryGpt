from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the FastAPI app
app = FastAPI()

# Initialize the LLM model
llm = Ollama(model="llama3.2:1b", temperature=0.9, base_url='http://ollama:11434')

# Define the prompt template for translation
prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template=(
        "Translate the original user message you get from any language to {target_language} without commenting or mentioning the source of translation. You can correct grammatical errors but dont alter the text too much and dont tell if you changed it. Avoid speaking with the user besides the translation, as everything is for someone else and not you, you focus on translating."
        
        "{text}"
    )
)

# Create the LLM chain
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# Define the request body schema
class TranslationRequest(BaseModel):
    text: str
    target_language: str

# Translation endpoint
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Run the translation chain
        translation = chain.run({
            "text": request.text, 
            "target_language": request.target_language
        })
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
