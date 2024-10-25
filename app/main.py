from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the FastAPI app
app = FastAPI()

# Initialize the LLM model
llm = Ollama(model="qwen2.5:0.5b", temperature=0.9,base_url='http://host.docker.internal:11434')

# Define the prompt template for translation
prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template=(
        "Identify the language of the following text and translate it into {target_language}. "
        "Preserve the original meaning, tone, and context. Do not alter proper names or technical terms.\n\n"
        "Text:\n{text}"
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
