# Load web page
import argparse
import os

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Filter out URL argument.')
    parser.add_argument('--url', type=str, default='https://valiantlynx.com', required=True, help='The URL to filter out.')

    args = parser.parse_args()
    url = args.url
    print(f"using URL: {url}")

    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")


    print(f"Loaded {len(data)} documents")

    from langchain import hub
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


    llm = Ollama(model="llama2-uncensored",
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    print(f"Loaded LLM model {llm.model}")

    from langchain.vectorstores import PGVector
     
    connection_string = os.getenv("PG_CONNECTION_STRING")
    collection_name = os.getenv("PG_COLLECTION_NAME")
    

    embeddings = OllamaEmbeddings(
        model="llama3",
    )
        
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=[],
        collection_name=collection_name,
        connection_string=connection_string,
    )

    retriever = db.as_retriever()  
        
    from langchain.chains import RetrievalQA
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},

    )

    question = f"summarize what this blog is trying to say? {url}?"
    result = qa_chain({"query": question})
    print(result) 


if __name__ == "__main__":
    main()
