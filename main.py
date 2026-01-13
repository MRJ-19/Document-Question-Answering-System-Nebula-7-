from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# 1. FIX CORS (Put this first!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DEFINE THE RAG COMPONENTS
# This must happen BEFORE the @app.post function
llm = ChatOllama(model="gemma3:4b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- THIS IS THE 'chain' THE ERROR WAS MISSING ---
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. DEFINE THE API ENDPOINT
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_ai(query: Query):
    # Now 'chain' is defined globally, so this will work!
    response = chain.invoke(query.question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)