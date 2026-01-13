from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup (Same reliable core)
llm = ChatOllama(model="gemma3:4b", temperature=0.1) # Added slight creativity
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 2. Modern Prompt Template
template = """You are a helpful assistant for Project Nebula. 
Use the context below to answer questions. If the answer isn't there, say you don't know.

Context: {context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 3. The Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. The Interactive Loop
print("\n--- 🤖 NEBULA-7 CHATBOT ONLINE ---")
print("Type 'exit' to quit or 'reindex' if you changed your data file.")

while True:
    user_input = input("\nUser: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    
    if user_input.lower() == 'reindex':
        print("Updating database... please run ingest.py again and restart.")
        break

    print("AI Thinking...")
    response = chain.invoke(user_input)
    print(f"AI: {response}")