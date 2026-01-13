from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Model and DB
llm = ChatOllama(model="gemma3:4b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# 2. Define the Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. The "Chain" (This is the modern way)
# This says: Get context, keep the question, pass to prompt, pass to LLM, parse result.
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Run and Debug
question = "What is the emergency override password?"
print(f"DEBUG: Searching for '{question}'...")

# First, let's see if the retriever finds anything at all
docs = retriever.invoke(question)
if docs:
    print(f"DEBUG: Found Context -> {docs[0].page_content}")
    # Now run the full chain
    response = chain.invoke(question)
    print(f"\nFinal Answer: {response}")
else:
    print("DEBUG: The retriever found nothing. Check your dummy_data.txt content.")