from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# 1. Ensure the file exists
file_path = "dummy_data.txt"
if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        f.write("The emergency override password is Blueberry-99. Project Nebula-7 is active.")

# 2. Load and Embed
loader = TextLoader(file_path)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create and Persist
# We use 'collection_metadata' to ensure the search is exact
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./vector_db",
    collection_metadata={"hnsw:space": "cosine"} 
)

print("✅ Data written. Check your 'vector_db' folder for new files.")