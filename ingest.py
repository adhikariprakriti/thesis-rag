import shutil, os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv() 

# 1. Load the PDF document
loader = PyMuPDFLoader("thesis.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # ~500 characters per chunk
    chunk_overlap=500,      # 50 chars overlap between chunks (keeps context)
    separators=["\n\n", "\n", ".", " "]  # tries to split on paragraphs first
)
chunks = splitter.split_documents(pages)

print(f"Split into {len(chunks)} chunks")
print(f"{chunks[5].page_content}")

# 3. Embed & store in ChromaDB
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Deleted old chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   
)

print("All chunks embedded and saved to ./chroma_db")


