from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. Load the existing vectorstore from disk
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Ask a question and retrieve relevant chunks
question = "What clustering techniques were used?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # fetch top 3 chunks
results = retriever.invoke(question)

# 3. See what was retrieved 
print(f"Question: {question}\n")
print(f"Top {len(results)} retrieved chunks:\n")

for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} (page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content)
    print()
