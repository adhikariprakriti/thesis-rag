import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Thesis Q&A", page_icon="📄")
st.title("Chat with my Thesis")
st.caption("Software Development Practices in Simulation Modeling")

DB_DIR = "./chroma_db"
PDF_PATH = "thesis.pdf"

@st.cache_resource
def get_rag_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if the database exists and has files. If not, build it dynamically!
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        st.info("Database not found on server. Initializing vector store from thesis.pdf...")
        
        if not os.path.exists(PDF_PATH):
            st.error(f"Critical Error: '{PDF_PATH}' was not found in the root directory of your repository. Please upload it to GitHub.")
            st.stop()
            
        # Load and split
        loader = PyMuPDFLoader(PDF_PATH)
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(pages)
        
        # Build vector store on the server's disk
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
    else:
        # Load existing vectorstore from the deployed directory
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
    You are an expert research assistant. Use the context below (extracted from a thesis)
    to answer the question. Be specific and cite details from the context.
    If the answer is not in the context, say "I don't have enough information from the thesis."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Initialize the chain safely
chain = get_rag_chain()

# Chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input 
if question := st.chat_input("Ask a question about the thesis..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching thesis..."):
            answer = chain.invoke(question)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})