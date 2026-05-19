from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Load vectorstore 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. Prompt 
prompt = PromptTemplate.from_template("""
You are an expert research assistant. Use the context below (extracted from a thesis)
to answer the question. Be specific and cite details from the context.
If the answer is not in the context, say "I don't have enough information from the thesis."

Context:
{context}

Question: {question}

Answer:
""")

# 3. LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask questions
questions = [
    "What clustering techniques were used and why?",
    "What is the main finding of this thesis?",
    "How many GitHub repositories were analyzed?"
]

for question in questions:
    print(f"\n{'='*50}")
    print(f"❓ {question}")
    print('='*50)
    answer = chain.invoke(question)
    print(f"{answer}")