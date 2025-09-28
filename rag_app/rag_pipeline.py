import os
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# === Load PDFs and preprocess ===
def load_pdf_files(pdf_path: str) -> List[Document]:
    loader = DirectoryLoader(
        pdf_path,
        glob="**/*.pdf",
        loader_cls=lambda path: PyPDFLoader(path),
        show_progress=True
    )
    return loader.load()

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")}
        ) for doc in docs
    ]

def text_split(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(docs)

# === Embedding ===
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

# === Vector DB ===
def get_vector_store(embedding, docs):
    vector_store = Chroma(
        collection_name="rag-collection",
        embedding_function=embedding,
        persist_directory="./chroma_db"
    )
    vector_store.add_documents(docs)
    return vector_store

# === Reranker ===
def get_compression_retriever(base_retriever):
    cohere_api_key = os.getenv("COHERE_API_KEY")
    compressor = CohereRerank(
        model="rerank-english-v3.0", 
        cohere_api_key=cohere_api_key
    )
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

# === LLM ===
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0
    )

# === Prompt ===
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === RAG Chain ===
def build_rag_chain():
    docs = load_pdf_files("../industrial-safety-pdfs")
    minimal_docs = filter_to_minimal_docs(docs)
    chunks = text_split(minimal_docs)
    
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings, chunks)
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    compression_retriever = get_compression_retriever(retriever)
    
    llm = get_llm()
    
    parallel_chain = RunnableParallel({
        'context': compression_retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    rag_chain = parallel_chain | prompt | llm | StrOutputParser()
    return rag_chain
