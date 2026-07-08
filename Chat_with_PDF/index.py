from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

pdf_path = Path(__file__).parent / 'AI.pdf'

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
print(docs[2])

# Split docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents=docs)

# Vector Embeddings via OpenRouter (OpenAI-compatible /v1/embeddings)
embedding_model = OpenAIEmbeddings(
    model='nvidia/llama-nemotron-embed-vl-1b-v2:free',
    api_key=os.environ['OPENROUTER_API_KEY'],
    base_url='https://openrouter.ai/api/v1',
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url='http://localhost:6333',
    collection_name='learning_rag'
)

print('Indexing of documents done.....')