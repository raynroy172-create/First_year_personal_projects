from pathlib import Path
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

video_url = 'https://www.youtube.com/watch?v=qcFdpBi5i38'

loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info = False,
)

docs = loader.load()
print(docs[0])


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

chunks = text_splitter.split_documents(documents = docs)

embedding_model = OpenAIEmbeddings(
    model = 'nvidia/llama-nemotron-embed-vl-1b-v2:free',
    api_key = os.environ['OPENROUTER_API_KEY'],
    base_url = 'https://openrouter.ai/api/v1',
    check_embedding_ctx_length = False,
    model_kwargs={"encoding_format": "float"},
)

vector_store = QdrantVectorStore.from_documents(
    documents = chunks,
    embedding=embedding_model,
    url='http://localhost:6333',
    collection_name='new_rag'
)

print('Indexing of video transcript done.....')