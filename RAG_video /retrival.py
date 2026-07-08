from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model='nvidia/llama-nemotron-embed-vl-1b-v2:free',
    api_key=os.environ['OPENROUTER_API_KEY'],
    base_url='https://openrouter.ai/api/v1',
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)


vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url='http://localhost:6333',
    collection_name='learning_rag',
)


while True:
    user_query = input('➡️ Ask a question: ')

    search_results = vector_store.similarity_search(
        query=user_query,
        k=5,  
    )

    def format_doc(doc):
        meta = doc.metadata
        if "page_label" in meta or "page" in meta:
            # PDF-origin chunk
            location = f"Page: {meta.get('page_label', meta.get('page'))}"
            source = meta.get("source", "unknown")
        else:
            # Video-origin chunk
            location = f"Video: {meta.get('title', 'unknown')} by {meta.get('author', 'unknown')}"
            source = meta.get("source", "unknown")  # video_id for YoutubeLoaderDL

        return f"Content: {doc.page_content}\n{location}\nSource: {source}"

    context = "\n\n---\n\n".join(format_doc(doc) for doc in search_results)

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers user queries based on the available
    context retrieved from a video, you can give some extra content but stick to the context of the video
    , Also mention the timestamp you are reffering to.

    You should only answer the user based on the following context, and guide the
    user to open the right page number to know more.

    Context:
    {context}
    """

    load_dotenv()
    client = OpenAI( base_url = 'https://openrouter.ai/api/v1',
                    api_key=os.environ['OPENROUTER_API_KEY'])

    response = client.chat.completions.create(
        model = 'nvidia/nemotron-3-ultra-550b-a55b:free',
        messages = [{'role':'system', 'content':SYSTEM_PROMPT},
                {'role':'user', 'content':user_query}]
    )

    print(f'🧠✅{response.choices[0].message.content}')

