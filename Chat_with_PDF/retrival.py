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
    user_query = input('Ask a question: ')

    search_results = vector_store.similarity_search(
        query=user_query,
        k=5,  
    )

    context = "\n\n---\n\n".join(
        f"Page Content: {doc.page_content}\n"
        f"Page Number: {doc.metadata.get('page_label', doc.metadata.get('page'))}\n"
        f"Source: {doc.metadata.get('source')}"
        for doc in search_results
    )

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers user queries based on the available
    context retrieved from a PDF file, along with page numbers.

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

    print(response.choices[0].message.content)

