from typing import Union
from fastapi import FastAPI

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import google.generativeai as gemini_client

import os
from dotenv import load_dotenv

load_dotenv()

qdrant_host = os.environ.get("QDRANT_HOST")
qdrant_port = os.environ.get("QDRANT_PORT")

print(qdrant_host)
print(qdrant_port)

client = QdrantClient(host=qdrant_host, port=qdrant_port)

collection_name="test_recommend"

# client.recreate_collection(
#    collection_name=collection_name,
#    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )

GEMINI_KEY=api_key=os.environ.get("GEMINI_API_KEY")
print(GEMINI_KEY)
gemini_client.configure(api_key=GEMINI_KEY)

app = FastAPI()

def vectorize_sentence(sentence):
    return gemini_client.embed_content(
        model="models/embedding-001",
        content=sentence,
        task_type="retrieval_document",
        title="sentence",
    )


@app.get("/rag")
def read_root():
    return {"rag": "manager"}

@app.get("/rag/sentence")
def read_item(sentence: str):
    return vectorize_sentence(sentence)

@app.get("/rag/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/rag/sentence/insert")
def insert_sentence(id: int, sentence: str):

    texts = [
        sentence
    ]
    print(texts)
    results = [
        gemini_client.embed_content(
            model="models/embedding-001",
            content=sentence,
            task_type="retrieval_document",
            title="sentence",
        )
        for sentence in texts
    ]
    points = [
        PointStruct(
            id=id,
            vector=response['embedding'],
            payload={"text": text},
        )
        for idx, (response, text) in enumerate(zip(results, texts))
    ]
    return client.upsert(collection_name, points)

@app.get("/rag/sentence/search")
def find_similar_sentences(query_sentence, top_k=10):
    results = client.search(
        collection_name=collection_name,
        query_vector=gemini_client.embed_content(
            model="models/embedding-001",
            content=query_sentence,
            task_type="retrieval_query",
        )["embedding"],
    )
    print(results)
    return results  # Return list of sentence IDs
