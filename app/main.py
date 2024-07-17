from typing import Union, List
from fastapi import FastAPI
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import google.generativeai as gemini_client

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

qdrant_host = os.environ.get("QDRANT_HOST")
qdrant_port = os.environ.get("QDRANT_PORT")

print(qdrant_host)
print(qdrant_port)

client = QdrantClient(host=qdrant_host, port=qdrant_port)

collection_recommend_name="test_recommend"

# client.recreate_collection(
#    collection_name=collection_name,
#    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )

collection_db_name="test_db"

# client.recreate_collection(
#    collection_name=collection_db_name,
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

rag_domain = os.getenv("RAG_DOMAIN")

# API 주소 (RAG_DOMAIN 환경 변수 사용)


@app.get("/rag")
def read_root():
    return {"rag": "manager"}

@app.get("/rag/sentence")
def read_item(sentence: str):
    return vectorize_sentence(sentence)

@app.get("/rag/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

class SentenceRequest(BaseModel):
    sentences: List[object]


@app.get("/rag/db_sync")
def sync_db_schema():
    # API 호출
    url = f"{rag_domain}/schema/tables"
    response = requests.get(url)

    # 응답 확인
    if response.status_code == 200:
        data = response.json()

        tables_info = []

        # 테이블 목록 반복
        for table in data["tables"]:
            # 각 테이블 상세 정보 API 호출
            table_info_url = f"{rag_domain}/schema/tables/{table}"
            table_info_response = requests.get(table_info_url)

            if table_info_response.status_code == 200:
                table_info_data = table_info_response.json()

                # 테이블 정보 추출
                table_info_str = ""  # 테이블 정보 문자열 저장 변수
                for key, value in table_info_data.items():
                    if key != "state":  # "state" 키는 제외
                        table_info_str += f"{key}: {value}\n"

                # 테이블 정보 딕셔너리 생성
                table_info_dict = {"title": table, "content": table_info_str}

                # 테이블 정보 목록에 추가
                tables_info.append(table_info_dict)
            else:
                print(f"테이블 상세 정보 API 호출 오류 ({table}): {table_info_response.status_code}")
        # 전체 테이블 정보 출력
        print(tables_info)
        qdrant_insert_sentences(tables_info, collection_db_name)


        return tables_info
    else:
        # 오류 발생 시 에러 메시지 출력
        print(f"API 호출 오류: {response.status_code}")



@app.post("/rag/sentences")
def insert_sentences(sentence_request: SentenceRequest):
    print(sentence_request)
    return qdrant_insert_sentences(sentence_request.sentences, collection_recommend_name)


@app.get("/rag/sentence/insert")
def insert_sentence(id: int, sentence: str):
    texts = [
        { 'title': sentence, 'content': sentence }
    ]  
    print(texts)
    return qdrant_insert_sentences(texts, collection_recommend_name)


def qdrant_insert_sentences(datas, collection_name):
    results = [
        gemini_client.embed_content(
            model="models/embedding-001",
            content=data["content"],
            task_type="retrieval_document",
            title=data["title"],
        )
        for data in datas
    ]
    points = [
        PointStruct(
            id=idx,
            vector=response['embedding'],
            payload={"text": text},
        )
        for idx, (response, text) in enumerate(zip(results, datas))
    ]
    return client.upsert(collection_name, points)

@app.get("/rag/sentence/search")
def find_similar_sentences(query, top_k=10):
    results = client.search(
        collection_name=collection_recommend_name,
        query_vector=gemini_client.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query",
        )["embedding"],
    )
    print(results)
    return results  # Return list of sentence IDs


@app.get("/rag/db/search")
def find_similar_sentences(query, top_k=10):
    results = client.search(
        collection_name=collection_db_name,
        query_vector=gemini_client.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query",
        )["embedding"],
    )
    print(results)
    return results  # Return list of sentence IDs