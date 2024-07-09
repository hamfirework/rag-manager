from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/rag")
def read_root():
    return {"rag": "manager"}


@app.get("rag/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
