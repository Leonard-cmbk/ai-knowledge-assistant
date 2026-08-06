from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class EchoRequest(BaseModel):
    message: str
    times: int = 1

@app.get("/")
async def root():
    return {"message": "This is AI-Knowledge-Assistant!"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/students/{student_id}")
def read_student(student_id: int):
    return {"student_id": student_id}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/echo")
def echo(req: EchoRequest):
    return {"result": req.message * req.times}

@app.get("/slow")
async def slow():
    await asyncio.sleep(2)
    return {"done": True}