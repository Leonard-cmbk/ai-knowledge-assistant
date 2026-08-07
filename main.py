from contextlib import asynccontextmanager
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import asyncio
import config
import httpx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    配置 延迟、连接池
    """
    app.state.client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        timeout=httpx.Timeout(10.0, connect=5.0, pool=10.0)
    )
    yield
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

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

@app.get('/sync')
def sync_call():
    with httpx.Client() as client:
        r = client.get("http://127.0.0.1:8000/get")
        return {"result": r.json()} 

@app.get('/async')
async def async_call():
    async with httpx.AsyncClient() as client:
        await asyncio.sleep(10)
        r = await client.get("http://127.0.0.1:8000/get")
        return {"result": r.json()}

async def openai_api():
    client = OpenAI(
        api_key=config.API_KEY,
        base_url=config.DEEPSEEK_URL
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的助手"},
            {"role": "user", "content": "你好，介绍一下你自己"},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    return response.choices[0].message.content

@app.get("/test_openai_api")
async def test_openai():
    r = await openai_api()
    return {"result": r}