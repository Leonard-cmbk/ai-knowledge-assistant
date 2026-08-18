from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from llm_client import LLMClient, LLMError, llm_client
from contextlib import asynccontextmanager
from pydantic import BaseModel
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


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/test_openai_api")
async def test_openai():
    try:
        result = await llm_client.chat(
            [
                {"role": "system", "content": "你是一个乐于助人的助手"},
                {"role": "user", "content": "你好,介绍一下你自己"},
            ]
        )
        return {"result": result}
    except LLMError as exc:
        #502 表示「上游(模型服务)出错」
        return JSONResponse(status_code=502, content={"detail": str(exc)}) 

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        try:
            messages = [m.model_dump() for m in req.messages]
            async for text in llm_client.stream_chat(messages):
                yield f"data: {text}\n\n" # EventSource 只认 data: 开头、空行结尾的事件帧
            yield "data: [DONE]\n\n"
        except LLMError as exc:
            yield f"data: [错误]{exc}\n\n"
    return StreamingResponse(gen(), media_type='text/event-stream') # 立刻返回响应,HTTP 头发出




# @app.get("/")
# async def root():
#     return {"message": "This is AI-Knowledge-Assistant!"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#     return {"item_id": item_id}

# @app.get("/students/{student_id}")
# def read_student(student_id: int):
#     return {"student_id": student_id}

# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.post("/echo")
# def echo(req: EchoRequest):
#     return {"result": req.message * req.times}

# @app.get("/slow")
# async def slow():
#     await asyncio.sleep(2)
#     return {"done": True}

# @app.get('/sync')
# def sync_call():
#     with httpx.Client() as client:
#         r = client.get("http://127.0.0.1:8000/get")
#         return {"result": r.json()} 

# @app.get('/async')
# async def async_call():
#     async with httpx.AsyncClient() as client:
#         await asyncio.sleep(10)
#         r = await client.get("http://127.0.0.1:8000/get")
#         return {"result": r.json()}

# async def openai_api():
#     client = OpenAI(
#         api_key=settings.deepseek_api_key,
#         base_url=settings.deepseek_url
#     )

#     response = client.chat.completions.create(
#         model=settings.deepseek_model,
#         messages=[
#             {"role": "system", "content": "你是一个乐于助人的助手"},
#             {"role": "user", "content": "你好，介绍一下你自己"},
#         ],
#         stream=False,
#         reasoning_effort="high",
#         extra_body={"thinking": {"type": "enabled"}}
#     )
#     return response.choices[0].message.content

