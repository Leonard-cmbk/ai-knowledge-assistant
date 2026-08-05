from fastapi import FastAPI


app = FastAPI(title="AI Knowledge Assistant")


@app.get("/")
def read_root() -> dict:
    return {"message": "Hello from AI Knowledge Assistant"}
