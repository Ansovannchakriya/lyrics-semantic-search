from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.search_engine import SemanticSearchEngine, SearchEngineError

APP_DIR = Path(__file__).parent
MAX_TOP_K = 50

app = FastAPI(title="Spotify Semantic Search")

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

engine: SemanticSearchEngine | None = None
engine_load_error: str | None = None
try:
    engine = SemanticSearchEngine()
except SearchEngineError as e:
    engine_load_error = str(e)
    print(f"WARNING: search engine did not load: {e}")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "engine_ready": engine is not None,
            "engine_error": engine_load_error,
            "num_songs": engine.index.ntotal if engine else 0,
            "embedding_dim": engine.dimension if engine else 0,
        },
    )


@app.post("/search")
def search(payload: SearchRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Search index is not available: {engine_load_error}",
        )

    query = payload.query.strip() if payload.query else ""
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    top_k = payload.top_k
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    if top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be at most {MAX_TOP_K}.",
        )

    try:
        results = engine.search(query, top_k=top_k)
    except SearchEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover -- unexpected engine failure
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return {"query": query, "results": results}
