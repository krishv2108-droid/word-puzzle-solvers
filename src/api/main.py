"""FastAPI app: JSON API under /api, static web UI at /.

    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .wordle_service import InputError, WordleService

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class GuessEntry(BaseModel):
    guess: str = Field(min_length=5, max_length=5, examples=["soare"])
    feedback: str = Field(min_length=5, max_length=5, description="'.' gray, 'y' yellow, 'g' green",
                          examples=["..y.g"])


class SuggestRequest(BaseModel):
    history: list[GuessEntry] = Field(default_factory=list, max_length=12)
    hard_mode: bool = False
    solver: str = "entropy"
    top_k: int = Field(default=10, ge=1, le=25)


class SimulateRequest(BaseModel):
    answer: str | None = Field(default=None, description="omit for a random answer")
    solver: str = "entropy"
    hard_mode: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.wordle = WordleService()
    yield


app = FastAPI(title="NYT Games Solver", version="1.0", lifespan=lifespan)


@app.exception_handler(InputError)
async def input_error_handler(request, exc: InputError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/wordle/info")
def wordle_info() -> dict:
    svc: WordleService = app.state.wordle
    return {"answers": len(svc.data.answers), "allowed_guesses": len(svc.data.guesses),
            "solvers": sorted(svc.players), "rankers": sorted(svc.rankers)}


@app.post("/api/wordle/suggest")
def wordle_suggest(req: SuggestRequest) -> dict:
    svc: WordleService = app.state.wordle
    return svc.suggest([(h.guess, h.feedback) for h in req.history], req.hard_mode, req.solver, req.top_k)


@app.post("/api/wordle/simulate")
def wordle_simulate(req: SimulateRequest) -> dict:
    svc: WordleService = app.state.wordle
    return svc.simulate(req.answer, req.solver, req.hard_mode)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
