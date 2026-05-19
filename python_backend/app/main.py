from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_ai_db, init_vector_db
from app.routers import ai, app_info, contacts, messages, psych, python_config, training
from app.utils.logging import setup_logging


setup_logging()

app = FastAPI(
    title="MindTrace Python Backend",
    description="Local Python backend for WeChat chat analysis and psychological risk screening assistance.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3418",
        "http://127.0.0.1:3418",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_ai_db()
    init_vector_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(app_info.router, prefix="/api")
app.include_router(contacts.router, prefix="/api")
app.include_router(messages.router, prefix="/api")
app.include_router(psych.router, prefix="/api")
app.include_router(ai.router, prefix="/api")
app.include_router(python_config.router, prefix="/api")
app.include_router(training.router, prefix="/api")
