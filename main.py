from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.core.config import settings
from app.services.model_manager import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    model_manager.load_models()
    yield
    # Clean up resources if needed
    pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
