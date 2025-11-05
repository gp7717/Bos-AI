"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config.settings import settings
from app.config.logging_config import setup_logging, get_logger
import uvicorn

# Setup logging first
setup_logging(log_level=settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Agentic Assistant API",
    description="Analytics Q&A system with agentic architecture",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["query"])


@app.on_event("startup")
async def startup():
    """Startup event handler."""
    logger.info("🚀 Starting Agentic Assistant API...")
    db_info = settings.database_url.split('@')[1] if '@' in settings.database_url else 'configured'
    logger.info(f"📊 Database: {db_info}")
    logger.info(f"🤖 Azure OpenAI Deployment: {settings.azure_openai_deployment_name}")
    logger.info(f"🌐 Azure OpenAI Endpoint: {settings.azure_openai_endpoint}")
    logger.info(f"⚙️ Log Level: {settings.log_level}")
    logger.info("✅ Application startup complete")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown event handler."""
    logger.info("🛑 Shutting down Agentic Assistant API...")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers
    )

