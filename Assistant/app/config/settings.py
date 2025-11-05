"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql://tripti:Seleric789@selericdb.postgres.database.azure.com:5432/postgres"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # LLM - Azure OpenAI
    azure_openai_api_key: str  # Azure API key
    azure_openai_endpoint: str  # Azure OpenAI endpoint
    azure_openai_deployment_name: str = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    azure_openai_api_version: str = "2024-02-15-preview"
    openai_temperature: float = 0.1
    
    # Legacy OpenAI (for backward compatibility, optional)
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 120
    
    # Vector Store
    vector_db_url: Optional[str] = None
    vector_dimension: int = 1536
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False
    
    # Security
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Observability
    jaeger_endpoint: Optional[str] = None
    log_level: str = "INFO"
    enable_tracing: bool = True
    
    # Amazon Ads API
    amazon_ads_client_id: Optional[str] = None
    amazon_ads_client_secret: Optional[str] = None
    amazon_ads_refresh_token: Optional[str] = None
    
    # Timezone
    default_timezone: str = "Asia/Kolkata"
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

