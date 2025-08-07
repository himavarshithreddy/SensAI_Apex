import os
from os.path import join
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from functools import lru_cache
from api.config import UPLOAD_FOLDER_NAME
from phoenix.otel import register

root_dir = os.path.dirname(os.path.abspath(__file__))
env_path = join(root_dir, ".env.aws")
if os.path.exists(env_path):
    load_dotenv(env_path)


class Settings(BaseSettings):
    google_client_id: str
    openai_api_key: str
    google_gemini_api_key: str | None = None  # For Google Gemini AI chat
    google_cloud_project_id: str | None = None  # For Google Speech-to-Text
    google_application_credentials: str | None = None  # Path to Google service account key file
    s3_bucket_name: str | None = None  # only relevant when running the code remotely
    s3_folder_name: str | None = None  # only relevant when running the code remotely
    local_upload_folder: str = (
        UPLOAD_FOLDER_NAME  # hardcoded variable for local file storage
    )
    bugsnag_api_key: str | None = None
    env: str | None = None
    slack_user_signup_webhook_url: str | None = None
    slack_course_created_webhook_url: str | None = None
    slack_usage_stats_webhook_url: str | None = None
    phoenix_endpoint: str | None = None
    phoenix_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=join(root_dir, ".env"))


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()

if settings.phoenix_api_key is not None:
    os.environ["PHOENIX_API_KEY"] = settings.phoenix_api_key

# Only register Phoenix tracer if endpoint is configured
if settings.phoenix_endpoint:
    tracer_provider = register(
        protocol="http/protobuf",
        project_name=f"sensai-{settings.env}",
        auto_instrument=True,
        batch=True,
        endpoint=f"{settings.phoenix_endpoint}/v1/traces",
    )
    tracer = tracer_provider.get_tracer(__name__)
else:
    # Create a custom tracer that handles openinference_span_kind parameter
    from opentelemetry.trace import NoOpTracerProvider, Span, Tracer
    from contextlib import asynccontextmanager
    from typing import Any, Dict
    
    class DummySpan:
        def __init__(self, **kwargs):
            pass
        
        def set_input(self, *args, **kwargs):
            pass
        
        def set_output(self, *args, **kwargs):
            pass
        
        def record_exception(self, *args, **kwargs):
            pass
        
        def set_status(self, *args, **kwargs):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args, **kwargs):
            pass
    
    class DummyTracer(Tracer):
        def start_as_current_span(self, name: str, **kwargs) -> DummySpan:
            return DummySpan(**kwargs)
        
        def start_span(self, name: str, **kwargs) -> DummySpan:
            return DummySpan(**kwargs)
    
    tracer = DummyTracer()
