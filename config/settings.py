import os
from dataclasses import dataclass, field

@dataclass
class Settings:
    # Server Settings
    PORT: int = int(os.environ.get("PORT", 8000))
    HOST: str = "0.0.0.0"
    
    # vLLM Settings
    VLLM_MODEL: str = os.environ.get("VLLM_MODEL", "open-scorer-120b")
    VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8080/v1")
    VLLM_API_KEY: str = os.environ.get("VLLM_API_KEY", "sk-local")
    
    # Agent Settings
    MAX_TURNS: int = 50
    CONTEXT_TOKENS: int = 32_000
    TEMPERATURE: float = 1.0
    
    # Paths
    WORKING_DIR: str = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
    PROJECT_DIR: str = os.path.join(WORKING_DIR, "KaggleClaw")
    RUN_DIR: str = os.path.join(PROJECT_DIR, "run")
    INPUT_DIR: str = "/kaggle/input/"
    
settings = Settings()
