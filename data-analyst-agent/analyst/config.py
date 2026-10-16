from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  groq_api_key: str = ""
  model_name: str = "openai/gpt-oss-20b"
  memory_db: str = "./data/memory.sqlite"
  max_agent_steps: int = 8
  use_embeddings: bool = False  # optional: semantic memory recall (needs sentence-transformers)

  model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
