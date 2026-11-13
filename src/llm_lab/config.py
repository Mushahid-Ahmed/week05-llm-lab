import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
