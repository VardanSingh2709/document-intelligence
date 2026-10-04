"""Centralized access to environment variables (API keys, etc.), loaded once
from .env. Never hardcode secrets elsewhere in the codebase — import from here."""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Create a .env file in the project root with: "
        "GROQ_API_KEY=your_key_here (see .env.example)"
    )