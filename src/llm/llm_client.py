"""Minimal Ollama client using its local HTTP API and `requests`."""
import requests

import config


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Send a message to Ollama and return the response text."""
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": temperature},  # 0.0 is deterministic, ideal for code.
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def is_available() -> bool:
    """Return whether the Ollama server is responding."""
    try:
        return requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False
