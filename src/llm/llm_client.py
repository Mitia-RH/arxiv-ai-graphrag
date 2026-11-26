"""Minimal Ollama client using its local HTTP API and `requests`."""
import re

import requests

import config


def strip_thinking(text: str) -> str:
    """Remove complete or unbalanced Ollama thinking markers and their content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Send a message to Ollama and return the response text."""
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},  # 0.0 is deterministic, ideal for code.
            "messages": [
                {"role": "system", "content": f"{system}\n/no_think"},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return strip_thinking(content)


def embed(text: str) -> list[float]:
    """Return an embedding from Ollama for the first 2000 characters of `text`."""
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": text[:2000]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def is_available() -> bool:
    """Return whether the Ollama server is responding."""
    try:
        return requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False
