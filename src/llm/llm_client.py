"""Appel minimal à Ollama (API HTTP locale). Pas de framework, juste `requests`."""
import requests

import config


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Envoie un message à Ollama et retourne la réponse texte."""
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": temperature},  # 0.0 = déterministe, idéal pour du code
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
    """True si le serveur Ollama répond."""
    try:
        return requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False
