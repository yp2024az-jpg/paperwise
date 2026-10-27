import os


def get_nvidia_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY") 
    if not api_key:
        raise RuntimeError(
            "Set NVIDIA_API_KEY in backend/.env to use NVIDIA models."
        )
    return api_key


def get_tavily_api_key() -> str:
    api_key = os.environ.get("tavily")
    if not api_key:
        raise RuntimeError("Set TAVILY_API_KEY in backend/.env to enable web search.")
    return api_key
