import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
    model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
)
