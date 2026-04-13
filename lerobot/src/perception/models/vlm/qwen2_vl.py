from transformers import AutoProcessor
from loguru import logger
from .base_vlm import OpenAICompat
from ..base_model import BaseModel


@BaseModel.register_model("qwen2_vl")
class Qwen2VL(OpenAICompat):
    """
    Implementation of Qwen 2 Vision Language model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str = "-", proxy: str = None, base_url: str = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning(
                "No base URL provided for Qwen2VL, assuming http://127.0.0.1:8000/v1"
            )
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "qwen2vl"