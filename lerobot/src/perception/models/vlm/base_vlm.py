import httpx
import os
import glob
from abc import abstractmethod


from openai import OpenAI


from perception.utils.image_convert import encode_image_to_base64
from perception.utils.video import encode_video_to_base64
from ..base_model import BaseModel


@BaseModel.register_model("base_vlm")
class BaseVLM(BaseModel):
    def __init__(
        self,
    ):
        self.short_name = "unknown"

    @abstractmethod
    def image_infer(self, system_prompt: str, user_prompt: str, image_paths: str | list[str]) -> str:
        """
        This method sends the user prompt and image files (if any) to the Vision-LLM API and
        returns the model's response.

        @param system_prompt: A system prompt that gives context to the model.
        @param user_prompt: The question or task prompt provided by the user.
        @param image_path: A path-like object representing the image file to be sent to the API.

        @return: The model's response to the user prompt. Always a string.
        """
        pass
    
    @abstractmethod
    def video_infer(
        self,
        system_prompt: str,
        user_prompt: str,
        video_paths: str | list[str],
    ) -> str:
        """
        This method sends the user prompt and image files (if any) to the Vision-LLM API and
        returns the model's response.

        @param system_prompt: A system prompt that gives context to the model.
        @param user_prompt: The question or task prompt provided by the user.
        @param image_path: A path-like object representing the video file to be sent to the API.

        @return: The model's response to the user prompt. Always a string.
        """
        pass


    @abstractmethod
    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        """
        This method sends a list of messages to the Vision-LLM API and returns the model's response. The definition of the actual payload may vary from model to model, with behaviors defined within the VLLM implementation.

        @param messages: The payload to be sent to the Vision-LLM API, typically a list of dictionaries if using the OpenAI-compatible interface.

        @return: The model's response to the messages. Always a string.
        """
        pass


class OpenAICompat(BaseVLM):
    def __init__(
        self,
        api_key: str | None = "mllm_defake_key",
        proxy: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the OpenAICompat class of Vision-LLM models. Provide the API key and optionally a proxy URL. Will populate self.api_key, self.http_client, and self.client.

        This class is meant to be used in conjunction with a self-hosted VLLM server instance. It also assumes that only one model is available on the server, otherwise the first model will be used, since VLLM may use folder names with trailing slashes as model names. However, this can be overridden by setting the model_name attribute after initializing this class (as is done in the GPT4o and GPT4oMini classes).

        @param api_key: The API key for the VLLM server.
        @param proxy: The proxy URL to use for the HTTP client, defaults to None.
        @param base_url: The base URL for the VLLM server. If not provided, the default value (http://127.0.0.1:8000/v1) will be used.

        @return: None
        """
        super().__init__()
        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(
            api_key=api_key, http_client=self.http_client, base_url=base_url
        )

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=120,
        )
        return response.choices[0].message.content


    def video_infer(
        self,
        system_prompt: str,
        user_prompt: str,
        video_paths: str | list[str],
    ) -> str:
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        # Create content list starting with the text prompt
        content = [{"type": "text", "text": user_prompt}]
        # Add each image to the content list
        for video_path in video_paths:
            if video_path.endswith(".mp4"):
                video_url = encode_video_to_base64(video_path)
            else:
                image_files = []
                for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    image_files.extend(sorted(glob.glob(os.path.join(video_path, pattern))))
                video_url = encode_video_to_base64(image_files)
            content.append({"type": "video_url", "video_url": {"url": video_url}})
        # Create the messages list with system and user content
        if system_prompt is None:
            messages = [{"role": "user", "content": content}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]

        response = self.infer_raw(messages=messages)
        return response


    def image_infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: str | list[str],
    ) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        # Create content list starting with the text prompt
        content = [{"type": "text", "text": user_prompt}]
        # Add each image to the content list
        for image_path in image_paths:
            image_url = encode_image_to_base64(image_path)
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        # Create the messages list with system and user content
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        response = self.infer_raw(messages=messages)
        return response

    def query_model_name(self) -> str:
        # Get the model name from the API
        response = self.client.models.list()
        return response.data[0].id
