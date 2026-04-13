import cv2
import base64
import imghdr
import numpy as np
from pathlib import Path

def encode_image_to_base64(image_path_or_array: str | Path | np.ndarray) -> str:
    """
    Encodes an image from the given file path to base64 format.

    Args:
        image_path: The path to the image file.

    Returns:
        A string in the format "data:image/{image_type};base64,{base64_encoded_image}"
    """
    if isinstance(image_path_or_array, np.ndarray):
        image_array = image_path_or_array
        _, image_data = cv2.imencode(".png", image_array)
        base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/png;base64,{base64_encoded_image}"
    else:
        image_path = Path(image_path_or_array)
        if not image_path.is_file():
            raise FileNotFoundError(f"File not found: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_type = imghdr.what(image_path)
            if image_type:
                base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
                return f"data:image/{image_type};base64,{base64_encoded_image}"
            else:
                raise ValueError("Unsupported image format")


def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decodes a base64 string to an image.

    Args:
        base64_string: The base64 string to decode.

    Returns:
        A NumPy array representing the image.
    """
    b64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image