import base64
import io
import json
import logging
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


class NumpyEncoderFloat32(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pil_image_to_base64(image, format="JEPG"):
    """
    Convert a PIL.Image to a Base64 string.

    Args:
        image (PIL.Image): The PIL image object to convert.
        format (str): The image format (e.g., "PNG", "JPEG"). Default is "PNG".

    Returns:
        str: Base64-encoded string of the image.
    """
    # Create a byte-stream
    buffered = io.BytesIO()

    # Save the image to the byte-stream
    image.save(buffered, format=format)

    # Encode the byte-stream to Base64
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_string


def base64_to_pil_image(base64_string):
    """
    Convert a Base64 string to a PIL.Image.

    Args:
        base64_string (str): The Base64 string to convert.

    Returns:
        PIL.Image: The decoded image.
    """
    # Decode the Base64 string to a byte-stream
    decoded = base64.b64decode(base64_string)

    # Convert the byte-stream to a PIL image
    image = Image.open(io.BytesIO(decoded))

    return image


def mask_to_bounding_box(mask):
    """
    convert mask to bounding box。

    Args:
    - mask: [0, 255]

    Returns:
    - bounding_box: (x_min, y_min, x_max, y_max)
    """
    # 找到所有的白色像素的 x, y 坐标
    ys, xs = np.where(mask > 0)

    # 如果没有找到任何物体像素，返回 None
    if len(xs) == 0 or len(ys) == 0:
        logger.warning(
            "No object pixel found in the mask, so we use the attribute/affordance probability value of the previous image"
        )
        # A signal, use the attribute/affordance probability value of the previous image
        return (-1, -1, -1, -1)

    # 计算最小和最大 x, y 坐标来定义 bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return (x_min, y_min, x_max, y_max)


def encode_image_to_base64(img, target_size=-1, fmt="JPEG"):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode("utf-8")
    return ret


def encode_image_file_to_base64(image_path, target_size=-1):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)
