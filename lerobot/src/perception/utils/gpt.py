import base64
import re
from io import BytesIO

import cv2


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def call_openai_api(prompt_msg, client):
    params = {
        "model": "qwen-vl-max-latest",
        "messages": prompt_msg,
        "temperature": 0,
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


def get_object_list_with_mllm(video_path, client):
    # Use the first frame for encoding
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    frame_count = 0
    max_frames = 2  # Only process the first 2 frames

    while video.isOpened() and frame_count < max_frames:
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1

    video.release()
    print(len(base64Frames), "frames read.")
    prompt_messages_state = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a visual object detector. "
                        "Your task is to count and identify the objects in the provided image that are on the desk. "
                        "Focus on objects classified as grasped_objects and containers. "
                        "Do not include hand or gripper in your answer"
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64Frames[0]}"
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "There are two kinds of objects, grasped_objects and containers in the environment. We only care about objects on the desk."
                        " You must strictly follow the rules below: Even if there are multiple objects that appear identical, you must repeat their names in your answer according to their quantity. For example, if there are three wooden blocks, you must mention 'wooden block' three times in your answer."
                        " Be careful and accurate with the number. Do not miss or add additional object in your answer."
                        " Based on the input picture, answer:"
                        "1. How many objects are there in the environment?\n"
                        "2. What are these objects?\n"
                        "You should respond in the format of the following example:\n"
                        "Number: 3\n"
                        "Objects: red pepper, red tomato, white bowl\n"
                        "Number: 4\n"
                        "Objects: wooden block, wooden block, wooden block, wooden block\n"
                    ),
                },
            ],
        },
    ]

    # prompt_messages_state = [
    #     {
    #         "role": "system",
    #         "content": [
    #             "You are a visual object detector. Your task is to count and identify the objects in the provided image that are on the desk. Focus on objects classified as grasped_objects and containers.",
    #             "Do not include hand or gripper in your answer",
    #         ],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             "There are two kinds of objects, grasped_objects and containers in the environment. We only care about objects on the desk.",
    #             "You must strictly follow the rules below: Even if there are multiple objects that appear identical, you must repeat their names in your answer according to their quantity. For example, if there are three wooden blocks, you must mention 'wooden block' three times in your answer."
    #             "Be careful and accurate with the number. Do not miss or add additional object in your answer."
    #             "Based on the input picture, answer:",
    #             "1. How many objects are there in the environment?",
    #             "2. What are these objects?",
    #             "You should respond in the format of the following example:",
    #             "Number: 3",
    #             "Objects: red pepper, red tomato, white bowl",
    #             "Number: 4",
    #             "Objects: wooden block, wooden block, wooden block, wooden block",
    #             *map(lambda x: {"image": x, "resize": 768}, base64Frames[0:1]),
    #         ],
    #     },
    # ]

    response_state = call_openai_api(prompt_messages_state, client)
    return response_state


def extract_num_object(response_state):
    # Extract number of objects
    num_match = re.search(r"Number: (\d+)", response_state)
    num = int(num_match.group(1)) if num_match else 0

    # Extract objects
    objects_match = re.search(r"Objects: (.+)", response_state)
    objects_list = objects_match.group(1).split(", ") if objects_match else []

    # Construct object list
    objects = [obj for obj in objects_list]

    return num, objects
