import json
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import torch
from PIL import Image

from .result import DetectionResult


def annotate(
    image: Union[Image.Image, np.ndarray],
    detection_results: List[DetectionResult],
    attrs: Optional[List[List[str]]] = None,
    affs: Optional[List[List[str]]] = None,
) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
    if attrs is None:
        attrs = [None] * len(detection_results)
    if affs is None:
        affs = [None] * len(detection_results)
    # Iterate over detections and add bounding boxes and masks
    for detection, attr, aff in zip(detection_results, attrs, affs):
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask
        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(
            image_cv2,
            (box.xmin, box.ymin),
            (box.xmax, box.ymax),
            color.tolist(),
            2,
        )
        cv2.putText(
            image_cv2,
            f"{label}: {score:.2f}",
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color.tolist(),
            2,
        )
        cv2.putText(
            image_cv2,
            f"attr: {attr}",
            (box.xmin, box.ymin - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color.tolist(),
            2,
        )
        cv2.putText(
            image_cv2,
            f"aff: {aff}",
            (box.xmin, box.ymin - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color.tolist(),
            2,
        )

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    attrs: Optional[List[List[str]]] = None,
    affs: Optional[List[List[str]]] = None,
    save_name: Optional[str] = None,
) -> None:
    annotated_image = annotate(image, detections, attrs, affs)
    plt.imshow(annotated_image)
    plt.axis("off")
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert(
            "RGB"
        )
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def load_video(video_str: str) -> List[Image.Image]:
    if video_str.startswith("http"):
        cap = cv2.VideoCapture(video_str)
    else:
        cap = cv2.VideoCapture(video_str)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    return frames


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]

    # Sample random named CSS colors
    return random.sample(
        named_css_colors, min(num_colors, len(named_css_colors))
    )


# def plot_detections_slider_plotly(
#     image: np.ndarray,
#     detections: List[DetectionResult],
#     attrs: Optional[List[List[str]]] = None,
#     affs: Optional[List[List[str]]] = None,
#     class_colors: Optional[Dict[str, str]] = None,
# ) -> None:
#     # If class_colors is not provided, generate random colors for each class
#     if class_colors is None:
#         num_detections = len(detections)
#         colors = random_named_css_colors(num_detections)
#         class_colors = {}
#         for i in range(num_detections):
#             class_colors[i] = colors[i]

#     fig = px.imshow(image)
#     if attrs is None:
#         attrs = [None] * len(detections)
#     if affs is None:
#         affs = [None] * len(detections)

#     frames = []
#     for idx, (detection, attr, aff) in enumerate(zip(detections, attrs, affs)):
#         label = detection.label
#         box = detection.box
#         score = detection.score
#         mask = detection.mask

#         polygon = mask_to_polygon(mask)

#         frame = go.Frame(
#             data=[
#                 go.Scatter(
#                     x=[point[0] for point in polygon] + [polygon[0][0]],
#                     y=[point[1] for point in polygon] + [polygon[0][1]],
#                     mode="lines",
#                     line=dict(color=class_colors[idx], width=2),
#                     fill="toself",
#                     name=f"{label}: {score:.2f}<br>attr: {attr}<br>aff: {aff}",
#                 ),
#                 go.Scatter(
#                     x=[(box.xyxy[0] + box.xyxy[2]) / 2],
#                     y=[(box.xyxy[1] + box.xyxy[3]) / 2],
#                     text=[f"{label}: {score:.2f}"],
#                     mode="text",
#                     showlegend=False
#                 )
#             ],
#             name=f"frame{idx}"
#         )
#         frames.append(frame)

#     fig.frames = frames

#     sliders = [
#         {
#             "steps": [
#                 {
#                     "args": [
#                         [f"frame{idx}"],
#                         {
#                             "frame": {"duration": 300, "redraw": True},
#                             "mode": "immediate",
#                             "transition": {"duration": 300},
#                         },
#                     ],
#                     "label": f"{idx+1}",
#                     "method": "animate",
#                 }
#                 for idx in range(len(frames))
#             ],
#             "transition": {"duration": 300},
#             "x": 0.1,
#             "xanchor": "left",
#             "y": 0,
#             "yanchor": "top",
#         }
#     ]

#     fig.update_layout(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         margin=dict(l=0, r=0, t=0, b=0),
#         showlegend=True,
#         width=1700,
#         height=1400,
#         updatemenus=[
#             dict(
#                 type="buttons",
#                 showactive=False,
#                 buttons=[
#                     dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
#                     dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
#                 ]
#             )
#         ],
#         sliders=sliders,
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     )

#     # Show plot
#     fig.show()


def plot_image_with_detection(
    image: np.ndarray,
    idx: int,
    color_id :int,
    detections: List[DetectionResult],
    plot_lines: Optional[List[Dict]] = None,
) -> go.Figure:
    """
    for `multi_thread_plt_attr_aff.py` file
    需要颜色对应，把 `plot_lines` 即对应的 toml 文件的颜色设置为与 aff 颜色一致即可
    """
    if plot_lines is None:
        plot_lines = []
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        plot_line = {}
        for i in range(num_detections):
            plot_line["index"] = i
            plot_line["name"] = f"Object {i}"
            plot_line["color"] = colors[i]
            plot_lines.append(plot_line)

    fig = px.imshow(image)
    detection = detections[idx]
    detection.label
    box = detection.box
    detection.score
    mask = detection.mask
    polygon = mask_to_polygon(np.array(mask))

    fig.add_trace(
        go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode="lines",
            line=dict(color=plot_lines[color_id]["color"], width=2),
            fill="toself",
            name=plot_lines[color_id]["name"],
        )
    )

    xmin = box.xmin
    ymin = box.ymin
    xmax = box.xmax
    ymax = box.ymax
    shape = [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=xmin,
            y0=ymin,
            x1=xmax,
            y1=ymax,
            line=dict(color=plot_lines[color_id]["color"]),
        )
    ]

    # shapes.append(shape)
    # annotations.append(annotation)

    # # Update layout
    # button_shapes = [dict(label="None", method="relayout", args=["shapes", []])]
    # button_shapes = button_shapes + [
    #     dict(label=f"Detection {idx+1}", method="relayout", args=["shapes", shape])
    #     for idx, shape in enumerate(shapes)
    # ]
    # button_shapes = button_shapes + [
    #     dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])
    # ]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        width=640,
        height=480,
        shapes=shape,
        # updatemenus=[dict(type="buttons", direction="up", buttons=button_shapes)],
        legend=dict(
            orientation="h", yanchor="bottom", y=0.95, xanchor="left", x=0.05
        ),
    )

    return fig


def plot_detections_wo_atf_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    plot_lines: Optional[List[Dict]] = None,
) -> go.Figure:

    if plot_lines is None:
        plot_lines = []
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        plot_line = {}
        for i in range(num_detections):
            plot_line["index"] = i
            plot_line["name"] = f"Object {i}"
            plot_line["color"] = colors[i]
            plot_lines.append(plot_line)

    fig = px.imshow(image)
    shapes = []
    for idx, detection_json in enumerate(detections):
        detection = json.loads(detection_json)
        detection["label"]
        box = detection["box"]
        detection["score"]
        mask = detection["mask"]
        polygon = mask_to_polygon(np.array(mask))

        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in polygon] + [polygon[0][0]],
                y=[point[1] for point in polygon] + [polygon[0][1]],
                mode="lines",
                line=dict(color=plot_lines[idx]["color"], width=2),
                fill="toself",
                name=plot_lines[idx]["name"],
            )
        )

        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        shape = [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=xmin,
                y0=ymin,
                x1=xmax,
                y1=ymax,
                line=dict(color=plot_lines[idx]["color"]),
            )
        ]

        shapes.append(shape)
        # annotations.append(annotation)

    # Update layout
    button_shapes = [
        dict(label="None", method="relayout", args=["shapes", []])
    ]
    button_shapes = button_shapes + [
        dict(
            label=f"Detection {idx+1}",
            method="relayout",
            args=["shapes", shape],
        )
        for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])
    ]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        width=640,
        height=480,
        updatemenus=[
            dict(type="buttons", direction="up", buttons=button_shapes)
        ],
        legend=dict(
            orientation="h", yanchor="bottom", y=0.95, xanchor="left", x=0.05
        ),
    )

    return fig


def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    attrs: Optional[List[List[str]]] = None,
    affs: Optional[List[List[str]]] = None,
    class_colors: Optional[Dict[str, str]] = None,
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]

    fig = px.imshow(image)
    if attrs is None:
        attrs = [None] * len(detections)
    if affs is None:
        affs = [None] * len(detections)
    # Add bounding boxes
    shapes = []
    for idx, (detection, attr, aff) in enumerate(zip(detections, attrs, affs)):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in polygon] + [polygon[0][0]],
                y=[point[1] for point in polygon] + [polygon[0][1]],
                mode="lines",
                line=dict(color=class_colors[idx], width=2),
                fill="toself",
                name=f"{label}: {score:.2f}<br>attr: {attr}<br>aff: {aff}",
            )
        )

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=xmin,
                y0=ymin,
                x1=xmax,
                y1=ymax,
                line=dict(color=class_colors[idx]),
            )
        ]
        # annotation = [
        #     dict(
        #         x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
        #         xref="x", yref="y",
        #         text=f"{label}: {score:.2f}",
        #     )
        # ]

        shapes.append(shape)
        # annotations.append(annotation)

    # Update layout
    button_shapes = [
        dict(label="None", method="relayout", args=["shapes", []])
    ]
    button_shapes = button_shapes + [
        dict(
            label=f"Detection {idx+1}",
            method="relayout",
            args=["shapes", shape],
        )
        for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])
    ]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        width=1700,
        height=1400,
        updatemenus=[
            dict(type="buttons", direction="up", buttons=button_shapes)
        ],
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    # Show plot
    fig.show()


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
        )
    )


def show_text(label, box, ax):
    ax.text(
        box[2],
        box[1] - 2,
        label,
        va="center",
        ha="center",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7),
    )
