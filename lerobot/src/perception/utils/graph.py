import math

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image


PLOT_COLORs = {
    "border": "#E6770B",
    "tint_alpha": 0.0001,  # 左下角圆点右侧的线条的alpha（不能设为0）
    "affo": "#bc5090",
    "attr_neg": "#003F5C",
    "attr_pos": "#ffa600",
}

FILL_COLORs = {
    "border": PLOT_COLORs["border"] + "20",
    "affo": PLOT_COLORs["affo"],
    "attr_neg": PLOT_COLORs["attr_neg"],
    "attr_pos": PLOT_COLORs["attr_pos"],
}

CAUSAL_COLORs = {  # 右上角graph的颜色
    "positive": PLOT_COLORs["attr_pos"],
    "negative": PLOT_COLORs["attr_neg"],
    "default": PLOT_COLORs["affo"],
    "edge": "#D3290F",
}
EDGE_COLORs = {
    "positive": "#ff6361",
    "negative": "#58508d",
    "default": PLOT_COLORs["affo"],
    "edge": "#D3290F",
}


def prob2size(p):
    return 30 + 30 * p


def prob2color(is_positive):
    if is_positive:
        return CAUSAL_COLORs["positive"]
    else:
        return CAUSAL_COLORs["negative"]


def prob2edgecolor(is_positive):
    if is_positive:
        return EDGE_COLORs["positive"]
    else:
        return EDGE_COLORs["negative"]


def prob2sign(is_positive):
    if is_positive:
        return "+"
    else:
        return "-"


def prob2alpha(p, is_positive=False, activate_threshold=0.05):
    if p > activate_threshold:
        res = (p - activate_threshold) / (1 - activate_threshold)
        if res > 1:
            res = 1
        return res
    else:
        return 0

def draw_causal_graph(
    logic_chains: list[dict],
    attr_prob_dict: dict[str, float],
    aff_prob_dict: dict[str, float],
    affordance_name: str,
):

    related_edges = [
        edge for edge in logic_chains if edge["affordance"] == affordance_name
    ]
    pos_attribute_names = [
        edge["attribute"]
        for edge in related_edges
        if edge["is_positive_affect"]
    ]
    neg_attribute_names = [
        edge["attribute"]
        for edge in related_edges
        if not edge["is_positive_affect"]
    ]

    fig = go.Figure()
    ATTRIBUTE_UPPERBOUND = 8.6  # const variable
    ATTRIBUTE_LOWBOUND = 5
    AFFORDANCE_UPPERBOUND = 180
    AFFORDANCE_LOWBOUND = 20

    pos_attributes = {
        name: attr_prob_dict[name] for name in pos_attribute_names
    }
    neg_attributes = {
        name: attr_prob_dict[name] for name in neg_attribute_names
    }
    affordance_value = {affordance_name: aff_prob_dict[affordance_name]}

    pos_attribute_size = {
        name: max(
            ATTRIBUTE_LOWBOUND,
            min(math.log(value + 1) * 25, ATTRIBUTE_UPPERBOUND),
        )
        for name, value in pos_attributes.items()
    }
    neg_attribute_size = {
        name: max(
            ATTRIBUTE_LOWBOUND,
            min(math.log(value + 1) * 25, ATTRIBUTE_UPPERBOUND),
        )
        for name, value in neg_attributes.items()
    }
    size = max(
        AFFORDANCE_LOWBOUND,
        min(
            math.log(affordance_value[affordance_name] + 1) * 50,
            AFFORDANCE_UPPERBOUND,
        ),
    )

    avg_pos = (
        sum(pos_attributes.values()) / len(pos_attributes)
        if len(pos_attributes) > 0
        else 0
    )
    avg_neg = (
        sum(neg_attributes.values()) / len(neg_attributes)
        if len(neg_attributes) > 0
        else 0
    )

    y_aff = (
        0.9 * avg_pos
        - 0.9 * avg_neg
        + 1.1 * sum(affordance_value.values())
    ) * 5
    prob = max(0, min(100, (y_aff + 2) * 10))
    y_neg = -5
    x_mid_pos = 0

    y_pos = 5 + y_aff

    if y_aff > 0.2:
        y_neg = -5 + (y_aff - 0.2)

    # affordance气球
    fig.add_trace(
        go.Scatter(
            x=[x_mid_pos],
            y=[y_aff],
            mode="markers+text",
            marker=dict(size=24, color="#EFB6C8", symbol="square"),
            text=affordance_name + f" (able):{prob:.2f}%",
            textposition="middle right",
            textfont=dict(color="black", size=10),
            zorder=10,
        ),
    )

    # 加减号
    if pos_attribute_names:
        if len(pos_attribute_names) % 2 == 0:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos],
                    y=(
                        [(y_pos + y_aff + 1) / 2]
                        if (y_pos + y_aff + 1) / 2 < 8.5
                        else [8.5]
                    ),
                    mode="text",
                    text=["+"],
                    textposition="middle center",
                    textfont=dict(color="#FFB200", size=15, weight="bold"),
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos + 0.03],
                    y=(
                        [(y_pos + y_aff + 1) / 2]
                        if (y_pos + y_aff + 1) / 2 < 8.5
                        else [8.5]
                    ),
                    mode="text",
                    text=["+"],
                    textposition="middle right",
                    textfont=dict(color="#FFB200", size=15, weight="bold"),
                ),
            )
    if neg_attribute_names:
        if len(neg_attribute_names) % 2 == 0:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos],
                    y=(
                        [(y_neg + y_aff - 1) / 2]
                        if (y_neg + y_aff - 1) / 2 > -8.5
                        else [-8.5]
                    ),
                    mode="text",
                    text=["-"],
                    textposition="middle center",
                    textfont=dict(color="#EB5B00", size=15, weight="bold"),
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos + 0.04],
                    y=(
                        [(y_neg + y_aff - 1) / 2]
                        if (y_neg + y_aff - 1) / 2 > -8.5
                        else [-8.5]
                    ),
                    mode="text",
                    text=["-"],
                    textposition="middle right",
                    textfont=dict(color="#EB5B00", size=15, weight="bold"),
                ),
            )

    x_pos = -(len(pos_attributes) - 1) * 0.3 / 2
    balloon_image_path = "resources/balloon.png"
    balloon_image = Image.open(balloon_image_path)
    for attr in pos_attributes.keys():
        if y_pos + (pos_attribute_size[attr] - 5) * 0.54 > 9:
            y_pos_edited = 9 - (pos_attribute_size[attr] - 5) * 0.54
        else:
            y_pos_edited = y_pos
        fig.add_trace(
            go.Scatter(
                x=[x_pos, x_mid_pos],
                y=[y_pos_edited, y_aff],
                mode="lines",
                line=dict(color="#FFB200", width=2),
                zorder=2,
            ),
        )
        fig.add_layout_image(
            dict(
                source=balloon_image,
                xref="x",
                yref="y",
                x=x_pos,
                y=y_pos_edited + (pos_attribute_size[attr] - 5) * 0.27,
                sizex=pos_attribute_size[attr] * 1.1,
                sizey=pos_attribute_size[attr] * 1.1,
                xanchor="center",
                yanchor="middle",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_pos],
                y=[
                    y_pos_edited
                    + (pos_attribute_size[attr] - 5) * 0.54
                    + 0.75
                ],
                mode="text",
                text= "full" if attr == "empty" else "nearly full",
                textposition="top center",
                textfont=dict(color="black", size=10),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_pos, (x_pos + x_mid_pos) / 2, x_mid_pos],
                y=[y_pos_edited, (y_pos_edited + y_aff) / 2, y_aff],
                mode="markers",
                marker=dict(
                    symbol="arrow-down",
                    size=6,
                    color="#FFB200",
                    angleref="previous",
                ),
                zorder=2,
            ),
        )
        x_pos += 0.3

    # arrow lines and balloons of neg attr
    x_neg = -(len(neg_attributes) - 1) * 0.3 / 2
    weight_image_path = "resources/weight.png"
    weight_image = Image.open(weight_image_path)
    neg_line_width = (y_aff - y_neg) / 3
    for attr in neg_attributes.keys():
        fig.add_trace(
            go.Scatter(
                x=[x_neg, x_mid_pos],
                y=[y_neg, y_aff],
                mode="lines",
                line=dict(color="#EB5B00", width=neg_line_width),
                zorder=2,
            ),
        )
        fig.add_layout_image(
            dict(
                source=weight_image,
                xref="x",
                yref="y",
                x=x_neg,
                y=y_neg + (neg_attribute_size[attr] - 5) / 5.5,
                sizex=neg_attribute_size[attr] / 2.5,
                sizey=neg_attribute_size[attr] / 2.5,
                xanchor="center",
                yanchor="middle",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_neg],
                y=[
                    y_neg
                    + (neg_attribute_size[attr] - 5) / 5.5
                    - neg_attribute_size[attr] / 5
                ],
                mode="text",
                text="empty",
                textposition="bottom center",
                textfont=dict(color="black", size=10),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_neg, (x_neg + x_mid_pos) / 2, x_mid_pos],
                y=[y_neg, (y_neg + y_aff) / 2, y_aff],
                mode="markers",
                marker=dict(
                    symbol="arrow-down",
                    size=6,
                    color="#EB5B00",
                    angleref="previous",
                ),
                zorder=2,
            ),
        )
        x_neg += 0.3

    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-11.5, 11.5])
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=True),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    # Convert Plotly figure to image array with higher resolution
    img = fig.to_image(format="png", scale=3)
    img_array = np.frombuffer(img, np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Crop the image to half its width, centered
    height, width, _ = img_cv.shape
    start_x = width // 5
    start_y = height // 15
    end_y = height - int(height / 3.8)
    end_x = start_x + (width // 5) * 3
    cropped_img = img_cv[start_y:end_y, start_x:end_x]

    # Encode the cropped image back to bytes
    _, img_encoded = cv2.imencode(".png", cropped_img)
    return img_encoded.tobytes()