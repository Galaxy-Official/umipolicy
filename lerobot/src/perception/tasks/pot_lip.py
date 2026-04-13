import logging
import math
import os

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .base_task import BaseTask, TaskStatus

logger = logging.getLogger(__name__)


@BaseTask.register_task("pot_lip")
class PotLip(BaseTask):
    def decision_making(self, video_type="camera") -> TaskStatus:
        try:
            # Get image from streaming input
            if video_type == "camera":
                # TODO: add support for real time camera input
                raise NotImplementedError(
                    "Real-time camera input not supported."
                )
                img = self.camera_input()
            else:
                pil_img_dict = self.video_input()

            attr_prob_dict = {}
            aff_prob_dict = {}
            mask_dict = {}
            confunding_mask_dict = {}
            # Perform object detection
            for key, pil_img in pil_img_dict.items():
                if self.frame_index == 0:
                    if (
                        self.config["detector"]["threshold"].get(key)
                        is not None
                    ):
                        threshold = self.config["detector"]["threshold"][key]
                    else:
                        threshold = self.config["detector"]["threshold"]
                    detections = self.object_detector.object_detection(
                        image=pil_img,
                        labels=self.config["detector"]["labels"],
                        threshold=threshold,
                    )

                    if not detections:
                        logger.error("No objects detected.")
                        return TaskStatus.STOP
                else:
                    detections = None
                obj_id_len, video_segment = self.sam2_predictor_dict[
                    key
                ].streaming_sam_and_tracking(
                    image=pil_img,
                    bbox_data=detections,
                    output_dir=self.output_path_dict[key],
                    frame_idx=self.frame_index,
                )

                mask_dict[key] = video_segment[self.target_id_dict[key]]
                confunding_mask_dict[key] = video_segment[
                    self.confunding_target_id_dict[key]
                ]
                attr_probs, _, aff_probs_w = (
                    self.ocl_pipeline.streaming_calculate_text_probabilities(
                        frame=pil_img,
                        video_segment=video_segment,
                        obj_id_len=obj_id_len,
                        key_attr=self.key_attr,
                        key_aff=self.key_aff,
                        logic_chains=self.logic_chains,
                    )
                )
                self.attribute_buffer_dict[key].append(attr_probs)
                self.affordance_buffer_dict[key].append(aff_probs_w)
                logger.debug(self.affordance_buffer_dict)
                # Stores the length of the neighborhood `buffer_length`, attr / aff value for robust decision making.
                attr_probs = self.caculate_average_probs(
                    list(self.attribute_buffer_dict[key])
                )
                attr_prob_dict[key] = attr_probs
                aff_probs = self.caculate_average_probs(
                    list(self.affordance_buffer_dict[key])
                )
                aff_prob_dict[key] = aff_probs
                logger.debug(f"Attr buffer: {self.attribute_buffer_dict}")
                logger.debug(f"Aff buffer: {self.affordance_buffer_dict}")
            (
                attr_bool_dict,
                aff_bool_dict,
                confunding_attr_dict,
                confunding_aff_dict,
            ) = self.process_attributes_affordances(
                attr_prob_dict, aff_prob_dict
            )
            logger.info(f"{self.frame_index}: {attr_bool_dict}")
            logger.info(f"{self.frame_index}: {aff_bool_dict}")
            if self.show_image:
                self.visualize_images_and_affordances(
                    pil_img_dict,
                    attr_bool_dict,
                    aff_bool_dict,
                    mask_dict,
                    confunding_attr_dict,
                    confunding_aff_dict,
                    confunding_mask_dict,
                )
            return TaskStatus.CONTINUE
        except StopIteration:
            logger.info("没有更多的图像。")
            return TaskStatus.STOP

    def process_attributes_affordances(self, attr_prob_dict, aff_prob_dict):
        """
        Combine attributes and affordances from multiple perspectives.
        """
        aggregated_attrs = {key: [] for key in self.key_attr}
        aggregated_affs = {key: [] for key in self.key_aff}
        # Simple average example
        for key, obj_ttr_prob in attr_prob_dict.items():
            for attr, prob in obj_ttr_prob[self.target_id_dict[key]].items():
                aggregated_attrs[attr].append(prob)
        for key, obj_aff_prob in aff_prob_dict.items():
            for aff, prob in obj_aff_prob[self.target_id_dict[key]].items():
                aggregated_affs[aff].append(prob)
        attr_bool_dict = {}
        aff_bool_dict = {}
        for key, prob_list in aggregated_attrs.items():
            attr_bool_dict[key] = self.max_pool(prob_list)
        for key, prob_list in aggregated_affs.items():
            aff_bool_dict[key] = self.max_pool(prob_list)
        # Handle confounding attributes and affordances
        confounding_attrs = {key: [] for key in self.key_attr}
        confounding_affs = {key: [] for key in self.key_aff}

        for key, obj_attr_prob in attr_prob_dict.items():
            for attr, prob in obj_attr_prob[
                self.confunding_target_id_dict[key]
            ].items():
                confounding_attrs[attr].append(prob)

        for key, obj_aff_prob in aff_prob_dict.items():
            for aff, prob in obj_aff_prob[
                self.confunding_target_id_dict[key]
            ].items():
                confounding_affs[aff].append(prob)

        confounding_attr_bool_dict = {}
        confounding_aff_bool_dict = {}

        for key, prob_list in confounding_attrs.items():
            confounding_attr_bool_dict[key] = self.max_pool(prob_list)

        for key, prob_list in confounding_affs.items():
            confounding_aff_bool_dict[key] = self.max_pool(prob_list)
        return (
            attr_bool_dict,
            aff_bool_dict,
            confounding_attr_bool_dict,
            confounding_aff_bool_dict,
        )

    def run(self, video_type="camera"):
        assert video_type in [
            "camera",
            "video",
        ], f"Invalid image input type {video_type}."
        while True:
            status = self.decision_making(video_type)
            if status == TaskStatus.STOP:
                logger.info("Task stopped.")
                break
            elif status == TaskStatus.CONTINUE:
                logger.info("Task continues.")
            else:
                logger.error("Unknown status.")
                break

    def visualize_images_and_affordances(
        self,
        pil_img_dict,
        attr_prob_dict=None,
        aff_prob_dict=None,
        mask_dict=None,
        confunding_attr_prob_dict=None,
        confunding_aff_prob_dict=None,
        confunding_mask_dict=None,
        causal_graph=False,
    ):
        # Convert and prepare all images
        show_images = []
        if all(
            x is None
            for x in [
                attr_prob_dict,
                aff_prob_dict,
                mask_dict,
                confunding_attr_prob_dict,
                confunding_aff_prob_dict,
                confunding_mask_dict,
            ]
        ):
            # Calculate dimensions
            img_height = max(img.shape[0] for img in show_images)
            img_width = max(img.shape[1] for img in show_images)

            # Create 1x3 canvas with space for text
            canvas = (
                np.ones((img_height + 50, 3 * img_width, 3), dtype=np.uint8)
                * 255
            )

            # Add images and labels
            view_names = [f"{key} view" for key in pil_img_dict.keys()]
            for i, (img, text) in enumerate(zip(show_images, view_names)):
                # Place image
                x_start = i * img_width
                canvas[:img_height, x_start : x_start + img.shape[1]] = img

                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[
                    0
                ]

                text_x = x_start + (img_width - text_size[0]) // 2
                text_y = img_height + 30

                cv2.putText(
                    canvas,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness,
                )

            # Save canvas
            filename = f'frame_{self.frame_index // self.config["sam2"]["frame_interval"]:04d}.png'
            save_path = os.path.join(self.demo_subdir, filename)
            cv2.imwrite(save_path, canvas)
            return TaskStatus.CONTINUE

        # ...existing code for full visualization...
        for key, pil_img in pil_img_dict.items():
            cv_image = np.array(pil_img)
            cv_img = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            # Create colored mask overlay
            alpha = 0.3
            if key in mask_dict:
                mask = np.squeeze(
                    mask_dict[key]
                )  # Remove extra dimension (1,480,640) -> (480,640)
                colored_mask = np.zeros_like(cv_img)
                colored_mask[mask] = [0, 0, 255]  # Green mask
                cv_img = cv2.addWeighted(cv_img, 1, colored_mask, alpha, 0)
            if key in confunding_mask_dict:
                mask = np.squeeze(confunding_mask_dict[key])
                colored_mask = np.zeros_like(cv_img)
                colored_mask[mask] = [0, 255, 0]
                cv_img = cv2.addWeighted(cv_img, 1, colored_mask, alpha, 0)
            show_images.append(cv_img)

        # Calculate grid dimensions
        img_height = max(img.shape[0] for img in show_images)
        img_width = max(img.shape[1] for img in show_images)

        # Create 2x2 canvas with extra height for text
        canvas = (
            np.ones((2 * img_height + 50, 2 * img_width, 3), dtype=np.uint8)
            * 255
        )
        view_names = [f"{key} view" for key in pil_img_dict.keys()]
        # Place images in grid
        positions = [
            (0, 0),
            (0, 1),
            (1, 0),
        ]  # Top-left, top-right, bottom-left
        for i, img in enumerate(show_images[:3]):  # Limit to first 3 images
            row, col = positions[i]
            y_start = row * img_height
            x_start = col * img_width
            y_end = y_start + img.shape[0]
            x_end = x_start + img.shape[1]
            canvas[y_start:y_end, x_start:x_end] = img
            # Add text under each image
            text = view_names[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Calculate text position (centered under image)
            text_x = x_start + (img_width - text_size[0]) // 2
            text_y = y_end - 8  # 30 pixels below the image

            cv2.putText(
                canvas,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
            )

        if causal_graph is not None:
            # Draw original causal graph
            graph_img_bytes = self.draw_causal_graph(
                attr_prob_dict,
                aff_prob_dict,
            )  # Draw causal graph
            graph_img_array = np.frombuffer(graph_img_bytes, np.uint8)
            graph_img = cv2.imdecode(graph_img_array, cv2.IMREAD_COLOR)
            # Draw confounding causal graph
            conf_graph_img_bytes = self.draw_causal_graph(
                confunding_attr_prob_dict,
                confunding_aff_prob_dict,
            )
            conf_graph_img_array = np.frombuffer(
                conf_graph_img_bytes, np.uint8
            )
            conf_graph_img = cv2.imdecode(
                conf_graph_img_array, cv2.IMREAD_COLOR
            )

            # Resize both graphs to fit half of the bottom-right quadrant
            half_width = img_width // 2
            graph_img_resized = cv2.resize(
                graph_img,
                (half_width, img_height),
                interpolation=cv2.INTER_CUBIC,
            )
            conf_graph_img_resized = cv2.resize(
                conf_graph_img,
                (half_width, img_height),
                interpolation=cv2.INTER_CUBIC,
            )

            # Place graphs side by side in bottom-right quadrant
            canvas[
                img_height : 2 * img_height, img_width : img_width + half_width
            ] = graph_img_resized
            canvas[
                img_height : 2 * img_height,
                img_width + half_width : 2 * img_width,
            ] = conf_graph_img_resized

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                canvas,
                "Object A",
                (img_width + 10, img_height + 30),
                font,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Object B",
                (img_width + half_width + 10, img_height + 30),
                font,
                0.7,
                (0, 255, 0),
                2,
            )

            # graph_img_resized = cv2.resize(
            #     graph_img, (img_width, img_height)
            # )  # Resize to fit the shape
            # canvas[img_height : 2 * img_height, img_width : 2 * img_width] = (
            #     graph_img_resized
            # )

        # Add affordance text

        text_a = f"object A: {aff_prob_dict['eat']:.2f}"
        text_b = f"object B: {confunding_aff_prob_dict['eat']:.2f}"
        text_end = (
            "A"
            if aff_prob_dict["eat"] > confunding_aff_prob_dict["eat"]
            else "B"
        )
        color_end = (
            (0, 0, 255)
            if aff_prob_dict["eat"] > confunding_aff_prob_dict["eat"]
            else (0, 255, 0)
        )
        # Starting position
        x_pos = 10  # Adjust as needed
        y_pos = 2 * img_height + 30  # Adjust as needed

        # Put text with different colors
        cv2.putText(
            canvas,
            text_a,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )  # Red
        x_pos += cv2.getTextSize(text_a, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][
            0
        ]

        cv2.putText(
            canvas,
            ", ",
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        x_pos += cv2.getTextSize(", ", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]

        cv2.putText(
            canvas,
            text_b,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )  # Green
        x_pos += cv2.getTextSize(text_b, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][
            0
        ]
        cv2.putText(
            canvas,
            ", Choose ",
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        x_pos += cv2.getTextSize(
            ", Choose ", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )[0][0]
        cv2.putText(
            canvas,
            text_end,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_end,
            2,
        )
        # Generate filename with frame index
        filename = f'frame_{self.frame_index // self.config["sam2"]["frame_interval"]:04d}.png'
        save_path = os.path.join(self.demo_subdir, filename)

        # Save the image
        cv2.imwrite(save_path, canvas)
        return TaskStatus.CONTINUE

    def draw_causal_graph(self, attr_prob_dict, aff_prob_dict):
        affordance_name = "eat"
        related_edges = [
            edge for edge in self.logic_chains if edge["affordance"] == "eat"
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
        affordance_value = {"eat": aff_prob_dict["eat"]}

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
                textfont=dict(color="black", size=12),
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
            if y_pos + (pos_attribute_size[attr] - 5) * 0.54 > 9.5:
                y_pos_edited = 9.5 - (pos_attribute_size[attr] - 5) * 0.54
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
                        + 1.25
                    ],
                    mode="text",
                    text=attr,
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
                    text=attr,
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
        img = fig.to_image(format="png", scale=6)
        img_array = np.frombuffer(img, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Crop the image to half its width, centered
        height, width, _ = img_cv.shape
        start_x = width // 5
        start_y = height // 12
        end_y = height - int(height / 3.8)
        end_x = start_x + (width // 5) * 3
        cropped_img = img_cv[start_y:end_y, start_x:end_x]

        # Encode the cropped image back to bytes
        _, img_encoded = cv2.imencode(".png", cropped_img)
        return img_encoded.tobytes()
