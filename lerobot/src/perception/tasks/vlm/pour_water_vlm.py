import os
import textwrap


import cv2
import numpy as np
from PIL import Image
from loguru import logger


from .base_vlm_task import BaseVLMTask, TaskStatus, ViewType


@BaseVLMTask.register_task("pour_water_vlm")
class PourWaterVLM(BaseVLMTask):
    def decision_making(self, video_type="camera"):
        try:
            # Get image from streaming input
            if video_type == "camera":
                # TODO: add support for real time camera input
                raise NotImplementedError("Real-time camera input not supported.")
                img = self.camera_input()
            else:
                pil_img_dict = self.video_input(is_path=True)
            # logger.info(pil_img_dict.keys())
            # Prepare the prompt for pouring water task
            user_prompt = """\
                These are three different perspectives (third, bird, ego) of the same moment showing a dumping scene as well.
                Synthesize the images from the three viewpoints and consider the following questions:
                1. Is the cup already full or empty?
                2. Is the cup open or closed?
                3. Is there any spilling or overflow?
                Based on your analysis, provide detailed reasoning and conclude whether it is safe to continue pouring water (CONTINUE) or if you should stop (STOP).
                """
                
                # - "CONTINUE": If it's safe to continue pouring water (container not full, no spilling, open container)
                # - "STOP": If pouring should stop (container full or nearly full, or water spilling, or closed container)
                # Respond with just CONTINUE or STOP.
                
                # Based on your analysis, provide detailed reasoning and conclude whether it is safe to continue pouring water (CONTINUE) or if you should stop (STOP).
            user_prompt = textwrap.dedent(user_prompt)

            # Call VLM APIb
            response = self.vlm.image_infer(
                system_prompt="You are analyzing a scene to determine if it's safe to continue pouring water.",
                user_prompt=user_prompt,
                image_paths=pil_img_dict.values(),
            )
            # response = self.vlm.video_infer(
                
            # )
            # Extract the decision from response
            decision = response.strip().upper()
            # results[view_type] = decision
            logger.info(decision)
            if self.show_image:
                self.visualize_images_and_affordances(
                    pil_img_dict, decision
                )
                
            # For now, we'll continue the task
            return TaskStatus.CONTINUE
        except StopIteration:
            logger.info("没有更多的图像。")
            return TaskStatus.STOP


    def visualize_images_and_affordances(
        self,
        pil_img_dict,
        message,
        causal_graph=False,
    ):
        # Convert and prepare all images
        show_images = []
        # Convert and prepare all images
        show_images = []
        for key, pil_img_path in pil_img_dict.items():
            
            cv_image = np.array(Image.open(pil_img_path))
            cv_img = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            # # Create colored mask overlay
            # if key in mask_dict:
            #     mask = np.squeeze(
            #         mask_dict[key]
            #     )  # Remove extra dimension (1,480,640) -> (480,640)
            #     colored_mask = np.zeros_like(cv_img)
            #     colored_mask[mask] = [0, 0, 255]  # Green mask

            #     # Blend mask with original image
            #     alpha = 0.3
            #     cv_img = cv2.addWeighted(cv_img, 1, colored_mask, alpha, 0)

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

        # if causal_graph is not None:
        #     # TODO: Draw causal graph
        #     graph_img_bytes = self.draw_causal_graph(
        #         attr_prob_dict, aff_prob_dict
        #     )  # Draw causal graph
        #     graph_img_array = np.frombuffer(graph_img_bytes, np.uint8)
        #     graph_img = cv2.imdecode(graph_img_array, cv2.IMREAD_COLOR)
        #     graph_img_resized = cv2.resize(
        #         graph_img, (img_width, img_height)
        #     )  # Resize to fit the shape
        #     canvas[img_height : 2 * img_height, img_width : 2 * img_width] = (
        #         graph_img_resized
        #     )

        # Add affordance text
        # if aff_prob_dict["pour"] > 0.3:
        #     text = f"pour-able: {aff_prob_dict['pour']:.2f}, CONTINUE"
        #     text_color = (0, 0, 0)
        # else:
        #     text = f"pour-able: {aff_prob_dict['pour']:.2f}, STOP"
        #     text_color = (0, 0, 255)
        if message == "CONTINUE":
            text_color = (0, 0, 0)
        elif message == "STOP":
            text_color = (0, 0, 255)
        else:
            text_color = (255, 0, 0)
            
        # if aff_prob_dict["pour"] > 0.3:
        #     text = f"pour-able: {aff_prob_dict['pour']:.2f}, CONTINUE"
        #     text_color = (0, 0, 0)
        # else:
        #     text = f"pour-able: {aff_prob_dict['pour']:.2f}, STOP"
        #     text_color = (0, 0, 255)
        # active_items = [f"{aff}-able, {value:.2f}" for aff, value in aff_prob_dict.items()]
        # text += "; ".join(active_items) if active_items else "None"

        cv2.putText(
            canvas,
            message,
            (10, 2 * img_height + 30),  # Place text below images
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )
        # output_path = "demo_test"
        # if output_path is not None:
        # Create output directory if it doesn't exist
        # os.makedirs(output_path, exist_ok=True)

        # Generate filename with frame index
        filename = f'frame_{self.frame_index // self.config["sam2"]["frame_interval"]:04d}.png'
        save_path = os.path.join(self.demo_subdir, filename)

        # Save the image
        cv2.imwrite(save_path, canvas)

        # Show combined visualization
        # cv2.imshow("Images and Affordances", canvas)
        # if cv2.waitKey(300) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     return TaskStatus.STOP

        return TaskStatus.CONTINUE
