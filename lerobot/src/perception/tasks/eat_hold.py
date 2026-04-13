import os


import cv2
import numpy as np
from loguru import logger


from .base_task import BaseTask, TaskStatus

CKPT_DIR_DICT = {
    "orange.": "robot/checkpoints/rdt-170m-finetune-flexiv-eat-orange-0409_0525/checkpoint-24000",
    "pink cube.": "robot/checkpoints/rdt-170m-finetune-flexiv-eat-pinkcube-0414_0926/checkpoint-16000",
    "yellow cube.": "robot/checkpoints/rdt-170m-finetune-flexiv-eat-yellowcube-0420_1121/checkpoint-19500"
}

LANG_EMBEDINGS_PATH_DICT = {
    "orange.": "robot/embeddings/pick_orange.pt",
    "pink cube.": "robot/embeddings/pick_pinkcube.pt",
    "yellow cube.": "robot/embeddings/pick_pinkcube.pt"
}


@BaseTask.register_task("eat_hold")
class EatHold(BaseTask):
    def object_selection(self, video_type = "camera", **kwargs) -> tuple[str, str]:
        
        assert video_type == "camera"
        pil_img_dict = kwargs["pil_img_dict"]
        key = "third"
        pil_img = pil_img_dict[key]
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
        # 在图像上绘制检测框
        img_array = np.array(pil_img)
        for detection in detections:
            bbox = detection.box
            x1, y1, x2, y2 = map(int, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, f"{detection.label}: {detection.score:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存带有检测框的图像
        save_path = os.path.join("demo_test", f"{key}_detection.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        logger.info(f"检测结果已保存至: {save_path}")
        # 获取检测到的物体的属性和可操作性概率
        attr_probs, _, aff_probs_w = self.ocl_pipeline.boundingbox_calculate_text_probabilities(
            frame=pil_img,
            detections=detections,
            labels=self.config["detector"]["labels"],
            key_attr=self.key_attr,
            key_aff=self.key_aff,
            logic_chains=self.logic_chains,
        )
        prob_dict = {}
        # 在图像上添加属性和可操作性概率信息
        for detection in detections:
            bbox = detection.box
            label = detection.label
            x1, y1 = int(bbox.xmin), int(bbox.ymin)
            
            # 添加属性概率文本
            y_offset = 30
            # for attr, prob in aff_probs_w.items():
            #     text = f"{attr}: {prob["eat"]:.2f}"
            #     cv2.putText(img_array, text, (x1, y1-y_offset), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #     y_offset += 20
                
            # 添加可操作性概率文本
            # prob = prob_list[label]
            prob = aff_probs_w[label]['eat'][0]
            prob_dict[label] = prob
            text = f"eat: {prob:.2f}"
            cv2.putText(img_array, text, (x1, y1-y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 20
        
        # 保存带有所有信息的图像
        save_path = os.path.join("demo_test", f"{key}_detection_with_probs.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        logger.info(f"带有属性和可操作性概率的检测结果已保存至: {save_path}")
        
        
        # 找到prob_dict中最大值对应的key
        max_prob_key = max(prob_dict.items(), key=lambda x: x[1])[0]
        logger.info(f"最大概率的物体是: {max_prob_key}, 概率为: {prob_dict[max_prob_key]:.2f}")
        
        return CKPT_DIR_DICT[max_prob_key], LANG_EMBEDINGS_PATH_DICT[max_prob_key]
    
    def decision_making(
        self,
        video_type: str = "camera",
        **kwargs,
    ) -> TaskStatus:
        self.threshold = 0.35
        try:
            # Get image from streaming input
            if video_type == "camera":
                self.camera_input()
                pil_img_dict = kwargs["pil_img_dict"]
            else:
                pil_img_dict = self.video_input()

            attr_prob_dict = {}
            aff_prob_dict = {}
            mask_dict = {}
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
                    output_dir=self.output_path_dict[key] if hasattr(self, "output_path_dict") else f"demo_test/{key}",
                    frame_idx=self.frame_index,
                )
                mask_dict[key] = video_segment[self.target_id_dict[key]]
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
                # logger.debug(self.affordance_buffer_dict)
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
            attr_bool_dict, aff_bool_dict = (
                self.process_attributes_affordances(
                    attr_prob_dict, aff_prob_dict
                )
            )
            # logger.debug(f"{self.frame_index}: {attr_bool_dict}")
            # logger.debug(f"{self.frame_index}: {aff_bool_dict}")
            if self.show_image:
                self.visualize_images_and_affordances(
                    pil_img_dict, attr_bool_dict, aff_bool_dict, mask_dict
                )
            if aff_bool_dict['hold'] > self.threshold:
                return TaskStatus.CONTINUE
            else:
                return TaskStatus.STOP
        except StopIteration:
            logger.info("没有更多的图像。")
            return TaskStatus.STOP
    
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
        attr_prob_dict,
        aff_prob_dict,
        mask_dict,
        causal_graph=True,
    ):
        # Convert and prepare all images
        show_images = []
        # Convert and prepare all images
        show_images = []
        for key, pil_img in pil_img_dict.items():
            cv_image = np.array(pil_img)
            cv_img = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            # Create colored mask overlay
            if key in mask_dict:
                mask = np.squeeze(
                    mask_dict[key]
                )  # Remove extra dimension (1,480,640) -> (480,640)
                colored_mask = np.zeros_like(cv_img)
                colored_mask[mask] = [0, 0, 255]  # Green mask

                # Blend mask with original image
                alpha = 0.3
                cv_img = cv2.addWeighted(cv_img, 1, colored_mask, alpha, 0)

            show_images.append(cv_img)

        # Calculate grid dimensions
        img_height = max(img.shape[0] for img in show_images)
        img_width = max(img.shape[1] for img in show_images)

        # Create 2x2 canvas with extra height for text
        canvas = (
            np.ones((2 * img_height + 100, 2 * img_width, 3), dtype=np.uint8)
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
            # TODO: Draw causal graph
            graph_img_bytes = self.draw_causal_graph(
                attr_prob_dict, aff_prob_dict, "hold"
            )  # Draw causal graph
            graph_img_array = np.frombuffer(graph_img_bytes, np.uint8)
            graph_img = cv2.imdecode(graph_img_array, cv2.IMREAD_COLOR)
            graph_img_resized = cv2.resize(
                graph_img, (img_width, img_height)
            )  # Resize to fit the shape
            canvas[img_height : 2 * img_height, img_width : 2 * img_width] = (
                graph_img_resized
            )
        y_pos = 2 * img_height + 30  # Start position for text
        # Display attributes
        attr_text = "Attributes: "
        active_attrs = [f"{attr}: {value:.2f}" for attr, value in attr_prob_dict.items()]
        attr_text += ", ".join(active_attrs) if active_attrs else "None detected"
        
        cv2.putText(
            canvas,
            attr_text,
            (10, y_pos),  # Place text below images
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),  # Black color for attributes
            2,
        )
        
        # Add affordance text (moved down by 30 pixels)
        y_pos += 30  # Move down for affordance text
        if aff_prob_dict['hold'] > 0.35:
            text = f"hold-able: {aff_prob_dict['hold']:.2f}, CONTINUE"
            text_color = (0, 0, 0)
        else:
            text = f"hold-able: {aff_prob_dict['hold']:.2f}, STOP"
            text_color = (0, 0, 255)


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
            text,
            (10, y_pos),  # Place text below images
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