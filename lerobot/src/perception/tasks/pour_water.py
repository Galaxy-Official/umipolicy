# import logging
import os
import cv2
import numpy as np

from loguru import logger
from .base_task import BaseTask, TaskStatus

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# logger = logging.getLogger(__name__)


@BaseTask.register_task("pour_water")
class PourWater(BaseTask):
    def decision_making(
        self,
        video_type: str = "camera",
        **kwargs,
    ) -> TaskStatus:
        self.threshold = 0.35
        try:
            # Get image from streaming input
            if video_type == "camera":
                pil_img_dict = self.camera_input(list(self.target_id_dict.keys()))
            else:
                pil_img_dict = self.video_input()

            attr_prob_dict = {}
            aff_prob_dict = {}
            mask_dict = {}
            # Perform object detection
            for key, pil_img in pil_img_dict.items():
                if key == "ego":
                    # Skip ego view for object detection
                    continue
                # 判断是否需要执行检测 - 在第一帧或者需要重新检测时执行
                if self.frame_index[key] == 0 or getattr(self, 'force_detection', False):
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
                        logger.error(f"在 {key} 视角未检测到任何目标。")
                        # 当单个视角没有检测到目标时，记录但不立即停止
                        if key == 'ego':  # 如果是wrist视角，可能需要多次尝试
                            logger.info(f"手腕视角 (ego) 暂未检测到目标，跳过此视角")
                            continue
                        elif key == 'bird' or key == 'third':  # 如果是头部或侧面视角未检测到，则停止
                            return TaskStatus.STOP
                    
                    # 如果是强制重新检测，则重置标志
                    if getattr(self, 'force_detection', False):
                        self.force_detection = False
                        logger.info(f"已重新执行检测，重置跟踪状态")
                else:
                    detections = None
                
                # 执行跟踪，传递视角名称
                try:
                    obj_id_len, video_segment = self.sam2_predictor_dict[key].streaming_sam_and_tracking(
                        image=pil_img,
                        bbox_data=detections,
                        output_dir=self.output_path_dict[key] if hasattr(self, "output_path_dict") else f"demo_test/{key}",
                        frame_idx=self.frame_index[key],
                        view_name=key,  # 传递视角名称
                    )
                    
                    # 如果没有目标ID，则跳过此视角
                    if self.target_id_dict[key] not in video_segment:
                        logger.warning(f"{key} 视角中未找到目标ID {self.target_id_dict[key]}，跳过此视角")
                        continue
                        
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
                except ValueError as e:
                    # 处理初始化失败的情况
                    logger.warning(f"{key} 视角初始化失败: {str(e)}")
                    continue
            
            # 检查mask体积
            mask_volumes = {}
            small_mask_views = []  # 记录哪些视角的mask体积过小

            for key, mask in mask_dict.items():
                # 计算mask中非零像素的数量
                volume = np.count_nonzero(mask)
                mask_volumes[key] = volume
                logger.info(f"{key} 视角的mask体积: {volume}个像素点")
                if volume < 10:
                    small_mask_views.append(key)
                    logger.warning(f"{key} 视角的mask体积过小: {volume}个像素点")
            
            # 如果有两个或更多视角的mask体积过小，强制下一帧重新检测并返回RETURN信号
            if len(small_mask_views) >= 2:
                logger.warning(f"以下视角的mask体积过小: {small_mask_views}，将重新进行目标检测")
                # 设置标志，在下一帧强制执行检测
                self.force_detection = True
                
                # 只重置mask体积过小的视角的跟踪器状态
                for key in small_mask_views:
                    if key in self.sam2_predictor_dict:
                        logger.info(f"重置 {key} 视角的跟踪器")
                        if hasattr(self.sam2_predictor_dict[key], 'reset_tracker'):
                            # 调用重置方法时传递视角名称
                            self.sam2_predictor_dict[key].reset_tracker(view_name=key)
                        else:
                            logger.warning(f"跟踪器 {key} 没有 reset_tracker 方法，可能需要手动实现")
                
                return TaskStatus.RETURN
                
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
            if aff_bool_dict['pour'] > self.threshold:
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
        try:
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
        except StopIteration as e:
            logger.info(f"Video ended: {e}")
        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            raise
        finally:
            # Clean up video resources
            self.cleanup_video_resources()

    def visualize_images_and_affordances(
        self,
        pil_img_dict,
        attr_prob_dict,
        aff_prob_dict,
        mask_dict,
        causal_graph=True,
    ):
        # 定义固定的视角顺序
        view_order = ['bird', 'third', 'ego']
        # Convert and prepare all images
        show_images = []
        view_names = []
        
        # 按照固定顺序处理图片
        for view in view_order:
            if view in pil_img_dict:
                pil_img = pil_img_dict[view]
                cv_image = np.array(pil_img)
                cv_img = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                # Create colored mask overlay
                if view in mask_dict:
                    mask = np.squeeze(mask_dict[view])
                    colored_mask = np.zeros_like(cv_img)
                    colored_mask[mask] = [0, 0, 255]  # Green mask

                    # Blend mask with original image
                    alpha = 0.3
                    cv_img = cv2.addWeighted(cv_img, 1, colored_mask, alpha, 0)

                show_images.append(cv_img)
                view_names.append(f"{view} view")
            else:
                # 如果某个视角不存在，添加一个空白图像
                blank_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
                show_images.append(blank_img)
                view_names.append(f"{view} view")

        # Calculate grid dimensions
        img_height = max(img.shape[0] for img in show_images)
        img_width = max(img.shape[1] for img in show_images)

        # Create 2x2 canvas with extra height for text
        canvas = (
            np.ones((2 * img_height + 100, 2 * img_width, 3), dtype=np.uint8)
            * 255
        )

        # Place images in grid
        positions = [
            (0, 0),  # bird view
            (0, 1),  # side view
            (1, 0),  # ego view
        ]
        
        for i, img in enumerate(show_images[:3]):
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
                attr_prob_dict, aff_prob_dict
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
        if aff_prob_dict['pour'] > 0.4:
            text = f"pour-able: {aff_prob_dict['pour']:.2f}, CONTINUE"
            text_color = (0, 0, 0)
        else:
            text = f"pour-able: {aff_prob_dict['pour']:.2f}, STOP"
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
        filename = f'frame_{self.frame_index["bird"] // self.config["sam2"]["frame_interval"]:04d}.png'
        save_path = os.path.join(self.demo_subdir, filename)

        # Save the image
        cv2.imwrite(save_path, canvas)

        # Show combined visualization
        # cv2.imshow("Images and Affordances", canvas)
        # if cv2.waitKey(300) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     return TaskStatus.STOP

    def object_selection(self, video_type = "camera") -> tuple[str, str]:
        pour_water_ckpt = "robot/checkpoints/rdt-170m-finetune-flexiv-pour-water2-side-0326_2054/checkpoint-10500"
        lang_embeddings_path = "robot/embeddings/pour_water.pt"
        return pour_water_ckpt, lang_embeddings_path