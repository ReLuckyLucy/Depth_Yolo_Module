import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO
import os
import numpy as np

# 映射模型名称和模型文件路径
model_mapping = {
    "YOLOv11n": "model/yolo11n.pt"
}

def yolo_inference(image, video, model_name, image_size, conf_threshold):
    model_path = model_mapping[model_name]  # 从映射中获取模型路径
    model = YOLO(model_path)  # 使用本地路径加载模型
    if image is not None:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    elif video is not None:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolo_inference_for_examples(image, model_name, image_size, conf_threshold):
    annotated_image, _ = yolo_inference(image, None, model_name, image_size, conf_threshold)
    return annotated_image


# ==================== 应用入口 ====================
def app():
    # 示例图片路径
    test_image_path = os.path.abspath("img/test.png")
    example_samples = []
    if os.path.exists(test_image_path):
        example_samples.append([test_image_path, "YOLOv11n", 640, 0.25])
    
    # 使用Soft主题构建界面
    with gr.Blocks(title="YOLOv11目标检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 YOLOv11 目标检测系统")
        gr.Markdown("上传图像或视频进行目标检测，支持参数调节")
        
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1):
                input_type = gr.Radio(
                    choices=["图片", "视频"],
                    value="图片",
                    label="输入类型",
                    info="选择处理的媒体类型"
                )
                
                image = gr.Image(label="输入图像", type="pil", visible=True)
                video = gr.Video(label="输入视频", visible=False)
                
                with gr.Accordion("模型参数", open=False):
                    model_name = gr.Dropdown(
                        label="模型选择",
                        choices=list(model_mapping.keys()),
                        value="YOLOv11n",
                        info="选择要使用的YOLOv11模型版本"
                    )
                    
                    image_size = gr.Slider(
                        minimum=320, maximum=1280, value=640, step=32,
                        label="图像尺寸",
                        info="更大的尺寸通常能提高精度，但会降低速度"
                    )
                    
                    conf_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                        label="置信度阈值",
                        info="调整检测的置信度阈值，较低的值会增加检测数量但可能增加误报"
                    )
                
                submit_btn = gr.Button("🚀 开始检测", variant="primary")
            
            # 右侧结果展示
            with gr.Column(scale=2):
                output_image = gr.Image(label="检测结果", type="numpy", visible=True)
                output_video = gr.Video(label="检测结果", visible=False)
        
        # 示例区块
        if example_samples:
            gr.Examples(
                examples=example_samples,
                inputs=[image, model_name, image_size, conf_threshold],
                outputs=output_image,
                fn=yolo_inference_for_examples,
                cache_examples=True,
                label="快速示例"
            )
        
        # 交互逻辑
        def update_visibility(input_type):
            image_visibility = input_type == "图片"
            video_visibility = input_type == "视频"
            
            return (
                gr.update(visible=image_visibility),
                gr.update(visible=video_visibility),
                gr.update(visible=image_visibility),
                gr.update(visible=video_visibility),
            )

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_name, image_size, conf_threshold, input_type):
            if input_type == "图片":
                return yolo_inference(image, None, model_name, image_size, conf_threshold)
            else:
                return yolo_inference(None, video, model_name, image_size, conf_threshold)

        submit_btn.click(
            fn=run_inference,
            inputs=[image, video, model_name, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )
        
        return demo

# ==================== 启动应用 ====================
if __name__ == '__main__':
    demo = app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
