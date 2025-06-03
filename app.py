# ==================== 环境配置 ====================
import os
os.environ["XFORMERS_DISABLED"] = "1"  # 禁用xformers的Triton依赖，解决Windows兼容问题

# ==================== 库导入 ====================
import torch
import einops
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from PIL.Image import Resampling
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from ultralytics import YOLO  # 导入YOLO类
import cv2
import tempfile

# 导入自定义模块
from depthfm import DepthFM

# ==================== 工具函数 ====================
def get_dtype_from_str(dtype_str):
    """将字符串转换为对应的torch数据类型"""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16, 
        "bf16": torch.bfloat16
    }
    return dtype_map[dtype_str]

def resize_max_res(img, max_edge_resolution, resample=Resampling.BILINEAR):
    """
    保持宽高比调整图像尺寸，确保长边不超过指定分辨率
    参数：
        img: PIL图像对象
        max_edge_resolution: 最大边长（像素）
        resample: 重采样方法
    返回：
        调整后的图像和原始尺寸元组
    """
    original_w, original_h = img.size
    scale = min(max_edge_resolution/original_w, max_edge_resolution/original_h)
    
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    new_w = (new_w // 64) * 64  # 确保尺寸是64的倍数
    new_h = (new_h // 64) * 64
    
    return img.resize((new_w, new_h), resample=resample), (original_w, original_h)

def load_im(input_image, processing_res=-1):
    """
    图像预处理管道
    参数：
        input_image: Gradio传入的numpy数组格式图像
        processing_res: 处理分辨率
    返回：
        预处理后的张量和原始尺寸
    """
    # 转换输入格式
    pil_img = Image.fromarray(input_image).convert('RGB')
    
    # 自动确定处理分辨率
    if processing_res < 0:
        processing_res = max(pil_img.size)
        
    # 调整尺寸
    resized_img, orig_size = resize_max_res(pil_img, processing_res)
    
    # 归一化处理
    img_array = np.array(resized_img)
    img_tensor = einops.rearrange(img_array, 'h w c -> c h w')  # 调整维度顺序
    img_tensor = img_tensor / 127.5 - 1  # 归一化到[-1, 1]
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32)[None]  # 添加批次维度
    
    return img_tensor, orig_size

# ==================== 模型初始化 ====================
# 使用GPU如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型
depth_model = DepthFM("checkpoints/depthfm-v1.ckpt")
depth_model = depth_model.to(device).eval()

# 初始化YOLOv11模型
yolo_model = YOLO("checkpoints/yolo11n.pt")  # 使用ultralytics的YOLO类

# ==================== 处理函数 ====================
def process_image(input_img, 
                depth_enabled=True, 
                yolo_enabled=True,
                num_steps=2, 
                ensemble_size=4, 
                processing_res=-1, 
                depth_colormap="magma",
                confidence_threshold=0.5):
    """
    图像处理主函数
    参数:
        input_img: 输入图像 (numpy数组)
        depth_enabled: 是否启用深度估计
        yolo_enabled: 是否启用目标检测
        num_steps: DepthFM的ODE求解步数
        ensemble_size: DepthFM的集成大小
        processing_res: 处理分辨率
        depth_colormap: 深度图颜色映射名称
        confidence_threshold: 目标检测置信度阈值
    返回:
        原始图像, 深度图, 目标检测结果, 融合结果
    """
    if input_img is None:
        return None, None, None, None
    
    # 转换为PIL图像备用
    if isinstance(input_img, np.ndarray):
        pil_img = Image.fromarray(input_img)
    else:
        pil_img = input_img
        input_img = np.array(pil_img)
    
    # 结果初始化
    depth_result = None
    detection_result = None
    fusion_result = None
    
    # -------- 深度估计 --------
    if depth_enabled:
        # 数据预处理
        input_tensor, orig_size = load_im(input_img, processing_res)
        input_tensor = input_tensor.to(device)
        
        # 深度估计
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            depth_map = depth_model.predict_depth(
                input_tensor,
                num_steps=num_steps,
                ensemble_size=ensemble_size
            )
        
        # 后处理
        depth_np = depth_map.squeeze().cpu().numpy()
        result = plt.get_cmap(depth_colormap)(depth_np, bytes=True)[..., :3]
        
        # 恢复原始尺寸
        depth_result = Image.fromarray(result)
        if depth_result.size != orig_size:
            depth_result = depth_result.resize(orig_size, Resampling.BILINEAR)
    
    # -------- 目标检测 --------
    detections = []
    if yolo_enabled:
        # 使用ultralytics的YOLO进行检测
        results = yolo_model(input_img, conf=confidence_threshold)
        
        # 提取检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_model.names[cls_id]
                
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': cls_name
                })
        
        print(f"检测到 {len(detections)} 个对象: {detections}")  # 调试输出
        
        # 可视化检测结果
        detection_result = results[0].plot()  # 使用ultralytics的内置可视化
        detection_result = Image.fromarray(detection_result)
    
    # -------- 融合结果 --------
    if depth_enabled and yolo_enabled and len(detections) > 0:
        # 准备融合图像基于深度图
        fusion_img = depth_result.copy() if depth_result else pil_img.copy()
        
        # 在融合图像上绘制检测结果
        draw = ImageDraw.Draw(fusion_img)
        
        # 提取深度信息和目标检测信息
        for det in detections:
            box = det['box']
            cls_name = det['class_name']
            conf = det['confidence']
            
            # 使用类别名称作为颜色键
            color = (255, 0, 0)  # 默认红色
            
            # 绘制边界框
            draw.rectangle(box, outline=color, width=3)
            
            # 绘制标签
            label = f"{cls_name} {conf:.2f}"
            draw.text((box[0], box[1]-20), label, fill=(255, 255, 255))
            
            # 计算边界框中心点的深度值（如果有深度图）
            if depth_result:
                center_x = (box[0] + box[2]) // 2
                center_y = (box[1] + box[3]) // 2
                
                # 绘制中心点
                draw.ellipse([center_x-5, center_y-5, center_x+5, center_y+5], fill=(255, 0, 0))
        
        fusion_result = fusion_img
    
    # 如果只启用了一个功能，使用该功能的结果作为融合结果
    if fusion_result is None:
        if depth_enabled and depth_result is not None:
            fusion_result = depth_result
        elif yolo_enabled and detection_result is not None:
            fusion_result = detection_result
        else:
            fusion_result = pil_img
    
    return pil_img, depth_result, detection_result, fusion_result

def process_video(input_video, num_steps=2, ensemble_size=4, processing_res=512, depth_colormap="magma", confidence_threshold=0.5, progress=gr.Progress()):
    """
    视频处理主函数
    参数:
        input_video: 输入视频路径
        num_steps: DepthFM的ODE求解步数
        ensemble_size: DepthFM的集成大小
        processing_res: 处理分辨率
        depth_colormap: 深度图颜色映射名称
        confidence_threshold: 目标检测置信度阈值
        progress: Gradio进度条对象
    返回:
        处理后的视频路径
    """
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建临时输出文件
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 设置进度条
    progress(0, desc="初始化...")
    
    try:
        frame_idx = 0
        processed_idx = 0
        
        # 逐帧处理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 进度更新
            progress(processed_idx/frame_count, desc=f"处理帧 {processed_idx+1}/{frame_count}")
            
            # 转换为RGB（OpenCV使用BGR）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 深度估计
            depth_result = process_image(
                frame_rgb, 
                depth_enabled=True,
                yolo_enabled=True,
                num_steps=num_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                depth_colormap=depth_colormap,
                confidence_threshold=confidence_threshold
            )
            
            # 转换回OpenCV格式
            depth_frame = np.array(depth_result[3])  # 融合结果
            depth_frame_bgr = cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR)
            
            # 写入输出视频
            out.write(depth_frame_bgr)
            processed_idx += 1
            
            # 每100帧检查是否用户取消
            if frame_idx % 100 == 0 and progress.is_cancelled():
                break
                
    except Exception as e:
        raise RuntimeError(f"视频处理出错: {str(e)}")
    finally:
        # 释放资源
        cap.release()
        out.release()
        progress(1.0, desc="处理完成")
        
    return output_path

# ==================== Gradio界面设计 ====================
# 示例图片配置
EXAMPLE_DIR = "examples"  # 示例文件目录
demo_samples = [
    [os.path.join(EXAMPLE_DIR, "img/dog.png"), True, True, 2, 4, -1, "magma", 0.5],
]

# 颜色映射选项
COLORMAP_OPTIONS = ["magma", "viridis", "inferno", "plasma", "cividis", "turbo", "jet"]

with gr.Blocks(title="3D感知与目标检测系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 单目视觉3D感知目标检测系统")
    gr.Markdown("基于DepthFM和YOLOv11的3D感知检测系统")
    
    with gr.Row():
        # 左侧控制面板
        with gr.Column(scale=1):
            input_type = gr.Radio(
                choices=["图像", "视频"],
                value="图像",
                label="输入类型",
                info="选择处理的媒体类型"
            )
            
            img_input = gr.Image(label="输入图像", type="numpy", visible=True)
            video_input = gr.Video(label="输入视频", visible=False)
            
            with gr.Row():
                depth_enabled = gr.Checkbox(label="启用深度估计", value=True)
                yolo_enabled = gr.Checkbox(label="启用目标检测", value=True)
            
            with gr.Accordion("深度估计参数", open=False):
                num_steps = gr.Slider(
                    minimum=1, maximum=10, value=2,
                    label="ODE求解步数", step=1
                )
                ensemble_size = gr.Slider(
                    minimum=1, maximum=10, value=4,
                    label="集成次数", step=1
                )
                processing_res = gr.Slider(
                    minimum=64, maximum=2048, value=-1,
                    label="处理分辨率（-1=自动）", step=64
                )
                depth_colormap = gr.Dropdown(
                    COLORMAP_OPTIONS, value="magma", 
                    label="深度图颜色映射"
                )
            
            with gr.Accordion("目标检测参数", open=False):
                confidence_threshold = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5,
                    label="置信度阈值", step=0.05
                )
            
            submit_btn = gr.Button("🚀 开始处理", variant="primary", visible=True)
            video_submit_btn = gr.Button("🎬 开始处理视频", variant="primary", visible=False)
        
        # 右侧结果展示
        with gr.Column(scale=2):
            with gr.Tab("融合结果"):
                fusion_output = gr.Image(label="融合结果", type="pil", visible=True)
                video_output = gr.Video(label="视频输出", visible=False)
            with gr.Tab("单独结果"):
                with gr.Row():
                    original_output = gr.Image(label="原始图像", type="pil")
                    depth_output = gr.Image(label="深度估计", type="pil")
                with gr.Row():
                    detection_output = gr.Image(label="目标检测", type="pil")
    
    # 联动函数
    def switch_input(input_type):
        if input_type == "图像":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    
    input_type.change(
        switch_input,
        inputs=[input_type],
        outputs=[img_input, video_input, submit_btn, video_submit_btn, fusion_output, video_output]
    )

    # 示例区块
    gr.Examples(
        examples=demo_samples,
        inputs=[img_input, depth_enabled, yolo_enabled, num_steps, 
                ensemble_size, processing_res, depth_colormap, confidence_threshold],
        outputs=[original_output, depth_output, detection_output, fusion_output],
        fn=process_image,
        cache_examples=False,  # 禁用缓存避免路径问题
        label="示例图片"
    )

    # 按钮事件绑定
    submit_btn.click(
        fn=process_image,
        inputs=[img_input, depth_enabled, yolo_enabled, num_steps, 
                ensemble_size, processing_res, depth_colormap, confidence_threshold],
        outputs=[original_output, depth_output, detection_output, fusion_output]
    )
    
    video_submit_btn.click(
        fn=process_video,
        inputs=[video_input, num_steps, ensemble_size, processing_res, depth_colormap, confidence_threshold],
        outputs=video_output
    )

# ==================== 启动应用 ====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,  # 生成公共访问链接
        allowed_paths=[os.path.abspath(EXAMPLE_DIR)]  # 允许访问示例目录
    )