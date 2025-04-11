# ==================== 环境配置 ====================
import os
os.environ["XFORMERS_DISABLED"] = "1"  # 禁用xformers的Triton依赖，解决Windows兼容问题

# ==================== 库导入 ====================
import torch
import einops
import numpy as np
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt
import gradio as gr
import cv2
import tempfile
from tqdm import tqdm
import time

# 本地模型导入
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

# ==================== 核心处理函数 ====================
def process_image(input_img, num_steps=2, ensemble_size=4, processing_res=-1, no_color=False, dtype="fp16"):
    """
    深度估计处理主函数
    参数：
        input_img: 输入图像（numpy数组）
        num_steps: ODE求解步数
        ensemble_size: 集成次数
        processing_res: 处理分辨率
        no_color: 是否使用灰度输出
        dtype: 计算精度
    返回：
        深度图（PIL图像对象）
    """
    # ---------- 模型初始化 ----------
    if not hasattr(process_image, "model"):
        # 单例模式，避免重复加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        process_image.model = DepthFM("checkpoints/depthfm-v1.ckpt")
        process_image.model = process_image.model.to(device).eval()
        process_image.device = device
    
    # ---------- 数据预处理 ----------
    input_tensor, orig_size = load_im(input_img, processing_res)
    input_tensor = input_tensor.to(process_image.device)
    
    # ---------- 深度估计 ----------
    model_dtype = get_dtype_from_str(dtype)
    process_image.model.model.dtype = model_dtype
    
    with torch.autocast(device_type="cuda", dtype=model_dtype):
        depth_map = process_image.model.predict_depth(
            input_tensor,
            num_steps=num_steps,
            ensemble_size=ensemble_size
        )
    
    # ---------- 后处理 ----------
    depth_np = depth_map.squeeze().cpu().numpy()  # 移除批次和通道维度
    
    # 颜色映射
    if no_color:
        result = (depth_np * 255).astype(np.uint8)
    else:
        result = plt.get_cmap('magma')(depth_np, bytes=True)[..., :3]
    
    # 恢复原始尺寸
    result_pil = Image.fromarray(result)
    if result_pil.size != orig_size:
        result_pil = result_pil.resize(orig_size, Resampling.BILINEAR)
    
    return result_pil

# ==================== 视频处理功能 ====================
def process_video(input_video, num_steps=2, ensemble_size=4, processing_res=512, no_color=False, dtype="fp16", frame_stride=1, progress=gr.Progress()):
    """
    视频深度估计处理
    参数：
        input_video: 输入视频路径
        num_steps: ODE求解步数
        ensemble_size: 集成次数
        processing_res: 处理分辨率
        no_color: 是否使用灰度输出
        dtype: 计算精度
        frame_stride: 帧采样步长（每隔多少帧处理一帧）
        progress: Gradio进度条对象
    返回：
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
    
    # 根据帧采样步长计算实际处理的帧数
    processed_frames = frame_count // frame_stride
    
    # 创建临时输出文件
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/frame_stride, (width, height))
    
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
                
            # 按步长采样
            if frame_idx % frame_stride == 0:
                # 进度更新
                progress(processed_idx/processed_frames, desc=f"处理帧 {processed_idx+1}/{processed_frames}")
                
                # 转换为RGB（OpenCV使用BGR）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 深度估计
                depth_result = process_image(
                    frame_rgb, 
                    num_steps=num_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    no_color=no_color,
                    dtype=dtype
                )
                
                # 转换回OpenCV格式
                depth_frame = np.array(depth_result)
                depth_frame_bgr = cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR)
                
                # 写入输出视频
                out.write(depth_frame_bgr)
                processed_idx += 1
                
            frame_idx += 1
            
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

# ==================== Gradio界面配置 ====================
# 示例文件配置
EXAMPLE_DIR = "examples"  # 示例图片存放目录
demo_samples = [
    [os.path.join(EXAMPLE_DIR, "img/dog.png"), 2, 4, -1, False],
]

# 界面布局
with gr.Blocks(title="DepthFM 深度估计演示", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 DepthFM 单目深度估计系统")
    gr.Markdown(" 上传图片或视频生成深度图，支持参数调节和多种输出格式")
    
    with gr.Tabs():
        # === 图像处理选项卡 ===
        with gr.TabItem("图像深度估计"):
            with gr.Row():
                # 左侧控制面板
                with gr.Column(scale=1):
                    img_input = gr.Image(label="输入图像", type="numpy")
                    
                    with gr.Accordion("高级参数", open=False):
                        processing_res = gr.Slider(
                            minimum=64, maximum=2048, value=-1,
                            label="处理分辨率（-1=自动）", step=64
                        )
                        num_steps = gr.Slider(
                            minimum=1, maximum=10, value=2,
                            label="ODE求解步数", step=1
                        )
                        ensemble_size = gr.Slider(
                            minimum=1, maximum=10, value=4,
                            label="集成次数", step=1
                        )
                        dtype = gr.Dropdown(
                            ["fp16", "fp32", "bf16"], value="fp16",
                            label="计算精度"
                        )
                        no_color = gr.Checkbox(
                            label="灰度输出", info="勾选启用单通道深度图"
                        )
                    
                    submit_btn = gr.Button("🚀 开始计算", variant="primary")
                
                # 右侧结果展示
                with gr.Column(scale=2):
                    img_output = gr.Image(label="深度图结果", type="pil")
            
            # 示例区块
            gr.Examples(
                examples=demo_samples,
                inputs=[img_input, num_steps, ensemble_size, processing_res, no_color],
                outputs=img_output,
                fn=process_image,
                cache_examples=False,  # 禁用缓存避免路径问题
                label="快速示例"
            )

            # 按钮事件绑定
            submit_btn.click(
                fn=process_image,
                inputs=[img_input, num_steps, ensemble_size, processing_res, no_color, dtype],
                outputs=img_output
            )
        
        # === 视频处理选项卡 ===
        with gr.TabItem("视频深度估计"):
            with gr.Row():
                # 左侧控制面板
                with gr.Column(scale=1):
                    video_input = gr.Video(label="输入视频")
                    
                    with gr.Accordion("视频处理参数", open=True):
                        video_processing_res = gr.Slider(
                            minimum=128, maximum=1024, value=512,
                            label="处理分辨率", step=64
                        )
                        video_frame_stride = gr.Slider(
                            minimum=1, maximum=10, value=1,
                            label="帧采样步长", step=1,
                            info="每隔多少帧处理一帧（值越大处理越快，但流畅度降低）"
                        )
                    
                    with gr.Accordion("深度估计参数", open=False):
                        video_num_steps = gr.Slider(
                            minimum=1, maximum=10, value=2,
                            label="ODE求解步数", step=1
                        )
                        video_ensemble_size = gr.Slider(
                            minimum=1, maximum=10, value=4,
                            label="集成次数", step=1
                        )
                        video_dtype = gr.Dropdown(
                            ["fp16", "fp32", "bf16"], value="fp16",
                            label="计算精度"
                        )
                        video_no_color = gr.Checkbox(
                            label="灰度输出", info="勾选启用单通道深度图"
                        )
                    
                    video_submit_btn = gr.Button("🎬 开始处理视频", variant="primary")
                
                # 右侧结果展示
                with gr.Column(scale=2):
                    video_output = gr.Video(label="深度视频结果")
                    video_status = gr.Textbox(label="处理状态", value="等待处理...", interactive=False)
            
            # 视频处理事件绑定
            video_submit_btn.click(
                fn=process_video,
                inputs=[video_input, video_num_steps, video_ensemble_size, 
                       video_processing_res, video_no_color, video_dtype, video_frame_stride],
                outputs=video_output
            )

# ==================== 启动应用 ====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 生成公共访问链接
        allowed_paths=[os.path.abspath(EXAMPLE_DIR)]  # 允许访问示例目录
    )



