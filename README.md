<div align="center">
 <img alt="logo" height="300px" src="examples\img\logo_depth.png">
</div>



<h1 align="center">Depth_Yolo_Module</h1>
<h1 align="center">基于DepthFM模块与Yolo模块的单目视觉3d感知目标检测</h1>

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ReLuckyLucy/Depth_Yolo_Module">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ReLuckyLucy/Depth_Yolo_Module">
    <img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/ReLuckyLucy/Depth_Yolo_Module?include_prereleases">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ReLuckyLucy/Depth_Yolo_Module">
</p>
<p align="center">
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ReLuckyLucy/Depth_Yolo_Module">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/ReLuckyLucy/Depth_Yolo_Module">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/ReLuckyLucy/Depth_Yolo_Module?style=social">
</p>



## 项目介绍
###  DepthFM
 DepthFM是一种最先进的、多功能的、快速的单目深度估计模型。DepthFM 非常高效，可以在单个推理步骤中合成逼真的深度图。除了传统的深度估计任务外，DepthFM 还展示了下游任务（如深度修复和深度条件合成）的最新功能。

DepthFM通过强图像先验从基础图像合成扩散模型 （Stable Diffusion v2-1） 到流动匹配模型的成功转移。它不是从噪声开始，而是直接从输入图像映射到深度图。


### Yolo
Yolo(You Only Look Once）是一种流行的物体检测和图像分割模型，由华盛顿大学的约瑟夫-雷德蒙（Joseph Redmon）和阿里-法哈迪（Ali Farhadi）开发。YOLO 于 2015 年推出，因其高速度和高精确度而迅速受到欢迎。

Yolo 支持各种视觉人工智能任务，如检测、分割、姿态估计、跟踪和分类。其先进的架构确保了卓越的速度和准确性，使其适用于各种应用，包括边缘设备和云 API。
> 目前还没有yolo模块


## 开始使用
### 通过 Conda 创建环境（推荐）：
```bash
conda create --name dfm_yolo python=3.11

conda activate dfm

pip install -r requirements.txt
```
由于默认下载的pytorch总是会下载到cpu版本，建议是手动输入
```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

###  模型权重下载：
```bash
wget https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt -P checkpoints/
```

---

### **验证安装**
运行以下命令检查关键依赖版本：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import xformers; print(f'xFormers: {xformers.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

输出应类似：
```
PyTorch: 2.2.0+cu121, CUDA: 12.1
xFormers: 0.0.24
NumPy: 1.26.0
```

---

### **注意事项**
1. **GPU 驱动要求**：
   - 确保 NVIDIA 驱动版本 ≥ 530.30（支持 CUDA 12.1）。
   - 检查驱动兼容性：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

2. **Windows 特定问题**：
   - 如果 Conda 安装失败，优先使用 `environment.yml` 中的 pip 安装方式。
   - 确保已安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（C++ 编译环境）。

3. **显存优化**：
   - 在代码中添加以下配置减少显存占用：
     ```python
     import os
     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
     ```



## 项目结构

```
Depth_Yolo_Module/
├── app.py                 # 主应用程序入口
├── gradio_depth_app.py    # DepthFM深度估计模块封装
├── yolov11.py             # YOLOv11目标检测模块封装
├── checkpoints/           # 预训练模型存放目录
│   ├── depthfm-v1.ckpt    # DepthFM模型权重
│   └── yolov11.pt         # YOLOv11模型权重
├── examples/              # 示例图片
│   └── img/
│       ├── car.jpg
│       └── street.jpg
└── README.md              # 项目文档
```


## 参考文献

1. DepthFM: Fast Monocular Depth Estimation with Flow Matching
2. YOLOv11: An Incremental Improvement for Object Detection

### 启动
>推荐使用gradio库
```python
python3 gradio_depth_app.py
```

