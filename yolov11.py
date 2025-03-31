# ==================== 库导入 ====================
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import random

class YOLOv11:
    """YOLOv11模型包装类，用于目标检测和结果可视化"""
    
    def __init__(self, model_path="checkpoints/yolov11.pt", device=None):
        """
        初始化YOLOv11模型
        参数:
            model_path: 模型权重文件路径
            device: 运行设备，默认会自动选择
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"正在加载YOLOv11模型到{self.device}...")
        
        # 实际应用中，可能需要使用YOLOv5/8等已有实现或自定义实现
        try:
            # 尝试使用Ultralytics包加载模型 (YOLOv5/8等)
            import ultralytics
            self.model = ultralytics.YOLO(model_path)
            self.backend = "ultralytics"
        except (ImportError, FileNotFoundError):
            # 备选方案：使用自定义实现（模拟）
            print("Ultralytics未安装或模型未找到，使用模拟版本...")
            self.model = self._mock_model()
            self.backend = "mock"
            
        # 目标类别配置 (以COCO数据集车辆相关类别为例)
        self.vehicle_classes = {
            2: "car", 
            3: "motorcycle", 
            5: "bus", 
            7: "truck",
            8: "boat",
            16: "dog",    # 添加狗的类别
            17: "cat",    # 添加猫的类别
            0: "person",  # 添加人的类别
            1: "bicycle"  # 添加自行车的类别
        }
        
        # 颜色配置
        self.colors = {
            "car": (255, 0, 0),      # 红色
            "motorcycle": (0, 255, 0), # 绿色
            "bus": (0, 0, 255),      # 蓝色
            "truck": (255, 255, 0),  # 黄色
            "boat": (255, 0, 255),   # 紫色
            "dog": (0, 255, 255),    # 青色
            "cat": (255, 128, 0),    # 橙色
            "person": (128, 255, 0), # 黄绿色
            "bicycle": (0, 128, 255) # 天蓝色
        }
    
    def _mock_model(self):
        """创建一个模拟模型（仅用于演示）"""
        class MockModel:
            def __call__(self, img, size=640):
                """模拟检测结果"""
                # 创建随机检测结果
                detections = []
                vehicle_classes = [2, 3, 5, 7, 8, 16, 17, 0, 1]  # 车辆类别ID
                
                # 获取图像尺寸
                height, width = img.shape[:2] if isinstance(img, np.ndarray) else (img.height, img.width)
                
                # 随机生成1-3个车辆
                for _ in range(np.random.randint(1, 4)):
                    # 随机类别、位置和大小
                    cls = np.random.choice(vehicle_classes)
                    x1 = np.random.randint(0, width - 100)
                    y1 = np.random.randint(0, height - 100)
                    w = np.random.randint(100, min(300, width - x1))
                    h = np.random.randint(100, min(300, height - y1))
                    conf = np.random.uniform(0.6, 0.95)
                    
                    detections.append([x1, y1, x1 + w, y1 + h, conf, cls])
                
                return [{'boxes': torch.tensor(detections)}]
        
        return MockModel()
    
    def detect(self, img):
        """
        执行目标检测
        参数:
            img: 输入图像 (numpy数组或PIL图像)
        返回:
            检测结果字典，包含边界框、类别和置信度
        """
        # 确保输入格式正确
        if isinstance(img, np.ndarray):
            # 转为RGB（如果是BGR）
            if img.shape[2] == 3 and img[0,0,0] == img[0,0,2]:  # 简单检查是否为BGR
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
        else:
            # 如果是PIL图像，转为numpy数组
            img_rgb = np.array(img)
        
        # 执行检测
        if self.backend == "ultralytics":
            results = self.model(img_rgb)
            detections = self._parse_ultralytics_results(results)
        else:
            results = self.model(img_rgb)
            detections = self._parse_mock_results(results)
            
        return detections
    
    def _parse_ultralytics_results(self, results):
        """解析Ultralytics YOLO返回的结果"""
        detections = []
        
        # 提取结果
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                # 获取边界框、置信度和类别
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                
                # 过滤出车辆类别
                if cls_id in self.vehicle_classes:
                    cls_name = self.vehicle_classes[cls_id]
                    detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': cls_name
                    })
        
        return detections
    
    def _parse_mock_results(self, results):
        """解析模拟模型的结果"""
        detections = []
        
        for result in results:
            boxes = result['boxes']
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                
                # 过滤出车辆类别
                if int(cls_id) in self.vehicle_classes:
                    cls_name = self.vehicle_classes[int(cls_id)]
                    detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': cls_name
                    })
        
        return detections
        
    def visualize(self, img, detections, line_thickness=2, font_size=1):
        """
        在图像上可视化检测结果
        参数:
            img: 原始图像 (numpy数组或PIL图像)
            detections: 检测结果
            line_thickness: 边界框线宽
            font_size: 字体大小
        返回:
            标注后的图像
        """
        # 确保输入是PIL图像
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        else:
            img_pil = img.copy()
            
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制每个检测结果
        for det in detections:
            # 提取信息
            box = det['box']
            cls_name = det['class_name']
            conf = det['confidence']
            color = self.colors.get(cls_name, (255, 255, 255))
            
            # RGB转BGR (如果需要)
            rgb_color = tuple(color)
            
            # 绘制边界框
            draw.rectangle(box, outline=rgb_color, width=line_thickness)
            
            # 绘制标签背景
            label = f"{cls_name} {conf:.2f}"
            text_size = draw.textlength(label)
            draw.rectangle(
                [box[0], box[1]-20, box[0]+text_size+5, box[1]],
                fill=rgb_color
            )
            
            # 绘制文本
            draw.text((box[0], box[1]-20), label, fill=(255, 255, 255))
        
        return img_pil 