# 全自动947鬼图

这是一个**支持批量生成947鬼图的**的开源项目，结合了 YOLOv11 检测与 OpenCV Inpainting 技术，具备图形界面和 HTTP API 双重形态。你可以通过 Web 上传图片 / GIF 轻松生成947鬼图，也可以作为服务集成到你的系统中。

---

## 🚀 使用方法

### 1. 克隆仓库并进入目录
   git clone https://github.com/ryujou/auto-947.git
   cd face-swap-api

### 2. 创建并激活虚拟环境
   python3 -m venv venv
   source venv/bin/activate

### 3. 安装依赖
   pip install -r requirements.txt

### 4. 准备模型与资源
   - 将 best.pt（YOLOv11 动漫人脸检测模型）放入 static/best.pt
   - 确保以下替换图像在 static/ 目录中：
     static/947left.png
     static/947right.png
     static/947mouth.png

------------------------------------------------------------

## 📂 项目结构

```
face-swap-api/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                # Gradio Web 界面
├── server.py             # FastAPI HTTP API
├── test_api.py           # 本地测试 API 脚本
├── static/
│   ├── best.pt
│   ├── 947left.png
│   ├── 947right.png
│   └── 947mouth.png
└── face_swap/
    ├── __init__.py
    ├── config.py         # 路径与参数配置
    ├── model.py          # YOLO 加载
    ├── utils.py          # inpaint 与 overlay 函数
    └── core.py           # 主处理流程 API
```
------------------------------------------------------------

## 🧠 核心流程说明（见 core.py）：

1. 加载 YOLOv11 模型
2. 使用模型检测 face / eyes / mouth
3. 根据人脸中心判断左右眼 bbox
4. 使用 OpenCV inpaint 去除眼睛与嘴巴区域
5. 将替换图像（带 alpha 通道）贴到对应位置
6. 返回处理后的图像

------------------------------------------------------------

## 🚀运行方式：

### 方式一：启动 Gradio Web 界面
   ```bash
   python app.py
   ```
   浏览器访问 http://127.0.0.1:7860

### 方式二：启动 FastAPI HTTP API 服务
   ```bash
   python server.py
   ```
   - POST /swap     上传图片并返回处理后结果，参数：file=图片文件, conf=置信度
   - GET  /health   检查服务状态

### 测试接口：

 - 准备一张 input/test.jpg
 - 运行：
   ```bash
   python test_api.py
   ```
 - 默认输出为 output/test_result.jpg

------------------------------------------------------------

## 📦 依赖列表（requirements.txt）：

- ultralytics>=8.0.0
- opencv-python
- numpy
- gradio
- fastapi
- uvicorn[standard]
- Pillow
- requests

------------------------------------------------------------

## 📌 TODO

- [ ] 接入QQ机器人实现在群聊中自动生成鬼图
- [ ] 支持视频输入（逐帧处理 + 拼接）


------------------------------------------------------------

LICENSE

MIT License © 2024 ryujou
