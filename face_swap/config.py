import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# 把模型权重也放到 static 目录下
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(STATIC_DIR, "best.pt")

# 替换五官图像路径
LEFT_EYE_PATH  = os.path.join(STATIC_DIR, "947left.png")
RIGHT_EYE_PATH = os.path.join(STATIC_DIR, "947right.png")
MOUTH_PATH     = os.path.join(STATIC_DIR, "947mouth.png")

# 检测类别索引
CLASS_FACE  = 1
CLASS_EYES  = 0
CLASS_MOUTH = 2

# Inpaint 参数
INPAINT_RADIUS = 3
INPAINT_METHOD = "telea"  # 可选 "telea" 或 "ns"
