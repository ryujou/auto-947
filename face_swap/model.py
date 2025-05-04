from ultralytics import YOLO
from .config import MODEL_PATH

_model = None

def get_yolo_model(force_reload: bool = False):
    """
    单例加载 YOLOv8 模型
    """
    global _model
    if _model is None or force_reload:
        _model = YOLO(MODEL_PATH)
    return _model
