import cv2
import numpy as np
from .model import get_yolo_model
from .utils import remove_face_parts_by_inpaint, replace_with_overlay
from .config import CLASS_FACE, CLASS_EYES, CLASS_MOUTH, LEFT_EYE_PATH, RIGHT_EYE_PATH, MOUTH_PATH

def swap_face_parts_api(
    image_bgr: np.ndarray,
    conf_threshold: float = 0.5
) -> np.ndarray:
    """
    核心 API：输入 BGR 图像，返回处理后 BGR 图像。
    """
    model = get_yolo_model()
    h, w = image_bgr.shape[:2]
    results = model.predict(source=image_bgr, conf=conf_threshold, save=False, show=False)[0].boxes

    # 收集 bbox
    face_box = None
    eyes_boxes = []
    mouth_box = None
    for box in results:
        cls = int(box.cls.cpu().numpy())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        if cls == CLASS_FACE:
            face_box = (x1, y1, x2, y2)
        elif cls == CLASS_EYES:
            eyes_boxes.append((x1, y1, x2, y2))
        elif cls == CLASS_MOUTH:
            mouth_box = (x1, y1, x2, y2)

    if face_box is None:
        return image_bgr  # 未检测到人脸

    # 判断左右眼（以人脸中线为分界）
    fx1, fy1, fx2, fy2 = face_box
    face_cx = (fx1 + fx2) / 2
    left, right = None, None
    for eb in eyes_boxes:
        ex1, ey1, ex2, ey2 = eb
        ecx = (ex1 + ex2) / 2
        if ecx < face_cx:
            left = eb
        else:
            right = eb

    # 1) 去除五官并 inpaint
    cleaned = remove_face_parts_by_inpaint(image_bgr.copy(), [left, right], mouth_box)

    # 2) 载入替换图
    left_img  = cv2.imread(LEFT_EYE_PATH,  cv2.IMREAD_UNCHANGED)
    right_img = cv2.imread(RIGHT_EYE_PATH, cv2.IMREAD_UNCHANGED)
    mouth_img = cv2.imread(MOUTH_PATH,     cv2.IMREAD_UNCHANGED)

    out = cleaned.copy()
    # 批量粘贴
    for box, patch in [(left,  left_img),
                       (right, right_img),
                       (mouth_box, mouth_img)]:
        if box and patch is not None:
            x1, y1, x2, y2 = box
            p = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = replace_with_overlay(out[y1:y2, x1:x2], p)

    return out
