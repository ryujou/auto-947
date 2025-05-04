import cv2
import numpy as np
from .config import INPAINT_RADIUS, INPAINT_METHOD

def remove_face_parts_by_inpaint(image, eyes_boxes, mouth_box):
    """
    对 image 上的眼睛和嘴巴区域做 inpaint 修复。
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # 标记眼睛和嘴巴的 mask 区域
    for box in eyes_boxes:
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    if mouth_box:
        mx1, my1, mx2, my2 = mouth_box
        cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)

    method_flag = cv2.INPAINT_TELEA if INPAINT_METHOD == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, mask, INPAINT_RADIUS, method_flag)

def replace_with_overlay(roi, overlay):
    """
    将带 alpha 通道的 overlay 融合到 roi。
    """
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        alpha = a.astype(float) / 255.0
        dst = roi.astype(float)
        for c, ch in enumerate((b, g, r)):
            dst[:, :, c] = alpha * ch + (1 - alpha) * dst[:, :, c]
        return dst.astype(np.uint8)
    else:
        return overlay
