import io
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from face_swap.core import swap_face_parts_api

app = FastAPI(
    title="Face Swap API",
    description="HTTP API for removing facial features + Inpainting + overlay replacement",
    version="1.0"
)

@app.post("/swap", summary="上传图片并返回换脸结果")
async def swap_endpoint(
    file: UploadFile = File(..., description="要处理的图片文件"),
    conf: float = Form(0.5, description="YOLO 置信度阈值，0.1~1.0")
):
    # 1. 读取并验证上传的图片
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片")

    # 2. 调用核心换脸 API
    try:
        out_bgr = swap_face_parts_api(img, conf_threshold=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{e}")

    # 3. 编码为 JPEG 并返回
    success, encoded = cv2.imencode(".jpg", out_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="图像编码失败")
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


@app.get("/health", summary="健康检查")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # 在命令行直接运行 python server.py 即可启动 API 服务
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
