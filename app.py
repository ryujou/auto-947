# app.py

import os
import io
import zipfile
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageSequence
import gradio as gr
from face_swap.core import swap_face_parts_api

def process_files(filepaths, conf):
    """
    接收用户上传的多文件（图片/GIF），返回单个 GIF 或者 ZIP 文件路径。
    """
    tmpdir = tempfile.mkdtemp()
    out_paths = []

    # Gradio 会把多个文件传成列表；如果只有一个文件，也会是列表
    for file_path in filepaths:
        _, ext = os.path.splitext(file_path.lower())

        # 处理 GIF
        if ext == ".gif":
            gif = Image.open(file_path)
            frames_out = []
            for frame in ImageSequence.Iterator(gif):
                rgb = frame.convert("RGB")
                arr = np.array(rgb)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                out_bgr = swap_face_parts_api(bgr, conf_threshold=conf)
                out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                frames_out.append(Image.fromarray(out_rgb))

            out_gif_path = os.path.join(tmpdir, os.path.basename(file_path))
            frames_out[0].save(
                out_gif_path,
                save_all=True,
                append_images=frames_out[1:],
                loop=0
            )
            out_paths.append(out_gif_path)

        # 处理静态图
        else:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            out_bgr = swap_face_parts_api(img_bgr, conf_threshold=conf)
            out_path = os.path.join(tmpdir, os.path.basename(file_path))
            cv2.imwrite(out_path, out_bgr)
            out_paths.append(out_path)

    # 如果只有一个 GIF，直接返回它
    if len(out_paths) == 1 and out_paths[0].lower().endswith(".gif"):
        return out_paths[0]

    # 否则打包成 ZIP
    zip_path = os.path.join(tmpdir, "results.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in out_paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path

with gr.Blocks() as demo:
    gr.Markdown("## 自动947（支持 GIF & 批量）")

    with gr.Row():
        file_input = gr.File(
            file_count="multiple",
            type="filepath",
            label="上传图片/GIF（可多选）"
        )
        result_file = gr.File(
            type="filepath",
            label="下载处理结果（GIF 或 ZIP）"
        )

    with gr.Row():
        conf = gr.Slider(
            minimum=0.1, maximum=1.0, step=0.1, value=0.5,
            label="YOLO 检测置信度阈值"
        )
        run_btn = gr.Button("运行")

    run_btn.click(
        fn=process_files,
        inputs=[file_input, conf],
        outputs=result_file
    )

if __name__ == "__main__":
    demo.launch()
