import os
import zipfile
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageSequence
import gradio as gr
from face_swap.core import swap_face_parts_api  

def process_files_for_preview(filepaths, conf):
    """
    批量处理输入图像/GIF，返回：[(预览图路径, 标签)...], 以及最终下载文件路径（zip或gif）
    """
    tmpdir = tempfile.mkdtemp()
    preview_images = []
    out_paths = []

    for file_path in filepaths:
        name = os.path.basename(file_path)
        _, ext = os.path.splitext(name.lower())

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

            out_gif_path = os.path.join(tmpdir, name)
            frames_out[0].save(
                out_gif_path,
                save_all=True,
                append_images=frames_out[1:],
                duration=gif.info.get('duration', 100),
                loop=0
            )
            out_paths.append(out_gif_path)
            preview_images.append((out_gif_path, f"🎞️ GIF - {name}"))

        # 处理静态图
        else:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            out_bgr = swap_face_parts_api(img_bgr, conf_threshold=conf)
            out_path = os.path.join(tmpdir, name)
            cv2.imwrite(out_path, out_bgr)
            out_paths.append(out_path)
            preview_images.append((out_path, f"🖼️ {name}"))

    # 单个 GIF 直接返回
    if len(out_paths) == 1 and out_paths[0].endswith(".gif"):
        return preview_images, out_paths[0]

    # 否则打包 ZIP
    zip_path = os.path.join(tmpdir, "results.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in out_paths:
            zf.write(p, arcname=os.path.basename(p))
    return preview_images, zip_path

# 启动 Gradio 页面
with gr.Blocks(css="#gallery img {height: 200px !important}") as demo:
    gr.Markdown("## 自动947")
    gr.Markdown("📂 支持上传图片或 GIF（可多选），右侧展示每个结果，支持动画预览。")

    with gr.Row():
        file_input = gr.File(
            file_count="multiple",
            type="filepath",
            label="📂 上传图片或GIF（可多选）"
        )
        with gr.Column():
            result_gallery = gr.Gallery(
                label="📸 结果预览（图/GIF）",
                show_label=True,
                columns=2,
                object_fit="contain"
            )
            result_file = gr.File(
                label="⬇️ 下载压缩结果文件",
                type="filepath"
            )

    with gr.Row():
        conf = gr.Slider(
            minimum=0.1, maximum=1.0, step=0.1, value=0.5,
            label="YOLO 检测置信度"
        )
        run_btn = gr.Button("🚀 运行")

    run_btn.click(
        fn=process_files_for_preview,
        inputs=[file_input, conf],
        outputs=[result_gallery, result_file]
    )

if __name__ == "__main__":
    # demo.launch()
    demo.launch(share=True)  # 如果需要公开访问，可以取消注释这一行
