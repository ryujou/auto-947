# test_api.py

import os
import requests

def test_swap_api(
    server_url="http://127.0.0.1:8000/swap",
    input_path="input/test.jpg",
    output_path="output/test_result.jpg",
    conf=0.5
):
    """
    测试 /swap 接口：
    - server_url: FastAPI 服务地址
    - input_path: 待测试的输入图片路径
    - output_path: 保存返回图片的路径
    - conf: YOLO 置信度阈值
    """
    if not os.path.isfile(input_path):
        print(f"ERROR: 输入文件不存在：{input_path}")
        return

    # 打开文件并发起 POST 请求
    with open(input_path, "rb") as f:
        files = {"file": (os.path.basename(input_path), f, "image/jpeg")}
        data = {"conf": str(conf)}
        print(f"Uploading {input_path} to {server_url} (conf={conf}) ...")
        resp = requests.post(server_url, files=files, data=data)

    # 检查状态码
    if resp.status_code != 200:
        print(f"Request failed: {resp.status_code}\n{resp.text}")
        return

    # 保存返回的图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as out_f:
        out_f.write(resp.content)
    print(f"Success! Result saved to {output_path}")

if __name__ == "__main__":
    # 1. 启动你的 FastAPI 服务 (server.py)，确保在 8000 端口运行
    #    uvicorn server:app --reload
    #
    # 2. 准备一张测试图 input/test.jpg
    # 3. 运行此脚本：
    #    python test_api.py

    test_swap_api(
        server_url="http://127.0.0.1:8000/swap",
        input_path="input/test.jpg",
        output_path="output/test_result.jpg",
        conf=0.5
    )
