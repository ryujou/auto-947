import io
import base64
from PIL import Image, ImageSequence
import httpx
from nonebot import on_regex, logger
from nonebot.exception import FinishedException
from nonebot.matcher import Matcher
from nonebot.adapters.onebot.v11 import MessageSegment, MessageEvent
from nonebot_plugin_waiter import waiter

API_URL = "http://127.0.0.1:8880/swap"  # 后端 FastAPI 服务地址
DEFAULT_CONF = "0.5"                   # 默认置信度参数
WAIT_TIMEOUT = 30                      # 等待用户发送图片的超时时间（秒）

swap = on_regex(r"^鬼图$", priority=5, block=True)


async def extract_images(event: MessageEvent) -> list[MessageSegment]:
    """
    从当前 MessageEvent 中提取所有 image segment。
    """
    return [seg for seg in event.message if seg.type == "image"]


async def get_image_segment(matcher: Matcher, event: MessageEvent) -> MessageSegment:
    """
    如果当前 event 已包含图片，则直接返回第一张；
    否则启动 waiter，等待用户在 WAIT_TIMEOUT 秒内发送新消息，若超时或未发送图片则取消操作。
    """
    imgs = await extract_images(event)
    if imgs:
        return imgs[0]

    @waiter(waits=["message"], keep_session=True)
    async def _waiter(next_event: MessageEvent):
        return next_event

    resp = await _waiter.wait(
        f"请在 {WAIT_TIMEOUT} 秒内发送你要处理的图片，超时将取消操作",
        timeout=WAIT_TIMEOUT
    )
    if not resp:
        await matcher.finish("操作超时，已取消。")
    imgs = await extract_images(resp)
    if not imgs:
        await matcher.finish("未检测到图片，已取消。")
    return imgs[0]


async def download_image(seg: MessageSegment) -> bytes:
    """
    下载图片内容，HTTP 状态非 200 或网络错误时抛出 httpx.HTTPError。
    """
    url = seg.data.get("url")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


async def call_swap_api(client: httpx.AsyncClient, image_bytes: bytes) -> bytes:
    """
    将 JPEG bytes 上传到后端换脸接口，返回处理后的 bytes；异常时抛出 httpx.HTTPError。
    """
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data = {"conf": DEFAULT_CONF}
    resp = await client.post(API_URL, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.content


async def process_and_send(matcher: Matcher, content: bytes):
    """
    根据内容是否为 GIF，选择逐帧或静态处理流程，最后调用 matcher.finish 发送结果。
    """
    is_gif = content[:6] in (b"GIF87a", b"GIF89a")
    async with httpx.AsyncClient() as client:
        if is_gif:
            # 处理多帧 GIF
            try:
                pil_gif = Image.open(io.BytesIO(content))
            except Exception:
                await matcher.finish("无法解析 GIF 文件。")
            frames_out, durations = [], []
            loop = pil_gif.info.get("loop", 0)

            for frame in ImageSequence.Iterator(pil_gif):
                buf = io.BytesIO()
                frame.convert("RGB").save(buf, format="JPEG")
                try:
                    out_bytes = await call_swap_api(client, buf.getvalue())
                except httpx.HTTPError as e:
                    await matcher.finish(f"处理帧失败：{e}")
                try:
                    out_frame = Image.open(io.BytesIO(out_bytes)).convert("RGB")
                except Exception:
                    await matcher.finish("处理后的 GIF 帧解码失败。")
                frames_out.append(out_frame)
                durations.append(frame.info.get("duration", 100))

            # 合成输出 GIF
            out_buf = io.BytesIO()
            frames_out[0].save(
                out_buf,
                format="GIF",
                save_all=True,
                append_images=frames_out[1:],
                duration=durations,
                loop=loop,
                disposal=2
            )
            b64 = base64.b64encode(out_buf.getvalue()).decode()
            await matcher.finish(MessageSegment.image(f"base64://{b64}"))

        else:
            # 处理静态图片
            try:
                pil = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception:
                await matcher.finish("无法解析图片。")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            try:
                out_bytes = await call_swap_api(client, buf.getvalue())
            except httpx.HTTPError as e:
                await matcher.finish(f"处理失败：{e}")
            b64 = base64.b64encode(out_bytes).decode()
            await matcher.finish(MessageSegment.image(f"base64://{b64}"))


@swap.handle()
async def _(matcher: Matcher, event: MessageEvent):
    try:
        # 1. 获取或等待图片
        seg = await get_image_segment(matcher, event)

        # 2. 下载图片字节
        content = await download_image(seg)

        # 3. 调用处理并发送结果
        await process_and_send(matcher, content)

    except FinishedException:
        # 透传框架的 finish 结束信号
        raise

    except httpx.HTTPError as e:
        # 网络或接口调用错误
        logger.exception(e)
        await matcher.finish(f"网络或下载出错：{e}")

    except Exception as e:
        # 未知异常
        logger.exception(e)
        await matcher.finish("未知错误，请稍后重试。")
