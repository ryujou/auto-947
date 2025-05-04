import io
import base64
from PIL import Image, ImageSequence
import httpx
from nonebot import on_regex, logger
from nonebot.matcher import Matcher
from nonebot.adapters.onebot.v11 import MessageSegment, MessageEvent
from nonebot_plugin_waiter import waiter

API_URL = "http://127.0.0.1:8000/swap"  # FastAPI 服务地址
DEFAULT_CONF = "0.5"
WAIT_TIMEOUT = 30  # 等待用户发送图片的超时时间（秒）


swap = on_regex(r"^鬼图$", priority=5, block=True)


async def extract_images(event: MessageEvent) -> list[MessageSegment]:
    """
    从当前消息中提取所有 image segment。
    """
    return [seg for seg in event.message if seg.type == "image"]


async def get_image_segment(matcher: Matcher, event: MessageEvent) -> MessageSegment:
    """
    如果当前 event 包含图片则直接返回，
    否则等待用户在 WAIT_TIMEOUT 秒内发送下一条带图消息。
    """
    imgs = await extract_images(event)
    if imgs:
        return imgs[0]

    # 没有图片就让用户在超时内补发
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
    下载图片，失败时抛出异常。
    """
    url = seg.data.get("url")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


async def call_swap_api(client: httpx.AsyncClient, image_bytes: bytes) -> bytes:
    """
    将单张 JPEG bytes 上传到后端，返回处理后 bytes，失败时抛出异常。
    """
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data = {"conf": DEFAULT_CONF}
    resp = await client.post(API_URL, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.content


async def process_and_send(matcher: Matcher, content: bytes):
    """
    根据内容是否 GIF 分支处理并发送，所有关键步骤都在外层 catch。
    """
    is_gif = content[:6] in (b"GIF87a", b"GIF89a")
    async with httpx.AsyncClient() as client:
        if is_gif:
            # GIF 逐帧处理
            try:
                gif = Image.open(io.BytesIO(content))
            except Exception:
                await matcher.finish("无法解析 GIF 文件。")
            frames_out, durations = [], []
            loop = gif.info.get("loop", 0)
            for frame in ImageSequence.Iterator(gif):
                buf = io.BytesIO()
                frame.convert("RGB").save(buf, format="JPEG")
                out_bytes = await call_swap_api(client, buf.getvalue())
                try:
                    out_frame = Image.open(io.BytesIO(out_bytes)).convert("RGB")
                except Exception:
                    await matcher.finish("处理后的 GIF 帧解码失败。")
                frames_out.append(out_frame)
                durations.append(frame.info.get("duration", 100))
            # 合成并发送 GIF
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
            # 静态图处理
            try:
                pil = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception:
                await matcher.finish("无法解析图片。")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            out_bytes = await call_swap_api(client, buf.getvalue())
            b64 = base64.b64encode(out_bytes).decode()
            await matcher.finish(MessageSegment.image(f"base64://{b64}"))


@swap.handle()
async def _(matcher: Matcher, event: MessageEvent):
    try:
        # 1. 统一提取或等待图片
        seg = await get_image_segment(matcher, event)

        # 2. 下载
        content = await download_image(seg)

        # 3. 处理并发送
        await process_and_send(matcher, content)

    except httpx.HTTPError as e:
        logger.exception(e)
        await matcher.finish(f"网络或下载出错：{e}")
    except Exception as e:
        logger.exception(e)
        await matcher.finish("未知错误，请稍后重试。")
