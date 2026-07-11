from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def es_context(
    *,
    url: str = "",
    cloud_id: str = "",
    api_key: str = "",
    username: str = "",
    password: str = "",
):
    from elasticsearch import AsyncElasticsearch

    kwargs: dict = {"verify_certs": True, "request_timeout": 120, "max_retries": 3, "retry_on_timeout": True}
    if cloud_id:
        kwargs["cloud_id"] = cloud_id.strip()
    elif url:
        kwargs["hosts"] = [url.strip().rstrip("/")]
    else:
        raise ValueError("必须提供 es_cloud_id 或 es_url")
    if api_key:
        kwargs["api_key"] = api_key.strip()
    elif username and password:
        kwargs["basic_auth"] = (username.strip(), password.strip())
    client = AsyncElasticsearch(**kwargs)
    try:
        yield client
    finally:
        await client.close()


def run_async(coro):
    """在同步 CLI 上下文中执行协程。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop().run_until_complete(coro)
    raise RuntimeError("不能在已有事件循环中同步调用 ES 方法")


def response_body(response):
    return getattr(response, "body", response)
