"""
LLM 记忆提取器：从对话中异步提取值得长期记忆的内容。

工作流程:
    1. LLM 响应完成后，异步触发提取（fire-and-forget，不阻塞用户回复）
    2. 调用 LLM，把最近几轮对话喂进去，让它输出 JSON 格式的记忆条目
    3. 解析 JSON，调用 storage.add_memory 写入
    4. 每条新记忆使用 SM-2 的初始调度（30 分钟后复习）

提取目标:
    - 用户的稳定偏好（喜欢/讨厌的东西）
    - 重要的个人信息（职业、生日、项目代号等）
    - 关键决策和约定
    - 用户明确告诉助手"记住"的内容

不提取:
    - 一次性问答中的信息（除非用户强调要记住）
    - 临时上下文（当前对话的中间产物）
    - 助手的回复内容
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import MemoryStorage

logger = logging.getLogger("astrbot_plugin_mementy")


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
你是一个记忆提取器，负责从对话中提取值得长期记住的信息。

请从下面的对话历史中，提取出"长期有价值"的记忆条目。判断标准：
✅ 用户的稳定偏好（喜欢/讨厌的东西、习惯、风格）
✅ 用户的重要个人信息（职业、项目代号、纪念日、住址等）
✅ 用户明确说"记住"、"别忘了"、"以后用到"的内容
✅ 关键决策和约定（如"下周我们一起做X"）
❌ 一次性问答中的临时信息
❌ 助手的回复内容
❌ 当前对话的中间产物（如代码片段、计算过程）

对每条记忆，请输出:
- content: 简洁的记忆内容（不超过100字），用第一人称"我"或"用户"描述事实
- tags: 相关标签数组，用于分类（如 ["偏好", "饮食"], ["项目", "代号"]）
- importance: 0~1 的重要性评分

只输出 JSON，不要输出任何解释文字。格式:
```json
[
  {
    "content": "用户喜欢喝冰美式，不加糖",
    "tags": ["偏好", "饮食"],
    "importance": 0.8
  }
]
```

对话历史:
{conversation}
"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    从 LLM 响应中提取 JSON 数组。
    支持 ```json ... ``` 包裹，或裸 JSON。
    """
    # 尝试找 markdown 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = match.group(1) if match else text

    # 找第一个 [ ... ] 数组
    bracket_match = re.search(r"\[[\s\S]*\]", candidate)
    if not bracket_match:
        return None

    try:
        return json.loads(bracket_match.group())
    except json.JSONDecodeError:
        logger.warning("[mementy] 解析 LLM 提取的 JSON 失败: %s", text[:200])
        return None


def _validate_items(items: Any) -> List[Dict[str, Any]]:
    """校验并过滤 LLM 输出的条目。"""
    if not isinstance(items, list):
        return []

    valid: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content or len(content) > 200:
            continue
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        importance = float(item.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))
        valid.append({
            "content": content,
            "tags": [str(t)[:20] for t in tags if str(t).strip()][:5],
            "importance": importance,
        })
    return valid


# ---------------------------------------------------------------------------
# 提取器类
# ---------------------------------------------------------------------------
class MemoryExtractor:
    """异步记忆提取器。"""

    def __init__(
        self,
        context: Any,
        storage: "MemoryStorage",
        enabled: bool = True,
        llm_provider_id: Optional[str] = None,
        max_items_per_extract: int = 5,
        max_conversation_turns: int = 10,
    ):
        self.context = context
        self.storage = storage
        self.enabled = enabled
        self.llm_provider_id = llm_provider_id
        self.max_items_per_extract = max_items_per_extract
        self.max_conversation_turns = max_conversation_turns

        # 防重复：记录最近处理过的用户+消息哈希，避免同一轮多次触发
        self._processed: set = set()
        self._processed_max = 1000

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    async def extract_and_store(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        provider_id: Optional[str] = None,
    ):
        """
        从本轮对话中提取记忆并存储。
        fire-and-forget：异常会被捕获并记录，不影响主流程。
        """
        if not self.enabled:
            return

        # 简单的去重 key
        dedupe_key = f"{user_id}|{user_message[:50]}|{assistant_response[:50]}"
        if dedupe_key in self._processed:
            return
        self._processed.add(dedupe_key)
        if len(self._processed) > self._processed_max:
            self._processed.clear()

        try:
            items = await self._call_llm_extract(
                user_message, assistant_response, conversation_history, provider_id
            )
            if not items:
                logger.debug("[mementy] 本轮未提取到新记忆 (user=%s)", user_id)
                return

            stored = 0
            for item in items[: self.max_items_per_extract]:
                try:
                    self.storage.add_memory(
                        user_id=user_id,
                        session_id=session_id,
                        content=item["content"],
                        tags=item["tags"],
                        importance=item["importance"],
                    )
                    stored += 1
                except Exception as exc:
                    logger.error("[mementy] 写入记忆失败: %s", exc)

            logger.info(
                "[mementy] 提取到 %d 条记忆，写入 %d 条 (user=%s)",
                len(items), stored, user_id,
            )
        except Exception as exc:
            logger.error("[mementy] 记忆提取异常: %s", exc)

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------
    async def _call_llm_extract(
        self,
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]],
        provider_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """调用 LLM 执行提取。"""
        # 组装对话历史文本
        lines: List[str] = []
        if conversation_history:
            for msg in conversation_history[-self.max_conversation_turns:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                role_label = "用户" if role == "user" else "助手"
                lines.append(f"{role_label}: {content}")
        # 加入本轮
        lines.append(f"用户: {user_message}")
        lines.append(f"助手: {assistant_response}")
        conversation_text = "\n".join(lines)

        prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

        # 获取 provider
        pid = provider_id or self.llm_provider_id
        if not pid:
            try:
                pid = await self.context.get_current_chat_provider_id()
            except Exception:
                logger.warning("[mementy] 无法获取 LLM provider，跳过提取")
                return []

        # 调用 LLM
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=pid,
                prompt=prompt,
            )
            raw_text = getattr(resp, "completion_text", str(resp))
        except Exception as exc:
            logger.error("[mementy] LLM 调用失败: %s", exc)
            return []

        parsed = _extract_json(raw_text)
        if parsed is None:
            return []
        return _validate_items(parsed)
