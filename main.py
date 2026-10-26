"""
AstrBot 插件：Mementy - Agent 长期记忆（艾宾浩斯间隔重复）

核心机制:
    - on_llm_request: 在 LLM 请求前，将到期记忆注入 extra_user_content_parts
    - on_llm_response: 在 LLM 响应后，异步调用 LLM 提取新记忆

记忆调度:
    - 使用 SM-2 算法（艾宾浩斯遗忘曲线的工程实现）
    - 新记忆: 30 分钟后第一次复习
    - 成功复习: 间隔逐步放大 (1d -> 3d -> 7d -> 15d -> 30d)
    - 回忆失败: 重置为 30 分钟
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, List, Optional

from astrbot.api import AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api.star import Context, Star

logger = logging.getLogger("astrbot_plugin_mementy")

PLUGIN_NAME = "astrbot_plugin_mementy"


class MementyPlugin(Star):
    """基于艾宾浩斯间隔重复的 Agent 长期记忆插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)

        # AstrBot 原生配置注入（AstrBotConfig 继承自 dict）
        self.config = config

        # 存储层
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", PLUGIN_NAME
        )
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "memories.db")

        from .storage import MemoryStorage
        self.storage = MemoryStorage(db_path)

        # 提取器
        from .extractor import MemoryExtractor
        self.extractor = MemoryExtractor(
            context=self.context,
            storage=self.storage,
            enabled=self.config.get("extraction_enabled", True),
            llm_provider_id=self.config.get("extraction_provider_id") or None,
            max_items_per_extract=self.config.get("max_items_per_extract", 5),
            max_conversation_turns=self.config.get("max_conversation_turns", 10),
        )

        logger.info("[mementy] 插件初始化完成，数据库: %s", db_path)

    # ==================================================================
    # 事件钩子：LLM 请求前 → 注入到期记忆
    # ==================================================================
    @filter.on_llm_request()
    async def inject_due_memories(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        """
        在 LLM 请求前，查找到期需要复习的记忆，注入到 extra_user_content_parts。

        使用 TextPart 包裹，放入 extra_user_content_parts 而不是修改 system_prompt，
        避免破坏 prompt 缓存（官方文档强调此举可避免 7-20 倍成本增加）。
        """
        try:
            user_id = self._get_user_id(event)
            if not user_id:
                return

            max_injected = int(self.config.get("max_memories_injected", 8))
            min_importance = float(self.config.get("min_importance_threshold", 0.3))

            # 1. 查找到期记忆
            due = self.storage.query_due(user_id)
            due = [m for m in due if m.importance >= min_importance][:max_injected]

            # 2. 兜底：如果到期记忆不够，补一些近期高重要性的
            if len(due) < max_injected:
                recent = self.storage.query_recent(user_id, limit=max_injected * 2)
                recent = [
                    m for m in recent
                    if m.importance >= min_importance
                    and (not due or m.id not in {d.id for d in due})
                ]
                due.extend(recent[: max_injected - len(due)])

            if not due:
                return

            # 3. 构造记忆块
            memory_lines = ["以下是我长期记住的关于你的内容，结合上下文参考："]
            for m in due:
                memory_lines.append(f"- {m.content}")

            memory_block = "\n".join(memory_lines)

            # 4. 注入
            from astrbot.core.agent.message import TextPart
            req.extra_user_content_parts.append(
                TextPart(
                    text=(
                        "<long_term_memories>\n"
                        f"{memory_block}\n"
                        "</long_term_memories>"
                    )
                ).mark_as_temp()
            )

            logger.debug(
                "[mementy] 为 user=%s 注入 %d 条记忆", user_id, len(due)
            )

            # 5. 标记这些记忆为"已复习"，更新 SM-2 调度
            await self._mark_reviewed(due, quality=4)

        except Exception as exc:
            logger.error("[mementy] 注入记忆异常: %s", exc)

    async def _mark_reviewed(
        self,
        memories: list,
        quality: int = 4,
    ):
        """
        标记一批记忆为已复习，更新 SM-2 调度。
        quality 默认给 4（有点困难但成功想起）—— 被召回即视为成功。
        """
        try:
            from .ebbinghaus import update_schedule
            for m in memories:
                if m.schedule is None:
                    continue
                new_state = update_schedule(m.schedule, quality=quality)
                self.storage.update_schedule(m.id, new_state)
        except Exception as exc:
            logger.error("[mementy] 更新复习调度失败: %s", exc)

    # ==================================================================
    # 事件钩子：LLM 响应后 → 异步提取新记忆
    # ==================================================================
    @filter.on_llm_response()
    async def extract_new_memories(
        self, event: AstrMessageEvent, resp: LLMResponse
    ):
        """
        在 LLM 响应完成后，异步触发记忆提取。
        fire-and-forget，不阻塞主流程。
        """
        if not self.config.get("extraction_enabled", True):
            return

        try:
            user_id = self._get_user_id(event)
            session_id = self._get_session_id(event)
            if not user_id:
                return

            user_msg = event.message_str or ""
            assistant_resp = getattr(resp, "completion_text", "") or str(resp)

            # 获取 provider_id
            provider_id = None
            try:
                pid = await self.context.get_current_chat_provider_id(
                    umo=getattr(event, "unified_msg_origin", None)
                )
                provider_id = pid
            except Exception:
                pass

            # 获取对话历史（多轮上下文提升提取质量）
            conversation_history = await self._get_recent_conversation(event)

            asyncio.create_task(
                self.extractor.extract_and_store(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_msg,
                    assistant_response=assistant_resp,
                    conversation_history=conversation_history,
                    provider_id=provider_id,
                )
            )

        except Exception as exc:
            logger.error("[mementy] 触发提取异常: %s", exc)

    async def _get_recent_conversation(
        self, event: AstrMessageEvent
    ) -> Optional[List[Dict[str, str]]]:
        """
        尝试从 AstrBot 会话管理器获取最近的对话历史。
        返回 List[{'role': 'user'|'assistant', 'content': str}] 或 None。
        """
        try:
            umo = getattr(event, "unified_msg_origin", None)
            if not umo:
                return None

            conv_mgr = getattr(self.context, "conversation_manager", None)
            if not conv_mgr:
                return None

            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if not curr_cid:
                return None

            conversation = await conv_mgr.get_conversation(umo, curr_cid)
            if conversation is None:
                return None

            # conversation.history 是一个字符串（AstrBot 内部格式），
            # 这里尝试简单解析为 messages 列表。
            raw = getattr(conversation, "history", "")
            if not raw:
                return None

            # 尝试 JSON 解析（新版 AstrBot 可能存为 JSON）
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        role = item.get("role", "user")
                        content = item.get("content", "")
                        if content:
                            result.append({"role": role, "content": str(content)})
                    if result:
                        return result[-self.config.get("max_conversation_turns", 10):]
            except (json.JSONDecodeError, AttributeError):
                pass

            # 兜底：返回 None，让 extractor 只用当前轮次
            return None

        except Exception as exc:
            logger.debug("[mementy] 获取对话历史失败 (可忽略): %s", exc)
            return None

    # ==================================================================
    # 用户指令
    # ==================================================================
    @filter.command_group("mementy")
    def mementy_group(self):
        """记忆管理指令组。"""
        pass

    @mementy_group.command("help")
    async def mementy_help(self, event: AstrMessageEvent):
        """帮助信息。"""
        yield event.plain_result(
            "📝 Mementy 记忆指令:\n"
            "/mementy list    - 查看我记住的内容\n"
            "/mementy count   - 查看记忆统计\n"
            "/mementy clear   - 清空所有记忆 (管理员)\n"
            "/mementy help    - 查看帮助"
        )

    @mementy_group.command("list")
    async def mementy_list(self, event: AstrMessageEvent):
        """查看当前用户的记忆列表。"""
        user_id = self._get_user_id(event)
        if not user_id:
            yield event.plain_result("无法识别你的身份。")
            return

        records = self.storage.query_by_user(user_id, limit=30)
        if not records:
            yield event.plain_result("还没有记住任何东西~")
            return

        lines = [f"📝 记住了 {len(records)} 条内容:"]
        for i, r in enumerate(records, 1):
            tags_str = ", ".join(r.tags) if r.tags else "-"
            lines.append(
                f"{i}. {r.content} [重要性 {r.importance:.1f}]"
                f"\n   标签: {tags_str}"
            )
        yield event.plain_result("\n".join(lines))

    @mementy_group.command("count")
    async def mementy_count(self, event: AstrMessageEvent):
        """查看记忆统计。"""
        user_id = self._get_user_id(event)
        if not user_id:
            yield event.plain_result("无法识别你的身份。")
            return

        active = self.storage.count(user_id, status="active")
        forgotten = self.storage.count(user_id, status="forgotten")
        due = len(self.storage.query_due(user_id))
        yield event.plain_result(
            f"📊 记忆统计:\n"
            f"  活跃记忆: {active}\n"
            f"  已遗忘:   {forgotten}\n"
            f"  到期待复习: {due}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mementy_group.command("clear")
    async def mementy_clear(self, event: AstrMessageEvent):
        """清空当前用户的所有记忆。"""
        user_id = self._get_user_id(event)
        if not user_id:
            yield event.plain_result("无法识别用户。")
            return

        conn = self.storage._conn
        conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        yield event.plain_result(f"已清空 user={user_id} 的所有记忆。")

    # ==================================================================
    # 内部工具
    # ==================================================================
    @staticmethod
    def _get_user_id(event: AstrMessageEvent) -> str:
        """从事件中获取稳定的用户 ID。"""
        try:
            sender = getattr(event.message_obj, "sender", None)
            if sender:
                uid = getattr(sender, "user_id", None)
                if uid:
                    return str(uid)
        except Exception:
            pass

        try:
            umo = getattr(event, "unified_msg_origin", None)
            if umo:
                return str(umo)
        except Exception:
            pass

        return ""

    @staticmethod
    def _get_session_id(event: AstrMessageEvent) -> str:
        try:
            sid = getattr(event.message_obj, "session_id", None)
            if sid:
                return str(sid)
        except Exception:
            pass
        return ""
