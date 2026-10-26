"""
基于 SM-2 算法的艾宾浩斯间隔重复调度器。

SM-2 是 SuperMemo 系列算法的第二个版本，已被 Anki 等广泛使用。
它根据每次复习的质量评分动态调整复习间隔和难度因子，
是艾宾浩斯遗忘曲线在工程上的成熟翻译。

核心公式:
    R(t) = e^(-t/S)     # 艾宾浩斯遗忘曲线
    next_interval = prev_interval * easiness_factor   # SM-2 调度规则

字段说明:
    repetitions       已成功复习次数
    easiness_factor   难度因子 (初始 2.5, 范围 [1.3, ∞))
    interval_days     当前复习间隔 (天)
    last_review_time  上次复习时间
    next_review_time  下次复习到期时间
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


# 艾宾浩斯经典复习节点（天）
EBBINGHAUS_INTERVALS_DAYS = [
    0.02,   # 约 30 分钟
    0.5,    # 12 小时
    1,      # 1 天
    2,      # 2 天
    4,      # 4 天
    7,      # 7 天
    15,     # 15 天
    30,     # 30 天
    60,     # 60 天
]

# 难度因子边界
MIN_EASINESS_FACTOR = 1.3
INITIAL_EASINESS_FACTOR = 2.5


@dataclass
class ScheduleState:
    """一条记忆的调度状态。"""

    repetitions: int = 0
    easiness_factor: float = INITIAL_EASINESS_FACTOR
    interval_days: float = 0.0
    last_review_time: Optional[datetime] = None
    next_review_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "repetitions": self.repetitions,
            "easiness_factor": self.easiness_factor,
            "interval_days": self.interval_days,
            "last_review_time": self.last_review_time.isoformat() if self.last_review_time else None,
            "next_review_time": self.next_review_time.isoformat() if self.next_review_time else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduleState":
        return cls(
            repetitions=data.get("repetitions", 0),
            easiness_factor=data.get("easiness_factor", INITIAL_EASINESS_FACTOR),
            interval_days=data.get("interval_days", 0.0),
            last_review_time=(
                datetime.fromisoformat(data["last_review_time"])
                if data.get("last_review_time")
                else None
            ),
            next_review_time=(
                datetime.fromisoformat(data["next_review_time"])
                if data.get("next_review_time")
                else None
            ),
        )


def compute_retention(time_elapsed_hours: float, stability_hours: float = 24.0) -> float:
    """
    根据艾宾浩斯遗忘曲线计算记忆保留率。

    R = e^(-t/S)
    R: 保留率 (0~1)
    t: 经过的时间 (小时)
    S: 记忆稳定性参数 (默认 24h, 即 24 小时后保留约 36%)
    """
    if time_elapsed_hours <= 0:
        return 1.0
    if stability_hours <= 0:
        return 0.0
    retention = math.exp(-time_elapsed_hours / stability_hours)
    return max(0.0, min(1.0, retention))


def initial_schedule(now: Optional[datetime] = None) -> ScheduleState:
    """创建新记忆的初始调度状态。"""
    now = now or datetime.now()
    # 第一次复习安排在约 30 分钟后
    interval = EBBINGHAUS_INTERVALS_DAYS[0]
    return ScheduleState(
        repetitions=0,
        easiness_factor=INITIAL_EASINESS_FACTOR,
        interval_days=interval,
        last_review_time=None,
        next_review_time=now + timedelta(days=interval),
    )


def update_schedule(
    state: ScheduleState,
    quality: int,
    now: Optional[datetime] = None,
) -> ScheduleState:
    """
    SM-2 算法核心：根据复习质量更新调度状态。

    Args:
        state: 当前调度状态
        quality: 复习质量评分 (0-5)
            5 = 完美回忆，毫不费力
            4 = 有点困难但回忆起来了
            3 = 困难但最终回忆起来
            2 = 错误，但看到答案后想起来了
            1 = 错误，答案似曾相识
            0 = 错误，完全不记得
        now: 当前时间（默认使用 datetime.now()）

    Returns:
        更新后的 ScheduleState
    """
    now = now or datetime.now()

    # Step 1: 更新难度因子
    # EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    new_ef = state.easiness_factor + (
        0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )
    new_ef = max(MIN_EASINESS_FACTOR, new_ef)

    # Step 2: 计算新间隔
    if quality < 3:
        # 回忆失败：重置复习次数，间隔回到第一个节点
        new_reps = 0
        new_interval = EBBINGHAUS_INTERVALS_DAYS[0]
    else:
        # 回忆成功
        new_reps = state.repetitions + 1
        if new_reps == 1:
            new_interval = 1.0  # 第一次成功复习 -> 1 天
        elif new_reps == 2:
            new_interval = 3.0  # 第二次成功 -> 3 天
        else:
            # 之后按 EF 增长
            new_interval = round(state.interval_days * new_ef)
            # 封顶保护
            new_interval = min(new_interval, EBBINGHAUS_INTERVALS_DAYS[-1] * 2)

    return ScheduleState(
        repetitions=new_reps,
        easiness_factor=new_ef,
        interval_days=new_interval,
        last_review_time=now,
        next_review_time=now + timedelta(days=new_interval),
    )


def is_due(state: ScheduleState, now: Optional[datetime] = None) -> bool:
    """判断一条记忆是否到期需要复习。"""
    now = now or datetime.now()
    if state.next_review_time is None:
        return True
    return now >= state.next_review_time


def hours_until_due(state: ScheduleState, now: Optional[datetime] = None) -> float:
    """距离下次到期还剩多少小时（负数表示已过期）。"""
    now = now or datetime.now()
    if state.next_review_time is None:
        return 0.0
    delta = state.next_review_time - now
    return delta.total_seconds() / 3600.0
