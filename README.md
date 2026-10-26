# Mementy · Agent 长期记忆插件

> 仓库：<https://github.com/luoyu23333/dsh-memory>

> 基于**艾宾浩斯遗忘曲线**（SM-2 间隔重复算法）的 AstrBot 长期记忆插件。
> 让 Agent 记住重要的东西，并在"快忘掉之前"自动想起。

---

## 你好，记忆

普通 Agent 的对话窗口是有限的，关掉会话就忘了你是谁。
Mementy 让 Agent 拥有**跨会话、按科学节奏强化**的长期记忆：

- 用户说过的**偏好、习惯、个人信息**自动沉淀
- 每条记忆都有**下次到期复习时间**，在遗忘临界点被重新注入上下文
- 每次被成功召回 → 间隔放大 → 变成长久记忆；长期未被唤起 → 自然沉底

背后的核心是 19 世纪艾宾浩斯提出的遗忘曲线 + 1980 年代成熟的 SM-2 间隔重复算法。

---

## 工作原理

```
┌─────────────────────────────────────────────────────────┐
│                  LLM 请求前 (on_llm_request)            │
│                                                         │
│  查 SQLite → 到期记忆 + 近期高重要性记忆                  │
│      ↓                                                  │
│  构造 <long_term_memories> 块                           │
│      ↓                                                  │
│  req.extra_user_content_parts.append(TextPart).mark_as_temp() │
│      ↓                                                  │
│  标记"已复习" → SM-2 算出新的下次到期时间                │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
                   LLM 正常回复
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  LLM 响应后 (on_llm_response)           │
│                                                         │
│  asyncio.create_task() fire-and-forget                  │
│      ↓                                                  │
│  调 LLM：从对话历史里提取值得记住的条目                   │
│      ↓                                                  │
│  解析 JSON → 校验 → 存入 SQLite                          │
│      ↓                                                  │
│  初始 SM-2 调度：30 分钟后第一次复习                     │
└─────────────────────────────────────────────────────────┘
```

**为什么用 `extra_user_content_parts` 而不是 `system_prompt`？**
官方文档明确警告：每轮变化的内容塞进 `system_prompt` 会破坏服务端缓存，**成本和延迟暴增 7-20 倍**。`extra_user_content_parts` 才是动态上下文的正确位置。

---

## SM-2 调度算法（艾宾浩斯的工程化翻译）

### 遗忘曲线公式

```
R(t) = e^(-t/S)
```

| 时间 t | 保留率 R |
|--------|---------|
| 20 分钟 | ~60% |
| 1 小时 | ~45% |
| 9 小时 | ~36% |
| 1 天 | ~33% |
| 2 天 | ~28% |
| 6 天 | ~25% |
| 31 天 | ~21% |

### SM-2 调度规则

每条记忆维护 **4 个核心字段**：

| 字段 | 含义 | 初始值 |
|------|------|--------|
| `repetitions` | 成功复习次数 | 0 |
| `easiness_factor` | 难度因子 EF | 2.5（范围 [1.3, +∞)） |
| `interval_days` | 当前复习间隔（天） | 0.02（≈30 分钟） |
| `next_review_time` | 下次到期时间 | 创建后 30 分钟 |

每次复习后，根据质量评分 `quality`（0-5）更新：

```python
# 新的难度因子
EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
EF' = max(1.3, EF')

if q < 3:          # 回忆失败
    reps = 0
    interval = 30min
else:              # 回忆成功
    reps += 1
    if reps == 1:   interval = 1 天
    elif reps == 2: interval = 3 天
    else:           interval = interval * EF'
```

### 复习节奏一览

```
新记忆 ──30min──▶ 1天 ──3天──▶ 7天 ──15天──▶ 30天 ──60天──▶ ...
  │                │          │          │          │
  └── 忘了? 重置 ──┘          └── 忘了? 重置 ──┘
```

记忆越常被用到 → EF 越大 → 间隔越长 → 形成稳定的长期记忆。

---

## 目录结构

```
astrbot_plugin_mementy/
├── __init__.py             # 包标识（空文件即可）
├── main.py                 # 插件入口：Star 子类 + 事件钩子 + 用户指令
├── ebbinghaus.py           # SM-2 调度引擎（纯算法，无外部依赖）
├── storage.py              # SQLite 存储层（MemoryStorage + MemoryRecord + ScheduleState）
├── extractor.py            # LLM 异步记忆提取器
├── _conf_schema.json       # AstrBot 可视化配置 schema
├── metadata.yaml           # 插件市场元数据（可选）
└── requirements.txt        # 第三方依赖（本插件为空）
```

### 各模块职责

| 模块 | 做什么 | 依赖 |
|------|--------|------|
| `ebbinghaus.py` | SM-2 算法实现、保留率计算、调度状态数据类 | 仅 `datetime` / `math` / `dataclasses` |
| `storage.py` | SQLite CRUD、到期查询、线程安全连接、JSON 序列化 | 仅 `sqlite3` / `json` / 标准库 |
| `extractor.py` | 调 LLM → 解析 JSON → 校验 → 写入 storage | AstrBot `context.llm_generate` |
| `main.py` | 事件钩子（on_llm_request / on_llm_response）、指令组 | AstrBot API |

### 数据库表结构

```sql
CREATE TABLE memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,    -- 记忆的主人
    session_id      TEXT    NOT NULL,    -- 来源会话
    content         TEXT    NOT NULL,    -- 记忆内容（不超过 200 字）
    tags            TEXT    DEFAULT '[]', -- JSON 标签列表
    importance      REAL    DEFAULT 0.5, -- 重要性 0~1
    created_time    TEXT    NOT NULL,    -- ISO 格式
    schedule        TEXT    NOT NULL,    -- ScheduleState 的 JSON
    status          TEXT    DEFAULT 'active'  -- active / archived / forgotten
);

-- 索引
idx_memories_user      ON user_id
idx_memories_session   ON session_id
idx_memories_status    ON status
idx_memories_next_rev  ON (status, created_time)
```

---

## 部署

1. 把 `astrbot_plugin_mementy/` 整个目录放到 AstrBot 的插件目录下（通常是 `data/plugins/`）
2. 重启 AstrBot
3. 在 WebUI → 扩展 → Mementy 里配置参数（首次加载自动创建数据库 `data/plugin_data/astrbot_plugin_mementy/memories.db`）

### 前置条件

- AstrBot >= v4.5.7（使用了 `context.llm_generate`、`extra_user_content_parts`、`mark_as_temp()` 等新 API）
- 有可用的 LLM Provider（提取新记忆需要调用一次 LLM）

---

## 配置项

所有配置都在 WebUI 可视化管理，也可以手动改 `data/astrbot_plugin_mementy_config.json`。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `extraction_enabled` | bool | true | 是否自动提取新记忆 |
| `extraction_provider_id` | string | "" | 提取专用 Provider ID；留空用当前会话默认 |
| `max_items_per_extract` | int | 5 | 单次最多提取几条 |
| `max_conversation_turns` | int | 10 | 喂给提取 LLM 的对话历史轮数 |
| `max_memories_injected` | int | 8 | 每轮请求最多注入几条 |
| `min_importance_threshold` | float | 0.3 | 注入最低重要性阈值 |

### 推荐配置

- **轻度使用**（日常聊天，偶尔需要记住偏好）：默认值即可
- **重度使用**（Agent 需长期记住大量用户信息）：
  - `max_memories_injected` 调到 12-15
  - `min_importance_threshold` 保持 0.3 或略降
  - `max_conversation_turns` 调到 15-20

---

## 用户指令

所有指令在聊天里直接输入即可（需要 wake_prefix，默认 `/`）。

| 指令 | 权限 | 说明 |
|------|------|------|
| `/mementy help` | 所有人 | 帮助信息 |
| `/mementy list` | 所有人 | 查看我记住的内容（最多 30 条） |
| `/mementy count` | 所有人 | 记忆统计（活跃数 / 已遗忘 / 到期待复习） |
| `/mementy clear` | 管理员 | 清空当前用户所有记忆（**不可恢复**） |

---

## 提取 Prompt 里的判断标准

### ✅ 会被记住的

- 用户稳定偏好：喜欢/讨厌的东西、习惯、风格
- 重要个人信息：职业、项目代号、纪念日、住址
- 用户明确说过"记住""别忘了""以后用到"的内容
- 关键决策和约定

### ❌ 不会被记住的

- 一次性问答里的临时信息
- 助手的回复内容
- 当前对话中间产物（代码片段、计算过程）

如果你觉得"这个应该记住但没记住"，可以在消息里加上明确引导，比如：

> "**记住**：以后我都用 Python 3.11，不要用旧版本"

---

## 技术细节 & 设计决策

### 为什么是 SM-2 而不是原始遗忘曲线？

艾宾浩斯的实验数据只描述"不复习会怎样衰减"，没有给"复习后间隔怎么增长"的公式。SM-2 是 SuperMemo 系列（Anki、Quizlet 都用）验证了几十年的工程化版本，核心是**每次复习后用评分修正间隔**——刚好匹配 LLM 记忆场景：一条记忆这次被模型"成功想起"（在上下文中被召回），就给它一个 success 评分，自动放大下次间隔。

### 为什么不做语义相似度检索？

早期的 AI 记忆系统（向量库 + 相似度）容易遇到两个坑：

1. **时效性丢失**：三个月前的正确偏好和昨天刚纠正的旧信息，语义距离几乎相同
2. **上下文膨胀**：存得越多越乱，反而稀释了真正重要的信息

Mementy 用**时间调度信号优先**（到期才召回到上下文），相似度兜底（近期高重要性记忆补位），两条信号协同。

### 为什么提取用 `fire-and-forget`？

一次提取需要额外调一次 LLM，如果阻塞主流程，用户体验会差很多。提取对"必须每轮都完成"没有强一致性要求——这一轮没提取，下一轮还会触发，而且 LLM 提取本身有幂等性（相似内容大概率会合并到已有记忆里）。

### 存储层为什么用 SQLite 而不是向量库？

当前版本用 SM-2 时间调度做主动召回，核心查询是"按 user_id + next_review_time 取到期条目"——这是 SQLite B-tree 索引最擅长的场景。向量库要额外部署、额外依赖，对 V1 版本性价比不高。

未来如果需要做**语义相似度兜底**（当到期记忆不够时，用 embedding 找相关记忆），可以平滑升级——存储层抽象了 `MemoryStorage`，替换实现即可。

### 调度状态为什么存在 memory 表内？

因为 schedule 是**每条记忆的固有属性**——不同记忆的复习节奏可能完全不同。如果单独建 `review_log` 表再 join，不仅查询复杂，还会引入事务一致性问题。JSON 字段在 SQLite 里可以 `json_extract` 建索引，性能足够。

---

## 常见问题

### Q: 为什么 Agent 好像"记不住"？

1. 确认 `extraction_enabled` 没被关掉
2. 看日志里有没有 `[mementy] 提取到 X 条记忆`
3. 用 `/mementy list` 确认记忆是否真的存入
4. 新记忆默认 30 分钟后才会到期注入，第一轮对话看不到是正常的

### Q: 为什么 Agent 好像"记住了不该记住的"？

新提取的记忆重要性默认 0.5，可以用 `/mementy clear` 清理，或把 `min_importance_threshold` 调高。

### Q: 能跨用户隔离吗？

是的。`user_id` 是每条记忆的隔离主键，不同用户的记忆不会互相污染。

### Q: 能跨会话（session）记住吗？

是的。检索时优先按 `user_id` 查，session_id 是来源追溯字段，不影响召回。

### Q: 数据库在哪里？

`data/plugin_data/astrbot_plugin_mementy/memories.db`，SQLite WAL 模式，可直接复制备份。

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1.1 | 2026-10-26 | 合规化：数据目录改为 `data/plugin_data/astrbot_plugin_mementy`，统一使用 AstrBot logger，更新 README |
| 0.1.0 | 2026-10-26 | 初版：SM-2 调度 + SQLite + LLM 异步提取 + on_llm_request/response 钩子 |

---

## License

MIT
