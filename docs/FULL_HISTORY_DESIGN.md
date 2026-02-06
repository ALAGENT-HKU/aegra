# 完整历史保留方案设计

## 问题背景

DeepAgents 的 `SummarizationMiddleware` 在上下文接近限制时会永久替换旧消息为摘要：
- 默认配置：`trigger=("fraction", 0.85)`, `keep=("fraction", 0.10)` 
- 当上下文达到 85% 时触发，仅保留 10% 的最新消息
- 这导致前端无法恢复完整的 `thinkingSteps`、`tool_calls` 等历史

## 解决方案：消息归档 + Merge History API

### 架构概览

```
用户消息 ──▶ Agent ──▶ SummarizationMiddleware ──▶ Checkpoint (可能被压缩)
                │
                └──▶ MessageArchive (完整历史，独立存储)
                
前端恢复历史:
  GET /threads/{id}/full-history 
    ├── 从 MessageArchive 获取完整历史
    ├── 从 Checkpoint 获取当前 state
    └── Merge: 以 Archive 为主，用 Checkpoint 补充最新消息
```

### 核心设计原则

1. **不修改 DeepAgents 包** - 通过 Middleware 拦截消息
2. **利用现有 EventStore 模式** - 复用 Postgres 存储
3. **兼容现有前端** - 返回格式与 `/threads/{id}/state` 一致
4. **渐进式增强** - 老会话降级为原有行为

---

## 实现步骤

### Step 1: 数据库表 - `message_archive`

```sql
CREATE TABLE IF NOT EXISTS message_archive (
    id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    thread_id TEXT NOT NULL REFERENCES thread(thread_id) ON DELETE CASCADE,
    message_index INTEGER NOT NULL,  -- 消息在历史中的位置
    message_id TEXT,                 -- LangGraph 消息 ID
    message_type TEXT NOT NULL,      -- 'human', 'ai', 'tool', 'system'
    content JSONB NOT NULL,          -- 完整消息内容 (序列化)
    tool_calls JSONB,                -- AI 消息的 tool_calls
    tool_call_id TEXT,               -- Tool 消息的 tool_call_id
    metadata JSONB,                  -- 额外元数据
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(thread_id, message_index)  -- 防止重复
);

CREATE INDEX idx_message_archive_thread ON message_archive(thread_id);
CREATE INDEX idx_message_archive_created ON message_archive(thread_id, created_at);
```

### Step 2: ORM 模型

```python
# 添加到 aegra/src/agent_server/core/orm.py

class MessageArchive(Base):
    __tablename__ = "message_archive"

    id: Mapped[str] = mapped_column(
        Text, primary_key=True, server_default=text("uuid_generate_v4()::text")
    )
    thread_id: Mapped[str] = mapped_column(
        Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=False
    )
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    message_id: Mapped[str | None] = mapped_column(Text)
    message_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    tool_calls: Mapped[dict | None] = mapped_column(JSONB)
    tool_call_id: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )

    __table_args__ = (
        Index("idx_message_archive_thread", "thread_id"),
        Index("idx_message_archive_created", "thread_id", "created_at"),
        # Unique constraint on (thread_id, message_index)
    )
```

### Step 3: 归档服务

```python
# 新建 aegra/src/agent_server/services/message_archive_service.py

from typing import Any
import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from ..core.orm import MessageArchive

logger = structlog.get_logger(__name__)


class MessageArchiveService:
    """Service for archiving complete message history"""
    
    async def archive_messages(
        self, 
        session: AsyncSession, 
        thread_id: str, 
        messages: list[Any]
    ) -> None:
        """
        Archive messages from LangGraph state.
        Uses UPSERT to handle re-runs and avoid duplicates.
        """
        if not messages:
            return
            
        for idx, msg in enumerate(messages):
            msg_data = self._serialize_message(msg, idx)
            
            stmt = insert(MessageArchive).values(
                thread_id=thread_id,
                message_index=idx,
                message_id=msg_data.get("id"),
                message_type=msg_data["type"],
                content=msg_data["content"],
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
                metadata_json=msg_data.get("metadata"),
            ).on_conflict_do_update(
                index_elements=["thread_id", "message_index"],
                set_={
                    "content": msg_data["content"],
                    "tool_calls": msg_data.get("tool_calls"),
                    "metadata_json": msg_data.get("metadata"),
                }
            )
            await session.execute(stmt)
        
        await session.commit()
        logger.info(
            "archived_messages", 
            thread_id=thread_id, 
            count=len(messages)
        )
    
    async def get_full_history(
        self, 
        session: AsyncSession, 
        thread_id: str
    ) -> list[dict]:
        """Get complete archived history for a thread"""
        stmt = (
            select(MessageArchive)
            .where(MessageArchive.thread_id == thread_id)
            .order_by(MessageArchive.message_index)
        )
        result = await session.scalars(stmt)
        
        return [self._deserialize_message(m) for m in result]
    
    async def get_archive_count(
        self, 
        session: AsyncSession, 
        thread_id: str
    ) -> int:
        """Get count of archived messages"""
        stmt = (
            select(func.count())
            .select_from(MessageArchive)
            .where(MessageArchive.thread_id == thread_id)
        )
        return await session.scalar(stmt) or 0
    
    def _serialize_message(self, msg: Any, index: int) -> dict:
        """Serialize a LangGraph message for storage"""
        # Handle different message formats (BaseMessage, dict, etc.)
        if hasattr(msg, "model_dump"):
            data = msg.model_dump()
        elif hasattr(msg, "dict"):
            data = msg.dict()
        elif isinstance(msg, dict):
            data = msg
        else:
            data = {"content": str(msg), "type": "unknown"}
        
        return {
            "id": data.get("id"),
            "type": data.get("type", "unknown"),
            "content": data.get("content"),
            "tool_calls": data.get("tool_calls"),
            "tool_call_id": data.get("tool_call_id"),
            "metadata": {
                "created_at": data.get("created_at"),
                "additional_kwargs": data.get("additional_kwargs"),
            }
        }
    
    def _deserialize_message(self, archive: MessageArchive) -> dict:
        """Deserialize archived message back to LangGraph format"""
        return {
            "id": archive.message_id,
            "type": archive.message_type,
            "content": archive.content,
            "tool_calls": archive.tool_calls,
            "tool_call_id": archive.tool_call_id,
            "created_at": archive.created_at.isoformat() if archive.created_at else None,
            **(archive.metadata_json or {}),
        }


message_archive_service = MessageArchiveService()
```

### Step 4: 归档 Middleware (用于 QSA Agent)

```python
# 在 qsa/agent.py 中添加归档中间件

from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command
from typing import Any

class HistoryArchiveMiddleware:
    """
    Middleware that archives messages BEFORE they are potentially summarized.
    
    This runs as the first middleware in the chain, capturing the full state
    before SummarizationMiddleware can compress it.
    """
    
    async def __call__(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any] | Command:
        # Extract thread_id from config
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return state
        
        # Archive messages if present
        messages = state.get("messages", [])
        if messages:
            # Use async task to avoid blocking
            import asyncio
            from agent_server.services.message_archive_service import message_archive_service
            from agent_server.core.orm import get_session
            
            async def archive():
                async for session in get_session():
                    await message_archive_service.archive_messages(
                        session, thread_id, messages
                    )
                    break
            
            asyncio.create_task(archive())
        
        return state
```

### Step 5: Full History API 端点

```python
# 添加到 aegra/src/agent_server/api/threads.py

@router.get("/threads/{thread_id}/full-history")
async def get_full_history(
    thread_id: str,
    merge_checkpoint: bool = Query(True, description="Merge with checkpoint state"),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Get full message history for a thread.
    
    This returns archived messages that may have been compressed by 
    SummarizationMiddleware in the checkpoint.
    
    Args:
        thread_id: Thread identifier
        merge_checkpoint: If True, merge archived history with checkpoint
                         to include any messages added after last archive
    
    Returns:
        ThreadState with complete message history in values.messages
    """
    from ..services.message_archive_service import message_archive_service
    
    # Verify thread ownership
    stmt = select(ThreadORM).where(
        ThreadORM.thread_id == thread_id,
        ThreadORM.user_id == user.identity
    )
    thread = await session.scalar(stmt)
    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    
    # Get archived history
    archived_messages = await message_archive_service.get_full_history(
        session, thread_id
    )
    
    if not merge_checkpoint:
        # Return only archived history
        return {
            "values": {"messages": archived_messages},
            "source": "archive",
            "archived_count": len(archived_messages),
        }
    
    # Get current checkpoint state
    try:
        thread_state = await get_thread_state(
            thread_id=thread_id,
            user=user,
            session=session
        )
        checkpoint_messages = thread_state.values.get("messages", [])
    except HTTPException as e:
        if e.status_code == 404:
            checkpoint_messages = []
        else:
            raise
    
    # Merge: archived history + any new messages from checkpoint
    merged_messages = _merge_histories(archived_messages, checkpoint_messages)
    
    # Return merged state
    return {
        "values": {"messages": merged_messages},
        "source": "merged",
        "archived_count": len(archived_messages),
        "checkpoint_count": len(checkpoint_messages),
        "merged_count": len(merged_messages),
    }


def _merge_histories(
    archived: list[dict], 
    checkpoint: list[dict]
) -> list[dict]:
    """
    Merge archived and checkpoint histories.
    
    Strategy:
    1. Start with full archived history
    2. Find where checkpoint diverges (after summary message)
    3. Append any new messages from checkpoint that aren't in archive
    """
    if not archived:
        return checkpoint
    if not checkpoint:
        return archived
    
    # Build index of archived message IDs
    archived_ids = {m.get("id") for m in archived if m.get("id")}
    
    # Find new messages in checkpoint (not in archive)
    # These are messages added AFTER the last archive
    new_messages = []
    for msg in checkpoint:
        msg_id = msg.get("id")
        
        # Skip summary messages (they replace real history)
        if _is_summary_message(msg):
            continue
        
        # Add messages not in archive
        if msg_id and msg_id not in archived_ids:
            new_messages.append(msg)
        elif not msg_id:
            # Messages without ID - check by content/type match
            if not _message_exists_in_archive(msg, archived):
                new_messages.append(msg)
    
    return archived + new_messages


def _is_summary_message(msg: dict) -> bool:
    """Check if message is a summary from SummarizationMiddleware"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.startswith("Here is a summary of the conversation")
    return False


def _message_exists_in_archive(msg: dict, archived: list[dict]) -> bool:
    """Check if message exists in archive by content comparison"""
    msg_content = msg.get("content")
    msg_type = msg.get("type")
    
    for arch_msg in archived:
        if arch_msg.get("type") == msg_type and arch_msg.get("content") == msg_content:
            return True
    return False
```

### Step 6: 前端集成

修改 `useLangGraphChat.ts` 中的 recovery 逻辑：

```typescript
// 在 useLangGraphChat.ts 中

// 新增：获取完整历史的 helper
async function loadFullHistory(
  client: Client,
  threadId: string,
  accessToken?: string
): Promise<any[]> {
  try {
    // 优先使用 full-history API
    const response = await fetch(
      `${getApiUrl()}/threads/${threadId}/full-history?merge_checkpoint=true`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    
    if (response.ok) {
      const data = await response.json();
      console.log('[loadFullHistory] Source:', data.source, 
                  'archived:', data.archived_count, 
                  'merged:', data.merged_count);
      return data.values?.messages || [];
    }
    
    // Fallback to standard state API
    console.log('[loadFullHistory] Falling back to standard state API');
    const state = await client.threads.getState(threadId);
    return (state.values as any)?.messages || [];
    
  } catch (error) {
    console.warn('[loadFullHistory] Error, falling back to state:', error);
    const state = await client.threads.getState(threadId);
    return (state.values as any)?.messages || [];
  }
}

// 修改 recovery 逻辑，使用 loadFullHistory
// 在 ~L649 和 ~L804 处，将：
//   const stateMessages = (threadState.values as any)?.messages || [];
// 改为：
//   const stateMessages = await loadFullHistory(client, threadId, session?.accessToken);
```

---

## 集成方式

### 方式 A: QSA Agent 中添加 Middleware（推荐）

在 `qsa/agent.py` 的 `create_deep_agent` 调用中添加 `HistoryArchiveMiddleware`：

```python
# qsa/agent.py

from your_archive_middleware import HistoryArchiveMiddleware

agent = create_deep_agent(
    model,
    tools=tools,
    middleware=[
        HistoryArchiveMiddleware(),  # 在 SummarizationMiddleware 之前
        WorkspaceSetupMiddleware(workspace_path),
    ],
    # ...
)
```

**注意**: DeepAgents 会将用户 middleware **追加**到内置 middleware 之后。所以需要换一种方式。

### 方式 B: Graph Streaming Hook（更优雅）

在 `graph_streaming.py` 中，每次运行时归档消息：

```python
# aegra/src/agent_server/services/graph_streaming.py

async def _archive_messages_on_checkpoint(
    thread_id: str,
    state_snapshot: Any,
    session: AsyncSession
):
    """Archive messages when a checkpoint is saved"""
    messages = state_snapshot.values.get("messages", [])
    if messages:
        await message_archive_service.archive_messages(
            session, thread_id, messages
        )
```

在 streaming 完成后调用此函数。

### 方式 C: Checkpoint Saver Hook（最底层）

如果使用 PostgresSaver，可以包装它：

```python
class ArchivingPostgresSaver(PostgresSaver):
    async def aput(self, config, checkpoint, metadata, new_versions):
        # 在保存前归档 messages
        if "messages" in checkpoint.get("channel_values", {}):
            await archive_messages(config, checkpoint)
        return await super().aput(config, checkpoint, metadata, new_versions)
```

---

## 迁移策略

### 对于已有会话

1. 老会话没有归档数据，API 自动降级到 checkpoint-only 模式
2. 前端的 `loadFullHistory` 会 fallback 到标准 state API
3. 用户体验无感知降级

### 新会话

1. 自动归档所有消息
2. 即使被 SummarizationMiddleware 压缩，也能恢复完整历史

---

## 总结

| 方案 | 修改范围 | 复杂度 | 推荐度 |
|------|---------|--------|--------|
| 方式 B: Graph Streaming Hook | aegra | 中 | ⭐⭐⭐⭐⭐ |
| 方式 A: QSA Middleware | qsa + aegra | 中 | ⭐⭐⭐⭐ |
| 方式 C: Checkpoint Saver | 底层 | 高 | ⭐⭐⭐ |

**推荐方式 B**：在 `graph_streaming.py` 中 hook，不需要修改 agent 代码，对所有 graph 通用。

---

## 实现状态 ✅

所有核心组件已实现并集成完成。

### 已完成的实现

#### 1. 后端 (aegra)

| 文件 | 状态 | 描述 |
|------|------|------|
| `src/agent_server/core/orm.py` | ✅ | 添加 MessageArchive ORM 模型 |
| `src/agent_server/services/message_archive_service.py` | ✅ | 完整的归档服务 (UPSERT, 序列化, 反序列化) |
| `alembic/versions/20260128_add_message_archive.py` | ✅ | 数据库迁移文件 |
| `src/agent_server/api/threads.py` | ✅ | `/threads/{thread_id}/full-history` API 端点 |
| `src/agent_server/api/runs.py` | ✅ | 在 run 完成后自动归档消息 |

#### 2. 前端 (new-frontend)

| 文件 | 状态 | 描述 |
|------|------|------|
| `lib/hooks/useLangGraphChat.ts` | ✅ | `loadFullHistoryMessages()` helper 和恢复逻辑修改 |

### 数据流

```
1. 用户发送消息 → Agent 处理 → 完成/中断
                                    │
2. runs.py: _archive_messages_from_checkpoint() 
   └── 获取 thread state
   └── 调用 message_archive_service.archive_messages()
   └── UPSERT 到 message_archive 表
                                    │
3. SummarizationMiddleware 可能在下次运行时压缩历史
                                    │
4. 前端刷新 → useLangGraphChat recovery
   └── loadFullHistoryMessages(threadId)
       └── GET /threads/{id}/full-history?merge_checkpoint=true
           └── 从 message_archive 获取完整历史
           └── 从 checkpoint 获取当前消息
           └── merge: archive 优先 + checkpoint 补充新消息
   └── reconstructHistoryFromMessages(mergedMessages)
   └── 前端显示完整历史（包括所有 thinkingSteps, tool_calls）
```

### 部署步骤

1. **运行数据库迁移**:
   ```bash
   cd aegra
   alembic upgrade head
   ```

2. **重启后端服务** (Docker):
   ```bash
   docker-compose restart aegra
   ```

3. **重新构建前端**:
   ```bash
   cd Frontend/new-frontend/alagent-new
   npm run build
   ```

### 向后兼容

- ✅ 老会话没有归档数据时，API 返回 checkpoint 消息（降级行为）
- ✅ 前端 `loadFullHistoryMessages` 在 API 失败时使用 fallback
- ✅ 不影响现有 run 执行流程
