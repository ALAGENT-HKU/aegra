# AEGRA 自定义修改清单

> **用途**：本文档完整记录 ALAGENT-HKU 在 upstream ibbybuilds/aegra 之上所做的全部定制修改。  
> 每次与 upstream merge 时，请参照此表逐项检查冲突和功能完整性。

## 概览

| 分支 | 说明 |
|---|---|
| `dev_ALAGENT-HKU` | 原始自定义分支 (基于旧版 upstream, 包名 `agent_server`) |
| `dev_ALAGENT-HKU-merged` | 合并到 upstream v0.8.3 后的版本 (包名 `aegra_api`)，基线 = `ba53449` + `485277e` + `74eb16b` |
| upstream baseline | `ibbybuilds/aegra` v0.8.3 (commit `ba53449`) |

> **什么是 "upstream ba53449 (v0.8.3) + 自定义 commit 485277e"？**  
> `485277e` 是将所有 `dev_ALAGENT-HKU` 自定义功能移植到 upstream v0.8.3（包名从 `agent_server` → `aegra_api`）的大合并 commit。  
> 之后 `74eb16b` 修复了 custom message dispatch 的 `graph_streaming.py` 逻辑。  
> `dev_ALAGENT-HKU-merged` 分支即为该三者的叠加结果，文档中凡提及该基线均指此状态。

**修改统计**: 20 个文件, ~2000 行自定义代码, 7 大功能类别

---

## 一、Commit 时间线

### dev_ALAGENT-HKU（原始自定义分支，包名 `agent_server`）

| 时间 | 作者 | Commit | 功能 |
|---|---|---|---|
| 2026-01-10 | Patrick_Lew | `3f044c2` | Docker 基础配置 + QSA 集成 |
| 2026-01-13 | Patrick_Lew | `a15812f` | 开发环境配置 |
| 2026-01-13 | Patrick_Lew | `587e0a3` | 修复 Docker-in-Docker |
| 2026-01-16 | Patrick_Lew | `c1ec18e` | Bug 修复 + Docker Compose 改进 |
| 2026-01-19 | Patrick_Lew | `eb93d9c` | 数据库连接池配置 |
| 2026-01-19 | Ziqi Yuan | `a922715` | GCP 凭证挂载 (Gemini) |
| 2026-01-19 | Patrick_Lew | `7bc6377` | **adispatch_custom_event 流式支持** |
| 2026-01-19 | Patrick_Lew | `6de041a` | 进度恢复机制 (event store + REST) |
| 2026-01-21 | Ziqi Yuan | `75bf88a` | RAG/项目基础目录配置 |
| 2026-01-21 | five0502 | `8195c10` | Redis 依赖 |
| 2026-01-21 | Ziqi Yuan | `409aef6` | recursion_limit 放宽到 100 |
| 2026-01-23 | Patrick_Lew | `0b2e066` | 文件日志系统 (RotatingFileHandler) |
| 2026-01-27 | Patrick_Lew | `7402177` | 用户注册邮件验证 (Resend) |
| 2026-01-27 | Patrick_Lew | `0467fef` | .env.example 敏感信息清理 |
| 2026-02-06 | Patrick_Lew | `ed32453` | **消息归档系统** (完整历史恢复) |
| 2026-03-07 | pingpongdragon | `ede4395` | 日志 user_id/session_id/thread_id 注入 |

### dev_ALAGENT-HKU-merged（移植到 upstream v0.8.3，包名 `aegra_api`）

| 时间 | 作者 | Commit | 功能 |
|---|---|---|---|
| 2026-03-22 | Patrick_Lew | `485277e` | **移植所有自定义功能到 upstream v0.8.3**（消息归档、JWT 认证、文件日志、custom event 存储格式、recursion_limit、on_disconnect 默认值等） |
| 2026-03-23 | Patrick_Lew | `74eb16b` | **修复 custom message dispatch**（`graph_streaming.py` 改用 `parent_ids` 过滤 `on_chain_stream`，新增 `on_custom_event` 拦截器） |

---

## 二、按功能分类的详细说明

### A. 流式管道 — adispatch_custom_event 支持

**目的**: 让 QSA 子图节点通过 `adispatch_custom_event()` 发送的进度事件能穿透到前端，即使 `subgraphs=False`。

**原理**:  
- `adispatch_custom_event` 走 langchain callback 系统，只有 `astream_events(v2)` 能捕获其 `on_custom_event`
- langgraph 的 `astream(stream_mode=["custom"])` 只能捕获 `get_stream_writer()` 的输出
- AEGRA 将 `"custom"` 作为虚拟 stream mode，自动切换到 `astream_events`，拦截 `on_custom_event` 转发

**修改文件**:

| 文件 (aegra_merged 路径) | 改动 |
|---|---|
| `libs/aegra-api/src/aegra_api/services/graph_streaming.py` | 1) `"custom"` 从 stream_modes 剥离 (与 `"events"` 同级)<br>2) `custom_events_requested` 触发 `astream_events`<br>3) 新增 `on_custom_event` 拦截器 → yield `("custom", {name, data, ...})`<br>4) `on_chain_stream` 过滤改用 `parent_ids` (替代不可靠的 `run_id`) |
| `libs/aegra-api/src/aegra_api/services/streaming_service.py` | `"custom"` 事件存储格式: `{type: "custom_event", payload: {...}, node_path}`<br>前端通过 `payload.name` + `payload.data` 解析 |

**前端匹配**: `useLangGraphChat.ts` → `handleCustomEvent()` 检查 `chunk.event === 'custom'`，支持两种格式 (stream/joinStream)

---

### B. 消息归档系统 — 完整历史恢复

**目的**: LangGraph 的 SummarizationMiddleware 会压缩旧消息。归档系统在压缩前保存完整历史，支持前端"查看完整对话"功能。

**修改文件**:

| 文件 | 说明 |
|---|---|
| `libs/aegra-api/src/aegra_api/core/orm.py` | +`MessageArchive` ORM model (thread_id, message_index, content JSONB, tool_calls JSONB 等) |
| `libs/aegra-api/src/aegra_api/services/message_archive_service.py` | **新文件 (350行)**: `archive_messages()`, `get_full_history()`, `cleanup_summary_messages()`<br>INSERT-only，message_id 去重，跳过 Summary 消息 |
| `libs/aegra-api/alembic/versions/20260128_add_message_archive.py` | **新文件**: 创建 `message_archive` 表，`down_revision` 依赖上游 migration 链 |
| `libs/aegra-api/src/aegra_api/api/runs.py` | +`_archive_messages_from_checkpoint()` 函数<br>run 完成/中断时自动归档 |
| `libs/aegra-api/src/aegra_api/api/threads.py` | +`GET /threads/{id}/full-history` 端点<br>合并 Archive + Checkpoint 去重返回完整历史 |
| `docs/FULL_HISTORY_DESIGN.md` | 设计文档 (612行，含架构图和 API 说明) |

---

### C. 进度恢复 — Event Store + REST 端点

**目的**: 前端断连后重新连接时，能恢复之前的 pipeline 进度事件。

**修改文件**:

| 文件 | 说明 |
|---|---|
| `libs/aegra-api/src/aegra_api/api/runs.py` | +`GET /threads/{id}/runs/{id}/events` 端点<br>按 `event_type` 过滤 (如 `custom`)，返回存储的 SSE 事件 |
| `libs/aegra-api/src/aegra_api/api/runs.py` | `on_disconnect` 默认值 `"cancel"` → `"continue"`<br>长时间运行的量化分析不应因前端断连而取消 |

---

### D. 认证系统

**目的**: JWT 双模式认证 + 资源隔离。

**修改文件**:

| 文件 | 说明 |
|---|---|
| `auth.py` | **新文件 (162行)**: `AUTH_TYPE=noop` (无认证) / `AUTH_TYPE=custom` (JWT)<br>从 Bearer token 解码 user_id/email/plan<br>`authorize()` 实现 owner 过滤器隔离资源 |
| `aegra.docker.json` | `auth.path: /app/auth.py:auth`, `enable_custom_route_auth: false`, CORS `allow_origins: ["*"]` |

---

### E. 日志增强

**目的**: 文件日志持久化 + 请求级用户/线程信息注入。

**修改文件**:

| 文件 | 说明 |
|---|---|
| `libs/aegra-api/src/aegra_api/utils/setup_logging.py` | +`structlog.contextvars.merge_contextvars` processor (使 context vars 生效)<br>+`RotatingFileHandler`: 通过 `LOG_TO_FILE`, `LOG_FILE_PATH`, `LOG_FILE_MAX_BYTES`, `LOG_FILE_BACKUP_COUNT` 控制 |
| `libs/aegra-api/src/aegra_api/middleware/logger_middleware.py` | +从 URL path 提取 `thread_id`/`session_id` 绑定 structlog context<br>+从 JWT header 提取 `user_id` 绑定 structlog context |
| `.gitignore` | +`logs/` 和 `*.log` 忽略规则 |

---

### F. 部署 & 依赖管理

**修改文件**:

| 文件 | 说明 |
|---|---|
| `docker-compose.custom.yml` | **新文件 (107行)**: ALAGENT 生产部署 compose<br>• 端口可配: `${PORT:-2024}`, `${PG_PORT:-5442}`, `${REDIS_PORT:-6379}`<br>• Volume: QSA 包, data, Docker socket, GCP 凭证, auth.py, logs<br>• `AUTH_TYPE=custom`, DinD 环境变量, 连接池配置<br>• uvicorn `--reload --reload-dir` |
| `docker-compose.yml` | +PYTHONPATH 扩展 (`/app/quantitative_strategy_agent`)<br>+Volume 挂载 (auth.py, logs, QSA) |
| `deployments/docker/Dockerfile` | • uv 版本 → 0.10.12<br>• +`--extra qsa` (QSA 依赖烘焙进镜像)<br>• 移除 `--compile-bytecode` (避免 fd 耗尽)<br>• +build-essential, git (psycopg2 编译 + git clone) |
| `libs/aegra-api/pyproject.toml` | • langgraph `>=1.1.0,<2.0.0`<br>• +`python-jose[cryptography]` (JWT)<br>• +`[qsa]` optional-deps: deepagents, langchain全家桶, torch, pandas-ta, statsmodels 等 36 个包 |
| `uv.lock` | 268 包完整锁定 (含 torch + CUDA) |
| `.env.example` | 环境变量模板 (日志, JWT, 数据库, Redis 等) |

---

### G. 运行时修复

| 文件 | 说明 |
|---|---|
| `libs/aegra-api/src/aegra_api/services/langgraph_service.py` | `cfg.setdefault("recursion_limit", 100)` (上游默认 25 不够用) |

---

## 三、合并风险矩阵

每次与 upstream merge 时，按风险等级处理：

### 🔴 高风险 — 必须手动 merge

| 文件 | 原因 |
|---|---|
| `deployments/docker/Dockerfile` | 上游活跃维护，多处修改 |
| `libs/aegra-api/src/aegra_api/api/runs.py` | 核心文件，+150 行深度修改 |
| `libs/aegra-api/src/aegra_api/services/graph_streaming.py` | 核心流式逻辑，+40 行深度修改 |
| `libs/aegra-api/pyproject.toml` | 版本范围修改 + qsa optional-deps |
| `uv.lock` | 无法 3-way merge，merge 后 `uv lock` 重新生成 |

### 🟡 中风险 — 可能需要手动调整

| 文件 | 原因 |
|---|---|
| `docker-compose.yml` | 上游会更新，建议自定义全部移到 custom.yml |
| `libs/aegra-api/src/aegra_api/api/threads.py` | +165 行追加，上游可能重构 |
| `libs/aegra-api/src/aegra_api/core/orm.py` | +32 行追加 |
| `libs/aegra-api/src/aegra_api/services/streaming_service.py` | payload 格式变更 |
| `libs/aegra-api/src/aegra_api/utils/setup_logging.py` | 返回值重构 + file handler |
| `libs/aegra-api/src/aegra_api/middleware/logger_middleware.py` | +30 行注入逻辑 |
| `libs/aegra-api/alembic/versions/20260128_*` | `down_revision` 依赖上游链 |

### 🟢 低风险 — 独立文件，不冲突

| 文件 |
|---|
| `auth.py` |
| `aegra.docker.json` |
| `docker-compose.custom.yml` |
| `.env.example` |
| `libs/aegra-api/src/aegra_api/services/message_archive_service.py` |
| `docs/FULL_HISTORY_DESIGN.md` |
| `docs/INTEGRATION_TEST_TABLE.md` |
| `run_server.py` |
| `.gitignore` |
| `libs/aegra-api/src/aegra_api/services/langgraph_service.py` (+3 行) |

---

## 四、Merge 操作手册

```bash
# 1. 添加上游 remote (如果没有)
git remote add upstream https://github.com/ibbybuilds/aegra.git

# 2. Fetch 最新上游
git fetch upstream main

# 3. 在 dev_ALAGENT-HKU-merged 分支上 merge
git checkout dev_ALAGENT-HKU-merged
git merge upstream/main

# 4. 处理冲突 (按上面的风险矩阵优先处理 🔴 文件)

# 5. 重新生成 uv.lock
cd libs/aegra-api
uv lock

# 6. 检查 alembic migration 链
# 确认 20260128_add_message_archive.py 的 down_revision 仍然有效

# 7. 重新构建 Docker 镜像
docker compose -f docker-compose.custom.yml build --no-cache

# 8. 测试
docker compose -f docker-compose.custom.yml up -d
# 验证: custom events, 消息归档, 认证, 日志
```

---

## 五、功能依赖关系

```
                    ┌─────────────────┐
                    │   auth.py       │
                    │ (JWT 认证)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ logger_middleware│
                    │ (user_id 注入)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼───┐  ┌──────▼──────┐  ┌───▼──────────┐
    │graph_streaming│  │  runs.py    │  │ threads.py   │
    │(custom event │  │(event store │  │(full-history)│
    │ 拦截+转发)   │  │ +归档集成)  │  │              │
    └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
           │                │                │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼───────┐
    │streaming_svc │  │ event_store │  │archive_svc   │
    │(SSE 存储)    │  │ (Postgres)  │  │(消息归档)    │
    └─────────────┘  └─────────────┘  └──────────────┘
```
