# aegra_merged 集成测试表 — QSA on upstream v0.8.3

> **分支**: `dev_ALAGENT-HKU-merged`  
> **基线**: upstream `ba53449` (v0.8.3) + 自定义 commit `485277e` + fix `74eb16b`  
> **日期**: 2026-03-23

---

## 目录

1. [P0 — 已修复阻塞项（验证修复生效）](#p0--已修复阻塞项验证修复生效)
2. [P1 — 核心自定义功能集成验证](#p1--核心自定义功能集成验证)
3. [P2 — 配置 & 环境适配](#p2--配置--环境适配)
4. [P3 — 回归风险点（upstream 新功能 vs 自定义冲突）](#p3--回归风险点upstream-新功能-vs-自定义冲突)
5. [⭐ Message Archive 机制详细验证](#-message-archive-机制详细验证)
6. [验证执行优先级建议](#验证执行优先级建议)

---

## P0 — 已修复阻塞项（验证修复生效）

| # | 测试项 | 验证方式 | 预期结果 | 涉及文件 |
|---|--------|---------|---------|---------|
| P0-1 | Volume 挂载路径（monorepo 结构） | `docker compose -f docker-compose.custom.yml config` 检查 volumes | `./libs/aegra-api/src:/app/src:ro` 等路径正确 | `docker-compose.custom.yml` |
| P0-2 | uvicorn 启动模块路径 | 容器内 `python -c "import aegra_api.main"` | 无 ModuleNotFoundError | `docker-compose.custom.yml` |
| P0-3 | Dockerfile final stage 构建工具 | `docker build` 后容器内 `gcc --version` | `build-essential` / `libpq-dev` / `git` 就位 | `deployments/docker/Dockerfile` |
| P0-4 | docker-compose.yml QSA 挂载 | `docker compose config` 查看 volumes | `quantitative_strategy_agent` 挂载 + `PYTHONPATH` | `docker-compose.yml` |
| P0-5 | aegra.docker.json auth 配置 | 启动服务查看日志 | `"auth.path": "/app/auth.py:auth"` 被正确加载 | `aegra.docker.json` |
| P0-6 | CORS allow_credentials 修复 | 发送 OPTIONS 预检请求 | `allow_credentials: false` + `allow_origins: ["*"]` 不冲突 | `aegra.docker.json` |

---

## P1 — 核心自定义功能集成验证

### Graph 加载 & 运行

| # | 测试项 | 验证方式 | 预期结果 | 风险说明 | 涉及文件 |
|---|--------|---------|---------|---------|---------|
| P1-1 | quant_agent graph 加载 | `GET /assistants` | `quant_agent` 出现在图列表 | `_load_graph_from_file()` 解析 `/app/.../agent.py:agent` | `aegra.docker.json` → `langgraph_service.py` |
| P1-2 | paper_pipeline graph 加载 | `GET /assistants` | `paper_pipeline` 出现在图列表 | 同上，路径 `pipeline_graph.py:pipeline_graph` | 同上 |
| P1-3 | Factory graph 检测 | 日志查看 graph type | 如果 `agent` 是 callable → factory；是 CompiledGraph → 普通 | `_load_graph_from_file()` 通过 `inspect.signature()` 判断 | `langgraph_service.py:501-549` |
| P1-4 | webapp.py 自定义路由 | `GET /qsa/...` 等 QSA 端点 | HTTP 200 + 正确响应 | `load_custom_app()` 要求 `FastAPI` 实例；`enable_custom_route_auth: false` → QSA 自定义路由**无平台级 auth** | `core/app_loader.py`、`aegra.docker.json` |

### Auth 链路

| # | 测试项 | 验证方式 | 预期结果 | 风险说明 | 涉及文件 |
|---|--------|---------|---------|---------|---------|
| P1-5 | JWT Auth 链路 | 带有效 JWT `Authorization: Bearer <token>` 请求 `/threads` | 200 + 返回该用户线程 | `auth.py` → `_load_auth_instance()` → middleware 注入 user | `auth.py` |
| P1-6 | Auth noop 模式 | `AUTH_TYPE=noop` 启动，不带 token | 200 OK | auth.py `noop` 分支跳过 JWT 验证 | `auth.py` |
| P1-7 | **`langgraph_auth_user` 注入** | graph 运行中查看 `config["configurable"]["langgraph_auth_user"]` | dict 含 `{identity, is_authenticated, email, subscription_plan}` | `inject_user_context()` L552 调用 `user.to_dict()` 注入 | `langgraph_service.py:552-575` |
| P1-8 | **`user_id` 注入** | graph 运行中查看 `config["configurable"]["user_id"]` | 非空字符串 = auth handler 的 `identity` | QSA `agent.py`、`config.py`、`pipeline_graph.py` 等多处读取 | `langgraph_service.py:559` |
| P1-9 | **QSA billing 用户获取** | 触发计费逻辑 | `configurable.get("langgraph_auth_user")` 返回有效对象 | `qsa/billing/middleware.py:82` 依赖此字段 | QSA `billing/middleware.py` |
| P1-10 | 资源授权过滤 | 用户 A 创建 thread，用户 B 查询 | B 看不到 A 的 thread | `auth.py` 的 `on_read`/`on_search` 按 `owner` metadata 过滤 | `auth.py:135-145` |

### ⭐ Message Archive 系统（详见下方独立章节）

| # | 测试项 | 验证方式 | 预期结果 | 风险说明 | 涉及文件 |
|---|--------|---------|---------|---------|---------|
| P1-11 | archive 写入 — run 成功 | 调用 quant_agent，查 DB `message_archive` 表 | 消息被归档，summary 被跳过 | `archive_messages()` 用 `message_id` 去重，INSERT only | `message_archive_service.py` |
| P1-12 | archive 写入 — run interrupt | HITL 中断场景 | interrupt 后也归档 | `execute_run_async:1101` interrupt 路径也调用 archive | `runs.py:1101` |
| P1-13 | full-history 读取 | `GET /threads/{id}/full-history` | `{values: {messages: [...]}, source: "merged"}` | 依赖 MessageArchive ORM + migration 正确 | `threads.py:385-490` |
| P1-14 | migration 自动执行 | 启动时 auto-migrate | `message_archive` 表创建成功 | `down_revision=d042a0ca1cb5` 必须匹配 upstream 链尾 | `20260128_add_message_archive.py` |

### Streaming & Events

| # | 测试项 | 验证方式 | 预期结果 | 风险说明 | 涉及文件 |
|---|--------|---------|---------|---------|---------|
| P1-15 | SSE streaming + on_disconnect=continue | `POST /threads/{id}/runs/stream` 后断开 | 运行继续，不因断连取消 | 自定义 `on_disconnect` 参数 | `runs.py:299-462` |
| P1-16 | Custom event streaming | QSA 通过 `get_stream_writer()` 发送 pipeline 进度 | SSE 收到 `event: custom` | `streaming_service.py:133-145` 处理 custom mode | `streaming_service.py` |
| P1-17 | Events 端点 | `GET /threads/{id}/runs/{run_id}/events?event_type=custom` | 200 + 返回存储的事件 | 依赖 event_store | `runs.py:527-562` |
| P1-18 | recursion_limit=100 | 长推理链 graph 运行 | 不在 25 步停止 | `create_run_config()` L637 写死 100 | `langgraph_service.py:637` |

### Logging

| # | 测试项 | 验证方式 | 预期结果 | 涉及文件 |
|---|--------|---------|---------|---------|
| P1-19 | File logging | 查看 `BASE_DATA_DIR/logs/` | 日志文件被写入 | `utils/setup_logging.py` |

---

## P2 — 配置 & 环境适配

| # | 测试项 | 验证方式 | 预期结果 | 涉及文件 |
|---|--------|---------|---------|---------|
| P2-1 | PORT 端口 | `curl localhost:${PORT}/ok` | 正确响应 | `docker-compose.custom.yml` |
| P2-2 | DATABASE_URL 连接 | 启动日志无 connection refused | asyncpg 连接成功 | `docker-compose.custom.yml` |
| P2-3 | SA_POOL_SIZE / LG_POOL_SIZE | 高并发请求 | 无 pool exhausted | `docker-compose.custom.yml` |
| P2-4 | USER_DATABASE_URL（QSA 用户库） | QSA billing/auth 查询 | 外部用户库连接正常 | `docker-compose.custom.yml` |
| P2-5 | Langfuse observability | 运行 graph 后查看 Langfuse UI | trace 显示 | `docker-compose.custom.yml` |
| P2-6 | Redis 连接 | graph 使用 Redis 时 | 无 connection error | `docker-compose.custom.yml` |
| P2-7 | GCP credentials mount | QSA 访问 BigQuery 等 | credential file 可读 | `docker-compose.custom.yml` |
| P2-8 | Docker Socket (DinD) | QSA `execute_backtest` tool | 容器内可 `docker run` | `/var/run/docker.sock` mount |
| P2-9 | DATA_DIR / WORKSPACE_DIR / MEMORIES_DIR | QSA 文件操作 | 目录存在且可写 | `docker-compose.custom.yml` |
| P2-10 | PostgreSQL 镜像兼容性 | custom 用 `postgres:15-alpine` vs base 用 `pgvector/pgvector:pg18` | 两种皆可 migrate + 运行 | 两个 compose 文件 |
| P2-11 | PYTHONPATH | 容器内 `python -c "import qsa"` | QSA 可导入 | `docker-compose.custom.yml` |
| P2-12 | QSA runtime pip install | 启动日志 | `pip install -e /app/quantitative_strategy_agent` 成功 | `docker-compose.custom.yml` |

---

## P3 — 回归风险点（upstream 新功能 vs 自定义冲突）

| # | 风险项 | 风险描述 | 检验方式 | 严重度 | 涉及文件 |
|---|--------|---------|---------|--------|---------|
| P3-1 | **DoubleEncodedJSONMiddleware 被移除** | aegra_my 有此中间件处理双重编码 JSON；upstream 用 `ContentTypeFixMiddleware` 替代。前端若仍发送双重编码 payload → 422 | 发送之前触发双重编码的请求，确认解析正常 | 🔴 高 | `main.py:231-243` |
| P3-2 | **ContentTypeFixMiddleware（新增）** | upstream 新增 `text/plain` → `application/json` 转换。可能影响 QSA 返回非 JSON 的端点（如 CSV 下载） | QSA 返回非 JSON 内容时检查 Content-Type 是否被篡改 | 🟡 中 | `middleware/content_type_fix.py` |
| P3-3 | **Stateless Runs Router（新增）** | upstream 新增 `POST /runs/wait` 和 `POST /runs/stream`（无 thread）。可能影响路由优先级 | 调用 `/runs/stream` 确认走 stateless 逻辑 | 🟢 低 | `api/stateless_runs.py` |
| P3-4 | **Auth 架构变化** | upstream 从 Starlette middleware 演进为 FastAPI Depends。auth.py handler 签名需匹配 `_load_auth_instance()` 接口 | 启动时无 signature mismatch error | 🔴 高 | `auth.py`、`langgraph_service.py` |
| P3-5 | **CORS credentials=false vs 前端 cookie** | `allow_credentials` 从 `true` → `false`（因 `allow_origins: ["*"]` + `credentials=true` 违反浏览器安全策略）。**如果前端用 cookie 认证而非 Authorization header，凭据将无法发送** | 确认前端认证方式：`Authorization: Bearer` → 无影响；cookies → 需改为指定域名 + `credentials=true` | 🔴 高 | `aegra.docker.json` |
| P3-6 | **Auto-migration chain** | upstream lifespan 自动运行 `alembic upgrade head`。自定义 migration 的 `down_revision` 必须精确匹配 upstream 链尾 | 启动查看 migration 日志，确认 chain 完整 | 🔴 高 | `main.py` lifespan、`alembic/versions/` |
| P3-7 | **`enable_custom_route_auth: false`** | QSA webapp.py 暴露的所有端点**无平台级认证**。如果 QSA 内部无 auth 逻辑 → 裸奔 | 检查 QSA webapp 是否自带鉴权；如无则改 `true` 或内部加 auth | 🔴 高 | `aegra.docker.json` |
| P3-8 | **pgvector 扩展依赖** | base compose 用 `pgvector/pgvector:pg18`，custom 用 `postgres:15-alpine`（无 pgvector）。若 upstream 新功能依赖 vector → custom 报错 | 检查 migration/store 是否用 `vector` 类型 | 🟡 中 | `docker-compose.custom.yml` |
| P3-9 | **`inject_user_context` 字段访问路径** | QSA 读 `email`、`subscription_plan` 需通过 `config["configurable"]["langgraph_auth_user"]["email"]`，**不在** configurable 顶层。若 QSA 用 `config["configurable"]["email"]` 直接读 → 取到 None | 全局搜索 QSA 中所有 `configurable.get("email")` 等直接读取 | 🟡 中 | QSA 各模块、`langgraph_service.py:552-575` |
| P3-10 | **python-jose 版本兼容** | `pyproject.toml` 要求 `>=3.3.0`，uv.lock 锁定版本。容器 pip install 可能引入不同版本 | `pip show python-jose` 确认版本一致 | 🟢 低 | `pyproject.toml` |
| P3-11 | **ORM model 注册顺序** | `MessageArchive` 在 `core/orm.py` 定义。alembic + app startup 都需发现此 model | 启动无 `NoReferencedTableError` | 🟡 中 | `core/orm.py` |
| P3-12 | **Event store cleanup** | upstream lifespan 新增 event store 清理。如果清理过早 → QSA pipeline progress 事件丢失 | 运行 pipeline 后等待，再查 events 是否存在 | 🟡 中 | `main.py` lifespan |
| P3-13 | **dependencies sys.path 重复** | `aegra.docker.json` `dependencies` + `PYTHONPATH` 可能导致重复路径 / import 优先级异常 | 容器内 `python -c "import sys; print(sys.path)"` 确认无重复 | 🟢 低 | `aegra.docker.json`、`langgraph_service.py` |

---

## ⭐ Message Archive 机制详细验证

### 机制概览

每次 run 结束后、下次 SummarizationMiddleware 发生前，将原始 messages 存入独立 PostgreSQL 表。该机制与 deepagents 的 SummarizationMiddleware **完全解耦**。

### 完整调用链路（已验证 ✅）

```
前端调用 POST /threads/{id}/runs/stream 或 /runs/wait 或 /runs (background)
    │
    ▼
create_and_stream_run() / wait_for_run() / create_run()
    │  (都通过 asyncio.create_task 调用)
    ▼
execute_run_async()                          ← runs.py:972
    │
    ├── 执行 graph (stream_graph_events)
    │
    ├── [interrupt 路径]
    │   └── _archive_messages_from_checkpoint()  ← runs.py:1101
    │
    └── [success 路径]
        └── _archive_messages_from_checkpoint()  ← runs.py:1107
                │
                ▼
        get_thread_state()  ← 读取 LangGraph checkpoint
                │
                ▼
        thread_state.values.get("messages")
                │
                ▼
        message_archive_service.archive_messages()
                │
                ├── _get_existing_message_ids()  ← 获取已归档 message_id set
                ├── _get_max_index()             ← 获取当前最大 index
                │
                ├── 遍历 messages:
                │   ├── _serialize_message()     ← 支持 Pydantic / dict / BaseMessage
                │   ├── _is_summary_message()    ← 跳过 "Here is a summary..." 前缀
                │   ├── message_id in existing?  ← 跳过已存在（dedup by message_id）
                │   └── INSERT + on_conflict_do_nothing
                │
                └── session.commit()

前端调用 GET /threads/{id}/full-history
    │
    ▼
get_full_history() endpoint                 ← threads.py:385
    │
    ├── 验证 thread ownership (user_id filter)
    │
    ├── message_archive_service.get_full_history()
    │   └── SELECT * FROM message_archive WHERE thread_id=? ORDER BY message_index
    │       └── 再次过滤 summary 消息（defensive）
    │
    ├── [merge_checkpoint=false] → 仅返回 archive
    │
    └── [merge_checkpoint=true]  → 同时获取 checkpoint state
        │
        ▼
    _merge_histories(archived, checkpoint)  ← threads.py:492
        │
        ├── 以 archived 为基础
        ├── 用 message_id set 做 dedup
        ├── 从 checkpoint 中找出 archive 没有的新消息
        ├── 跳过 summary 消息
        └── 返回 archived + new_messages
```

### 关键设计验证点

| # | 验证点 | 当前实现 | 状态 | 验证细节 |
|---|--------|---------|------|---------|
| MA-1 | **归档时机** | `execute_run_async` 在 interrupt 和 success **两条路径都归档** | ✅ 正确 | `runs.py:1101` (interrupt) 和 `runs.py:1107` (success) |
| MA-2 | **归档在 summarization 之前** | 归档读取 **当前 run 结束后** 的 checkpoint，此时 SummarizationMiddleware 已在本轮 run 中执行完毕，但**下次 run 还未开始**，所以新产生的消息一定是原始的 | ✅ 正确 | `_archive_messages_from_checkpoint` 调用 `get_thread_state()` 读取最新 checkpoint |
| MA-3 | **去重机制 — message_id** | 用 `message_id`（LangGraph 为每条消息生成的 UUID）作为去重 key，而非 index | ✅ 正确 | `archive_messages()` L80: `if msg_id and msg_id in existing_ids: continue` |
| MA-4 | **INSERT only, 不 UPDATE** | 使用 `INSERT ... ON CONFLICT DO NOTHING`，永不覆盖已归档消息 | ✅ 正确 | L96: `insert(MessageArchive).values(...).on_conflict_do_nothing()` |
| MA-5 | **跳过 summary 消息** | 写入时通过 `_is_summary_message()` 检测 "Here is a summary of the conversation" 前缀 | ✅ 正确 | L76: `if self._is_summary_message(msg_data): skipped_summary += 1; continue` |
| MA-6 | **读取时二次过滤 summary** | `get_full_history()` 读取时再次过滤 summary（防御老数据） | ✅ 正确 | `message_archive_service.py:173`: `if not self._is_summary_message_content(...)` |
| MA-7 | **合并策略 — archived 优先** | `_merge_histories()` 以 archived 为主体，仅追加 checkpoint 中 archive 没有的消息 | ✅ 正确 | `threads.py:492-530`: `filtered_archived + new_messages` |
| MA-8 | **合并 — summary 消息不混入** | `_merge_histories` 明确跳过 checkpoint 中的 summary 消息 | ✅ 正确 | `threads.py:515`: `if _is_summary_message(msg_dict): continue` |
| MA-9 | **ORM model 完备** | `MessageArchive` 含 thread_id, message_index, message_id, message_type, content (JSONB), tool_calls, tool_call_id, metadata_json, created_at | ✅ 正确 | `core/orm.py:143-176` |
| MA-10 | **Migration chain 完整** | `down_revision = "d042a0ca1cb5"` 精确匹配 upstream 最后一个 migration (`migrate_run_status_to_langgraph_`) | ✅ 正确 | 完整链: `None → 7b79bfd12626 → ... → d042a0ca1cb5 → 20260128_message_archive` |
| MA-11 | **Foreign key + CASCADE** | `thread_id` FK → `thread.thread_id` ON DELETE CASCADE，删除 thread 自动清理归档 | ✅ 正确 | `orm.py:157` 和 migration L42-46 |
| MA-12 | **Unique index 防数据膨胀** | `(thread_id, message_index)` UNIQUE index | ✅ 正确 | `orm.py:175`: `unique=True` |
| MA-13 | **序列化覆盖率** | `_serialize_message()` 支持 Pydantic model_dump、dict()、raw dict 三种格式 | ✅ 正确 | `message_archive_service.py:259-313` |
| MA-14 | **反序列化兼容前端** | `_deserialize_message()` 还原 content（dict→text 展开）、保留 tool_calls/tool_call_id/additional_kwargs | ✅ 正确 | `message_archive_service.py:315-348` |
| MA-15 | **错误容忍 — 归档失败不阻塞 run** | `_archive_messages_from_checkpoint` 中 archive 异常仅 `logger.warning`，不抛出 | ✅ 正确 | `runs.py:963-967`: `except Exception as e: logger.warning(...)` |

### 需要实际部署验证的风险点

| # | 风险点 | 描述 | 验证方法 |
|---|--------|------|---------|
| MA-R1 | **SummarizationMiddleware 压缩时机** | 如果某些 graph 在单次 run 的**中间节点**就触发 summarization（而非 run 结束后），archive 读到的已经是压缩后的 messages | 在 summarization 前后分别打印 `state["messages"]` 长度，确认 archive 时消息完整 |
| MA-R2 | **并发 run 归档竞争** | 如果同一 thread 有并发 run（multitask_strategy="enqueue"），可能导致 `message_index` 唯一约束冲突 | `on_conflict_do_nothing` 已处理；但需确认不会丢消息 |
| MA-R3 | **tool_calls 序列化完整性** | LangGraph 的 `ToolMessage` 含 `tool_call_id` 和 `artifact` 字段，`_serialize_message` 是否完整保留 | 执行含 tool call 的 run，检查归档中 tool_calls 字段完整性 |
| MA-R4 | **content 类型多样性** | 消息 content 可能是 `str`、`dict`、`list[dict]`（多模态）。`_serialize_message` 将 `str` → `{"text": str}`，但 `list` 类型未处理 | 发送多模态消息（如带图片），检查归档 content 格式 |
| MA-R5 | **`uuid_generate_v4()` 扩展依赖** | `MessageArchive.id` 的 `server_default` 使用 `public.uuid_generate_v4()`，需要 PostgreSQL 的 `uuid-ossp` 扩展 | 确认 `postgres:15-alpine` 镜像默认包含此扩展，或在 migration 中 `CREATE EXTENSION IF NOT EXISTS "uuid-ossp"` |
| MA-R6 | **大消息量性能** | 每次 archive 都先查 `_get_existing_message_ids()`（全量 message_id SET），超长对话（>1000条）可能有性能问题 | 压测长对话场景下 archive 耗时 |

---

## 验证执行优先级建议

### 第一轮 — 启动即可验证

> P0 全部、P1-1、P1-2、P1-4、P1-5、P1-14、P3-4、P3-6、MA-10、MA-R5

重点确认：**服务能启动、graph 能加载、auth 能工作、migration chain 不断裂**。

### 第二轮 — 单次 graph 执行

> P1-7、P1-8、P1-9、P1-11、P1-12、P1-15、P1-16、P1-18、MA-1~MA-8

重点确认：**configurable 注入完整、archive 机制触发且正确、streaming 正常**。

### 第三轮 — 完整功能回归

> P1-13、P1-17、P1-10、P1-19、P2 全部、MA-R1~MA-R4

重点确认：**full-history 端点正确合并、环境配置完备**。

### 第四轮 — 边界 & 回归

> P3 全部、MA-R5、MA-R6

重点确认：**upstream 变更不影响自定义功能、边角场景不出问题**。

---

## 附录：Migration Chain

```
20250817 initial_schema    revision=7b79bfd12626  down=None
20250830 add_context        revision=5931cd77e93b  down=7b79bfd12626
20250831 cascade_delete     revision=11b2402d4be1  down=5931cd77e93b
20250913 version_table      revision=76dfdbe90d2b  down=11b2402d4be1
20250913 assistant_metadata revision=aee821a02fc8  down=76dfdbe90d2b
20251115 run_status_migrate revision=d042a0ca1cb5  down=aee821a02fc8
20260128 message_archive    revision=20260128_...  down=d042a0ca1cb5  ← 自定义
```

## 附录：关键文件索引

| 文件 | 用途 |
|------|------|
| `aegra.docker.json` | Docker 部署配置（graphs, auth, http, cors） |
| `aegra.json` | 开发环境配置 |
| `auth.py` | JWT/noop 双模式认证 |
| `docker-compose.custom.yml` | 完整 QSA 生产部署 |
| `docker-compose.yml` | 基础开发部署 |
| `deployments/docker/Dockerfile` | Docker 镜像构建 |
| `libs/aegra-api/src/aegra_api/api/runs.py` | Run CRUD + streaming + archive 触发 |
| `libs/aegra-api/src/aegra_api/api/threads.py` | Thread CRUD + full-history 端点 |
| `libs/aegra-api/src/aegra_api/services/langgraph_service.py` | Graph 加载 + inject_user_context |
| `libs/aegra-api/src/aegra_api/services/message_archive_service.py` | 消息归档服务 |
| `libs/aegra-api/src/aegra_api/services/streaming_service.py` | SSE streaming + event broker |
| `libs/aegra-api/src/aegra_api/core/orm.py` | ORM models (含 MessageArchive) |
| `libs/aegra-api/src/aegra_api/core/app_loader.py` | 自定义 FastAPI app 加载 |
| `libs/aegra-api/src/aegra_api/main.py` | App 入口 + middleware + lifespan |
| `libs/aegra-api/alembic/versions/20260128_add_message_archive.py` | 归档表 migration |
