<p align="center">
  <img src="docs/images/banner.png" alt="Aegra banner" />
</p>

<<<<<<< HEAD
# Aegra - Open Source LangGraph Platform Alternative

<p align="center">
  <strong>Self-hosted AI agent backend. LangGraph power without vendor lock-in.</strong>
</p>

<p align="center">
  <a href="https://github.com/ibbybuilds/aegra/stargazers"><img src="https://img.shields.io/github/stars/ibbybuilds/aegra" alt="GitHub stars"></a>
  <a href="https://github.com/ibbybuilds/aegra/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ibbybuilds/aegra" alt="License"></a>
  <a href="https://github.com/ibbybuilds/aegra/issues"><img src="https://img.shields.io/github/issues/ibbybuilds/aegra" alt="Issues"></a>
  <a href="https://discord.com/invite/D5M3ZPS25e"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://www.reddit.com/r/aegra/"><img src="https://img.shields.io/badge/Reddit-Join-orange?logo=reddit&logoColor=white" alt="Reddit"></a>
  <a href="https://x.com/intent/user?screen_name=ibbyybuilds"><img src="https://img.shields.io/twitter/follow/ibbyybuilds?style=social" alt="Follow on X"></a>
</p>

Replace LangGraph Platform with your own infrastructure. Built with FastAPI + PostgreSQL for developers who demand complete control over their agent orchestration.

**🔌 Agent Protocol Compliant**: Aegra implements the [Agent Protocol](https://github.com/langchain-ai/agent-protocol) specification, an open-source standard for serving LLM agents in production.

**🎯 Perfect for:** Teams escaping vendor lock-in • Data sovereignty requirements • Custom deployments • Cost optimization

## 🆕 What's New

- **🗂️ [Semantic Store](docs/semantic-store.md)**: Vector embeddings with pgvector for semantic similarity search in your agent memory
- **📦 [Dependencies Config](docs/dependencies.md)**: Add shared utility modules to Python path for graph imports
- **🎨 LangGraph Studio Support**: Full compatibility with LangGraph Studio for visual graph debugging and development
- **🤖 AG-UI / CopilotKit Support**: Seamless integration with AG-UI and CopilotKit-based clients for enhanced user experiences
- **⬆️ LangGraph v1.0.0**: Upgraded to LangGraph and LangChain v1.0.0 with latest features and improvements
- **🤝 Human-in-the-Loop**: Interactive agent workflows with approval gates and user intervention points
- **📊 [Langfuse Integration](docs/langfuse-usage.md)**: Complete observability and tracing for your agent runs


## 🔥 Why Aegra vs LangGraph Platform?

| Feature                | LangGraph Platform         | Aegra (Self-Hosted)                               |
| ---------------------- | -------------------------- | ------------------------------------------------- |
| **Cost**               | $$$+ per month             | **Free** (self-hosted), infra-cost only           |
| **Data Control**       | Third-party hosted         | **Your infrastructure**                           |
| **Vendor Lock-in**     | High dependency            | **Zero lock-in**                                  |
| **Customization**      | Platform limitations       | **Full control**                                  |
| **API Compatibility**  | LangGraph SDK              | **Same LangGraph SDK**                            |
| **Authentication**     | Lite: no custom auth       | **Custom auth** (JWT/OAuth/Firebase/NoAuth)       |
| **Database Ownership** | No bring-your-own database | **BYO Postgres** (you own credentials and schema) |
| **Tracing/Telemetry**  | Forced LangSmith in SaaS   | **Your choice** (Langfuse/None)                   |

## ✨ Core Benefits

- **🏠 Self-Hosted**: Run on your infrastructure, your rules
- **🔄 Drop-in Replacement**: Use existing LangGraph Client SDK without changes
- **🛡️ Production Ready**: PostgreSQL persistence, streaming, authentication
- **📊 Zero Vendor Lock-in**: Apache 2.0 license, open source, full control
- **🚀 Fast Setup**: 5-minute deployment with Docker
- **🔌 Agent Protocol Compliant**: Implements the open-source [Agent Protocol](https://github.com/langchain-ai/agent-protocol) specification
- **💬 Agent Chat UI Compatible**: Works seamlessly with [LangChain's Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui)
=======
<h1 align="center">Aegra</h1>

<p align="center">
  <strong>Self-hosted LangSmith Deployments alternative. Your infrastructure, your rules.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/aegra-api/"><img src="https://img.shields.io/pypi/v/aegra-api?label=aegra-api&color=blue" alt="PyPI API"></a>
  <a href="https://pypi.org/project/aegra-cli/"><img src="https://img.shields.io/pypi/v/aegra-cli?label=aegra-cli&color=blue" alt="PyPI CLI"></a>
  <a href="https://github.com/ibbybuilds/aegra/actions/workflows/ci.yml"><img src="https://github.com/ibbybuilds/aegra/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://app.codecov.io/gh/ibbybuilds/aegra"><img src="https://codecov.io/gh/ibbybuilds/aegra/graph/badge.svg" alt="Codecov"></a>
</p>

<p align="center">
  <a href="https://github.com/ibbybuilds/aegra/stargazers"><img src="https://img.shields.io/github/stars/ibbybuilds/aegra" alt="GitHub stars"></a>
  <a href="https://github.com/ibbybuilds/aegra/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ibbybuilds/aegra" alt="License"></a>
  <a href="https://discord.com/invite/D5M3ZPS25e"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://patreon.com/aegra"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=github-sponsors&logoColor=white" alt="Sponsor"></a>
</p>

---
>>>>>>> origin/dev_ALAGENT-HKU-merged

Aegra is a drop-in replacement for LangSmith Deployments. Use the same LangGraph SDK, same APIs, but run it on your own infrastructure with PostgreSQL persistence.

**Works with:** [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) | [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) | [AG-UI / CopilotKit](https://github.com/CopilotKit/CopilotKit)

<<<<<<< HEAD
- Python 3.11+
- Docker (for PostgreSQL)
- uv (Python package manager)
=======
## 🚀 Quick Start
>>>>>>> origin/dev_ALAGENT-HKU-merged

### Using the CLI (Recommended)

**Prerequisites:** Python 3.12+, Docker (for PostgreSQL)

```bash
<<<<<<< HEAD
# Clone and setup
git clone https://github.com/ibbybuilds/aegra.git
cd aegra
# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync env and dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Mac/Linux
# OR .venv/Scripts/activate  # Windows

# Environment
cp .env.example .env

# Start everything (database + migrations + server)
docker compose up aegra
=======
pip install aegra-cli

# Initialize a new project — prompts for location, template, and name
aegra init

# Follow the printed next steps:
cd <your-project>
cp .env.example .env     # Add your OPENAI_API_KEY to .env
uv sync                  # Install dependencies
uv run aegra dev         # Start PostgreSQL + dev server
>>>>>>> origin/dev_ALAGENT-HKU-merged
```

> **Note:** Always install `aegra-cli` directly — not the `aegra` meta-package. The `aegra` package on PyPI is a convenience wrapper that does not support version pinning.

### From Source

```bash
<<<<<<< HEAD
# Health check
curl http://localhost:8000/health

# Interactive API docs
open http://localhost:8000/docs
```

You now have a self-hosted LangGraph Platform alternative running locally.

## 💬 Agent Chat UI Compatible

Aegra works seamlessly with [LangChain's Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui). Simply set `NEXT_PUBLIC_API_URL=http://localhost:8000` and `NEXT_PUBLIC_ASSISTANT_ID=agent` in your Agent Chat UI environment to connect to your Aegra backend.

## 👨‍💻 For Developers

**New to database migrations?** Check out our guides:

- **📚 [Developer Guide](docs/developer-guide.md)** - Complete setup, migrations, and development workflow
- **⚡ [Migration Cheatsheet](docs/migration-cheatsheet.md)** - Quick reference for common commands

**Quick Development Commands:**

```bash
# Docker development (recommended)
docker compose up aegra

# Local development
docker compose up postgres -d
python3 scripts/migrate.py upgrade
python3 run_server.py

# Create new migration
python3 scripts/migrate.py revision --autogenerate -m "Add new feature"
```

> **Note**: The current `docker-compose.yml` is optimized for **development** with hot-reload, volume mounts, and debug settings. For production deployment considerations, see [production-docker-setup.md](docs/production-docker-setup.md).

## 🧪 Try the Example Agent

Use the **same LangGraph Client SDK** you're already familiar with:

```python
import asyncio
from langgraph_sdk import get_client

async def main():
    # Connect to your self-hosted Aegra instance
    client = get_client(url="http://localhost:8000")

    # Create assistant (same API as LangGraph Platform)
    assistant = await client.assistants.create(
        graph_id="agent",
        if_exists="do_nothing",
        config={},
    )
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Stream responses (identical to LangGraph Platform)
    stream = client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={
            "messages": [
                {"type": "human", "content": [{"type": "text", "text": "hello"}]}
            ]
        },
        stream_mode=["values", "messages-tuple", "custom"],
        on_disconnect="cancel",
    )

    async for chunk in stream:
        print(f"event: {getattr(chunk, 'event', None)}, data: {getattr(chunk, 'data', None)}")

asyncio.run(main())
```

**Key Point**: Your existing LangGraph applications work without modification! 🔄

## 🏗️ Architecture

```text
Client → FastAPI → LangGraph SDK → PostgreSQL
 ↓         ↓           ↓             ↓
Agent    HTTP     State        Persistent
SDK      API    Management      Storage
```

### Components

- **FastAPI**: Agent Protocol-compliant HTTP layer
- **LangGraph**: State management and graph execution
- **PostgreSQL**: Durable checkpoints and metadata
- **Agent Protocol**: Open-source specification for LLM agent APIs
- **Config-driven**: `aegra.json` for graph definitions

## 🛣️ Custom Routes

Aegra supports adding custom FastAPI endpoints to extend your server with additional functionality. This is useful for webhooks, admin panels, custom UI, or any other endpoints you need.

### Configuration

Add custom routes by configuring the `http.app` field in your `aegra.json` or `langgraph.json`:

```json
{
  "graphs": {
    "agent": "./graphs/react_agent/graph.py:graph"
  },
  "http": {
    "app": "./custom_routes.py:app",
    "enable_custom_route_auth": false,
    "cors": {
      "allow_origins": ["https://example.com"],
      "allow_credentials": true
    }
  }
}
```

### Creating Custom Routes

Create a Python file (e.g., `custom_routes.py`) with your FastAPI app:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/custom/hello")
async def hello():
    return {"message": "Hello from custom route!"}

@app.post("/custom/webhook")
async def webhook(data: dict):
    return {"received": data, "status": "processed"}

# You can override shadowable routes like the root
@app.get("/")
async def custom_root():
    return {"message": "Custom Aegra Server", "custom": True}
```

### Route Priority

Custom routes follow this priority order:

1. **Unshadowable routes**: `/health`, `/ready`, `/live`, `/docs`, `/openapi.json` - always accessible
2. **Custom user routes**: Your endpoints take precedence
3. **Shadowable routes**: `/`, `/info` - can be overridden by custom routes
4. **Protected core routes**: `/assistants`, `/threads`, `/runs`, `/store` - cannot be overridden

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `app` | `string` | `None` | Import path to custom FastAPI/Starlette app (format: `"path/to/file.py:variable"`) |
| `enable_custom_route_auth` | `boolean` | `false` | Apply Aegra's authentication middleware to custom routes |
| `cors` | `object` | `None` | Custom CORS configuration |

### Example Use Cases

- **Webhooks**: Add endpoints to receive external webhooks
- **Admin Panel**: Build custom admin interfaces
- **Custom UI**: Serve additional frontend applications
- **Metrics**: Add custom monitoring endpoints
- **Integration**: Connect with third-party services

See [`custom_routes_example.py`](custom_routes_example.py) for a complete example.
=======
git clone https://github.com/ibbybuilds/aegra.git
cd aegra
cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker compose up
```

Open [http://localhost:2026/docs](http://localhost:2026/docs) to explore the API.

Your existing LangGraph code works without changes:

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2026")

assistant = await client.assistants.create(graph_id="agent")
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"messages": [{"type": "human", "content": "Hello!"}]},
):
    print(chunk)
```

## 🔥 Why Aegra?
>>>>>>> origin/dev_ALAGENT-HKU-merged

*Based on [LangChain pricing](https://www.langchain.com/pricing) as of February 2026. An enterprise tier with self-hosting is also available at custom pricing.*

<<<<<<< HEAD
```text
aegra/
├── aegra.json           # Graph configuration
├── auth.py              # Authentication setup
├── custom_routes.py     # Custom FastAPI endpoints (optional)
├── graphs/              # Agent definitions
│   └── react_agent/     # Example ReAct agent
├── src/agent_server/    # FastAPI application
│   ├── main.py         # Application entrypoint
│   ├── core/           # Database & infrastructure
│   ├── models/         # Pydantic schemas
│   ├── services/       # Business logic
│   └── utils/          # Helper functions
├── tests/              # Test suite
└── deployments/        # Docker & K8s configs
```
=======
| | LangSmith Deployments | Aegra |
|:--|:--|:--|
| **Deploy agents** | Local dev only (Free), paid cloud (Plus) | Free, unlimited |
| **Custom auth** | Not available (Free), available (Plus) | Python handlers (JWT/OAuth/Firebase) |
| **Self-hosted** | Enterprise only (license key required) | Always (Apache 2.0) |
| **Own database** | Managed only (Free/Plus), bring your own (Enterprise) | Bring your own Postgres |
| **Tracing** | LangSmith only | Any OTLP backend (Langfuse, Phoenix, etc.) |
| **Data residency** | LangChain cloud (Free/Plus), your infrastructure (Enterprise) | Your infrastructure |
| **SDK** | LangGraph SDK | Same LangGraph SDK |
>>>>>>> origin/dev_ALAGENT-HKU-merged

## ✨ Features

<<<<<<< HEAD
### Environment Variables

Copy `.env.example` to `.env` and configure values:

```bash
cp .env.example .env
```

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/aegra

# Authentication (extensible)
AUTH_TYPE=noop  # noop, custom

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=INFO
ENV_MODE=LOCAL # DEVELOPMENT, PRODUCTION, LOCAL (PRODUCTION outputs JSON logs)
LOG_VERBOSITY=standard # standard, verbose (verbose outputs request-id for each request)

# LLM Providers
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...
# TOGETHER_API_KEY=...

LANGFUSE_LOGGING=true
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Graph Configuration

`aegra.json` defines your agent graphs:

```json
{
  "graphs": {
    "agent": "./graphs/react_agent/graph.py:graph"
  }
}
```

### Dependencies (Optional)

Add shared utility modules to the Python path for graph imports:

```json
{
  "graphs": { ... },
  "dependencies": [
    "./shared",
    "./libs/common"
  ]
}
```

Paths are resolved relative to the config file. This matches LangGraph CLI behavior.

📚 **[Full Documentation](docs/dependencies.md)** - Path resolution, use cases, and examples.

### Semantic Store (Optional)

Enable semantic similarity search for your agent's memory using pgvector:

```json
{
  "graphs": { ... },
  "store": {
    "index": {
      "dims": 1536,
      "embed": "openai:text-embedding-3-small",
      "fields": ["$"]
    }
  }
}
```

**Options:** `dims` (required), `embed` (required), `fields` (optional, default `["$"]`)

**Supported embedding providers:**
- `openai:text-embedding-3-small` (1536 dims)
- `openai:text-embedding-3-large` (3072 dims)
- `bedrock:amazon.titan-embed-text-v2:0` (1024 dims)
- `cohere:embed-english-v3.0` (1024 dims)

📚 **[Full Documentation](docs/semantic-store.md)** - Configuration, usage examples, and troubleshooting.

## 🎯 What You Get

### ✅ **Core Features**

- [Agent Protocol](https://github.com/langchain-ai/agent-protocol)-compliant REST endpoints
- Persistent conversations with PostgreSQL checkpoints
- Streaming responses with network resilience
- Config-driven agent graph management
- Compatible with LangGraph Client SDK
- Human-in-the-loop support
- [Langfuse integration](docs/langfuse-usage.md) for observability and tracing

### ✅ **Production Ready**

- Docker containerization (development-focused setup; production considerations documented)
- Database migrations with Alembic
- Comprehensive test suite
- Authentication framework (JWT/OAuth ready)
- Health checks and monitoring endpoints

> **Production Deployment**: The included `docker-compose.yml` is optimized for development. For production deployment guidance, see [production-docker-setup.md](docs/production-docker-setup.md).

### ✅ **Developer Experience**

- Interactive API documentation (FastAPI)
- Hot reload in development
- Clear error messages and logging
- Extensible architecture
- **📚 [Developer Guide](docs/developer-guide.md)** - Complete setup, migrations, and development workflow
- **⚡ [Migration Cheatsheet](docs/migration-cheatsheet.md)** - Quick reference for common commands

## Star History

<a href="https://www.star-history.com/#ibbybuilds/aegra&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ibbybuilds/aegra&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ibbybuilds/aegra&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ibbybuilds/aegra&type=Date" />
 </picture>
</a>

## 🛣️ Roadmap

**✅ Completed**

- Agent Chat UI compatibility
- Agent Protocol API implementation
- PostgreSQL persistence and streaming
- Authentication framework
- Human-in-the-loop support
- Langfuse integration

**🎯 Next**

- Custom HTTP endpoints support
- Generative user interfaces support
- Redis-backed streaming buffers
- Advanced deployment recipes

**🚀 Future**

- Performance optimizations
- Custom UI themes and branding
- Aegra CLI for migration and image building

## 🤝 Contributing

We welcome contributions! Here's how you can help:

**🐛 Issues & Bugs**

- Report bugs with detailed reproduction steps
- Suggest new features and improvements
- Help with documentation

**💻 Code Contributions**

- Improve Agent Protocol spec alignment
- Add authentication backends
- Enhance testing coverage
- Optimize performance

**📚 Documentation**

- Deployment guides
- Integration examples
- Best practices

**Get Started**: Check out [CONTRIBUTING.md](CONTRIBUTING.md), our [Developer Guide](docs/developer-guide.md), and our [good first issues](https://github.com/ibbybuilds/aegra/labels/good%20first%20issue).

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

<p align=\"center\">
  <strong>⭐ If Aegra helps you escape vendor lock-in, please star the repo! ⭐</strong><br>
  <sub>Built with ❤️ by developers who believe in infrastructure freedom</sub>
</p>
=======
- **[Agent Protocol](https://github.com/langchain-ai/agent-protocol) compliant** - Works with Agent Chat UI, LangGraph Studio, CopilotKit
- **[Human-in-the-loop](https://docs.aegra.dev/guides/human-in-the-loop)** - Approval gates and user intervention points
- **[Streaming](https://docs.aegra.dev/guides/streaming)** - Real-time responses with network resilience
- **[Persistent state](https://docs.aegra.dev/guides/threads-and-state)** - PostgreSQL checkpoints via LangGraph
- **[Configurable auth](https://docs.aegra.dev/guides/authentication)** - JWT, OAuth, Firebase, or none
- **[Unified Observability](https://docs.aegra.dev/guides/observability)** - Fan-out tracing support via OpenTelemetry
- **[Semantic store](https://docs.aegra.dev/guides/semantic-store)** - Vector embeddings with pgvector
- **[Custom routes](https://docs.aegra.dev/guides/custom-routes)** - Add your own FastAPI endpoints

## 🛠️ CLI Commands

```bash
aegra init              # Interactive — asks for location, template, and name
aegra init ./my-agent   # Create at path (still prompts for template)

aegra dev               # Start development server (hot reload + auto PostgreSQL)
aegra serve             # Start production server (no reload)
aegra up                # Build and start all Docker services
aegra down              # Stop Docker services

aegra version           # Show version info
```

## 📚 Documentation

**[docs.aegra.dev](https://docs.aegra.dev)** — Full documentation with guides, API reference, and configuration.

| Topic | Description |
|-------|-------------|
| [Quickstart](https://docs.aegra.dev/quickstart) | Get a running server in under 5 minutes |
| [Configuration](https://docs.aegra.dev/reference/configuration) | aegra.json format and all options |
| [Authentication](https://docs.aegra.dev/guides/authentication) | JWT, OAuth, Firebase, or custom auth handlers |
| [Streaming](https://docs.aegra.dev/guides/streaming) | 8 SSE stream modes with reconnection |
| [Store](https://docs.aegra.dev/guides/store) | Key-value and semantic search storage |
| [Observability](https://docs.aegra.dev/guides/observability) | Fan-out tracing to Langfuse, Phoenix, or any OTLP backend |
| [Deployment](https://docs.aegra.dev/guides/deployment) | Docker, PaaS, and Kubernetes deployment |
| [Migration](https://docs.aegra.dev/migration) | Migrate from LangSmith Deployments |

## 💬 Community & Support

- **[Discord](https://discord.com/invite/D5M3ZPS25e)** - Chat with the community
- **[GitHub Discussions](https://github.com/ibbybuilds/aegra/discussions)** - Ask questions, share ideas
- **[GitHub Issues](https://github.com/ibbybuilds/aegra/issues)** - Report bugs

## 🏗️ Built With

- [FastAPI](https://fastapi.tiangolo.com/) - HTTP layer
- [LangGraph](https://github.com/langchain-ai/langgraph) - State management & graph execution
- [PostgreSQL](https://www.postgresql.org/) - Persistence & checkpoints
- [OpenTelemetry](https://opentelemetry.io/) - Observability standard
- [pgvector](https://github.com/pgvector/pgvector) - Vector embeddings

## 🤝 Contributing

We welcome contributions! See [Contributing guide](https://docs.aegra.dev/guides/contributing) and check out [good first issues](https://github.com/ibbybuilds/aegra/labels/good%20first%20issue).

## 💖 Support the Project

The best contribution is code, PRs, and bug reports - that's what makes open source thrive.

For those who want to support Aegra financially, whether you're using it in production or just believe in what we're building, you can [become a sponsor](https://patreon.com/aegra). Sponsorships help keep development active and the project healthy.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE).

---

<p align="center">
  <strong>⭐ Star us if Aegra helps you escape vendor lock-in ⭐</strong>
</p>

<a href="https://www.star-history.com/#ibbybuilds/aegra&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ibbybuilds/aegra&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ibbybuilds/aegra&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ibbybuilds/aegra&type=Date" />
  </picture>
</a>
>>>>>>> origin/dev_ALAGENT-HKU-merged
