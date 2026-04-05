<<<<<<< HEAD
FROM python:3.11-slim-bookworm AS base
=======
ARG PY_VERSION=3.12
FROM python:${PY_VERSION}-slim-bookworm AS base
ARG PY_VERSION
>>>>>>> origin/dev_ALAGENT-HKU-merged

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Create non-root user for runtime
RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage
# -----------------------------
FROM base AS builder
<<<<<<< HEAD
=======

# Retrieve the uv binary directly from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /bin/uv

# Install system build dependencies required for compiling Python extensions.
>>>>>>> origin/dev_ALAGENT-HKU-merged
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
# Copy project metadata and source
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install project (builds wheel and installs dependencies specified in pyproject)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir email-validator python-multipart psycopg2-binary bcrypt passlib python-jose
=======
# Copy aegra-api package files and workspace lockfile
COPY libs/aegra-api/pyproject.toml libs/aegra-api/README.md ./
COPY uv.lock ./

# Install dependencies from lockfile (includes dev deps for example agents + QSA deps).
# --extra qsa: include quantitative_strategy_agent optional dependencies (deepagents, langchain, etc.)
RUN uv export --frozen --no-emit-project --extra qsa --format=requirements-txt > requirements.txt && \
    uv pip install --system -r requirements.txt && \
    rm requirements.txt

# Export constraints file for runtime safety (prevents version drift from any runtime pip install)
RUN uv export --frozen --no-emit-project --extra qsa --format=requirements-txt --no-hashes > /constraints.txt

# Copy the actual project source code and forced includes (alembic).
COPY libs/aegra-api/src/ ./src/
COPY libs/aegra-api/alembic.ini ./alembic.ini
COPY libs/aegra-api/alembic/ ./alembic/

# Install the project package itself.
RUN uv pip install --system --no-deps .
>>>>>>> origin/dev_ALAGENT-HKU-merged

# -----------------------------
# Final, minimal runtime image
# -----------------------------
FROM base AS final

<<<<<<< HEAD
# Install only minimal runtime libs (psycopg[binary] used, so libpq not required at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and console scripts from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy runtime assets required by the app and alembic
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/
COPY aegra.json ./aegra.json
COPY auth.py ./auth.py
COPY graphs/ ./graphs/

# Copy src to keep compatibility with current compose command using 'src.agent_server.main:app'
# (We can switch compose to 'agent_server.main:app' later and drop this for a smaller image.)
COPY src/ ./src/

EXPOSE 8000
=======
# Install only minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libpq-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python${PY_VERSION}/site-packages/ /usr/local/lib/python${PY_VERSION}/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy constraints file for runtime pip install safety
COPY --from=builder /constraints.txt /constraints.txt

# Copy runtime assets required by the app and alembic
COPY libs/aegra-api/alembic.ini ./alembic.ini
COPY libs/aegra-api/alembic/ ./alembic/
COPY aegra.json ./aegra.json
COPY examples/ ./examples/

EXPOSE 2026
>>>>>>> origin/dev_ALAGENT-HKU-merged

# Run as non-root
USER app

<<<<<<< HEAD
# Leave default CMD empty so docker-compose command can run migrations then start uvicorn
# CMD ["uvicorn", "src.agent_server.main:app", "--host", "0.0.0.0", "--port", "8000"]


=======
# Default command - can be overridden by docker-compose
CMD ["aegra", "serve"]
>>>>>>> origin/dev_ALAGENT-HKU-merged
