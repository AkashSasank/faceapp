## FaceApp

FaceApp is a Python package for:
- face extraction and embedding generation from local images
- vector indexing in Qdrant
- queue-based local scaling (API producer + worker consumers)

The package exposes a CLI command: `faceapp`.

---

## Architecture

- **API service**: accepts uploads and enqueues tasks
- **Worker service/processes**: consumes tasks and runs extraction + indexing
- **Queue**: pluggable queue interface (Redis backend configured by default)
- **Vector DB**: Qdrant

In Docker Compose, this is split into containers:
- `api`
- `worker`
- `redis`
- `qdrant`

---

## Dependencies

### Python dependencies

Defined in `pyproject.toml` (installed with the package), including:
- `fastapi`, `uvicorn`
- `qdrant-client`
- `redis`
- `pydantic`
- `python-multipart`
- DeepFace stack (`deepface`, `tf-keras`)
- `python-magic`

### System dependencies

You need `libmagic` available on the runtime host/container for image MIME validation.

macOS (Homebrew):
```bash
brew install libmagic
```

Debian/Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y libmagic1
```

---

## Install and Build

From `faceapp/`:

```bash
uv sync
```

Run via uv (dev):
```bash
uv run faceapp --help
```

Build distribution:
```bash
uv build
```

Build Sphinx docs:
```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Or with Makefile:
```bash
make -C docs html
```

Open generated docs at:
`docs/_build/html/index.html`

Install built wheel example:
```bash
pip install dist/faceapp-0.1.0-py3-none-any.whl
```

---

## Configuration

Primary config files (repo root):
- `configs/projects.yaml`
- `configs/extraction.yaml`
- `configs/indexing.yaml`

### Environment variables

Queue:
- `TASK_QUEUE_BACKEND` (default: `redis`)
- `REDIS_URL` (default: `redis://localhost:6379/0`)
- `TASK_QUEUE_NAME` (default: `faceapp:tasks`)

Qdrant:
- `QDRANT_URL` (example: `http://qdrant:6333`)
- `QDRANT_API_KEY` (optional)
- `QDRANT_PATH` (optional local path mode)

Filesystem:
- `FACEAPP_SHARED_DIR` (default: `/shared`)
- `FACEAPP_CONFIG_DIR` (for worker config lookup)

---

## CLI Commands

### Core runtime

Start API:
```bash
faceapp runserver --host 0.0.0.0 --port 8000
```

Start worker consumers:
```bash
faceapp runworkers --source queue --workers 3
```

Run API + workers together:
```bash
faceapp runall --host 0.0.0.0 --port 8000 --source queue --workers 3
```

Grouped aliases are also available:
```bash
faceapp run server ...
faceapp run workers ...
faceapp run all ...
```

### Ingest interface (simplified)

Immediate single-file ingest:
```bash
faceapp ingest --file ./dataset/raw/sample.jpg --config-name foo
```

Folder ingest loop (auto-continuous):
```bash
faceapp ingest --folder ./dataset/raw --config-name foo --poll-interval 2
```

### Search interface (simplified)

Immediate single-file search:
```bash
faceapp search --file ./dataset/faces/query.jpg --config-name foo
```

Folder search loop (auto-continuous):
```bash
faceapp search --folder ./dataset/faces --config-name foo --poll-interval 2
```

Optional search tuning:
```bash
faceapp search \
	--file ./dataset/faces/query.jpg \
	--config-name foo \
	--embedding-models Facenet512 \
	--thresholds 0.4
```

---

## Worker task file format (task-file mode)

Sample file: `configs/worker_tasks.local.sample.json`

Run worker in task-file mode:
```bash
faceapp runworkers --source task-file --task-file ./configs/worker_tasks.local.sample.json
```

---

## Docker Compose Usage

From repo root:

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Redis: `localhost:6379`
- Qdrant: `http://localhost:6333`

Check status:
```bash
docker compose ps
```

Stop:
```bash
docker compose down
```

---

## Common Issues

### `failed to find libmagic`
Install system `libmagic` (see dependency section), then restart your environment.

### Queue connection refused
Ensure Redis is running and `REDIS_URL` is correct.

### No search matches
- Verify data was ingested for the same `project_id`
- Verify embedding model + threshold configuration
- Verify Qdrant is reachable via `QDRANT_URL`

