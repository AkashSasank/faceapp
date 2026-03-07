## FaceApp

FaceApp is a Python package for:
- face extraction and embedding generation from local images
- vector indexing in Qdrant
- CLI-based ingestion and search workflows

The package exposes a CLI command: `faceapp`.

---

## Architecture

- **CLI**: runs ingest and search commands
- **Processing runtime**: extraction + indexing pipeline
- **Vector DB**: Qdrant

---

## Dependencies

### Python dependencies

Defined in `pyproject.toml` (installed with the package), including:
- `qdrant-client`
- `pydantic`
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
- `configs/search.yaml`

### Environment variables

Qdrant:
- `QDRANT_URL` (example: `http://qdrant:6333`)
- `QDRANT_API_KEY` (optional)
- `QDRANT_PATH` (optional local path mode)

Filesystem:
- `FACEAPP_SHARED_DIR` (default: `/shared`)
- `FACEAPP_CONFIG_DIR` (for config lookup)

---

## CLI Commands

### Ingest interface

Immediate single-file ingest:
```bash
faceapp ingest --file ./dataset/raw/sample.jpg --config-name foo
```

Folder ingest loop (auto-continuous):
```bash
faceapp ingest --folder ./dataset/raw --config-name foo --poll-interval 2
```

Concurrent folder ingest:
```bash
faceapp ingest --folder ./dataset/raw --config-name foo --ingest-workers 4
```

Grouped alias:
```bash
faceapp run ingest --file ./dataset/raw/sample.jpg --config-name foo
```

### Search interface

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

Grouped alias:
```bash
faceapp run search --file ./dataset/faces/query.jpg --config-name foo
```

By default, `faceapp search` resolves model and threshold from
`configs/search.yaml` using the project's `config_refs.search_profile`
(or `config_refs.search.profile`), and can inherit models/thresholds from
`configs/extraction.yaml` via `extraction_profile` in search profile.

CLI flags `--embedding-models` and `--thresholds` override configured values.

---

## Docker Compose Usage

From repo root:

```bash
docker compose up -d qdrant
```

Service:
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

### No search matches
- Verify data was ingested for the same `project_id`
- Verify embedding model + threshold configuration
- Verify Qdrant is reachable via `QDRANT_URL`
