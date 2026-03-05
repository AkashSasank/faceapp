FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libmagic1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libx11-6 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libmagic1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libx11-6 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist /dist
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /dist/*.whl \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir opencv-python-headless

EXPOSE 8000
EXPOSE 8010
