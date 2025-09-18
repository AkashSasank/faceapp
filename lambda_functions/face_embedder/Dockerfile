# ---------- Builder Stage ----------
FROM ubuntu:22.04 AS builder

# Install Python and build tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Upgrade pip first
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements and install to /opt/python (Lambda layer style)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /opt/python

# If you have your in-house wheel(s), copy them in and install
ARG WHEEL_PATH=./dist
COPY ${WHEEL_PATH}/ /tmp
RUN pip install --no-cache-dir /tmp/*.whl -t /opt/python


# ---------- Runtime Stage ----------
FROM public.ecr.aws/lambda/python:3.11

# Copy only the dependencies from builder
COPY --from=builder /opt/python /opt/python

# Copy your Lambda handler
COPY app.py ${LAMBDA_TASK_ROOT}

# Lambda entrypoint
CMD ["app.handler"]
