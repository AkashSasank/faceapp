FROM python:3.11-slim

# Install Lambda Runtime Interface Client
RUN pip install --no-cache-dir awslambdaric

ARG WHEEL_PATH=./dist
COPY ${WHEEL_PATH}/ /tmp
RUN pip install  /tmp/*.whl
COPY requirements.txt .
RUN pip install -r requirements.txt
# Set workdir for Lambda code
WORKDIR /var/task

# Copy application code
COPY app.py .

# Entrypoint is RIC
ENTRYPOINT [ "python", "-m", "awslambdaric" ]

# Default handler
CMD [ "app.handler" ]
