FROM  ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
ARG WHEEL_PATH=./dist
COPY ${WHEEL_PATH}/ /tmp
RUN pip install --no-cache-dir /tmp/*.whl
# Copy lambda app
COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]
