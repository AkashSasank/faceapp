FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
ARG WHEEL_PATH=./dist
COPY ${WHEEL_PATH}/ /tmp
RUN pip install --no-cache-dir /tmp

# Copy lambda app
COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]
