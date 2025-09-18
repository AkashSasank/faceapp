# ---------- Builder Stage ----------
#FROM ubuntu/python:3.13-25.04_stable AS builder
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

#FROM public.ecr.aws/lambda/python:3.10
## Install Python and build tools
##RUN yum  install -y python3 python3-pip python3-dev build-essential
## Install build tools including g++
##RUN apk add --no-cache gcc musl-dev python3-dev \
##    && rm -rf /var/cache/apk/* \
#   # Install build essentials including g++
##RUN yum install -y build-essential gcc python3-dev
## Upgrade pip first
##RUN python3 -m pip install --upgrade pip setuptools wheel
#
## Copy requirements and install to /opt/python (Lambda layer style)
#COPY requirements.txt .
#RUN pip install -r requirements.txt
#
## If you have your in-house wheel(s), copy them in and install
#ARG WHEEL_PATH=./dist
#COPY ${WHEEL_PATH}/ /tmp
#RUN pip install  /tmp/*.whl
#
#
## ---------- Runtime Stage ----------
##FROM public.ecr.aws/lambda/python:3.10
#
## Copy only the dependencies from builder
##COPY --from=builder /opt/python /opt/python
#
## Copy your Lambda handler
#COPY app.py ${LAMBDA_TASK_ROOT}
#
## Lambda entrypoint
#CMD ["app.handler"]
