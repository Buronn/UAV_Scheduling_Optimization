FROM python:3.8-slim-buster
# Se utiliza slim-buster porque alpine no permite instalar algunas dependencias

WORKDIR /app

COPY ./app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy source code
COPY ./app /app

# Run container in background
CMD tail -f /dev/null