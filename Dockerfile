FROM python:3.11-slim

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN pip install --upgrade pip

COPY /app /app
COPY Dockerfile Dockerfile

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install fastapi uvicorn

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
