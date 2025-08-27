FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY app ./app
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["python", "-m", "app.main"]
