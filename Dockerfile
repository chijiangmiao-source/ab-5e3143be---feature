FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY app ./app
COPY tests ./tests
COPY verify ./verify

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=12 \
  CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz',timeout=2)"]

CMD ["python", "-m", "app.server"]
