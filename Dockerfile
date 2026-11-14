# One image for the router and the mock worker (the GPU deployment uses the official vLLM image for workers).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --uid 10001 --no-create-home app
USER 10001

EXPOSE 8000
# Router by default; the worker Deployment overrides the command.
ENTRYPOINT ["python", "-m"]
CMD ["llmserve.router", "--host", "0.0.0.0", "--port", "8000", "--worker-dns", "llmserve-worker:8001"]
