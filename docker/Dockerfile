FROM python:3.12-slim

WORKDIR /app

# Copy only what pip needs first so dependency layers cache across code edits.
COPY pyproject.toml README.md ./
COPY sketch/ ./sketch/
RUN pip install --no-cache-dir -e ".[streaming]"

COPY streaming/ ./streaming/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "streaming.consumer"]
