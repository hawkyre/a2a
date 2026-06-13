FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

COPY cs_agent/ .
COPY kb/ ./kb/

EXPOSE 9002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9002"]
