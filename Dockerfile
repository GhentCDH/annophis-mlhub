FROM ghcr.io/astral-sh/uv:0.11.7-alpine3.23

# dont include dev dependencies
ENV UV_NO_DEV=1

COPY . /app

WORKDIR /app

RUN uv sync --locked

CMD ["uv", "run", "main.py"]
