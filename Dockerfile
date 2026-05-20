# Capture or diff a snapshot from inside a container:
#
#   docker build -t mlenv .
#   docker run --rm mlenv snapshot
#
# Mount snapshots to diff them against a committed baseline:
#
#   docker run --rm -v "$PWD:/w" -w /w mlenv diff baseline.json current.json
#
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/jmweb-org/mlenv"
LABEL org.opencontainers.image.description="Snapshot the machine learning environment and diff two snapshots."
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

ENTRYPOINT ["mlenv"]
CMD ["--help"]
