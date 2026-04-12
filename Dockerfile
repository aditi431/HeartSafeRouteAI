FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "fastapi>=0.111" \
    "uvicorn>=0.30" \
    "requests>=2.32" \
    "pydantic>=2.7" \
    "numpy>=1.26" \
    "openai>=1.0.0" \
    "openenv-core>=0.2.0"

# Copy application code
COPY . .

# Set up user for HF Spaces
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
