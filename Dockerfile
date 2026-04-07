FROM python:3.11-slim

# HF Spaces requires non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/          ./src/
COPY server/       ./server/
COPY openenv.yaml  .

# Copy entry points used by judges
COPY inference.py  .
COPY validate.py   .

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src:/app
ENV PORT=7860
ENV ER_TASK=easy

EXPOSE 7860

# server/app.py is the OpenEnv-spec entry point
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
