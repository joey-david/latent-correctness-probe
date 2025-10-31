FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . /app

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "main"]
CMD []
