# Runtime image for the full app (Wordle + Connections). Built from the staged folder that
# deploy/build_space.py creates, so paths below are relative to that folder.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# CPU-only PyTorch (the default Linux wheel pulls ~2 GB of CUDA libraries we cannot use here)
COPY --chown=user requirements-app.txt .
RUN pip install --user torch --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements-app.txt

COPY --chown=user . .

# Bake the slow one-time work into the image so the app starts in seconds:
# the three embedding models, the Wordle pattern matrix, and the 2.6M-partition table.
RUN python -c "from sentence_transformers import SentenceTransformer as S; [S(n, device='cpu') for n in ('all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'BAAI/bge-large-en-v1.5')]" \
 && python -c "from src.wordle.solvers.base import WordleData; WordleData.load()" \
 && python -c "from src.connections.partition import all_partitions; all_partitions()"

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
