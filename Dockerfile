# Batch CLI image: generates the seeded synthetic dataset and runs the engine.
# Not a server, so there is no HEALTHCHECK. No API key is needed for `run`/`verify`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY code/requirements.txt code/requirements.txt
RUN pip install -r code/requirements.txt

COPY code/ code/
COPY data/ data/
COPY docs/ docs/

RUN useradd --create-home --uid 10001 engine \
    && python data/generate_synthetic.py \
    && chown -R engine:engine /app
USER engine

WORKDIR /app/code
ENTRYPOINT ["python", "main.py"]
CMD ["run"]
