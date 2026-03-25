FROM python:3.10-slim

ARG RUN_ID

RUN echo "Downloading model with RUN_ID=$RUN_ID"

CMD ["echo", "Model container started!"]