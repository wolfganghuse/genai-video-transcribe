ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg

COPY requirements.txt ./

USER 0

#RUN pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

ENV XDG_CACHE_HOME /app/model

RUN python -c 'import whisper; model = whisper.load_model("base")'
RUN useradd -m -u 1001 appuser && chown -R appuser $APP_HOME

USER 1001

COPY . ./


EXPOSE 8080

CMD python -m parliament .
