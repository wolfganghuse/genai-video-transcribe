ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the source code into the container.
COPY . .

RUN pip install -r requirements.txt

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser


# Switch to the non-privileged user to run the application.
USER appuser


# Expose the port that the application listens on.
EXPOSE 8504

HEALTHCHECK CMD curl --fail http://localhost:8504/_stcore/health
# Run the application.
CMD streamlit run app.py --server.port=8504 --server.address=0.0.0.0