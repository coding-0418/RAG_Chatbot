FROM python:3.11-slim

WORKDIR /app

# System deps for lxml/pypdf wheels, curl for the container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The vectorstore is data, not code: build it once with `docker run <image> python ingest.py`
# against a mounted ./vectorstore volume before (or instead of) starting the app, so
# re-ingesting doesn't require rebuilding the image and the app can start against a
# vectorstore built in a previous container.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
