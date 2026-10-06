# Base image: official Python 3.12 slim build. "slim" has fewer pre-installed
# OS packages than the default, keeping our image smaller, since we control
# exactly which system libraries we actually need below.
FROM python:3.12-slim

# Where our application code will live inside the container.
WORKDIR /app

# System-level dependencies some of our Python packages need under the hood.
# PaddleOCR/OpenCV (used internally by paddleocr) require these even though
# they're invisible from Python's side — this is exactly the kind of gap
# that only reveals itself when running on a truly clean machine.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the dependency list first, not the whole project yet.
# This is the layer-caching principle from our concept discussion: as long
# as requirements.txt doesn't change, Docker reuses this expensive layer
# (installing PyTorch, transformers, etc.) on every rebuild, instead of
# reinstalling everything just because we edited one Python file.
COPY requirements.txt .

# Install Python dependencies. --no-cache-dir keeps the image smaller by not
# retaining pip's download cache (irrelevant inside a container we won't
# reuse for further pip installs).
RUN pip install --no-cache-dir -r requirements.txt

# NOW copy the actual application code — this layer changes often (every
# time we edit a .py file), but because it's last, it doesn't invalidate
# the expensive dependency-install layer above it.
COPY app/ ./app/
COPY models/ ./models/

# The port our FastAPI app listens on inside the container.
EXPOSE 8000

# The command that runs when a container starts from this image.
# No --reload here: that's a development convenience, not something a
# deployed container should do.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]