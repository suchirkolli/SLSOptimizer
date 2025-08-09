# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for some packages (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy code
COPY . /app

# Expose streamlit default port
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]