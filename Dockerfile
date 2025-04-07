FROM python:3.10-slim

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit on container start
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.enableCORS=false"]