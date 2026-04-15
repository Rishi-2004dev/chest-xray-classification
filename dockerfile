# 1. The Base Machine (OS + Python)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file FIRST
COPY requirements.txt .

# 4. Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your Model and API script
COPY main.py .
COPY cnn_xray_model.keras .

# 6. Open the port for the world to connect
EXPOSE 8000

# 7. The boot command (Start the server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]