# Use an official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (modify if your app runs on another port)
EXPOSE 8000

CMD ["python", "app/main.py"]
