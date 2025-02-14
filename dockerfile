# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from your project directory to /app in the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (assuming your app runs on port 5000, change if needed)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]