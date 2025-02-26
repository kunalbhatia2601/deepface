# Emotion Detection API

This is a FastAPI application that provides real-time emotion detection using the DeepFace library.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### POST /detect-emotion

Upload an image to detect emotions. The API will return:

- Dominant emotion
- Scores for all emotions (angry, disgust, fear, happy, sad, surprise, neutral)

### Example Usage

```python
import requests

url = "http://localhost:8000/detect-emotion"
files = {"image": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Documentation

Access the API documentation at `http://localhost:8000/docs`
