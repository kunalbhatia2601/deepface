from fastapi import FastAPI, UploadFile, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pathlib
from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Emotion Detection API",
    description="Real-time face emotion detection using DeepFace",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=pathlib.Path(__file__).parent / "static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return pathlib.Path("live.html").read_text()

@app.get("/live", response_class=HTMLResponse)
async def get_live():
    return pathlib.Path("live.html").read_text()

@app.post("/detect-emotion")
async def detect_emotion(image: UploadFile):
    try:
        # Read and validate the uploaded image
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Convert uploaded image to format DeepFace can process
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
        
        # Analyze the image using DeepFace
        result = DeepFace.analyze(img_array, 
                                actions=['emotion'],
                                enforce_detection=False)
        
        # Extract emotion data
        if isinstance(result, list):
            result = result[0]
            
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        print("Emotions:", emotions)
        print("Dominant emotion:", dominant_emotion)
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-face")
async def analyze_face(image: UploadFile):
    try:
        # Read and validate the uploaded image
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Convert uploaded image to format DeepFace can process
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
        
        # Analyze the image using DeepFace for multiple attributes
        result = DeepFace.analyze(img_array, 
                                actions=['emotion', 'age', 'gender', 'race'],
                                enforce_detection=False)
        
        # Extract data
        if isinstance(result, list):
            result = result[0]
        print("Result:", result)
        return {
            "emotion": {
                "dominant_emotion": result['dominant_emotion'],
                "scores": result['emotion']
            },
            "age": result['age'],
            "gender": {
                "gender": result['gender'],
                "scores": result['gender']
            },
            "race": {
                "dominant_race": result['dominant_race'],
                "scores": result['race']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the base64 encoded image from the client
            data = await websocket.receive_text()
            
            # Process the image
            try:
                # Decode the base64 string to image
                encoded_data = data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Convert BGR (OpenCV) to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Analyze face
                result = DeepFace.analyze(img_rgb, 
                                      actions=['emotion', 'age', 'gender', 'race'],
                                      enforce_detection=False)
                
                if isinstance(result, list):
                    result = result[0]
                
                # Prepare and send response
                analysis_result = {
                    "emotion": {
                        "dominant_emotion": result['dominant_emotion'],
                        "scores": result['emotion']
                    },
                    "age": result['age'],
                    "gender": {
                        "gender": result['gender'],
                        "scores": result['gender']
                    },
                    "race": {
                        "dominant_race": result['dominant_race'],
                        "scores": result['race']
                    }
                }
                
                await websocket.send_json(analysis_result)
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
