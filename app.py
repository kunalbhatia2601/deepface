from fastapi import FastAPI, UploadFile, HTTPException
from deepface import DeepFace
import numpy as np
from PIL import Image
import io
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

@app.get("/")
async def root():
    return {"message": "Welcome to Emotion Detection API"}

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
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
