from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
import numpy as np
from keras import models

model = models.load_model('./model.h5')
app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file)
    
    if img.mode != 'RGB':
        return {"class_name": "Gambar tidak diperbolehkan"}
    
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = ['Bacterial Blight', 'Blast', 'Brownspot', 'Tungro'][class_index]
    class_prediction = float(prediction[0][class_index]) * 100  
    return {"class_name": class_name, "prediction": f"{class_prediction:.2f}%"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
