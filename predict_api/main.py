from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse 
import numpy as np
from PIL import Image, ImageOps
import os
import pickle
from io import BytesIO


model = pickle.load(open("./saved_model.pkl", 'rb'))

result_dict = {
    0 : "manga",
    1 : "roman",
    2 : "comics"
}


def proccess_image(image, img_size):
    img = ImageOps.grayscale(image.resize(img_size))
    array = np.asarray(img) / 255**2
    
    return np.array(array.flatten().tolist()).reshape((1,img_size[0]*img_size[1]))


def predict_infer(model, input_):
    return model.predict(input_).item()


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


app = FastAPI()

@app.post('/predict')
async def predict(file: UploadFile=File(...)):
    image = read_imagefile(await file.read())
    processed_image = proccess_image(image, (56,56))

    prediction = predict_infer(model= model, input_= processed_image)


    html_content = f"l'image est un: {result_dict[prediction]}"
    return HTMLResponse(content = html_content, status_code = 200)


#@app.get('/train')
    #def train     