import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from fastapi import File
from fastapi import UploadFile
import cv2
import json 
import requests
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
try:
    from PIL import Image
except ImportError:
    import Image
from array import array
import os
from PIL import Image
import sys
import time

def delete_extra_files():
    dir = 'files/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

#load fastapi
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#get function
@app.get('/')
def read_root():
    return {"Hello": "World"}

#post function
@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    img = cv2.imread(f"files/{file.filename}")
    height, width, color_scheme = img.shape
    print(height,width)
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    interpreter = tf.lite.Interpreter(model_path="card_ocr_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    new_img = cv2.resize(img, (224, 224))
    print(new_img.shape)
    img_tensor = image.img_to_array(new_img)   
    input_shape = input_details[0]['shape']
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data[0])
    print(f"Label: {output_data[0].argmax()}")
    labels = ['back', 'front', 'sim_back', 'sim_front']
    print(labels[int(output_data[0].argmax())])
    output_label = labels[int(output_data[0].argmax())]
    if output_label == 'sim_front':
        url = "https://im-cnic-scan.cognitiveservices.azure.com/vision/v3.1/ocr?language=en"
        files = {'file': open(f"files/{file.filename}", 'rb')}
        headers = {
        'Ocp-Apim-Subscription-Key': '7ae9515f51394b3781220f87cd218f11'
        }
        response = requests.post(url, headers=headers, files=files)
        data = json.loads(response.text)
        regions = data.get("regions")
        all_data = []
        for rg in regions:
            lines = rg.get("lines")
            lines_list = []
            for line in lines:
                words = line.get("words")
                words_list = []
                for word in words:
                    print(word.get("text"))
                    words_list.append(word.get("text"))
                words_string = " ".join(words_list)
                lines_list.append(words_string)
            lines_string = "\n".join(lines_list)
            all_data.append(lines_string)
        results = "\n".join(all_data)
        father_name = ""
        username = ""
        date_of_birth = ""
        date_of_expiry = ""
        # Name
        names = []
        name = ""
        name_reg = re.compile(r'(?:Name)(?:\s+|)(\w+.*)') 
        match = name_reg.finditer(results)
        for matches in match:
            name = matches.group(1) 
            names.append(name)   
        username = names[0]
        print(f"name: {username}")
        father_name = names[-1]
        print(f"father name: {father_name}")
        #cnic
        cnic_reg = re.compile(r'((?:\d{5}|\d{4})(?:-|\s+|)\d{7}(?:-|\s+|)(?:\d|))')
        match = cnic_reg.finditer(results)
        cnic = ""
        for matches in match:
            cnic = matches.group(0)
        print(cnic) 
        #dates
        dates = []
        date_reg = re.compile(r'(\d{2}[.]\d{2}[.]\d{4})')
        match = date_reg.finditer(results)
        date = ""
        for matches in match:
            date = matches.group(0)
            dates.append(date)
        date_of_birth = dates[0]
        date_of_birth = date_of_birth.replace(".", "-")
        print(f"date of birth: {date_of_birth}")
        date_of_expiry = dates[-1]
        date_of_expiry = date_of_expiry.replace(".", "-")
        print(f"date of expiry: {date_of_expiry}")  
        

        # delete_extra_files()

        return {
            "name":username,
            "father_name":father_name,
            "cnic_no":cnic,
            "date_of_birth":date_of_birth,
            "expiry_date":date_of_expiry
        }

    elif output_label == 'sim_back':
        address = gray[111:147, 165:360]
        save_img(address)
        address_text = urdu_ocr('pics/pic.jpg')
        delete_extra_files()

        return {
            "address":address_text
        }

    elif output_label == 'back':
        address = gray[49:154, 198:493]
        save_img(address)
        expiry_date = gray[303:338,152:264]
        address_text = urdu_ocr('pics/pic.jpg')
        print(address_text)
        expiry_date_text = ocr_sections(expiry_date)
        delete_extra_files()

        return {
            "expiry_date":expiry_date_text,
            "address":address_text
        }
    elif output_label == 'front':
        name = gray[203:243,261:390]
        save_img(name)
        name_text = urdu_ocr('pics/pic.jpg')
        f_name = gray[285:325, 261:390]
        save_img(f_name)
        father_name = urdu_ocr('pics/pic.jpg')  
        cnic_no = gray[163:195,219:384]
        D_B = gray[378:416,271:390]
        # father_name = urdu_ocr(f_name)
        cnic = ocr_sections(cnic_no)
        date_of_birth = ocr_sections(D_B)

        # delete_extra_files()

        return {
            "name":name_text,
            "father_name":father_name,
            "cnic_no":cnic,
            "date_of_birth":date_of_birth,
        }

