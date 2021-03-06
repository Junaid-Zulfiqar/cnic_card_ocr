import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from fastapi import File
from fastapi import UploadFile
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import pytesseract
pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
import shutil 
import random
try:
    from PIL import Image
except ImportError:
    import Image

# import easyocr
# reader = easyocr.Reader(['en','ur']) 

def ocr_sections(section):
    custom_config = r'--oem 3 --psm 6'
    ocr_result = pytesseract.image_to_string(section, config=custom_config)
    ocr_result = ocr_result.replace("\n\f","")
    try:
        ocr_result = ocr_result.split(":")
        ocr_result = ocr_result[1]
        ocr_result = ocr_result.replace("\n","")
    except Exception as e:
        print(ocr_result)
    ocr_result = "".join(ocr_result)     
    return ocr_result

# def urdu_ocr(img):
#     ocr_result = reader.readtext(img)
#     urdu_data = []
#     for i in ocr_result:
#         urdu_data.append(i[1])
#     urdu_data = " ".join(urdu_data)
#     return urdu_data

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
    img_resize = cv2.resize(img,(600,480))
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)    

    interpreter = tf.lite.Interpreter(model_path="cnic_identification_model.tflite")
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
        name = gray[111:147, 165:360]
        f_name = gray[213:247, 165:360]
        cnic_no = gray[378:406,165:308]
        D_B = gray[378:406,318:430]
        E_D = gray[435:465,318:430]


        name_text = ocr_sections(name)
        father_name = ocr_sections(f_name)
        cnic = ocr_sections(cnic_no)
        date_of_birth = ocr_sections(D_B)
        expiry_date = ocr_sections(E_D)

        return {
            "name":name_text,
            "father_name":father_name,
            "cnic_no":cnic,
            "date_of_birth":date_of_birth,
            "expiry_date":expiry_date
        }

    elif output_label == 'sim_back':
        address = gray[111:147, 165:360]
        address_text = urdu_ocr(address)

        return {
            "address":address_text
        }

    elif output_label == 'back':
        address = gray[49:154, 198:493]
        print(address)
        expiry_date = gray[303:338,152:264]
        # address_text = urdu_ocr(address)
        # print(address_text)
        # expiry_date_text = urdu_ocr(expiry_date)

        # return {
        #     "expiry_date":expiry_date_text,
        #     "address":address_text
        # }
        return {
            "data":"null",
        }