import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from fastapi import File
from fastapi import UploadFile
import cv2
import os
import requests
import re
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import pytesseract
pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
import pyocr
import pyocr.builders
import shutil 
import random
try:
    from PIL import Image
except ImportError:
    import Image
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()    


# import easyocr
# reader = easyocr.Reader(['en'])

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
def save_img(image):
    data = Image.fromarray(image)
    data.save('pics/pic.jpg')

def hsv(image):
    imgs = HSV(image)
    h_min = 34
    h_max = 140
    s_min = 0
    s_max = 67
    v_min = 0
    v_max = 175
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgs, lower, upper)
    data = Image.fromarray(mask)
    data.save('pics/pic.jpg')

def hsv_ocr_sections(section):
    hsv(section)
    section = cv2.imread('pics/pic.jpg')
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
    ocr_result = ocr_result.replace("\n","")
    if ocr_result.startswith(" "):
        ocr_result = ocr_result[1:]

    return ocr_result



def ocr_sections(section):
    section = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
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
    # ocr_result = ocr_result.replace("\n","")
    if ocr_result.startswith(" "):
        ocr_result = ocr_result[1:]


    return ocr_result

def pyocr_sections(section):
    section = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    data = Image.fromarray(section)
    data.save('pics/pic.jpg')
    img = Image.open('pics/pic.jpg')
    tools = pyocr.get_available_tools()[0]
    ocr_result = tools.image_to_string(
    img,
    lang='eng',
    builder=pyocr.builders.DigitBuilder())
    ocr_result = ocr_result.replace("\n\f","")
    try:
        ocr_result = ocr_result.split(":")
        ocr_result = ocr_result[1]
        ocr_result = ocr_result.replace("\n","")
    except Exception as e:
        print(ocr_result)
    ocr_result = "".join(ocr_result)   
    # ocr_result = ocr_result.replace("\n","")
    if ocr_result.startswith(" "):
        ocr_result = ocr_result[1:]

    return ocr_result

def keras_ocr_sections(section):
    section = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    data = Image.fromarray(section)
    data.save('pics/pic.jpg')
    images = [
    keras_ocr.tools.read(url) for url in [
        'pics/pic.jpg',
        ]
    ]
    prediction_groups = pipeline.recognize(images)
    print(prediction_groups[0][0])
    ocr_result = prediction_groups[0][0][0]
    if ocr_result.startswith(" "):
        ocr_result = ocr_result[1:]
    return ocr_result

def delete_extra_files():
    dir = 'pics/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    dir = 'files/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))    
# def easy_urdu_ocr(img):
#     ocr_result = reader.readtext(img)
#     urdu_data = []
#     for i in ocr_result:
#         print(i)
#         urdu_data.append(i[1])
#     urdu_data = " ".join(urdu_data)
#     return urdu_data

# def easy_ocr(img):
#     ocr_result = reader.readtext(img)
#     urdu_data = []
#     for i in ocr_result:
#         print(i)
#         urdu_data.append(i[1])
#     urdu_data = " ".join(urdu_data)
#     return urdu_data


def validate_name(name_text):
    name_text = name_text.lower().replace("name","")
    name_text = name_text.lower().replace("lame","")
    name_text = name_text.lower().replace("ame","")
    name_text = name_text.lower().replace("wame","")
    if "\n" in name_text:
        name_text = name_text.split("\n")
        name_text = name_text[1]
    name_text = re.sub(r"[-()\"‘#/@;:<>{}`+=~|.!?,“]", "", name_text)
    if len(name_text) > 21:
        name_text = ""
    return name_text

def validate_cnic(cnic_data):
    cnic_reg = re.compile(r'((?:\d{5}|\d{4})(?:-|\s+|)\d{7}(?:-|\s+|)(?:\d|))')
    cnic_data = cnic_data.lower().replace("b",'8')
    match = cnic_reg.finditer(cnic_data)
    cnic = ""
    for matches in match:
        cnic = matches.group(0)
        print(cnic)  
    return cnic     

def validate_fn(father_name):
    father_name = re.sub(r"[-()\"\'‘#/@;:<>{}`+=~|.!?,“]", "", father_name)
    father_name = father_name.lower().replace("name","")
    father_name = father_name.lower().replace("father name","")
    father_name = father_name.lower().replace("father name","")
    father_name = father_name.lower().replace("2","Z")
    if "\n" in father_name:
        father_name = father_name.split("\n")
        father_name = father_name[1]
    if len(father_name) > 21:
        father_name = ""
    return father_name

def validate_db(date_of_birth):
    date_reg = re.compile(r'(\d+(?:.|\s+|)\d{2}(?:.|\s+|)\d{4})')
    match = date_reg.finditer(date_of_birth)
    date_of_birth = ""
    for matches in match:
        date_of_birth = matches.group(0)
        print(date_of_birth)
    return date_of_birth        

def validate_ed(expiry_date):
    date_reg = re.compile(r'(\d+(?:.|\s+|)\d{2}(?:.|\s+|)\d{4})')
    match = date_reg.finditer(expiry_date)
    expiry_date = ""
    for matches in match:
        expiry_date = matches.group(0)
        print(expiry_date)
    expiry_date = expiry_date.replace(",", ".")    
    return expiry_date    



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
    img_resize = cv2.resize(img,(600,480))

    gray = img_resize    
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
        name = gray[95:150, 165:400]
        f_name = gray[198:255, 165:400]
        cnic_no = gray[360:414,165:308]
        D_B = gray[360:406,310:430]
        E_D = gray[420:465,310:430]

        #name
        name_text = ocr_sections(name)
        name_text = validate_name(name_text)
        if name_text == "":
            name_text = hsv_ocr_sections(name) 
            name_text = validate_name(name_text)
        if name_text == "":
            name = increase_brightness(name, value=100)
            name_text = ocr_sections(name)
            name_text = validate_name(name_text)
        if name_text == "":
            # name = increase_brightness(name, value=100)
            name_text = pyocr_sections(name)
            name_text = validate_name(name_text)
        if name_text == "":
            # name = increase_brightness(name, value=100)
            name_text = keras_ocr_sections(name)
            name_text = validate_name(name_text)        


    
        # if name_text == "":
        #     name_text = easy_ocr(name)
        #     name_text = validate_name(name_text)    
        
        #father_name
        father_name = ocr_sections(f_name)
        father_name = validate_fn(father_name)        
        if father_name== "":
            father_name = hsv_ocr_sections(f_name)
            father_name = validate_fn(father_name)
        if father_name == "":
            f_name = increase_brightness(f_name, value=100)
            father_name = ocr_sections(f_name)
            father_name = validate_fn(father_name) 
        if father_name == "":
            # f_name = increase_brightness(f_name, value=100)
            father_name = pyocr_sections(f_name)
            father_name = validate_fn(father_name)
        if father_name == "":
            father_name = keras_ocr_sections(f_name)
            # father_name = validate_fn(father_name)         
        # if father_name == "":
        #     father_name = easy_ocr(f_name)
        #     father_name = validate_fn(father_name)
        if father_name == "":
            father_name = keras_ocr_sections(f_name)
            father_name = validate_fn(father_name)
            

        #cnic
        cnic_data = ocr_sections(cnic_no)
        cnic = validate_cnic(cnic_data)
        if cnic == "":
            cnic = hsv_ocr_sections(cnic_no)
            cnic = validate_cnic(cnic)
        if cnic == "":
            cnic_no = increase_brightness(cnic_no, value=100)
            cnic = ocr_sections(cnic_no)
            cnic = validate_cnic(cnic)    
        if cnic == "":
            # cnic_no = increase_brightness(cnic_no, value=100)
            cnic = pyocr_sections(cnic_no)
            cnic = validate_cnic(cnic)    
               
        # if cnic == "":
        #     cnic = easy_ocr(cnic_no)
            # cnic = validate_cnic(cnic_data)
        if cnic == "":
            cnic = keras_ocr_sections(cnic_no)
            cnic = validate_cnic(cnic_data)    

        #date_of_birth
        date_of_birth = ocr_sections(D_B)
        date_of_birth = validate_db(date_of_birth)
        if date_of_birth == "":
            date_of_birth = hsv_ocr_sections(D_B)  
            date_of_birth = validate_db(date_of_birth)
        if date_of_birth == "":
            D_B = increase_brightness(D_B, value=100)
            date_of_birth = ocr_sections(D_B)
            date_of_birth = validate_db(date_of_birth)    
        if date_of_birth == "":
            date_of_birth = pyocr_sections(D_B)
            date_of_birth = validate_db(date_of_birth)  
        if date_of_birth == "":
            date_of_birth = keras_ocr_sections(D_B)
            date_of_birth = validate_db(date_of_birth)      

        # if date_of_birth == "":
        #     date_of_birth = easy_ocr(D_B)  
        #     date_of_birth = validate_db(date_of_birth)        
                  
        #expiry_date          
        expiry_date = ocr_sections(E_D)
        expiry_date = validate_ed(expiry_date)    
        if expiry_date == "":
            expiry_date = hsv_ocr_sections(E_D)
            expiry_date = validate_ed(expiry_date)
        if expiry_date == "":
            E_D = increase_brightness(E_D, value=100)
            expiry_date = ocr_sections(E_D)
            expiry_date = validate_ed(expiry_date)   
        if expiry_date == "":
            # E_D = increase_brightness(E_D, value=100)
            expiry_date = pyocr_sections(E_D)
            expiry_date = validate_ed(expiry_date)
        if expiry_date == "":
            expiry_date = keras_ocr_sections(E_D)
            expiry_date = validate_ed(expiry_date)          
        # if expiry_date == "":
        #     expiry_date = easy_ocr(E_D)
        #     expiry_date = validate_ed(expiry_date)    

        delete_extra_files()

        return {
            "name":name_text,
            "father_name":father_name,
            "cnic_no":cnic,
            "date_of_birth":date_of_birth,
            "expiry_date":expiry_date
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

        delete_extra_files()

        return {
            "name":name_text,
            "father_name":father_name,
            "cnic_no":cnic,
            "date_of_birth":date_of_birth,
        }

