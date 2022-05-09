import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response, status
from typing import Optional
from pydantic import BaseModel
from fastapi import File
from fastapi import UploadFile
import cv2
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
try:
    from PIL import Image
except ImportError:
    import Image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

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
@app.post("/uploadfile",status_code=400)
async def create_upload_file(response: Response,file: UploadFile = File(...)):
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
    predicted_label = output_data[0].max()
    print(predicted_label)
    output_label = labels[int(output_data[0].argmax())]
    if predicted_label > 0.7:
        if output_label == 'sim_front':
            print(labels[int(output_data[0].argmax())])
            subscription_key = "e9675160b2fb424988da9558ff23d440"
            endpoint = "https://im-cnic-scan.cognitiveservices.azure.com/"

            computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
            print("===== Read File - remote =====")
            # Get an image with text

            # Call API with URL and raw response (allows you to get the operation location)
            read_image_path = f"files/{file.filename}"
            # Open the image
            read_image = open(read_image_path, "rb")

            # Call API with image and raw response (allows you to get the operation location)
            read_response = computervision_client.read_in_stream(read_image, raw=True)
            # Get the operation location (URL with ID as last appendage)
            read_operation_location = read_response.headers["Operation-Location"]
            # Grab the ID from the URL
            operation_id = read_operation_location.split("/")[-1]

            # Call the "GET" API and wait for it to retrieve the results 
            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

            # Print the detected text, line by line
            # print(read_result)
            total_results = []
            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        # print(line.text)
                        total_results.append(line.text)
                        # print(line.bounding_box)
            results = "\n".join(total_results)
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
            if len(names) > 1:  
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
            if len(dates) > 1:
                date_of_birth = dates[0]
                date_of_birth = date_of_birth.replace(".", "-")
                print(f"date of birth: {date_of_birth}")
                date_of_expiry = dates[-1]
                date_of_expiry = date_of_expiry.replace(".", "-")
                print(f"date of expiry: {date_of_expiry}")  

            if cnic == "" and name == "":
                return {
                    response
                }
            else:
                response.status_code = 200


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
    else:
        delete_extra_files()
        return {
                response
                }


