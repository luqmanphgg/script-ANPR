import torch
import cv2
import matplotlib.pyplot as plt 
import easyocr
import os
import numpy as np
import re
import pytesseract
from paddleocr import PaddleOCR

plt.get_backend()  

model = torch.hub.load('ultralytics/yolov5', 'custom', 'C:\S_T_A_L\yolov5_license_plate\last.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

image = cv2.imread('C:/S_T_A_L/dataset/rebounddataset/license (106).jpg')
img = [image]

model.conf = 0.8

run_model = model(img)
try:
    element = run_model.xyxy[0][0]
    print("Object detected") 
    bbox_coord = run_model.xyxy[0][0]
    bbox = []

    #masukkan coordinate bbox dalam array bbox
    for bound in bbox_coord:
      bbox.append(int(bound.item()))

    #get only 4 first item from array bbox
    bbox = bbox[:4]

    new_image = image.copy()
    cv2.rectangle(new_image, (bbox[0],bbox[1],bbox[2],bbox[3]),(255,0,0), 5)

    cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)


    #resize
    resize_cropped = cv2.resize(
        img_rgb, None, fx=2, fy=2,
        interpolation = cv2.INTER_CUBIC
        )

    #===============image processing==============================================================================================================
    alpha = 1.5
   #turn it into gray
    grayscale_resize_test_license_plate = cv2.cvtColor(resize_cropped, cv2.COLOR_BGR2GRAY)
    
    #apply gaussian blur to the image to decrease noise
    gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)
    
    #remove noise
    noiseless_image_bw = cv2.fastNlMeansDenoising(gaussian_blur_license_plate, None, 20, 7, 21)
    
    #sharp the image
    kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])

    image_sharp = cv2.filter2D(noiseless_image_bw, ddepth=-1, kernel=kernel)


    #=========ocr=================================================================================================================================

    result = ocr.ocr(image_sharp, cls=False, det=False)
    ocr_res = result[0][0] #get first item in array
    ocr_no = ocr_res[0] #get first item in tuple in that array

    onlynumber_letter = "".join(re.findall("[A-Z0-9]", ocr_no))
    print('number plate:', onlynumber_letter)
#=============================================================================================================================================

except IndexError:  
    print("Object is not detected")

  



