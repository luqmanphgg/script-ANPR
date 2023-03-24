import torch
import cv2
import matplotlib.pyplot as plt 
import easyocr
import os, sys
import numpy as np
import time
import re
import glob
import keras_ocr
from paddleocr import PaddleOCR

plt.get_backend()  

#save output into text file
#sys.stdout = open('C:\S_T_A_L\yolov5_license_plate\output.txt','wt')

model = torch.hub.load('ultralytics/yolov5', 'custom', 'C:\S_T_A_L\yolov5_license_plate\last.pt')

ocr = PaddleOCR(use_angle_cls=True, lang='en')

input_path = 'C:\S_T_A_L\dataset\\aanother_100'


model.conf = 0.85

total_time = time.time()

#note: for ni untuk loop
for filename in os.listdir(input_path):
    start_time = time.time()
    image = cv2.imread(os.path.join(input_path,filename))
    print(filename)
    img = [image]

    alpha = 1.5
    run_model = model(img)

    try:
        element = run_model.xyxy[0][0]
        print("Object detected")  # try to access the element at index 0
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

    #(T, threshInv) = cv2.threshold(image_sharp, 200, 255, cv2.THRESH_BINARY_INV)


        adjusted = cv2.convertScaleAbs(image_sharp, alpha=alpha)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(adjusted,paragraph="True")
        print('')
        result
        print(result)
        type(result)
        print(result[0])
        firstitem = result[0]
        seconditem = firstitem[1]
        print(seconditem)
        uppero_l=seconditem.upper() #change small letter to upper letter
        onlynumber_letters = "".join(re.findall("[A-Z0-9]", uppero_l))
        print('number plate:', onlynumber_letters)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("")
        print("")

    #===============================================================================================================================       
    except IndexError:  # handle the IndexError exception
        print("Object is not detected")    
        print("")
        print("")
    #======ocr======================================================================================================================
'''
        result = ocr.ocr(adjusted, cls=False, det=False)
        ocr_res = result[0][0] #get first item in array
        ocr_no = ocr_res[0] #get first item in tuple in that array

        onlynumber_letter = "".join(re.findall("[A-Z0-9]", ocr_no))
        print('number plate:', onlynumber_letter)
        print("--- %s seconds ---" % (time.time() - start_time)) #print second run for each image
        print("")
        print("")
'''
seconds = time.time() - total_time
print("Total","--- %s seconds ---" % (seconds)) #print second run for each image
print("")
av_duration = seconds/100
print("Average duration ", "--- %s seconds ---" % (av_duration))
print("") 
print("Total","--- %s minutes ---" % (seconds // 60))
    



    

     

    




    




