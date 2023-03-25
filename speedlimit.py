import cv2
import numpy as np
import easyocr

def read_speed_limit(image, x,y,w,h):

    crop_img = image[y:y+h,x:x+w]
    # cv2.imshow("croped",crop_img)
    reader = easyocr.Reader(['en'])
    speed_limit = reader.readtext(crop_img,paragraph=False,allowlist ='0123456789')
    return int(speed_limit[0][1])



